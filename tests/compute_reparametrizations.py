import os
import sys
import time


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
from jaxlip.linear import SpectralLinear, OrthoLinear, ParametrizedLinear, l2_normalize
from jaxlip.newton_schulz import orthogonalize
import jax
from jax.numpy.linalg import norm
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from jaxlip.activations import GroupSort2
from jaxlip.reparametrizer import (
    collect_buckets,
    parametrize_vmapped_cached,
    parametrize_from_params_cached,
)
from jaxlip.utils import inject_biases


# Sync + timing helpers
def _sync(tree):
    """Block until device work is done; safe for pytrees."""
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None,
        tree,
    )


def bench(fn, *args, warmup=2, repeat=20, label=""):
    """Time a function: first (includes compile) and steady-state average."""
    t0 = time.perf_counter()
    y = fn(*args)
    _sync(y)
    first = time.perf_counter() - t0

    for _ in range(warmup):
        y = fn(*args)
    _sync(y)

    t0 = time.perf_counter()
    for _ in range(repeat):
        y = fn(*args)
    _sync(y)
    steady = (time.perf_counter() - t0) / repeat

    print(f"[{label}] first={first * 1e3:.2f} ms   steady/iter={steady * 1e3:.2f} ms")
    return first, steady


class TestVMappedReparams(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.key = jax.random.PRNGKey(0)
        self.rngs = nnx.Rngs(self.key)
        self.model = self._create_test_model()
        self.x = jax.random.uniform(self.key, (16, 10))
        self.buckets = collect_buckets(self.model)
        self.reparam_weights = parametrize_vmapped_cached(self.buckets)

    def _create_test_model(self):
        """Create test model with mixed parametrizations."""

        class Model(nnx.Module):
            def __init__(self, rngs):
                self.lin0 = ParametrizedLinear(
                    10, 20, parametrization="spectral", rngs=rngs
                )
                self.lin1 = ParametrizedLinear(
                    20, 20, parametrization="spectral", rngs=rngs
                )
                self.lin2 = ParametrizedLinear(
                    20, 20, parametrization="spectral", rngs=rngs
                )
                self.lin3 = ParametrizedLinear(
                    20, 20, parametrization="ortho", rngs=rngs
                )
                self.lin4 = ParametrizedLinear(
                    20, 20, parametrization="ortho", rngs=rngs
                )
                self.lin5 = ParametrizedLinear(
                    20, 2, parametrization="spectral", rngs=rngs
                )
                self.act = GroupSort2()

            def __call__(self, x, ws=None):
                x = self.lin0(x, ws)
                x = self.lin1(x, ws)
                x = self.lin2(x, ws)
                x = self.lin3(x, ws)
                x = self.lin4(x, ws)
                x = self.act(x)
                x = self.lin5(x, ws)
                return x

        return Model(self.rngs)

    def test_bucket_collection(self):
        """Test that bucket collection works correctly."""
        buckets = collect_buckets(self.model)

        # Check that we have the expected parametrization types
        self.assertIn("spectral", buckets)
        self.assertIn("ortho", buckets)

        # Check that buckets contain expected number of layers
        spectral_count = sum(len(items) for items in buckets["spectral"].values())
        ortho_count = sum(len(items) for items in buckets["ortho"].values())

        self.assertEqual(spectral_count, 4)  # lin0, lin1, lin2, lin5
        self.assertEqual(ortho_count, 2)  # lin3, lin4

    def test_vmap_reparametrization_correctness(self):
        """Test that vmapped reparametrization produces correct results."""
        outs_normal = self.model(self.x)
        outs_vmap = self.model(self.x, self.reparam_weights)
        _sync((outs_normal, outs_vmap))

        diff = jnp.abs(outs_vmap - outs_normal)
        max_abs = float(jnp.max(diff))
        rel = float(max_abs / (float(jnp.max(jnp.abs(outs_normal))) + 1e-12))

        if is_verbose():
            print(f"[compare] max_abs={max_abs:.3e}, rel_max={rel:.3e}")
            print(
                f"[compare] allclose (rtol=3e-3, atol=3e-3): {bool(jnp.allclose(outs_vmap, outs_normal, rtol=3e-3, atol=3e-3))}"
            )

        self.assertTrue(
            jnp.allclose(outs_vmap, outs_normal, rtol=3e-3, atol=3e-3),
            msg="Outputs are not close enough!",
        )

    def test_performance_benchmarking(self):
        """Test performance comparison between normal and vmap execution."""
        f_normal = lambda x: self.model(x)
        f_vm = lambda x: self.model(x, self.reparam_weights)

        if is_verbose():
            bench(f_normal, self.x, label="normal (eager)")
            bench(f_vm, self.x, label="vmap-weights (eager)")

        jit_normal = jax.jit(lambda x: self.model(x))
        jit_vm = jax.jit(lambda x: self.model(x, self.reparam_weights))

        with jax.default_matmul_precision("highest"):
            if is_verbose():
                bench(jit_normal, self.x, label="normal (jit, highest precision)")
                bench(jit_vm, self.x, label="vmap-weights (jit, highest precision)")

            # Just verify they run without error
            _ = jit_normal(self.x)
            _ = jit_vm(self.x)

    def test_gradient_computation(self):
        """Test that gradients can be computed through vmap reparametrization."""
        params_init = self._create_params_dict()
        target = jnp.zeros((self.x.shape[0], 2), dtype=jnp.float32)

        def loss_from_params(params, x):
            ws = parametrize_from_params_cached(self.buckets, params)
            inject_biases(self.model, params, ws)
            y = self.model(x, ws)
            return jnp.mean((y - target) ** 2)

        opt = optax.adam(1e-3)
        opt_state = opt.init(params_init)

        @jax.jit
        def train_step(params, opt_state, x):
            loss, grads = jax.value_and_grad(loss_from_params)(params, x)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Test gradient computation
        params, opt_state, loss0 = train_step(params_init, opt_state, self.x)
        _sync(loss0)
        params, opt_state, loss1 = train_step(params, opt_state, self.x)
        _sync(loss1)

        if is_verbose():
            print(
                f"[adam] step0 loss={float(loss0):.6f}  step1 loss={float(loss1):.6f}  finite={jnp.isfinite(loss1)}"
            )

        self.assertTrue(jnp.isfinite(loss1), msg="Loss after Adam step is not finite!")
        self.assertTrue(loss1 < loss0, msg="Loss after Adam step did not decrease!")

    def test_parameter_evolution_and_constraints(self):
        """Test that parameters evolve during training while maintaining constraints."""
        params_init = self._create_params_dict()
        target = jnp.zeros((self.x.shape[0], 2), dtype=jnp.float32)

        def loss_from_params(params, x):
            ws = parametrize_from_params_cached(self.buckets, params)
            inject_biases(self.model, params, ws)
            y = self.model(x, ws)
            return jnp.mean((y - target) ** 2)

        opt = optax.adam(1e-3)
        opt_state = opt.init(params_init)

        @jax.jit
        def train_step(params, opt_state, x):
            loss, grads = jax.value_and_grad(loss_from_params)(params, x)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        # Train for some steps and capture weights
        params = params_init
        for _ in range(500):
            params, opt_state, loss = train_step(params, opt_state, self.x)
        old_ws = parametrize_from_params_cached(self.buckets, params)

        # Train for more steps
        for _ in range(500):
            params, opt_state, loss = train_step(params, opt_state, self.x)
        new_ws = parametrize_from_params_cached(self.buckets, params)

        # Test parameter evolution and constraint maintenance
        ortho_uids = self._get_ortho_uids()

        for (k1, v1), (k2, v2) in zip(old_ws.items(), new_ws.items()):
            self.assertTrue(k1 == k2)
            self.assertTrue(
                (v1 != v2).any(), msg="Parameters should change during training"
            )

            if v1.ndim == 2:
                S1 = jnp.linalg.svd(v1, compute_uv=False)
                S2 = jnp.linalg.svd(v2, compute_uv=False)

                if is_verbose():
                    if v1.shape[0] == v1.shape[1] and k1 in ortho_uids:
                        print("Min sv old_params: ", jnp.min(S1))
                        print("Min sv new_params: ", jnp.min(S2))
                    print("Max sv old_params: ", jnp.max(S1))
                    print("Max sv new_params: ", jnp.max(S2))

                # Test spectral norm constraint (should be ≤ 1)
                self.assertTrue(
                    jnp.max(S1) < 1.01,
                    msg="Spectral norm constraint violated for old weights",
                )
                self.assertTrue(
                    jnp.max(S2) < 1.01,
                    msg="Spectral norm constraint violated for new weights",
                )

                # Test orthogonal constraint (singular values should be ≈ 1)
                if v1.shape[0] == v1.shape[1] and k1 in ortho_uids:
                    self.assertTrue(
                        jnp.min(S1) > 0.99,
                        msg="Orthogonal constraint violated for old weights",
                    )
                    self.assertTrue(
                        jnp.min(S2) > 0.99,
                        msg="Orthogonal constraint violated for new weights",
                    )

    def test_from_params_cached_consistency(self):
        """Test that parametrize_from_params_cached works consistently."""
        params_dict = self._create_params_dict()

        # Test multiple calls produce same results
        ws1 = parametrize_from_params_cached(self.buckets, params_dict)
        ws2 = parametrize_from_params_cached(self.buckets, params_dict)

        for k in ws1.keys():
            self.assertTrue(
                jnp.allclose(ws1[k], ws2[k]), msg=f"Inconsistent results for key {k}"
            )

    def _create_params_dict(self):
        """Create parameters dictionary for training."""
        return {
            **{
                uid: w.value
                for name, by_sig in self.buckets.items()
                for _, items in by_sig.items()
                for (uid, w, _h) in items
            },
            **{
                f"b:{m._uid}": m.b.value
                for _, m in self.model.iter_modules()
                if isinstance(m, ParametrizedLinear) and m.bias
            },
        }

    def _get_ortho_uids(self):
        """Get UIDs of orthogonal layers."""
        return [
            m._uid
            for _, m in self.model.iter_modules()
            if hasattr(m, "parametrization") and m.parametrization == "ortho"
        ]


if __name__ == "__main__":
    unittest.main()

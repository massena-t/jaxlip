import os
import sys


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import unittest
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from unittest.mock import patch

from jaxlip.zbp.assign import assign_owners_round_robin, _walk_modules
from jaxlip.zbp.layout import assign_reparam_groups
from jaxlip.zbp.pack import build_reparam_pack, apply_with_reparam
from jaxlip.zbp.distributed_op import reparam_distributed, reparam_distributed_vmap
from jaxlip.zbp.base import ReparametrizedModule
from jaxlip.linear import DistributedOrthoLinear
from models.mixer import MLPMixer
from utils.utils import get_model


def shard_batch(batch, num_devices):
    return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch)


class TestDistributedOrthoLinear(unittest.TestCase):
    def setUp(self):
        self.num_devices = jax.local_device_count()
        if self.num_devices < 2:
            self.skipTest("Multi-GPU tests require at least 2 devices")

    def test_distributed_ortho_linear_creation(self):
        """Test that DistributedOrthoLinear can be created and has correct attributes"""
        rng = nnx.Rngs(params=jax.random.key(0))

        layer = DistributedOrthoLinear(10, 5, rngs=rng)

        # Check basic attributes
        self.assertEqual(layer.din, 10)
        self.assertEqual(layer.dout, 5)
        self.assertEqual(layer.w.shape, (10, 5))
        self.assertFalse(layer.cached)
        self.assertEqual(layer.owner, -1)  # Not assigned yet

        # Check it's a ReparametrizedModule
        self.assertIsInstance(layer, ReparametrizedModule)

        if is_verbose():
            print(f"Created DistributedOrthoLinear: {layer.din} -> {layer.dout}")

    def test_assign_owners_round_robin(self):
        """Test that owners are assigned correctly to DistributedOrthoLinear layers"""
        rng = nnx.Rngs(params=jax.random.key(0))

        # Create a model with multiple DistributedOrthoLinear layers
        model = nnx.Sequential(
            DistributedOrthoLinear(10, 8, rngs=rng),
            *[
                DistributedOrthoLinear(8, 6, rngs=rng)
                for _ in range(self.num_devices - 1)
            ],
        )

        # Assign owners
        assign_owners_round_robin(model)

        # Collect all reparametrized modules
        reparam_modules = []
        for m in _walk_modules(model):
            if isinstance(m, ReparametrizedModule):
                reparam_modules.append(m)

        # Check that owners are assigned in round-robin fashion
        expected_owners = [i % self.num_devices for i in range(len(reparam_modules))]
        actual_owners = [m.owner for m in reparam_modules]

        self.assertEqual(
            actual_owners,
            expected_owners,
            f"Owners should be assigned round-robin: expected {expected_owners}, got {actual_owners}",
        )

        if is_verbose():
            print(
                f"DistributedOrthoLinear owners: {actual_owners} across {self.num_devices} devices"
            )

    def test_assign_reparam_groups(self):
        """Test that reparam groups are assigned correctly to DistributedOrthoLinear layers"""
        rng = nnx.Rngs(params=jax.random.key(0))

        # Create model with layers of different shapes
        model = nnx.Sequential(
            DistributedOrthoLinear(10, 8, rngs=rng),  # Group 0
            DistributedOrthoLinear(
                10, 8, rngs=rng
            ),  # Group 0 (same shape, same function)
            DistributedOrthoLinear(8, 6, rngs=rng),  # Group 1 (different shape)
            DistributedOrthoLinear(8, 6, rngs=rng),  # Group 1 (different shape)
            DistributedOrthoLinear(1, 3, rngs=rng),  # Group 2 (different shape)
        )

        assign_reparam_groups(model)

        # Collect modules and their group assignments
        modules_with_groups = []
        for m in _walk_modules(model):
            if isinstance(m, ReparametrizedModule) and hasattr(m, "w"):
                modules_with_groups.append((m, m._zbp_gid, m._zbp_idx, m.w.shape))

        seen_gid = {}
        seen_idx = {}

        for info in modules_with_groups:
            mod, gid, idx, shape = info
            if shape in seen_gid.keys():
                assert seen_gid[shape] == gid
            if shape in list(seen_idx.keys()):
                assert idx not in seen_idx[shape]
                seen_idx[shape].append(idx)
            else:
                seen_gid[shape] = gid
                seen_idx[shape] = [idx]
            self.assertEqual(mod.owner, -1)

    def test_build_reparam_pack_with_distributed_ortho(self):
        """Test that build_reparam_pack works with DistributedOrthoLinear layers"""
        rng = nnx.Rngs(params=jax.random.key(0))

        model = nnx.Sequential(
            DistributedOrthoLinear(6, 4, rngs=rng),
            DistributedOrthoLinear(6, 4, rngs=rng),  # Same group
            DistributedOrthoLinear(4, 3, rngs=rng),  # Different group
        )

        # Setup distributed reparam
        assign_owners_round_robin(model)
        assign_reparam_groups(model)

        # Collect modules and their group assignments
        # Build reparam pack with distributed=False first (easier to test)
        Qs_groups_local = build_reparam_pack(model, distributed=False)

        # Check structure
        self.assertEqual(len(Qs_groups_local), 2, "Should have 2 groups")
        self.assertEqual(
            Qs_groups_local[1].shape,
            (2, 6, 4),
            "First group should have 2 modules of shape (6, 4)",
        )
        self.assertEqual(
            Qs_groups_local[0].shape,
            (1, 4, 3),
            "Second group should have 1 module of shape (4, 3)",
        )

        # Check orthogonality
        for i, Qs in enumerate(Qs_groups_local):
            for j in range(Qs.shape[0]):
                Q = Qs[j]
                # For DistributedOrthoLinear, check that Q is orthogonal
                if Q.shape[0] >= Q.shape[1]:  # Tall matrix
                    should_be_identity = Q.T @ Q
                    identity = jnp.eye(Q.shape[1])
                    self.assertTrue(
                        jnp.allclose(should_be_identity, identity, atol=1e-2),
                        f"Group {i}, module {j} should produce orthogonal matrix",
                    )
                    if is_verbose:
                        print(f"{jnp.linalg.matrix_norm(Q, ord=2)}")
                else:  # Wide matrix
                    should_be_identity = Q @ Q.T
                    identity = jnp.eye(Q.shape[0])
                    self.assertTrue(
                        jnp.allclose(should_be_identity, identity, atol=1e-2),
                        f"Group {i}, module {j} should produce orthogonal matrix",
                    )
                    if is_verbose:
                        print(f"{jnp.linalg.matrix_norm(Q, ord=2)}")

        if is_verbose():
            print(
                f"DistributedOrthoLinear reparam pack shapes: {[Q.shape for Q in Qs_groups_local]}"
            )

    def test_distributed_ortho_lipschitz_constraint(self):
        """Test that DistributedOrthoLinear maintains 1-Lipschitz constraint"""
        rng = nnx.Rngs(params=jax.random.key(0))

        # Create layers of different shapes
        layers = [
            DistributedOrthoLinear(8, 6, rngs=rng),  # Tall
            DistributedOrthoLinear(6, 8, rngs=rng),  # Wide
            DistributedOrthoLinear(5, 5, rngs=rng),  # Square
        ]

        for layer in layers:
            assign_owners_round_robin(layer)
            assign_reparam_groups(layer)

            # Test data
            key = jax.random.key(123)
            x1 = jax.random.normal(key, (3, layer.din))
            x2 = jax.random.normal(jax.random.split(key)[0], (3, layer.din))

            # Get reparam pack and apply
            Qs_groups = build_reparam_pack(layer, distributed=False)
            y1 = apply_with_reparam(layer, x1, Qs_groups)
            y2 = apply_with_reparam(layer, x2, Qs_groups)

            # Check Lipschitz constraint: ||f(x1) - f(x2)|| <= ||x1 - x2||
            input_diff = jnp.linalg.norm(x1 - x2, axis=-1)
            output_diff = jnp.linalg.norm(y1 - y2, axis=-1)

            lipschitz_satisfied = jnp.all(
                output_diff <= input_diff + 1e-4
            )  # small tolerance

            self.assertTrue(
                lipschitz_satisfied,
                f"DistributedOrthoLinear ({layer.din}->{layer.dout}) should satisfy 1-Lipschitz constraint",
            )

            if is_verbose():
                max_ratio = jnp.max(output_diff / (input_diff + 1e-8))
                print(
                    f"DistributedOrthoLinear ({layer.din}->{layer.dout}) max Lipschitz ratio: {max_ratio:.6f}"
                )

    def test_distributed_ortho_multi_gpu_training(self):
        """Test DistributedOrthoLinear training across multiple GPUs (or 1) and
        assert spectral norms of reparam weights are ≤ 1 + tol."""
        rng = nnx.Rngs(params=jax.random.key(0))
        batch_size = 16
        num_devices = jax.local_device_count()
        assert batch_size % num_devices == 0

        # Small helper to shard (N, ...) -> (devices, N/devices, ...)
        def shard_batch(arr):
            return arr.reshape((num_devices, -1) + arr.shape[1:])

        # Model: three stacked DistributedOrthoLinear layers
        model = nnx.Sequential(
            DistributedOrthoLinear(10, 8, rngs=rng),
            DistributedOrthoLinear(8, 6, rngs=rng),
            DistributedOrthoLinear(6, 4, rngs=rng),
        )

        # Owner + (fn,shape)-group ids for reparam packing
        assign_owners_round_robin(model)
        assign_reparam_groups(model)

        optimizer = nnx.Optimizer(model, optax.adam(1e-3))

        state_axes = nnx.StateAxes(
            {
                nnx.Param: None,
                nnx.Cache: None,
                nnx.BatchStat: None,
            }
        )

        # A tiny pure forward that threads reparam overrides through the Sequential
        def forward_with_overrides(m, x, overrides):
            out = x
            # nnx.Sequential exposes the layers as .layers (in order)
            for layer in m.layers:
                out = layer(out, reparam_overrides=overrides)
            return out

        # pmapped training step (axis bound; gradients flow)
        @nnx.pmap(
            in_axes=(state_axes, None, 0, 0),
            out_axes=(state_axes, None, 0),
            axis_name="device",
        )
        def train_step(m, opt, x, y):
            def loss_fn(mm):
                Qs_groups = build_reparam_pack(mm, distributed=True)  # compute-on-owner
                preds = forward_with_overrides(mm, x, Qs_groups)
                return jnp.mean((preds - y) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            grads = jax.lax.pmean(grads, axis_name="device")
            opt.update(grads)
            return m, opt, loss

        # pmapped eval loss (bind axis; pure path)
        @nnx.pmap(in_axes=(state_axes, 0, 0), out_axes=0, axis_name="device")
        def eval_loss(m, x, y):
            Qs_groups = build_reparam_pack(m, distributed=True)
            preds = forward_with_overrides(m, x, Qs_groups)
            return jnp.mean((preds - y) ** 2)

        # pmapped spectral norm check (bind axis; no grads)
        token = jnp.arange(num_devices)  # mapped dummy so pmap has a non-None in_axes

        @nnx.pmap(in_axes=(state_axes, 0), out_axes=0, axis_name="device")
        def max_spectral_norm(m, _tok):
            Qs_groups = build_reparam_pack(m, distributed=True)
            if not Qs_groups:
                return jnp.array(0.0, dtype=jnp.float32)

            def group_norms(Qs):  # Qs: [G, din, dout]
                return jax.vmap(lambda Q: jnp.linalg.norm(Q, ord=2))(Qs)  # [G]

            norms = [group_norms(Qs) for Qs in Qs_groups]
            return jnp.max(jnp.concatenate(norms))

        # Data
        key = jax.random.key(789)
        x = jax.random.uniform(key, (batch_size, 10))
        y = jax.random.uniform(key, (batch_size, 4))
        x_sh, y_sh = shard_batch(x), shard_batch(y)

        # Initial checks
        init_loss = eval_loss(model, x_sh, y_sh).mean()
        spec_before = max_spectral_norm(model, token).mean()

        tol = 1e-3
        self.assertLessEqual(
            float(spec_before),
            1.0 + tol,
            f"spectral norm before > 1+tol: {float(spec_before):.6f}",
        )

        # Train a few steps
        for _ in range(5):
            model, optimizer, loss = train_step(model, optimizer, x_sh, y_sh)

        # Final checks
        fin_loss = eval_loss(model, x_sh, y_sh).mean()
        spec_after = max_spectral_norm(model, token).mean()

        if is_verbose():
            print(f"spectral norm before: ", float(spec_before))
            print(f"spectral norm after: ", float(spec_after))
            print(f"{fin_loss=}, {init_loss=}")

        self.assertTrue(jnp.isfinite(loss).all(), "Loss should be finite")
        self.assertLessEqual(
            float(spec_after),
            1.0 + tol,
            f"spectral norm after > 1+tol: {float(spec_after):.6f}",
        )
        self.assertTrue(
            fin_loss <= init_loss + 1e-3,
            f"Loss should decrease or stay ~stable (init {float(init_loss):.6f} -> final {float(fin_loss):.6f})",
        )

    def test_distributed_ortho_caching(self):
        """Test parameter caching functionality with DistributedOrthoLinear"""
        rng = nnx.Rngs(params=jax.random.key(0))

        layer = DistributedOrthoLinear(6, 4, rngs=rng)
        assign_owners_round_robin(layer)
        assign_reparam_groups(layer)

        # Test data
        key = jax.random.key(555)
        x = jax.random.uniform(key, (3, 6))

        # Get reparam pack for consistent testing
        Qs_groups = build_reparam_pack(layer, distributed=False)

        # Output without caching
        self.assertFalse(layer.cached, "Layer should start uncached")
        output1 = apply_with_reparam(layer, x, Qs_groups)

        # Cache parameters
        layer._cache_params()
        self.assertTrue(layer.cached, "Layer should be cached after _cache_params")

        # Output with caching (should use cached values, not reparam_overrides)
        output2 = layer(x, reparam_overrides=Qs_groups)

        # Uncache parameters
        layer._uncache()
        self.assertFalse(layer.cached, "Layer should be uncached after _uncache")

        # Output after uncaching
        output3 = apply_with_reparam(layer, x, Qs_groups)

        assert (layer.cache == Qs_groups[0][0]).all()

        # Outputs should be consistent
        self.assertTrue(
            jnp.allclose(output1, output3, atol=1e-5),
            "Output should be consistent before and after caching cycle",
        )

        if is_verbose():
            print(f"DistributedOrthoLinear caching test passed")


if __name__ == "__main__":
    unittest.main()

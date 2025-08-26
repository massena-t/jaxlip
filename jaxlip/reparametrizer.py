from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Tuple, List
import time

import jax
import jax.numpy as jnp
from flax import nnx
from jaxlip.linear import ParametrizedLinear, DICT_PARAMS
from jaxlip.utils import (
    _stack_tree,
    _array_sig,
    _tree_sig_structure,
    _tree_all,
    _all_equal_pytrees,
)


def collect_buckets(root: nnx.Module):
    """
    Returns:
      buckets[name][(shape, dtype)] = list of (uid, weight, hparams_tree)

    where:
      - `name` is a key into DICT_PARAMS,
      - `shape` is the parameter shape (tuple, no batch dim),
      - `hparams_tree` is any pytree of per-layer constants (or {} if unused).
    """
    buckets: Dict[
        str, Dict[Tuple[Tuple[int, ...], Any], List[Tuple[str, Any, Any]]]
    ] = defaultdict(lambda: defaultdict(list))

    visited: set[int] = set()

    def _walk(mod: nnx.Module):
        if id(mod) in visited:
            return
        visited.add(id(mod))

        if isinstance(mod, ParametrizedLinear):
            name = mod.parametrization  # key for DICT_PARAMS
            w = mod.w  # raw parameter tensor
            hparams = getattr(mod, "reparam_hparams", {})  # {} if not needed
            buckets[name][(w.shape, w.dtype)].append((mod._uid, w, hparams))

        # Recurse through typical containers
        for _, value in vars(mod).items():
            if isinstance(value, nnx.Module):
                _walk(value)
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, nnx.Module):
                        _walk(v)
            elif isinstance(value, (list, tuple)) and not isinstance(
                value, (str, bytes)
            ):
                for v in value:
                    if isinstance(v, nnx.Module):
                        _walk(v)

    _walk(root)
    return buckets


_VMAP_CACHE: Dict[Tuple, Any] = {}


def clear_vmap_cache():
    _VMAP_CACHE.clear()


def _get_compiled(
    name: str,
    per_shape: Tuple[int, ...],
    dtype: Any,
    mode: str,
    batch_n: int,
    hsig=None,
):
    """
    Retrieve or build a compiled vmap+jit function for a given bucket.

    Args:
      name: parametrization key (DICT_PARAMS[name])
      per_shape: shape of a single weight (no batch dim)
      dtype: dtype of a single weight
      mode: 'nohp' | 'const' | 'per'
      batch_n: number of items in the bucket (leading axis length)
      hsig: optional, structure signature for hparams (esp. for 'per')

    Keying on batch_n ensures we don't reuse a compilation for different bucket sizes.
    """
    key = (name, tuple(per_shape), str(dtype), mode, int(batch_n), hsig)
    cached = _VMAP_CACHE.get(key)
    if cached is not None:
        return cached

    fn = DICT_PARAMS[name]

    if mode == "nohp":
        compiled = jax.jit(jax.vmap(lambda w: fn(w), in_axes=0, out_axes=0))
    elif mode == "const":
        # Broadcast same hparams to every item (in_axes=(0, None))
        compiled = jax.jit(
            jax.vmap(lambda w, const: fn(w, **const), in_axes=(0, None), out_axes=0)
        )
    elif mode == "per":
        compiled = jax.jit(
            jax.vmap(lambda w, h: fn(w, **h), in_axes=(0, 0), out_axes=0)
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    _VMAP_CACHE[key] = compiled
    return compiled


def parametrize_vmapped_cached(buckets):
    """
    Vectorized, compiled, and cached reparam across all buckets.
    Returns dict: {uid: reparameterized_weight}
    """
    out = {}
    for name, by_sig in buckets.items():
        # sanity: ensure key exists
        _ = DICT_PARAMS[name]
        for (shape, dtype), items in by_sig.items():
            uids, ws, hs = zip(*items)
            W = jnp.stack(ws, 0)  # [N, ...shape]
            N = int(W.shape[0])

            if _all_equal_pytrees(hs):
                const = hs[0]
                if const == {}:
                    compiled = _get_compiled(name, shape, dtype, mode="nohp", batch_n=N)
                    R = compiled(W)
                else:
                    hsig = _tree_sig_structure(const)
                    compiled = _get_compiled(
                        name, shape, dtype, mode="const", batch_n=N, hsig=hsig
                    )
                    R = compiled(W, const)
            else:
                H = _stack_tree(hs)  # batch pytrees of hparams
                hsig = _tree_sig_structure(H)  # includes batch dimension in shapes
                compiled = _get_compiled(
                    name, shape, dtype, mode="per", batch_n=N, hsig=hsig
                )
                R = compiled(W, H)

            out.update({uid: r for uid, r in zip(uids, R)})
    return out


def parametrize_reference(buckets):
    """Slow reference: per-layer loop (no vmap/jit)."""
    out = {}
    for name, by_sig in buckets.items():
        fn = DICT_PARAMS[name]
        for _, items in by_sig.items():
            for uid, w, h in items:
                out[uid] = fn(w, **h) if h else fn(w)
    return out


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


def parametrize_from_params_cached(buckets, params: Dict[str, jnp.ndarray]):
    out = {}
    for name, by_sig in buckets.items():
        _ = DICT_PARAMS[name]
        for (shape, dtype), items in by_sig.items():
            uids, _ws_ignored, hs = zip(*items)
            W = jnp.stack([params[uid] for uid in uids], 0)
            N = int(W.shape[0])

            if _all_equal_pytrees(hs):
                const = hs[0]
                if const == {}:
                    compiled = _get_compiled(name, shape, dtype, mode="nohp", batch_n=N)
                    R = compiled(W)
                else:
                    hsig = _tree_sig_structure(const)
                    compiled = _get_compiled(
                        name, shape, dtype, mode="const", batch_n=N, hsig=hsig
                    )
                    R = compiled(W, const)
            else:
                H = _stack_tree(hs)
                hsig = _tree_sig_structure(H)
                compiled = _get_compiled(
                    name, shape, dtype, mode="per", batch_n=N, hsig=hsig
                )
                R = compiled(W, H)

            out.update({uid: r for uid, r in zip(uids, R)})
    return out


# ----------------------------
# Main (demo + benchmarks)
# ----------------------------

if __name__ == "__main__":
    # Example model with multiple buckets
    from activations import GroupSort2

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
                20, 20, parametrization="spectral", rngs=rngs
            )
            self.lin4 = ParametrizedLinear(
                20, 20, parametrization="spectral", rngs=rngs
            )
            self.lin5 = ParametrizedLinear(20, 20, parametrization="ortho", rngs=rngs)
            self.lin6 = ParametrizedLinear(20, 20, parametrization="ortho", rngs=rngs)
            self.lin7 = ParametrizedLinear(20, 20, parametrization="ortho", rngs=rngs)
            self.lin8 = ParametrizedLinear(20, 20, parametrization="ortho", rngs=rngs)
            self.lin9 = ParametrizedLinear(20, 20, parametrization="ortho", rngs=rngs)
            self.act = GroupSort2()
            self.lin10 = ParametrizedLinear(20, 2, parametrization="ortho", rngs=rngs)

        def __call__(self, x, ws=None):
            # No prints here for fair timing
            x = self.lin0(x, ws)
            x = self.lin1(x, ws)
            x = self.lin2(x, ws)
            x = self.lin3(x, ws)
            x = self.lin4(x, ws)
            x = self.lin5(x, ws)
            x = self.lin6(x, ws)
            x = self.lin7(x, ws)
            x = self.lin8(x, ws)
            x = self.lin9(x, ws)
            x = self.act(x)
            x = self.lin10(x, ws)
            return x

    # Build model & inputs
    key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(key)
    model = Model(rngs)

    x = jax.random.uniform(key, (16, 10))

    # Collect and reparametrize (also warms the vmap cache)
    buckets = collect_buckets(model)
    reparam_weights = parametrize_vmapped_cached(buckets)

    # Correctness check vs reference path
    outs_normal = model(x)
    outs_vmap = model(x, reparam_weights)
    _sync((outs_normal, outs_vmap))

    diff = jnp.abs(outs_vmap - outs_normal)
    max_abs = float(jnp.max(diff))
    rel = float(max_abs / (float(jnp.max(jnp.abs(outs_normal))) + 1e-12))
    print(f"[compare] max_abs={max_abs:.3e}, rel_max={rel:.3e}")
    print(
        f"[compare] allclose (rtol=3e-3, atol=3e-3): {bool(jnp.allclose(outs_vmap, outs_normal, rtol=3e-3, atol=3e-3))}"
    )

    # Timing: eager and jit forwards
    f_normal = lambda x: model(x)
    f_vm = lambda x: model(x, reparam_weights)

    bench(f_normal, x, label="normal (eager)")
    bench(f_vm, x, label="vmap-weights (eager)")

    jit_normal = jax.jit(lambda x: model(x))
    jit_vm = jax.jit(lambda x: model(x, reparam_weights))

    with jax.default_matmul_precision("highest"):
        bench(jit_normal, x, label="normal (jit, highest precision)")
        bench(jit_vm, x, label="vmap-weights (jit, highest precision)")

    # Adam gradient step (sanity)
    # Params tree = dict {uid -> raw W}; we optimize these, and compute ws from them.
    params_init = {
        uid: w
        for name, by_sig in buckets.items()
        for _, items in by_sig.items()
        for (uid, w, _h) in items
    }

    # Simple regression target: zeros (any scalar loss will do for the check)
    target = jnp.zeros((x.shape[0], 2), dtype=outs_normal.dtype)

    def loss_from_params(params, x):
        # Compute reparameterized weights from raw params, then forward
        ws = parametrize_from_params_cached(buckets, params)
        y = model(x, ws)
        return jnp.mean((y - target) ** 2)

    import optax

    opt = optax.adam(1e-3)
    opt_state = opt.init(params_init)

    @jax.jit
    def train_step(params, opt_state, x):
        loss, grads = jax.value_and_grad(loss_from_params)(params, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # One step to verify gradients are computable and finite
    params, opt_state, loss0 = train_step(params_init, opt_state, x)
    _sync(loss0)
    params, opt_state, loss1 = train_step(params, opt_state, x)
    _sync(loss1)

    print(
        f"[adam] step0 loss={float(loss0):.6f}  step1 loss={float(loss1):.6f}  finite={jnp.isfinite(loss1)}"
    )

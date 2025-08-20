from collections.abc import Mapping, Sequence
import jax
from flax import nnx
from jaxlip.linear import ParametrizedLinear
from jaxlip.batchop import BatchCentering2d, BatchCentering, LayerCentering
from jaxlip.conv import ParametrizedConv2d


def _stack_tree(list_of_trees):
    """Turn [tree0, tree1, ...] into a single tree stacked on axis 0."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, 0), *list_of_trees)


def _array_sig(a):
    """Return a hashable signature for a leaf (structure-only, no values)."""
    import numpy as np

    if isinstance(a, (jnp.ndarray, np.ndarray)):
        return ("arr", tuple(a.shape), str(a.dtype))
    elif isinstance(a, (int, float, bool, str)):
        return ("lit", type(a).__name__)
    else:
        return ("obj", type(a).__name__)


def _tree_sig_structure(tree):
    """Hashable signature of pytree structure + leaf shapes/dtypes (no values)."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    sig_leaves = tuple(_array_sig(x) for x in leaves)
    return (treedef.to_string(), sig_leaves)


def _tree_all(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(x) for x in leaves)


def _all_equal_pytrees(hs):
    """
    Strict equality of pytrees by value for scalars and by exact array equality for arrays.
    Used to decide between 'const' vs 'per' modes. If you only care about structure,
    replace this with a structure-only check.
    """
    import numpy as np

    def leaf_eq(a, b):
        if isinstance(a, (jnp.ndarray, np.ndarray)) and isinstance(
            b, (jnp.ndarray, np.ndarray)
        ):
            if a.shape != b.shape or str(a.dtype) != str(b.dtype):
                return False
            # jnp.array_equal returns a boolean scalar DeviceArray; cast to bool
            return bool(jnp.array_equal(a, b))
        return a == b

    ref = hs[0]
    for h in hs[1:]:
        eq_tree = jax.tree_util.tree_map(leaf_eq, ref, h)
        if not _tree_all(eq_tree):
            return False
    return True


def cache_model_params(root: nnx.Module, verbose: bool = True) -> None:
    """
    Recursively traverse `root` and invoke `specific_method()` on
    every `nnx.Linear` instance in the tree.

    Parameters
    ----------
    root : nnx.Module
        The top-level Flax nnx model (or sub-module) to traverse.

    Notes
    -----
    * A small `visited` set prevents re-visiting the same object when
      modules are shared.
    * The traversal looks inside standard Python containers
      (dict / list / tuple / set) so nested collections work too.
    """
    visited: set[int] = set()

    def _walk(mod: nnx.Module) -> None:
        if mod == root and verbose:
            print(
                "WARNING: Do not retrain this network without previously running ._uncache()"
            )
        if id(mod) in visited:  # avoid duplicates
            return
        visited.add(id(mod))

        if isinstance(mod, (ParametrizedLinear, ParametrizedConv2d)):
            mod._cache_params()
        if isinstance(
            mod,
            (
                BatchCentering2d,
                BatchCentering,
            ),
        ):
            mod.use_running_average = True

        for _, value in vars(mod).items():
            if isinstance(value, nnx.Module):
                _walk(value)
            elif isinstance(value, Mapping):
                for v in value.values():
                    if isinstance(v, nnx.Module):
                        _walk(v)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for v in value:
                    if isinstance(v, nnx.Module):
                        _walk(v)

    _walk(root)
    pass


def uncache_model_params(root: nnx.Module) -> None:
    """
    Recursively traverse `root` and invoke `specific_method()` on
    every `nnx.Linear` instance in the tree.

    Parameters
    ----------
    root : nnx.Module
        The top-level Flax nnx model (or sub-module) to traverse.

    Notes
    -----
    * A small `visited` set prevents re-visiting the same object when
      modules are shared.
    * The traversal looks inside standard Python containers
      (dict / list / tuple / set) so nested collections work too.
    """
    visited: set[int] = set()

    def _walk(mod: nnx.Module) -> None:
        if id(mod) in visited:  # avoid duplicates
            return
        visited.add(id(mod))

        if isinstance(mod, (ParametrizedLinear, ParametrizedConv2d)):
            mod._uncache()
        if isinstance(
            mod,
            (
                BatchCentering2d,
                BatchCentering,
            ),
        ):
            mod.use_running_average = False

        for _, value in vars(mod).items():
            if isinstance(value, nnx.Module):
                _walk(value)
            elif isinstance(value, Mapping):
                for v in value.values():
                    if isinstance(v, nnx.Module):
                        _walk(v)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for v in value:
                    if isinstance(v, nnx.Module):
                        _walk(v)

    _walk(root)
    pass


def load_params_into_model(
    model, params, *, strict: bool = True, clear_caches: bool = True
):
    """
    Overwrite the model's raw weights and biases with arrays from `params`.

    Expected keys in `params`:
      - weight: uid -> array
      - bias:   f"b:{uid}" -> array

    Args:
      model: nnx.Module containing ParametrizedLinear layers.
      params: dict[str, jax.Array] mapping uids/bias-keys to arrays.
      strict: if True, check shape and dtype match before assigning.
      clear_caches: if True, invalidate per-layer reparam cache after changing weights.
    """
    seen = set()

    def _assign(dst_param, new_val, name: str):
        if strict:
            if dst_param.value.shape != new_val.shape:
                raise ValueError(
                    f"{name} shape mismatch: got {new_val.shape}, expected {dst_param.value.shape}"
                )
            if dst_param.value.dtype != new_val.dtype:
                raise ValueError(
                    f"{name} dtype mismatch: got {new_val.dtype}, expected {dst_param.value.dtype}"
                )
        dst_param.value = new_val

    def walk(mod):
        if id(mod) in seen:
            return
        seen.add(id(mod))

        if isinstance(mod, ParametrizedLinear):
            uid = mod._uid

            # Load weight
            if uid in params:
                _assign(mod.w, params[uid], f"W[{uid}]")
                if clear_caches:
                    if hasattr(mod, "_uncache"):
                        mod._uncache()
                    elif hasattr(mod, "_cache_dirty"):
                        mod._cache_dirty = True

            # Load bias (if present)
            if getattr(mod, "bias", False):
                bkey = f"b:{uid}"
                if bkey in params:
                    _assign(mod.b, params[bkey], f"b[{uid}]")

        # Recurse into children
        for v in vars(mod).values():
            if isinstance(v, nnx.Module):
                walk(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, nnx.Module):
                        walk(vv)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, nnx.Module):
                        walk(vv)

    walk(model)


def inject_biases(model, params, ws):
    for _, m in model.iter_modules():
        if isinstance(m, ParametrizedLinear) and m.bias:
            key = f"b:{m._uid}"
            if key in params:
                ws[key] = params[key]
    pass

import jax
from flax import nnx
from collections.abc import Mapping, Sequence
from .base import ReparametrizedModule


def _iter_children(obj):
    """Yield child containers/modules from an arbitrary Python object."""
    # nnx.Module stores submodules/vars as normal attributes
    if isinstance(obj, nnx.Module):
        # vars(obj) is safe; filters callable/static stuff by type below
        for v in vars(obj).values():
            yield v
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield v
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield v
    # else: leaf (Param, ndarray, int, None, etc.)


def _walk_modules(root):
    """Iterative DFS that yields modules; avoids cycles by id()."""
    seen = set()
    stack = [root]
    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(obj, nnx.Module):
            yield obj  # includes ReparametrizedModule etc.

        for child in _iter_children(obj):
            # Only follow containers / modules to keep it cheap
            if isinstance(child, (nnx.Module, Mapping, list, tuple, set)):
                stack.append(child)


def assign_owners_round_robin(model, *, num_devices: int | None = None):
    """Set `owner` field on every ReparametrizedModule in `model`."""
    if num_devices is None:
        num_devices = jax.local_device_count()
    k = 0
    for mod in _walk_modules(model):
        if isinstance(mod, ReparametrizedModule):
            # only assign if not already set
            if getattr(mod, "owner", -1) < 0:
                mod.owner = k % num_devices
                k += 1

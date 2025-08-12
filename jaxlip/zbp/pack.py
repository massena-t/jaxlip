from typing import List
import jax
import jax.numpy as jnp
from .assign import _walk_modules
from .base import ReparametrizedModule
from .distributed_op import reparam_distributed_vmap
from ..newton_schulz import orthogonalize


def build_reparam_pack(model, *, distributed: bool = True) -> List[jnp.ndarray]:
    """
    Returns a list 'Qs_groups' where each item is an array of shape
    [group_size, din, dout] aligned with model._zbp_gid grouping.
    No mutation; pure.
    """
    # Collect per-group stacks in order of gid
    # First, discover how many groups exist and their sizes in a stable pass
    groups: dict[int, list[ReparametrizedModule]] = {}
    for m in _walk_modules(model):
        if isinstance(m, ReparametrizedModule) and hasattr(m, "w"):
            gid = getattr(m, "_zbp_gid", None)
            idx = getattr(m, "_zbp_idx", None)
            if gid is None or idx is None:
                raise ValueError(
                    "assign_reparam_groups(model) must be called before build_reparam_pack"
                )
            groups.setdefault(gid, []).append(m)
    # Ensure ordering by gid then idx
    max_gid = max(groups.keys()) if groups else -1
    Qs_groups: List[jnp.ndarray] = []
    for gid in range(max_gid + 1):
        mods = sorted(groups[gid], key=lambda m: m._zbp_idx)
        Ws = jnp.stack([m.w for m in mods], axis=0)  # [G, din, dout]
        owners = jnp.asarray([m.owner for m in mods], dtype=jnp.int32)  # [G]
        if distributed:
            Qs = reparam_distributed_vmap(Ws, owners)  # [G, din, dout]
        else:
            Qs = jax.vmap(orthogonalize, in_axes=0, out_axes=0)(Ws)  # local fallback
        Qs_groups.append(Qs)
    return Qs_groups


def apply_with_reparam(model, x, Qs_groups: List[jnp.ndarray]):
    """
    Call model(x, reparam_overrides=Qs_groups) purely; the layers
    will pick their Q as Qs_groups[self._zbp_gid][self._zbp_idx].
    """
    return model(x, reparam_overrides=Qs_groups)

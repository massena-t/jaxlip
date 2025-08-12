from collections import defaultdict
from typing import Dict, List, Tuple, Any
from flax import nnx
from .assign import _walk_modules
from .base import ReparametrizedModule


def assign_reparam_groups(model) -> None:
    """
    Deterministically assign (group_id, idx_in_group) to every
    ReparametrizedModule that has a .w.
    Groups are keyed by (reparam_fn identity, weight shape).
    """
    # Collect modules per group
    groups: Dict[Tuple[int, Tuple[int, ...]], List[ReparametrizedModule]] = defaultdict(
        list
    )
    for m in _walk_modules(model):
        if isinstance(m, ReparametrizedModule) and hasattr(m, "w"):
            # identify the function object; if you use reparam_fn=None, override _resolve_reparam_fn in base
            f = m._resolve_reparam_fn()
            key = (id(f), tuple(m.w.shape))
            groups[key].append(m)

    # Assign compact group ids and local indices
    for gid, (key, mods) in enumerate(
        sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    ):
        for idx, m in enumerate(mods):
            # static integers; set once (allowed)
            m._zbp_gid = gid
            m._zbp_idx = idx

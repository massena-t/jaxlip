from typing import Optional, Callable, List
import jax.numpy as jnp
from flax import nnx
from .distributed_op import reparam_distributed
from ..newton_schulz import orthogonalize


class ReparametrizedModule(nnx.Module):
    owner: int = -1
    reparam_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    _zbp_gid: int = -1
    _zbp_idx: int = -1

    def _resolve_reparam_fn(self):
        return self.reparam_fn or orthogonalize

    def distributed_reparam(self, W):
        # compute-on-owner; gradients via custom VJP
        return reparam_distributed(W, self.owner)

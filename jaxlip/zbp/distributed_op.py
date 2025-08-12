import jax
import jax.numpy as jnp
from jax import lax
from ..newton_schulz import orthogonalize

AXIS_NAME = "device"


def _owner_mask(owner: jnp.ndarray) -> jnp.ndarray:
    """JAX bool scalar: is this device the owner?"""
    owner = jnp.asarray(owner, jnp.int32)
    return jnp.asarray(lax.axis_index(AXIS_NAME) == owner, dtype=bool)


def _impl_forward(W: jnp.ndarray, owner: jnp.ndarray) -> jnp.ndarray:
    take = _owner_mask(owner)
    local = lax.cond(
        take,
        lambda _: orthogonalize(W),
        lambda _: jnp.zeros_like(W),
        operand=None,
    )
    # Owner contributes Q; psum broadcasts Q to all replicas.
    return lax.psum(local, axis_name=AXIS_NAME)


def _impl_backward(W: jnp.ndarray, owner: jnp.ndarray, ct: jnp.ndarray) -> jnp.ndarray:
    # Sum cotangents from all replicas -> total cotangent
    g_total = lax.psum(ct, axis_name=AXIS_NAME)

    def grad_on_owner(_):
        _, vjp = jax.vjp(lambda X: orthogonalize(X), W)
        (gW,) = vjp(g_total)
        return gW

    take = _owner_mask(owner)
    gW_local = lax.cond(take, grad_on_owner, lambda _: jnp.zeros_like(W), operand=None)
    gW_all = lax.psum(gW_local, axis_name=AXIS_NAME)  # identical ∇W on all replicas
    # Don't divide here; your train loop already pmeans grads.
    return gW_all


@jax.custom_vjp
def reparam_distributed(W: jnp.ndarray, owner: jnp.ndarray) -> jnp.ndarray:
    return _impl_forward(W, owner)


def _fwd(W, owner):
    Q = _impl_forward(W, owner)
    return Q, (W, owner)


def _bwd(res, ct):
    W, owner = res
    gW = _impl_backward(W, owner, ct)
    return (gW, None)


reparam_distributed.defvjp(_fwd, _bwd)


def reparam_distributed_vmap(Ws: jnp.ndarray, owners: jnp.ndarray) -> jnp.ndarray:
    owners = jnp.asarray(owners, dtype=jnp.int32)
    return jax.vmap(lambda w, o: reparam_distributed(w, o), in_axes=(0, 0), out_axes=0)(
        Ws, owners
    )

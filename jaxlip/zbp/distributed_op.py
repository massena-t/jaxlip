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


def reparam_distributed_vmap(Ws: jnp.ndarray, owners: jnp.ndarray) -> jnp.ndarray:
    """
    Batched version that limits peak memory by chunking with static slice sizes.
    Pads to a multiple of CHUNK_SIZE so dynamic_slice can use a constant size.
    """
    owners = jnp.asarray(owners, dtype=jnp.int32)

    CHUNK_SIZE = 5  # tune: smaller -> lower peak mem; larger -> faster

    K = int(Ws.shape[0])  # static at trace time
    num_chunks = (K + CHUNK_SIZE - 1) // CHUNK_SIZE
    M = num_chunks * CHUNK_SIZE
    pad = M - K

    if pad:
        pad_width = [(0, pad)] + [(0, 0)] * (Ws.ndim - 1)
        Ws_pad = jnp.pad(Ws, pad_width, mode="constant")
        # Use -1 so no device will match _owner_mask for padded entries.
        owners_pad = jnp.pad(owners, ((0, pad),), mode="constant", constant_values=-1)
    else:
        Ws_pad = Ws
        owners_pad = owners

    out0 = jnp.empty_like(Ws_pad)

    def body(i, out):
        start = i * CHUNK_SIZE
        Ws_chunk = lax.dynamic_slice_in_dim(Ws_pad, start, CHUNK_SIZE, axis=0)
        owners_chunk = lax.dynamic_slice_in_dim(owners_pad, start, CHUNK_SIZE, axis=0)

        chunk = jax.vmap(reparam_distributed, in_axes=(0, 0), out_axes=0)(
            Ws_chunk, owners_chunk
        )

        out = lax.dynamic_update_slice_in_dim(out, chunk, start, axis=0)
        return out

    out_pad = lax.fori_loop(0, num_chunks, body, out0)

    # Trim padding back to shape (K, ...). K is static, so this is fine.
    out = lax.dynamic_slice_in_dim(out_pad, 0, K, axis=0)
    return out

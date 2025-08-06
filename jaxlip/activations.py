import jax
import jax.numpy as jnp
import flax.nnx as nnx

from functools import partial

# Taken from https://github.com/Algue-Rythme/CertifiedQuantileRegression/blob/main/cnqr/_src/lipschitz.py


def channelwise_groupsort2(x, axis):
    """GroupSort2 activation function.

    Args:
      x: array of shape (C,). C must be an even number.

    Returns:
      array of shape (C,) with groupsort2 applied.
    """
    assert x.shape[axis] % 2 == 0
    a, b = jnp.split(x, 2, axis=axis)
    min_ab = jnp.minimum(a, b)
    max_ab = jnp.maximum(a, b)
    return jnp.concatenate([min_ab, max_ab], axis=axis)


def groupsort2(x):
    """Apply GroupSort2 activation across batch dimension.

    Args:
        x (jax.Array): Input tensor of shape (batch_size, ..., channels).
                      The last dimension (channels) must be even.

    Returns:
        jax.Array: Output tensor with the same shape as input,
                  with GroupSort2 activation applied.
    """
    return jax.vmap(partial(channelwise_groupsort2, axis=-1), in_axes=0)(x)


class GroupSort2(nnx.Module):
    """GroupSort2 activation function module.

    GroupSort2 is a Lipschitz-preserving activation function that maintains
    the Lipschitz constant of 1. It works by splitting channels into pairs,
    computing min and max for each pair, and concatenating them back.

    This activation is particularly useful in Lipschitz neural networks
    as it preserves the network's overall Lipschitz constant.

    Examples:
        >>> import jax
        >>> from flax import nnx
        >>> activation = GroupSort2()
        >>> x = jax.random.normal(jax.random.PRNGKey(0), (32, 64))  # 64 channels (even)
        >>> y = activation(x)  # shape: (32, 64)
    """

    def __call__(self, x):
        """Apply GroupSort2 activation.

        Args:
            x (jax.Array): Input tensor with even number of channels in the last dimension.

        Returns:
            jax.Array: Output tensor with same shape as input.

        Raises:
            AssertionError: If the last dimension is not even.
        """
        return groupsort2(x)

import jax
from flax import nnx

import jax.numpy as jnp
from jax.numpy.linalg import norm
from jaxlip.newton_schulz import orthogonalize
from jax.nn.initializers import orthogonal


def l2_normalize(weight):
    """Normalize a 2D weight matrix by its spectral norm (L2 operator norm).

    Args:
        weight (jax.Array): A 2D weight matrix of shape (input_dim, output_dim).

    Returns:
        jax.Array: The weight matrix normalized by its spectral norm, maintaining the same shape.

    Raises:
        AssertionError: If the input weight is not 2-dimensional.

    Notes:
        A small epsilon (1e-10) is added to the denominator for numerical stability.
        The spectral norm is computed using the L2 matrix norm (largest singular value).
    """
    assert len(weight.shape) == 2
    return weight / (norm(weight, ord=2) + 1e-10)


class SpectralLinear(nnx.Module):
    """Linear layer with spectral normalization for Lipschitz constraint.

    This module implements a linear transformation with weights normalized by their spectral norm,
    ensuring the layer has a Lipschitz constant of at most 1. The spectral normalization is applied
    either on-the-fly during forward pass or can be cached for efficiency during evaluation.

    The layer computes: y = x @ (W / ||W||_2) + b (if bias=True)
    where ||W||_2 is the spectral norm (largest singular value) of W.

    Attributes:
        w (nnx.Param): Weight matrix of shape (din, dout).
        b (nnx.Param): Bias vector of shape (dout,), only if bias=True.
        cache (jax.Array): Cached normalized weights for evaluation mode.
        cached (bool): Whether the weights are currently cached.
        din (int): Input dimension.
        dout (int): Output dimension.
        bias (bool): Whether to include bias term.
    """

    def __init__(self, din: int, dout: int, bias: bool = True, *, rngs: nnx.Rngs):
        """Initialize the SpectralLinear layer.

        Args:
            din (int): Input dimension (number of input features).
            dout (int): Output dimension (number of output features).
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            rngs (nnx.Rngs): Random number generator state for parameter initialization.

        Examples:
            >>> import jax
            >>> from flax import nnx
            >>> rngs = nnx.Rngs(42)
            >>> layer = SpectralLinear(784, 128, rngs=rngs)
            >>> x = jax.random.normal(jax.random.PRNGKey(0), (32, 784))
            >>> y = layer(x)  # shape: (32, 128)
        """
        key = rngs.params()
        self.w = nnx.Param(orthogonal()(key, (din, dout)))
        self.cache = nnx.Cache(jax.random.uniform(key, (din, dout)), collection="cache")

        self.bias = bias

        if bias:
            self.b = nnx.Param(jnp.zeros((dout,)))

        self.din, self.dout = din, dout
        self.cached = False

    def _cache_params(self):
        """Cache the normalized weights for efficient evaluation.

        This method pre-computes the spectral-normalized weights and stores them in the cache.
        Subsequent forward passes will use the cached weights instead of recomputing the normalization.
        This is particularly useful during evaluation when weights are frozen.
        """
        self.cache.value = l2_normalize(self.w)
        self.cached = True
        pass

    def _uncache(self):
        """Disable weight caching, forcing recomputation of normalized weights.

        This method sets the cached flag to False, ensuring that subsequent forward passes
        will recompute the spectral normalization on-the-fly. This is typically used when
        switching from evaluation mode back to training mode.
        """
        self.cached = False
        pass

    def __call__(self, x: jax.Array):
        """Forward pass through the spectral normalized linear layer.

        Args:
            x (jax.Array): Input tensor of shape (batch_size, din) or (..., din).

        Returns:
            jax.Array: Output tensor of shape (batch_size, dout) or (..., dout).

        Notes:
            If weights are cached, uses the pre-computed normalized weights.
            Otherwise, computes spectral normalization on-the-fly.
        """
        if not self.cached:
            y = x @ l2_normalize(self.w)
        else:
            y = x @ self.cache.value
        return y + self.b if self.bias else y


class OrthoLinear(nnx.Module):
    """Linear layer with orthogonal weight constraints for Lipschitz control.

    This module implements a linear transformation with orthogonal weight matrices,
    ensuring the layer has a Lipschitz constant of exactly 1. The orthogonality constraint
    is enforced using the Newton-Schulz iteration method.

    The layer computes: y = x @ W_orth + b (if bias=True)
    where W_orth is the orthogonalized version of the weight matrix W.

    Attributes:
        w (nnx.Param): Weight matrix of shape (din, dout).
        b (nnx.Param): Bias vector of shape (dout,), only if bias=True.
        cache (jax.Array): Cached orthogonalized weights for evaluation mode.
        cached (bool): Whether the weights are currently cached.
        din (int): Input dimension.
        dout (int): Output dimension.
        bias (bool): Whether to include bias term.
    """

    def __init__(self, din: int, dout: int, bias: bool = True, *, rngs: nnx.Rngs):
        """Initialize the OrthoLinear layer.

        Args:
            din (int): Input dimension (number of input features).
            dout (int): Output dimension (number of output features).
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            rngs (nnx.Rngs): Random number generator state for parameter initialization.

        Examples:
            >>> import jax
            >>> from flax import nnx
            >>> rngs = nnx.Rngs(42)
            >>> layer = OrthoLinear(784, 128, rngs=rngs)
            >>> x = jax.random.normal(jax.random.PRNGKey(0), (32, 784))
            >>> y = layer(x)  # shape: (32, 128)
        """
        key = rngs.params()
        self.w = nnx.Param(orthogonal()(key, (din, dout)))
        self.cache = nnx.Cache(jax.random.uniform(key, (din, dout)), collection="cache")

        self.bias = bias

        if bias:
            self.b = nnx.Param(jnp.zeros((dout,)))

        self.din, self.dout = din, dout
        self.cached = False

    def _cache_params(self):
        """Cache the orthogonalized weights for efficient evaluation.

        This method pre-computes the orthogonalized weights using Newton-Schulz iteration
        and stores them in the cache. Subsequent forward passes will use the cached weights
        instead of recomputing the orthogonalization.
        """
        self.cache.value = orthogonalize(self.w)
        self.cached = True
        pass

    def _uncache(self):
        """Disable weight caching, forcing recomputation of orthogonalized weights.

        This method sets the cached flag to False, ensuring that subsequent forward passes
        will recompute the orthogonalization on-the-fly. This is typically used when
        switching from evaluation mode back to training mode.
        """
        self.cached = False
        pass

    def __call__(self, x: jax.Array, *args):
        """Forward pass through the orthogonal linear layer.

        Args:
            x (jax.Array): Input tensor of shape (batch_size, din) or (..., din).
            *args: Additional arguments (unused, for compatibility).

        Returns:
            jax.Array: Output tensor of shape (batch_size, dout) or (..., dout).

        Notes:
            If weights are cached, uses the pre-computed orthogonalized weights.
            Otherwise, computes orthogonalization on-the-fly using Newton-Schulz iteration.
        """
        if not self.cached:
            w_orth = orthogonalize(self.w)
            y = x @ w_orth
        else:
            y = x @ self.cache.value
        return y + self.b if self.bias else y

import jax
from flax import nnx

import uuid
import jax.numpy as jnp
from jax.numpy.linalg import norm
from jaxlip.newton_schulz import orthogonalize
from jax.nn.initializers import orthogonal
from functools import partial


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


PARAMETRIZATIONS = ["spectral", "ortho"]
DICT_PARAMS = {"spectral": l2_normalize, "ortho": orthogonalize}


class ParametrizedLinear(nnx.Module):
    def __init__(
        self,
        din: int,
        dout: int,
        bias: bool = True,
        parametrization="spectral",
        *,
        rngs: nnx.Rngs,
    ):
        assert parametrization in PARAMETRIZATIONS, "Unknown parametrization"
        key = rngs.params()
        self.w = nnx.Param(orthogonal()(key, (din, dout)))
        self.cache = nnx.Cache(jax.random.uniform(key, (din, dout)), collection="cache")
        self.bias = bias
        self.parametrization = parametrization

        if bias:
            self.b = nnx.Param(jnp.zeros((dout,)))

        self.din, self.dout = din, dout
        self.cached = False
        self._uid = uuid.uuid4().hex

    def get_weights(self, weight):
        return DICT_PARAMS[self.parametrization](weight)

    def _cache_params(self):
        self.cache.value = self.get_weights(self.w)
        self.cached = True
        pass

    def _uncache(self):
        self.cached = False
        pass

    def __call__(self, x: jax.Array, ws: jax.Array = None):
        if ws is not None:
            y = x @ ws[self._uid]
            if self.bias:
                b = ws.get(f"b:{self._uid}", self.b.value)
                return y + b
            else:
                return y
        else:
            if not self.cached:
                y = x @ self.get_weights(self.w)
            else:
                y = x @ self.cache.value
            return y + self.b if self.bias else y


OrthoLinear = partial(ParametrizedLinear, parametrization="ortho")
SpectralLinear = partial(ParametrizedLinear, parametrization="spectral")

"""JaxLip: Lipschitz Neural Networks in JAX."""

from .activations import GroupSort2
from .batchop import BatchCentering, BatchCentering2d, LayerCentering
from .conv import SpectralConv2d, AOLConv2d
from .linear import OrthoLinear, SpectralLinear
from .utils import cache_model_params, uncache_model_params

__version__ = "0.1.0"
__all__ = [
    "GroupSort2",
    "BatchCentering",
    "BatchCentering2d",
    "LayerCentering",
    "SpectralConv2d",
    "AOLConv2d",
    "OrthoLinear",
    "SpectralLinear",
    "cache_model_params",
    "uncache_model_params",
]

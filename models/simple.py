import jax
import jax.numpy as jnp
from flax import nnx
from jaxlip.conv import SpectralConv2d
from jaxlip.linear import OrthoLinear, SpectralLinear
from jaxlip.batchop import BatchCentering2d
from jaxlip.activations import GroupSort2
import einops


class SimpleConvNet(nnx.Module):
    def __init__(self, rngs, mean=0.5, std=0.5):
        self.stem = nnx.Sequential(
            SpectralConv2d(
                3,
                32,
                3,
                strides=3,
                num_iters_train=20,
                num_iters_eval=40,
                detach_iter=10,
                rngs=rngs,
                bias=False,
            ),
            BatchCentering2d(32),
            GroupSort2(),
            SpectralConv2d(
                32,
                64,
                3,
                strides=2,
                num_iters_train=20,
                num_iters_eval=40,
                detach_iter=10,
                rngs=rngs,
                bias=False,
            ),
            GroupSort2(),
            BatchCentering2d(64),
        )
        self.head = nnx.Sequential(
            OrthoLinear(2304, 128, rngs=rngs),
            GroupSort2(),
            SpectralLinear(128, 10, rngs=rngs),
        )
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x = x - jnp.expand_dims(self.mean, axis=(0, 1, 2))
        x = x / jnp.expand_dims(self.std, axis=(0, 1, 2))
        lipconstant = 1 / jnp.min(self.std)
        x = self.stem(x)
        x = einops.rearrange(x, "b h w c ->  b (h w c)")
        x = self.head(x)
        return x / lipconstant

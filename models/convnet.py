import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import einops
from jaxlip.conv import SpectralConv2d, AOLConv2d
from jaxlip.linear import OrthoLinear, SpectralLinear
from jaxlip.batchop import LipDyT, BatchCentering2d, LayerCentering
from jaxlip.activations import GroupSort2
from flax import nnx

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
import math
from functools import partial


class ConvNet(nnx.Module):
    def __init__(
        self,
        rngs,
        input_shape,
        dim_repeats_stride,
        stem_stride,
        num_classes,
        num_iters_train=20,
        num_iters_eval=80,
        detach_iter=10,
        mean=None,
        std=None,
    ):
        h, w, in_features = input_shape
        conv_layer = partial(
            SpectralConv2d,
            # AOLConv2d,
            bias=False,
            num_iters_train=num_iters_train,
            num_iters_eval=num_iters_eval,
            detach_iter=detach_iter,
        )  # SpectralConv2d or nnx.Conv for debug purposes
        stem = conv_layer(
            in_features=in_features,
            out_features=dim_repeats_stride[0][0],
            kernel_size=(3, 3),
            strides=stem_stride,
            rngs=rngs,
        )
        previous_dim = dim_repeats_stride[0][0]
        h, w = (
            math.ceil(h / stem_stride),
            math.ceil(w / stem_stride),
        )
        layers = [stem]
        for dim, repeats, stride in dim_repeats_stride:
            layers.append(
                conv_layer(
                    in_features=previous_dim,
                    out_features=dim,
                    kernel_size=(3, 3),
                    strides=1,
                    rngs=rngs,
                )
            )
            layers.append(GroupSort2())
            layers.append(BatchCentering2d(num_channels=dim, momentum=0.95))
            for _ in range(repeats - 1):
                layers.append(
                    SpectralConv2d(
                        in_features=dim,
                        out_features=dim,
                        kernel_size=(3, 3),
                        strides=1,
                        rngs=rngs,
                    )
                )
                layers.append(GroupSort2())
                layers.append(BatchCentering2d(num_channels=dim, momentum=0.95))
            layers.append(
                SpectralConv2d(
                    in_features=dim,
                    out_features=dim,
                    kernel_size=(3, 3),
                    strides=stride,
                    rngs=rngs,
                )
            )
            layers.append(GroupSort2())
            layers.append(BatchCentering2d(num_channels=dim, momentum=0.95))
            h, w = (
                math.ceil(h / stride),
                math.ceil(w / stride),
            )
            previous_dim = dim

        self.conv = nnx.Sequential(*layers)
        self.head = OrthoLinear(
            din=previous_dim * h * w,
            dout=num_classes,
            rngs=rngs,
        )
        self.mean = (
            nnx.Cache(jnp.array(mean))
            if mean is not None
            else nnx.Cache(jnp.array(0.0))
        )
        self.std = (
            nnx.Cache(jnp.array(std)) if std is not None else nnx.Cache(jnp.array(1.0))
        )

    def __call__(self, x):
        lipconstant = 1.0
        x = x - jnp.expand_dims(self.mean, axis=(0, 1, 2))
        x = x / jnp.expand_dims(self.std, axis=(0, 1, 2))
        lipconstant *= 1 / jnp.min(self.std)
        x = self.conv(x)
        x = einops.rearrange(x, "b h w c -> b (h w c)")
        x = self.head(x)
        return x / lipconstant


if __name__ == "__main__":
    rng = nnx.Rngs(params=jax.random.key(0))

    dummy = jnp.ones((200, 32, 32, 3))
    model = ConvNet(
        rngs=rng,
        input_shape=(32, 32, 3),
        dim_repeats_stride=[(8, 2, 1), (16, 2, 2), (32, 2, 2)],
        stem_stride=2,
        num_classes=1000,
        num_iters_train=1,
        num_iters_eval=40,
    )
    logits = model(dummy)
    model.eval()
    logits = model(dummy)
    print("Logits shape:", logits.shape)  # (1, 1000)

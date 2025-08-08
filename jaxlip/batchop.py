import jax
import typing as tp
from flax import nnx
import jax.numpy as jnp
from flax.nnx.nn import dtypes
from typing import Optional, Sequence, Union


class BatchCentering(nnx.Module):
    """
    Subtracts an exponential moving average of per-feature means,
    then adds a learnable bias.  No variance/scale is used.
    """

    def __init__(
        self,
        num_features: int,
        *,
        use_running_average: bool = False,
        axis: Union[int, Sequence[int]] = -1,
        momentum: float = 0.9,
        # rngs: nnx.Rngs,
    ):
        feature_shape = (num_features,)
        # running mean state
        self.mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
        # bias parameter
        self.bias = nnx.Param(jnp.zeros(feature_shape, jnp.float32))
        self.use_running_average = use_running_average
        self.axis = axis
        self.momentum = momentum
        self.num_features = num_features

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # normalize axis spec to a tuple
        if self.axis == -1:
            feature_axes = (x.ndim - 1,)
        else:
            feature_axes = (
                (self.axis,) if isinstance(self.axis, int) else tuple(self.axis)
            )
        # all axes except features
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

        if self.use_running_average:
            mean = self.mean
        else:
            curr_mean = jnp.mean(x, axis=reduction_axes)
            self.mean.value = (
                self.momentum * self.mean.value + (1.0 - self.momentum) * curr_mean
            )
            mean = self.mean.value

        # reshape mean and bias for broadcasting
        stats_shape = [1] * x.ndim
        for ax in feature_axes:
            stats_shape[ax] = x.shape[ax]
        mean = mean.reshape(stats_shape)
        bias = self.bias.reshape(stats_shape)
        return x - mean + bias


class BatchCentering2d(nnx.Module):
    def __init__(
        self,
        num_channels: int,
        *,
        use_running_average: bool = False,
        momentum: float = 0.9,
    ):
        channel_shape = (num_channels,)
        self.mean = nnx.BatchStat(
            jnp.zeros(channel_shape, jnp.float32), collection="batch_stats"
        )
        self.bias = nnx.Param(jnp.zeros(channel_shape, jnp.float32))
        self.use_running_average = use_running_average
        self.momentum = momentum

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Expect NHWC: channels last
        # Reduction over N, H, W to get per-channel mean
        reduction_axes = tuple(i for i in range(x.ndim) if i != (x.ndim - 1))
        if self.use_running_average:
            mean = self.mean.value
        else:
            curr_mean = jnp.mean(x, axis=reduction_axes)
            self.mean.value = (
                self.momentum * self.mean.value + (1.0 - self.momentum) * curr_mean
            )
            mean = self.mean.value

        # reshape/broadcast mean & bias and apply…
        mean_b = mean.reshape((1,) * (x.ndim - 1) + (x.shape[-1],))
        bias_b = self.bias.reshape(mean_b.shape)
        return x - mean_b + bias_b


class LipDyT(nnx.Module):
    """Lipschitz‐controlled tanh nonlinearity with eager‐init in Flax NNX."""

    def __init__(
        self,
        num_features: int,
        init_alpha: float = 1.0,
        compression: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        # create all trainable variables right here
        self.alpha = nnx.Param(jnp.full((1,), init_alpha))
        self.feat_coefs = nnx.Param(-5.0 * jnp.ones((num_features,)))
        self.bias = nnx.Param(jnp.zeros((num_features,)))
        # non‐trainable config
        self.compression = compression

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, C]
        gate = 1.0 - self.compression * nnx.sigmoid(self.feat_coefs.value)
        return nnx.tanh(self.alpha.value * gate * x) + self.bias.value

    def get_lipconstant(self) -> jnp.ndarray:
        """L = α * max_i [1 – compression * σ(feat_coefs)_i]."""
        gate = 1.0 - self.compression * nnx.sigmoid(self.feat_coefs.value)
        return self.alpha.value * jnp.max(gate)


class LayerCentering(nnx.Module):
    """LayerCentering module for per-feature centering (no running mean)."""

    def __init__(
        self,
        *,
        reduction_axes: tp.Union[int, tuple[int, ...]] = -1,
        dtype: tp.Optional[jnp.dtype] = None,
    ):
        super().__init__()
        self.reduction_axes = reduction_axes
        self.dtype = dtype

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        axes = self._canonical_axes(x.ndim, self.reduction_axes)
        mean = jnp.mean(x, axis=axes, keepdims=True)
        if self.dtype is not None:
            mean = mean.astype(self.dtype)
        return x - mean

    @staticmethod
    def _canonical_axes(rank: int, axes: tp.Union[int, tp.Iterable[int]]):
        if not isinstance(axes, tp.Iterable):
            axes = (axes,)
        return tuple(sorted({rank + a if a < 0 else a for a in axes}))

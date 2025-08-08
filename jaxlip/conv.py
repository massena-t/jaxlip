import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import random

from jaxlip.bound import compute_tensor_norm, tensor_norm, reshape_strided_kernel
from jax.nn.initializers import orthogonal

NUM_ITERS = 10


class SpectralConv2d(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        *,
        strides: tuple[int, int] = (1, 1),
        padding: str = "SAME",
        bias: bool = True,
        num_iters_train: int = 20,
        num_iters_eval: int = 80,
        detach_iter: int = 10,
        rngs: nnx.Rngs,
    ):
        """
        :param in_features:  number of input channels
        :param out_features: number of output channels
        :param kernel_size:  (height, width) of the conv kernel
        :param strides:       (sy, sx) convolution stride
        :param padding:      "SAME" or "VALID" (or explicit pads)
        :param rngs:         nnx.Rngs for parameter initialization
        """
        key = rngs.params()
        # split for weight and bias if you like
        w_key, b_key = jax.random.split(key, 2)

        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh, kw = kernel_size, kernel_size

        if isinstance(strides, int):
            strides = (strides, strides)

        # weight shape: [out_features, in_features, kh, kw]
        w_shape = (out_features, in_features, kh, kw)
        self.w = nnx.Param(orthogonal()(w_key, w_shape))
        self.cache = nnx.Cache(jax.random.uniform(w_key, w_shape), collection="cache")

        self.bias = bias

        if bias:
            self.b = nnx.Param(jnp.zeros((out_features,)))

        K = reshape_strided_kernel(self.w, strides).astype(jnp.complex64)
        cout, cin, h, w = K.shape
        key, k1, k2, k3 = random.split(key, 4)

        self.u1 = nnx.Cache(
            random.normal(k1, (cout, 1, 1, 1), dtype=jnp.complex64), collection="pi_u1"
        )
        self.u2 = nnx.Cache(
            random.normal(k2, (1, cin, 1, 1), dtype=jnp.complex64), collection="pi_u2"
        )
        self.u3 = nnx.Cache(
            random.normal(k3, (1, 1, h, 1), dtype=jnp.complex64), collection="pi_u3"
        )

        self.stride = strides
        self.padding = padding
        self.in_features = in_features
        self.out_features = out_features
        self.key = nnx.Cache(key, collection="cache_key")
        self.num_iters_train = num_iters_train
        self.num_iters_eval = num_iters_eval
        self.detach_iter = detach_iter
        self.cached = False
        self.deterministic = False

        # Initialize w, u1, u2, u3 properly
        sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
            self.w,
            self.u1.value,
            self.u2.value,
            self.u3.value,
            num_iters=500,
            s=self.stride,
        )

    def _cache_params(self):
        sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
            self.w,
            self.u1.value,
            self.u2.value,
            self.u3.value,
            num_iters=self.num_iters_eval,
            s=self.stride,
        )
        self.cache.value = self.w / sigma
        self.cached = True

    def _uncache(self):
        self.cached = False
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        :param x: input array of shape [batch, height, width, in_features]
        :returns: output array of shape [batch, new_h, new_w, out_features]
        """
        if not self.cached:
            num_iters = (
                self.num_iters_eval if self.deterministic else self.num_iters_train
            )
            sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
                self.w,
                self.u1.value,
                self.u2.value,
                self.u3.value,
                num_iters=num_iters,
                detach_iter=self.detach_iter,
                s=self.stride,
            )
            kernel = self.w / sigma
        else:
            kernel = self.cache.value

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        # add bias (broadcast over batch and spatial dims)
        return y + self.b[None, None, None, :] if self.bias else y

    def get_ortho_regul(self):
        ratio = (
            self.kernel_size
            * compute_tensor_norm(
                self.key,
                self.w,
                num_iters=self.num_iters_train,
                s=self.stride,
                return_time=False,
            )
        ) / jnp.sqrt(jnp.vdot(self.w, self.w))
        return ratio


class AOLConv2d(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: tuple[int, int],
        *,
        strides: tuple[int, int] = (1, 1),
        padding: str = "SAME",
        bias: bool = True,
        num_iters_train: int = 10,
        num_iters_eval: int = 80,
        detach_iter: int = 5,
        rngs: nnx.Rngs,
    ):
        """
        :param in_features:  number of input channels
        :param out_features: number of output channels
        :param kernel_size:  (height, width) of the conv kernel
        :param stride:       (sy, sx) convolution stride
        :param padding:      "SAME" or "VALID" (or explicit pads)
        :param rngs:         nnx.Rngs for parameter initialization
        """
        key = rngs.params()
        # split for weight and bias if you like
        w_key, b_key = jax.random.split(key, 2)

        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh, kw = kernel_size, kernel_size

        if isinstance(strides, int):
            strides = (strides, strides)

        # weight shape: [out_features, in_features, kh, kw]
        w_shape = (out_features, in_features, kh, kw)
        self.w = nnx.Param(orthogonal()(w_key, w_shape))
        self.cache = nnx.Cache(jax.random.uniform(w_key, w_shape), collection="cache")

        self.bias = bias

        if bias:
            self.b = nnx.Param(jnp.zeros((out_features,)))

        K = reshape_strided_kernel(self.w, strides).astype(jnp.complex64)
        cout, cin, h, w = K.shape
        key, k1, k2, k3 = random.split(key, 4)

        self.u1 = nnx.Cache(
            random.normal(k1, (cout, 1, 1, 1), dtype=jnp.complex64), collection="pi_u1"
        )
        self.u2 = nnx.Cache(
            random.normal(k2, (1, cin, 1, 1), dtype=jnp.complex64), collection="pi_u2"
        )
        self.u3 = nnx.Cache(
            random.normal(k3, (1, 1, h, 1), dtype=jnp.complex64), collection="pi_u3"
        )

        self.stride = strides
        self.padding = padding
        self.in_features = in_features
        self.out_features = out_features
        self.key = nnx.Cache(key, collection="cache_key")
        self.num_iters_train = num_iters_train
        self.num_iters_eval = num_iters_eval
        self.cached = False
        self.deterministic = False
        self.detach_iter = detach_iter

        # Initialize u1, u2, u3 properly
        kernel = aol_conv2d_rescale(self.w)
        sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
            kernel,
            self.u1.value,
            self.u2.value,
            self.u3.value,
            num_iters=500,
            s=self.stride,
        )

    def _cache_params(self):
        kernel = aol_conv2d_rescale(self.w)
        if self.stride != 1:
            num_iters = (
                self.num_iters_eval if self.deterministic else self.num_iters_train
            )
            sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
                kernel,
                self.u1.value,
                self.u2.value,
                self.u3.value,
                num_iters=num_iters,
                s=self.stride,
            )
            kernel /= sigma
        self.cache.value = kernel
        self.cached = True

    def _uncache(self):
        self.cached = False
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        :param x: input array of shape [batch, height, width, in_features]
        :returns: output array of shape [batch, new_h, new_w, out_features]
        """
        if not self.cached:
            kernel = aol_conv2d_rescale(self.w)
            if self.stride != 1:
                num_iters = (
                    self.num_iters_eval if self.deterministic else self.num_iters_train
                )
                sigma, self.u1.value, self.u2.value, self.u3.value = tensor_norm(
                    kernel,
                    self.u1.value,
                    self.u2.value,
                    self.u3.value,
                    num_iters=num_iters,
                    detach_iter=self.detach_iter,
                    s=self.stride,
                )
                kernel /= sigma
        else:
            kernel = self.cache.value

        y = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=kernel,
            window_strides=self.stride,
            padding=self.padding,
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        # add bias (broadcast over batch and spatial dims)
        return y + self.b[None, None, None, :] if self.bias else y


def aol_conv2d_rescale(kernel_params):
    channel_rescaling_vals = get_aol_conv2d_rescale(kernel_params)
    return kernel_params * channel_rescaling_vals[None, :, None, None]


def get_aol_conv2d_rescale(kernel_params, epsilon=1e-6):
    w = kernel_params  # shape: [co, ci, k1, k2]
    # To the format used in https://github.com/berndprach/AOL/blob/main/src/aol_code/layers/aol/aol_conv2d_rescale.py
    w = jnp.transpose(w, (2, 3, 1, 0))  # shape: [k1, k2, ci, co]

    w_input_dimension_as_batch = jnp.transpose(w, (2, 0, 1, 3))
    w_input_dimension_as_output = jnp.transpose(w, (0, 1, 3, 2))

    p1 = w.shape[0] - 1
    p2 = w.shape[1] - 1

    v = jax.lax.conv_general_dilated(
        lhs=w_input_dimension_as_batch,
        rhs=w_input_dimension_as_output,
        window_strides=(1, 1),
        padding=[(p1, p1), (p2, p2)],
        dimension_numbers=("NHWC", "HWIO", "NHWC"),  # tf like layout
    )

    # Sum the abs value of v over one of the input
    lipschitz_bounds_squared = jax.lax.reduce_sum(
        jnp.abs(v), axes=(1, 2, 3)
    )  # shape [co]
    return (lipschitz_bounds_squared + epsilon) ** (-1 / 2)


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [0, 3, 4, 1, 2])  # g, k1, k2, ci, co
    transforms = np.fft.fft2(kernel, input_shape, axes=[1, 2])  # g, k1, k2, ci, co
    try:
        svs = np.linalg.svd(
            transforms, compute_uv=False, full_matrices=False
        )  # g, k1, k2, min(ci, co)
        stable_rank = (np.mean(svs) ** 2) / svs.max()
        return svs.min(), svs.max(), stable_rank
    except np.linalg.LinAlgError:
        print("numerical error in svd, returning only largest singular value")
        return None, np.linalg.norm(transforms, axis=(1, 2), ord=2), None

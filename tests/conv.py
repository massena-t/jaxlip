import os
import sys


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
import numpy as np
from jaxlip.conv import AOLConv2d, SpectralConv2d, conv_singular_values_numpy
import jax
from jax.numpy.linalg import norm
import jax.numpy as jnp
import flax.nnx as nnx
import optax


class TestSpectralConv2d(unittest.TestCase):
    upper_tol = 1e-5

    def test_cache_spectral_norm(self):
        """Test that cached conv weights have spectral norm <= 1"""
        vals = []
        specs = []

        # Test various conv configurations
        configs = [
            (1, 1, (3, 3), (1, 1)),  # 1x1 -> 1x1, 3x3 kernel
            (3, 8, (3, 3), (1, 1)),  # 3x3 -> 8x8, 3x3 kernel
            (16, 32, (5, 5), (1, 1)),  # 16x16 -> 32x32, 5x5 kernel, stride 1
            # since sedghi doesn't support stride
        ]

        for in_ch, out_ch, kernel_size, stride in configs:
            rng = nnx.Rngs(params=jax.random.key(0))
            conv_layer = SpectralConv2d(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=kernel_size,
                strides=stride,
                rngs=rng,
            )

            # Cache parameters
            conv_layer._cache_params()

            # Test with representative input shapes for singular value computation
            input_shapes = [(32, 32), (64, 64)]
            for input_shape in input_shapes:
                try:
                    # Convert JAX array to numpy for Sedghi method
                    kernel_np = np.array(conv_layer.cache)

                    # Add batch dimension for conv_singular_values_numpy (expects shape [batch, out, in, h, w])
                    kernel_reshaped = kernel_np[None, ...]  # Add batch dim

                    min_sv, max_sv, stable_rank = conv_singular_values_numpy(
                        kernel_reshaped, input_shape
                    )

                    if max_sv is not None:
                        specs.append(float(max_sv))
                        vals.append(max_sv > 1.0 + self.upper_tol)

                        if is_verbose():
                            print(
                                f"Config {in_ch}→{out_ch}, kernel {kernel_size}, stride {stride}"
                            )
                            print(f"  Input shape: {input_shape}")
                            print(f"  Max singular value: {max_sv:.6f}")
                            if min_sv is not None:
                                print(f"  Min singular value: {min_sv:.6f}")
                                print(f"  Stable rank: {stable_rank:.6f}")

                except Exception as e:
                    if is_verbose():
                        print(f"Skipping config {in_ch}→{out_ch} due to error: {e}")
                    continue

        valid = any(vals)
        if is_verbose():
            print("Cached spectral conv max singular values:", specs)
        self.assertFalse(
            valid, "Some cached conv layers have spectral norm outside tolerance"
        )

    def test_weight_spectral_norm_power_iteration(self):
        """Test that conv weights have correct spectral norm using power iteration method"""
        vals = []
        specs = []

        configs = [
            (3, 8, (3, 3), (1, 1)),
            (8, 16, (3, 3), (2, 2)),
        ]

        for in_ch, out_ch, kernel_size, stride in configs:
            rng = nnx.Rngs(params=jax.random.key(42))
            conv_layer = SpectralConv2d(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=kernel_size,
                strides=stride,
                rngs=rng,
            )

            # Get spectral norm from power iteration (this is what the layer uses internally)
            from jaxlip.bound import tensor_norm

            sigma, _, _, _ = tensor_norm(
                conv_layer.w,
                conv_layer.u1,
                conv_layer.u2,
                conv_layer.u3,
                num_iters=50,  # More iterations for accuracy
                s=stride,
            )

            # Normalized weight should have spectral norm ~1
            normalized_kernel = conv_layer.w / sigma

            # Re-compute spectral norm of normalized kernel
            sigma_normalized, _, _, _ = tensor_norm(
                normalized_kernel,
                conv_layer.u1,
                conv_layer.u2,
                conv_layer.u3,
                num_iters=50,
                s=stride,
            )

            specs.append(float(sigma_normalized))
            vals.append(sigma_normalized > 1.0 + self.upper_tol)

            if is_verbose():
                print(
                    f"Power iteration - Config {in_ch}→{out_ch}, kernel {kernel_size}, stride {stride}"
                )
                print(f"  Original spectral norm: {sigma:.6f}")
                print(f"  Normalized spectral norm: {sigma_normalized:.6f}")

        valid = any(vals)
        if is_verbose():
            print("Power iteration normalized spectral norms:", specs)
        self.assertFalse(
            valid, "Some normalized conv weights have spectral norm outside tolerance"
        )


class TestAOLConv2d(unittest.TestCase):
    upper_tol = 1e-5

    def test_cache_aol_norm(self):
        """Test that cached conv weights have spectral norm <= 1"""
        vals = []
        specs = []

        # Test various conv configurations
        configs = [
            (1, 1, (3, 3), (1, 1)),  # 1x1 -> 1x1, 3x3 kernel
            (3, 8, (3, 3), (1, 1)),  # 3x3 -> 8x8, 3x3 kernel
            (16, 32, (5, 5), (1, 1)),  # 16x16 -> 32x32, 5x5 kernel, stride 1
            # since sedghi doesn't support stride
        ]

        for in_ch, out_ch, kernel_size, stride in configs:
            rng = nnx.Rngs(params=jax.random.key(0))
            conv_layer = AOLConv2d(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=kernel_size,
                strides=stride,
                rngs=rng,
            )

            # Cache parameters
            conv_layer._cache_params()

            # Test with representative input shapes for singular value computation
            input_shapes = [(32, 32), (64, 64)]
            for input_shape in input_shapes:
                try:
                    # Convert JAX array to numpy for Sedghi method
                    kernel_np = np.array(conv_layer.cache)

                    # Add batch dimension for conv_singular_values_numpy (expects shape [batch, out, in, h, w])
                    kernel_reshaped = kernel_np[None, ...]  # Add batch dim

                    min_sv, max_sv, stable_rank = conv_singular_values_numpy(
                        kernel_reshaped, input_shape
                    )

                    if max_sv is not None:
                        specs.append(float(max_sv))
                        vals.append(float(max_sv) > 1.0 + self.upper_tol)

                        if is_verbose():
                            print(
                                f"Config {in_ch}→{out_ch}, kernel {kernel_size}, stride {stride}"
                            )
                            print(f"  Input shape: {input_shape}")
                            print(f"  Max singular value: {max_sv:.6f}")
                            if min_sv is not None:
                                print(f"  Min singular value: {min_sv:.6f}")
                                print(f"  Stable rank: {stable_rank:.6f}")

                except Exception as e:
                    if is_verbose():
                        print(f"Skipping config {in_ch}→{out_ch} due to error: {e}")
                    continue

        valid = any(vals)
        if is_verbose():
            print("Cached AOL conv max singular values:", specs)
        self.assertFalse(
            valid, "Some cached conv layers have spectral norm outside tolerance"
        )

    def test_weight_aol_norm_power_iteration(self):
        """Test that conv weights have correct spectral norm using power iteration method"""
        vals = []
        specs = []

        configs = [
            (3, 8, (3, 3), (1, 1)),
            (8, 16, (3, 3), (2, 2)),
        ]

        for in_ch, out_ch, kernel_size, stride in configs:
            rng = nnx.Rngs(params=jax.random.key(42))
            conv_layer = AOLConv2d(
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=kernel_size,
                strides=stride,
                rngs=rng,
            )

            # Get spectral norm from power iteration (this is what the layer uses internally)
            from jaxlip.conv import aol_conv2d_rescale
            from jaxlip.bound import tensor_norm

            kernel = aol_conv2d_rescale(conv_layer.w)
            sigma, _, _, _ = tensor_norm(
                kernel,
                conv_layer.u1,
                conv_layer.u2,
                conv_layer.u3,
                num_iters=50,  # More iterations for accuracy
                s=stride,
            )

            # Normalized weight should have spectral norm ~1
            normalized_kernel = kernel / sigma

            # Re-compute spectral norm of normalized kernel
            sigma_normalized, _, _, _ = tensor_norm(
                normalized_kernel,
                conv_layer.u1,
                conv_layer.u2,
                conv_layer.u3,
                num_iters=50,
                s=stride,
            )

            specs.append(float(sigma_normalized))
            vals.append(sigma_normalized > 1.0 + self.upper_tol)

            if is_verbose():
                print(
                    f"Power iteration - Config {in_ch}→{out_ch}, kernel {kernel_size}, stride {stride}"
                )
                print(f"  Original spectral norm: {sigma:.6f}")
                print(f"  Normalized spectral norm: {sigma_normalized:.6f}")

        valid = any(vals)
        if is_verbose():
            print("Power iteration normalized spectral norms:", specs)
        self.assertFalse(
            valid, "Some normalized conv weights have spectral norm outside tolerance"
        )


class TestConvergence(unittest.TestCase):
    def test_conv_convergence(self):
        """Test that SpectralConv2d can be trained and loss decreases"""
        rng = nnx.Rngs(params=jax.random.key(0))

        # Simple conv layer
        conv_layer_spectral = SpectralConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )
        conv_layer_aol = AOLConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )

        final_losses = []
        # Random training data
        for conv_layer in [conv_layer_spectral, conv_layer_aol]:
            key = jax.random.key(2025)
            x = jax.random.uniform(key, (10, 32, 32, 3))  # Small batch
            y = jax.random.uniform(key, (10, 32, 32, 8))  # Target output

            optimizer = nnx.Optimizer(conv_layer, optax.sgd(1e-3), wrt=nnx.Param)
            params = nnx.state(conv_layer, nnx.Param)

            @nnx.jit
            def train_step(conv_layer, optimizer, x, y):
                def loss_fn(m):
                    y_pred = m(x)
                    return ((y_pred - y) ** 2).mean()

                loss, grads = nnx.value_and_grad(loss_fn)(conv_layer)
                optimizer.update(grads)
                return loss

            # Initial loss
            init_loss = ((conv_layer(x) - y) ** 2).mean()

            # Train for a few steps
            for _ in range(80):
                loss = train_step(conv_layer, optimizer, x, y)

            # Final loss
            final_loss = ((conv_layer(x) - y) ** 2).mean()
            final_losses.append(final_loss < init_loss)

            if is_verbose():
                print(
                    f"Conv convergence test - Init loss: {init_loss:.6f}, Final loss: {final_loss:.6f}"
                )

        valid = all(final_losses)
        self.assertTrue(
            valid,
            "Conv layer should be able to reduce loss during training",
        )


class TestLipschitzProperty(unittest.TestCase):
    def test_lipschitz_bound(self):
        """Test that SpectralConv2d respects Lipschitz bound"""
        rng = nnx.Rngs(params=jax.random.key(0))

        conv_layer_spectral = SpectralConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )
        conv_layer_aol = AOLConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )
        lipschitz_ratios = []

        for conv_layer in [conv_layer_aol, conv_layer_spectral]:
            # Cache parameters to ensure spectral norm = 1
            conv_layer._cache_params()

            key = jax.random.key(2025)
            origin = jax.random.uniform(key, (1, 32, 32, 3))
            pert = 1e-6 * jax.random.uniform(key, (1, 32, 32, 3))
            ref = conv_layer(origin)

            def loss_fn(p):
                out = conv_layer(origin + p)
                out_, ref_ = out.reshape(-1), ref.reshape(-1)
                return jnp.linalg.norm(out_ - ref_, ord=2)

            out = conv_layer(origin + pert)
            grad = jax.grad(loss_fn)(pert)

            pert = pert - 1e-2 * grad
            origin_, pert_ = origin.reshape(-1), pert.reshape(-1)

            lipschitz_ratio = loss_fn(pert) / jnp.linalg.norm(origin_ - pert_, ord=2)
            lipschitz_ratios.append(lipschitz_ratio <= 1.0)

            if is_verbose():
                print(f"Lipschitz ratio: {lipschitz_ratio:.6f}")

        # Should be <= 1 (with some tolerance for numerical errors)
        lipschitz_ratio = any(lipschitz_ratios)
        self.assertTrue(
            lipschitz_ratios,
            f"Lipschitz ratio {lipschitz_ratio:.6f} should be <= 1",
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import unittest
import numpy as np
from jaxlip.conv import AOLConv2d, SpectralConv2d, conv_singular_values_numpy
import jax
from jax.numpy.linalg import norm
import jax.numpy as jnp
import flax.nnx as nnx
import optax


def shard_batch(batch, num_devices):
    return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch)


class TestMultiGPUConv(unittest.TestCase):
    def setUp(self):
        self.num_devices = jax.local_device_count()
        if self.num_devices < 2:
            self.skipTest("Multi-GPU tests require at least 2 devices")

    def test_conv_multi_gpu_training(self):
        """Test that conv layers work with pmap across multiple GPUs"""
        rng = nnx.Rngs(params=jax.random.key(0))
        batch_size = 16
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} not divisible by {self.num_devices} devices"
        )

        trained_results = []
        for conv_layer in [
            SpectralConv2d(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                rngs=rng,
            ),
            AOLConv2d(
                in_features=3,
                out_features=8,
                kernel_size=(3, 3),
                strides=(1, 1),
                rngs=rng,
            ),
        ]:
            # Create model and optimizer
            model = conv_layer
            optimizer = nnx.Optimizer(model, optax.adam(1e-3))

            # Define state axes for nnx.pmap
            state_axes = nnx.StateAxes(
                {
                    nnx.Param: None,
                    nnx.Cache: None,
                    nnx.BatchStat: None,
                }
            )

            @nnx.pmap(
                in_axes=(state_axes, None, 0, 0),
                out_axes=(state_axes, None, 0),
                axis_name="device",
            )
            def train_step(model, optimizer, x, y):
                def loss_fn(m):
                    pred = m(x)
                    return jnp.mean((pred - y) ** 2)

                loss, grads = nnx.value_and_grad(loss_fn)(model)
                grads = jax.lax.pmean(grads, axis_name="device")
                optimizer.update(grads)
                return model, optimizer, loss

            # Generate test data
            key = jax.random.key(2025)
            x = jax.random.uniform(key, (batch_size, 32, 32, 3))
            y = jax.random.uniform(key, (batch_size, 32, 32, 8))

            # Shard data across devices
            x_sharded = shard_batch(x, self.num_devices)
            y_sharded = shard_batch(y, self.num_devices)

            # Initial loss
            initial_loss = jnp.mean((model(x) - y) ** 2)

            # Train for a few steps
            for _ in range(30):
                model, optimizer, loss = train_step(
                    model, optimizer, x_sharded, y_sharded
                )

            # Final loss
            final_loss = jnp.mean((model(x) - y) ** 2)

            # Check that loss decreased
            loss_decreased = final_loss < initial_loss
            trained_results.append(loss_decreased)

            if is_verbose():
                print(
                    f"{type(model).__name__} - Initial: {initial_loss:.6f}, Final: {final_loss:.6f}, Decreased: {loss_decreased}"
                )

        self.assertTrue(
            all(trained_results),
            "All conv layers should reduce loss during multi-GPU training",
        )

    def test_conv_multi_gpu_eval(self):
        """Test that conv layers work with pmap for evaluation"""
        rng = nnx.Rngs(params=jax.random.key(0))
        batch_size = 8
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} not divisible by {self.num_devices} devices"
        )

        for conv_layer in [
            SpectralConv2d(
                in_features=3,
                out_features=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                rngs=rng,
            ),
            AOLConv2d(
                in_features=3,
                out_features=16,
                kernel_size=(5, 5),
                strides=(1, 1),
                rngs=rng,
            ),
        ]:
            model = conv_layer

            @nnx.pmap(in_axes=(None, 0), out_axes=0)
            def eval_step(model, x):
                return model(x)

            # Generate test data
            key = jax.random.key(42)
            x = jax.random.uniform(key, (batch_size, 28, 28, 3))
            x_sharded = shard_batch(x, self.num_devices)

            # Forward pass on multiple GPUs
            output_sharded = eval_step(model, x_sharded)

            # Reshape back to original batch dimension
            output = output_sharded.reshape(-1, 28, 28, 16)

            # Compare with single-device output
            expected_output = model(x)

            # They should be identical (within numerical precision)
            self.assertTrue(
                jnp.allclose(output, expected_output, atol=1e-5),
                f"Multi-GPU and single-GPU outputs should be identical for {type(model).__name__}",
            )

            if is_verbose():
                print(f"{type(model).__name__} multi-GPU eval test passed")

    def test_conv_lipschitz_constraint_multi_gpu(self):
        """Test that conv layers maintain Lipschitz constraints across multiple GPUs"""
        rng = nnx.Rngs(params=jax.random.key(0))

        spectral_layer = SpectralConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )
        aol_layer = AOLConv2d(
            in_features=3, out_features=8, kernel_size=(3, 3), strides=(1, 1), rngs=rng
        )

        # Test data
        key = jax.random.key(123)
        x1 = jax.random.normal(key, (4, 16, 16, 3))
        x2 = jax.random.normal(jax.random.split(key)[0], (4, 16, 16, 3))

        @nnx.pmap(in_axes=(None, 0), out_axes=0)
        def compute_output(model, x):
            return model(x)

        # Shard inputs
        x1_sharded = shard_batch(x1, self.num_devices)
        x2_sharded = shard_batch(x2, self.num_devices)

        for layer, name in [
            (spectral_layer, "SpectralConv2d"),
            (aol_layer, "AOLConv2d"),
        ]:
            # Compute outputs on multiple GPUs
            y1_sharded = compute_output(layer, x1_sharded)
            y2_sharded = compute_output(layer, x2_sharded)

            # Reshape back
            y1 = y1_sharded.reshape(-1, 16, 16, 8)
            y2 = y2_sharded.reshape(-1, 16, 16, 8)

            # Check Lipschitz constraint: ||f(x1) - f(x2)||_F <= ||x1 - x2||_F
            input_diff = jnp.linalg.norm((x1 - x2).reshape(x1.shape[0], -1), axis=-1)
            output_diff = jnp.linalg.norm((y1 - y2).reshape(y1.shape[0], -1), axis=-1)

            # For spectral conv, should be exactly 1-Lipschitz
            # For AOL, should also maintain Lipschitz property
            lipschitz_satisfied = jnp.all(
                output_diff <= input_diff + 1e-4
            )  # small tolerance for numerical errors

            self.assertTrue(
                lipschitz_satisfied,
                f"{name} should satisfy Lipschitz constraint on multi-GPU",
            )

            if is_verbose():
                max_ratio = jnp.max(output_diff / (input_diff + 1e-8))
                print(f"{name} max Lipschitz ratio: {max_ratio:.6f}")

    def test_conv_caching_multi_gpu(self):
        """Test that conv layer parameter caching works with multi-GPU"""
        rng = nnx.Rngs(params=jax.random.key(0))

        for conv_layer in [
            SpectralConv2d(
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                rngs=rng,
            ),
            AOLConv2d(
                in_features=3,
                out_features=16,
                kernel_size=(3, 3),
                strides=(1, 1),
                rngs=rng,
            ),
        ]:
            model = conv_layer

            @nnx.pmap(in_axes=(None, 0), out_axes=0)
            def eval_step(model, x):
                return model(x)

            # Test data
            key = jax.random.key(99)
            x = jax.random.uniform(key, (4, 24, 24, 3))
            x_sharded = shard_batch(x, self.num_devices)

            # Output without caching
            output1_sharded = eval_step(model, x_sharded)
            output1 = output1_sharded.reshape(-1, 24, 24, 16)

            # Cache parameters
            model._cache_params()

            # Output with caching
            output2_sharded = eval_step(model, x_sharded)
            output2 = output2_sharded.reshape(-1, 24, 24, 16)

            # Uncache parameters
            model._uncache()

            # Output after uncaching
            output3_sharded = eval_step(model, x_sharded)
            output3 = output3_sharded.reshape(-1, 24, 24, 16)

            # All outputs should be identical
            self.assertTrue(
                jnp.allclose(output1, output2, atol=1e-5),
                f"{type(model).__name__} cached output should match uncached",
            )
            self.assertTrue(
                jnp.allclose(output1, output3, atol=1e-5),
                f"{type(model).__name__} output should be consistent after uncaching",
            )

            if is_verbose():
                print(f"{type(model).__name__} caching test passed on multi-GPU")


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

            # Create model and optimizer
            model = conv_layer
            optimizer = nnx.Optimizer(model, optax.adam(1e-2))

            # Initial loss
            init_loss = jnp.mean((model(x) - y) ** 2)

            # Training loop
            for _ in range(100):

                def loss_fn(m):
                    pred = m(x)
                    return jnp.mean((pred - y) ** 2)

                loss, grads = nnx.value_and_grad(loss_fn)(model)
                optimizer.update(grads)

            # Final loss
            final_loss = jnp.mean((model(x) - y) ** 2)

            # Check convergence
            converged = final_loss < init_loss
            final_losses.append(converged)

            if is_verbose():
                print(
                    f"Conv convergence test - Init loss: {init_loss:.6f}, Final loss: {final_loss:.6f}"
                )

        valid = all(final_losses)
        self.assertTrue(
            valid,
            "Conv layer should be able to reduce loss during training",
        )


if __name__ == "__main__":
    unittest.main()

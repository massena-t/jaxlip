import os
import sys


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import unittest
from jaxlip.linear import SpectralLinear, OrthoLinear, l2_normalize
from jaxlip.newton_schulz import orthogonalize
import jax
from jax.numpy.linalg import norm
import jax.numpy as jnp
import flax.nnx as nnx
import optax


def shard_batch(batch, num_devices):
    return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch)


class TestMultiGPULinear(unittest.TestCase):
    def setUp(self):
        self.num_devices = jax.local_device_count()
        if self.num_devices < 2:
            self.skipTest("Multi-GPU tests require at least 2 devices")

    def test_linear_multi_gpu_training(self):
        """Test that linear layers work with pmap across multiple GPUs"""
        rng = nnx.Rngs(params=jax.random.key(0))
        batch_size = 32
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} not divisible by {self.num_devices} devices"
        )

        trained_results = []
        for lin_layer in [
            SpectralLinear(128, 64, rngs=rng),
            OrthoLinear(128, 64, rngs=rng),
        ]:
            # Create model and optimizer
            model = lin_layer
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
            x = jax.random.uniform(key, (batch_size, 128))
            y = jax.random.uniform(key, (batch_size, 64))

            # Shard data across devices
            x_sharded = shard_batch(x, self.num_devices)
            y_sharded = shard_batch(y, self.num_devices)

            # Initial loss
            initial_loss = jnp.mean((model(x) - y) ** 2)

            # Train for a few steps
            for _ in range(50):
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
            "All linear layers should reduce loss during multi-GPU training",
        )

    def test_linear_multi_gpu_eval(self):
        """Test that linear layers work with pmap for evaluation"""
        rng = nnx.Rngs(params=jax.random.key(0))
        batch_size = 16
        assert batch_size % self.num_devices == 0, (
            f"Batch size {batch_size} not divisible by {self.num_devices} devices"
        )

        for lin_layer in [
            SpectralLinear(64, 32, rngs=rng),
            OrthoLinear(64, 32, rngs=rng),
        ]:
            model = lin_layer

            @nnx.pmap(in_axes=(None, 0), out_axes=0)
            def eval_step(model, x):
                return model(x)

            # Generate test data
            key = jax.random.key(42)
            x = jax.random.uniform(key, (batch_size, 64))
            x_sharded = shard_batch(x, self.num_devices)

            # Forward pass on multiple GPUs
            output_sharded = eval_step(model, x_sharded)

            # Reshape back to original batch dimension
            output = output_sharded.reshape(-1, 32)

            # Compare with single-device output
            expected_output = model(x)

            # They should be identical (within numerical precision)
            self.assertTrue(
                jnp.allclose(output, expected_output, atol=1e-6),
                f"Multi-GPU and single-GPU outputs should be identical for {type(model).__name__}",
            )

            if is_verbose():
                print(f"{type(model).__name__} multi-GPU eval test passed")

    def test_lipschitz_constraint_multi_gpu(self):
        """Test that Lipschitz constraints are maintained across multiple GPUs"""
        rng = nnx.Rngs(params=jax.random.key(0))

        spectral_layer = SpectralLinear(100, 50, rngs=rng)
        ortho_layer = OrthoLinear(100, 50, rngs=rng)

        # Test data
        key = jax.random.key(123)
        x1 = jax.random.normal(key, (8, 100))
        x2 = jax.random.normal(jax.random.split(key)[0], (8, 100))

        @nnx.pmap(in_axes=(None, 0), out_axes=0)
        def compute_output(model, x):
            return model(x)

        # Shard inputs
        x1_sharded = shard_batch(x1, self.num_devices)
        x2_sharded = shard_batch(x2, self.num_devices)

        for layer, name in [
            (spectral_layer, "SpectralLinear"),
            (ortho_layer, "OrthoLinear"),
        ]:
            # Compute outputs on multiple GPUs
            y1_sharded = compute_output(layer, x1_sharded)
            y2_sharded = compute_output(layer, x2_sharded)

            # Reshape back
            y1 = y1_sharded.reshape(-1, 50)
            y2 = y2_sharded.reshape(-1, 50)

            # Check Lipschitz constraint: ||f(x1) - f(x2)|| <= ||x1 - x2||
            input_diff = jnp.linalg.norm(x1 - x2, axis=-1)
            output_diff = jnp.linalg.norm(y1 - y2, axis=-1)

            lipschitz_satisfied = jnp.all(
                output_diff <= input_diff + 1e-5
            )  # small tolerance

            self.assertTrue(
                lipschitz_satisfied,
                f"{name} should satisfy Lipschitz constraint on multi-GPU",
            )

            if is_verbose():
                max_ratio = jnp.max(output_diff / (input_diff + 1e-8))
                print(f"{name} max Lipschitz ratio: {max_ratio:.6f}")


class TestConvergence(unittest.TestCase):
    def test_linear(self):
        rng = nnx.Rngs(params=jax.random.key(0))
        trained = []
        for lin_layer in [
            SpectralLinear(10, 2, rngs=rng),
            OrthoLinear(10, 2, rngs=rng),
        ]:
            key = jax.random.key(2025)
            a = jax.random.uniform(key, (100, 10))
            b = jax.random.uniform(key, (100, 2))

            # Create model and optimizer
            model = lin_layer
            optimizer = nnx.Optimizer(model, optax.adam(1e-2))

            # Initial loss
            initial_loss = jnp.mean((model(a) - b) ** 2)

            # Training loop
            for _ in range(100):

                def loss_fn(m):
                    pred = m(a)
                    return jnp.mean((pred - b) ** 2)

                loss, grads = nnx.value_and_grad(loss_fn)(model)
                optimizer.update(grads)

            # Final loss
            final_loss = jnp.mean((model(a) - b) ** 2)

            # Check convergence
            converged = final_loss < initial_loss
            trained.append(converged)

            if is_verbose():
                print(
                    f"{type(model).__name__} - Initial: {initial_loss:.6f}, Final: {final_loss:.6f}"
                )

        trained = all(trained)
        self.assertTrue(trained)


if __name__ == "__main__":
    unittest.main()

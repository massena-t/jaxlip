# tests/batch_operations.py
import os
import sys
import unittest


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


# ensure project root is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from jaxlip.batchop import BatchCentering, BatchCentering2d, LayerCentering
import jax.numpy as jnp


class TestBatchCentering(unittest.TestCase):
    tol = 1e-3

    def _close(self, actual, expected, msg=None):
        """Elementwise comparison within tolerance, with useful error on failure."""
        under = actual < expected - self.tol
        over = actual > expected + self.tol
        correct = ~(under | over)
        if not correct.all():
            diff = actual - expected
            raise AssertionError(
                f"{msg or 'Mismatch'}:\n"
                f"expected:\n{expected}\n"
                f"actual:\n{actual}\n"
                f"diff:\n{diff}"
            )

    def test_1d(self):
        layer = BatchCentering(num_features=3)

        a = jnp.array(
            [
                [3, 2, 1],
                [3, 2, 1],
                [3, 2, 1],
            ],
        )  # shape (3,3)

        # training mode: EMA should track per-feature mean over reduction axes
        layer.train()
        out = layer(a)

        # Since all rows are identical, centering should zero them (bias is zero)
        self._close(
            out, jnp.zeros_like(a), msg="Initial centering failed for identical input"
        )
        self.assertEqual(a.shape, out.shape)

        # Check that the running mean has expected shape (per-feature)
        self.assertEqual(
            layer.mean.shape, (a.shape[-1],)
        )  # feature axis is last by default

        # Run many times to stabilize EMA
        for _ in range(1000):
            _ = layer(a)

        # Capture stabilized mean and switch to eval
        mean_val = layer.mean
        layer.eval()

        # In eval, output should be a - mean
        expected_eval = a - mean_val.reshape((1, -1))  # broadcast over batch
        actual_eval = layer(a)
        self._close(actual_eval, expected_eval, msg="Eval-mode centering mismatch")

        # New input with different per-feature values
        b = jnp.array(
            [
                [1, 2, 1],
                [1, 2, 1],
                [1, 2, 1],
            ],
        )
        # For this b, per-feature mean is [1,2,1]; subtracting stabilized previous mean (from a)
        expected_b = b - mean_val.reshape((1, -1))
        actual_b = layer(b)
        self._close(actual_b, expected_b, msg="Centering failed for second input")

    def test_2d(self):
        # NHWC: batch=3, height=2, width=4, channels=3 (varying dims to avoid all 2s)
        layer = BatchCentering2d(num_channels=3)

        # a has constant per-channel values [5,1,7]
        channel_vals_a = jnp.array([5, 1, 7], dtype=jnp.float32)  # shape (3,)
        a = (
            jnp.ones((3, 2, 4, 1), dtype=jnp.float32) * channel_vals_a
        )  # broadcast to (3,2,4,3)

        layer.train()
        out = layer(a)

        # Identical per-channel values => centering subtracts the per-channel mean, yielding zeros
        self.assertEqual(out.shape, a.shape)
        self._close(
            out, jnp.zeros_like(a), msg="Initial centering failed for identical input"
        )

        # Mean should be per-channel (C,)
        self.assertEqual(layer.mean.shape, (a.shape[-1],))

        # Stabilize EMA on 'a'
        for _ in range(1000):
            _ = layer(a)

        mean_val = layer.mean  # expected to be close to [5,1,7]
        layer.eval()

        # Eval-mode: subtract per-channel mean broadcast to NHWC
        mean_broadcast = mean_val.reshape((1, 1, 1, -1))
        expected_eval = a - mean_broadcast
        actual_eval = layer(a)
        self._close(actual_eval, expected_eval, msg="Eval-mode centering mismatch")

        # Now test a new input 'b' with different constant per-channel values [2,4,6].
        # To get zero after centering, re-enter train mode and stabilize on b, then eval.
        b_channel_vals = jnp.array([2, 4, 6], dtype=jnp.float32)
        b = (
            jnp.ones((3, 2, 4, 1), dtype=jnp.float32) * b_channel_vals
        )  # shape (3,2,4,3)

        layer.train()
        for _ in range(1000):
            _ = layer(b)

        layer.eval()
        actual_b = layer(b)
        expected_b = jnp.zeros_like(b)
        self._close(actual_b, expected_b, msg="Centering failed for second input")


class TestLayerCentering(unittest.TestCase):
    tol = 1e-3

    def _close(self, actual, expected, msg=None):
        """Elementwise comparison within tolerance, with useful error on failure."""
        under = actual < expected - self.tol
        over = actual > expected + self.tol
        correct = ~(under | over)
        if not correct.all():
            diff = actual - expected
            raise AssertionError(
                f"{msg or 'Mismatch'}:\n"
                f"expected:\n{expected}\n"
                f"actual:\n{actual}\n"
                f"diff:\n{diff}"
            )

    def test_1d_default_reduction(self):
        # Default reduction_axes=-1 (last axis)
        layer = LayerCentering()

        a = jnp.array(
            [
                [3, 2, 1],
                [6, 4, 2],
                [9, 6, 3],
            ]
        )  # shape (3,3)

        # Training mode: EMA should track mean over reduction axes
        layer.use_running_average = False
        out = layer(a)

        # For reduction_axes=-1, mean is computed per row, then subtracted
        # Row means: [2, 4, 6], so expected output is a - [[2],[4],[6]]
        expected = a - jnp.mean(a, axis=-1, keepdims=True)
        self._close(
            out, expected, msg="Initial centering failed with default reduction"
        )
        self.assertEqual(a.shape, out.shape)

        # Check that running mean has correct shape (keeps dims from keepdims=True)
        self.assertEqual(layer.mean.shape, (a.shape[0], 1))

        # Run many times to stabilize EMA
        for _ in range(1000):
            _ = layer(a)

        # Capture stabilized mean and switch to eval
        mean_val = layer.mean
        layer.use_running_average = True

        # In eval, output should be a - stored_mean
        expected_eval = a - mean_val
        actual_eval = layer(a)
        self._close(actual_eval, expected_eval, msg="Eval-mode centering mismatch")

        # New input with different values
        b = jnp.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        expected_b = b - mean_val
        actual_b = layer(b)
        self._close(actual_b, expected_b, msg="Centering failed for second input")

    def test_2d_batch_reduction(self):
        # Test with reduction_axes=0 (batch axis)
        layer = LayerCentering(reduction_axes=0)

        # Create input where batch mean is easy to compute
        a = jnp.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ]
        )  # shape (3,2,2)

        layer.use_running_average = False
        out = layer(a)

        # Mean over axis 0: [[5,6], [7,8]]
        expected = a - jnp.mean(a, axis=0, keepdims=True)
        self._close(out, expected, msg="Batch reduction centering failed")
        self.assertEqual(a.shape, out.shape)

        # Check mean shape with keepdims
        self.assertEqual(layer.mean.shape, (1, a.shape[1], a.shape[2]))

        # Stabilize EMA
        for _ in range(1000):
            _ = layer(a)

        mean_val = layer.mean
        layer.use_running_average = True

        # Test eval mode
        expected_eval = a - mean_val
        actual_eval = layer(a)
        self._close(
            actual_eval, expected_eval, msg="Eval-mode batch centering mismatch"
        )

    def test_multiple_reduction_axes(self):
        # Test with reduction_axes=(0,2)
        layer = LayerCentering(reduction_axes=(0, 2))

        a = jnp.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ]
        )  # shape (2,2,3)

        layer.use_running_average = False
        out = layer(a)

        # Mean over axes (0,2): shape should be (1,2,1) due to keepdims
        expected = a - jnp.mean(a, axis=(0, 2), keepdims=True)
        self._close(out, expected, msg="Multiple axes reduction failed")
        self.assertEqual(a.shape, out.shape)

        # Check mean shape
        self.assertEqual(layer.mean.shape, (1, a.shape[1], 1))

        # Test stability and eval mode
        for _ in range(1000):
            _ = layer(a)

        mean_val = layer.mean
        layer.use_running_average = True

        expected_eval = a - mean_val
        actual_eval = layer(a)
        self._close(
            actual_eval, expected_eval, msg="Multi-axis eval centering mismatch"
        )

    def test_momentum_update(self):
        # Test that momentum parameter affects EMA updates
        layer = LayerCentering(momentum=0.5)  # Lower momentum = faster updates

        a = jnp.array([[1.0, 2.0, 3.0]])
        b = jnp.array([[10.0, 20.0, 30.0]])

        layer.use_running_average = False

        # First pass with 'a'
        _ = layer(a)
        mean_after_a = layer.mean.copy()

        # Second pass with 'b' - should update mean significantly with momentum=0.5
        _ = layer(b)
        mean_after_b = layer.mean.copy()

        # Check that mean moved substantially toward b's mean
        b_mean = jnp.mean(b, axis=-1, keepdims=True)
        expected_update = 0.5 * mean_after_a + 0.5 * b_mean
        self._close(mean_after_b, expected_update, msg="Momentum update failed")


if __name__ == "__main__":
    unittest.main()

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
            out, a - 0.1 * a, msg="Initial centering failed for identical input"
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
            out, a - 0.1 * a, msg="Initial centering failed for identical input"
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

    def test_default_reduction_axes(self):
        layer = LayerCentering()

        # Test with 2D input (batch, features)
        x = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )

        out = layer(x)

        # Default reduction_axes=-1 means centering along last axis (features)
        # Each row should be centered around its mean
        expected = x - jnp.mean(x, axis=-1, keepdims=True)
        self._close(out, expected, msg="Default reduction axes centering failed")
        self.assertEqual(out.shape, x.shape)

    def test_specific_reduction_axes(self):
        layer = LayerCentering(reduction_axes=0)

        x = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )

        out = layer(x)

        # reduction_axes=0 means centering along axis 0 (batch)
        expected = x - jnp.mean(x, axis=0, keepdims=True)
        self._close(out, expected, msg="Specific reduction axes centering failed")
        self.assertEqual(out.shape, x.shape)

    def test_multiple_reduction_axes(self):
        layer = LayerCentering(reduction_axes=(0, 1))

        # 3D input: batch, height, width
        x = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        out = layer(x)

        # Centering along axes (0, 1)
        expected = x - jnp.mean(x, axis=(0, 1), keepdims=True)
        self._close(out, expected, msg="Multiple reduction axes centering failed")
        self.assertEqual(out.shape, x.shape)

    def test_dtype_conversion(self):
        layer = LayerCentering(dtype=jnp.float16)

        x = jnp.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=jnp.float32,
        )

        out = layer(x)

        # Mean should be converted to float16
        mean = jnp.mean(x, axis=-1, keepdims=True).astype(jnp.float16)
        expected = x - mean
        self._close(out, expected, msg="Dtype conversion centering failed")
        self.assertEqual(out.shape, x.shape)

    def test_negative_axes(self):
        layer = LayerCentering(reduction_axes=-2)

        # 3D input
        x = jnp.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        out = layer(x)

        # -2 should be converted to axis 1 for 3D input
        expected = x - jnp.mean(x, axis=1, keepdims=True)
        self._close(out, expected, msg="Negative axes centering failed")
        self.assertEqual(out.shape, x.shape)

    def test_1d_input(self):
        layer = LayerCentering()

        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        out = layer(x)

        # For 1D input with default reduction_axes=-1, center around global mean
        expected = x - jnp.mean(x, axis=-1, keepdims=True)
        self._close(out, expected, msg="1D input centering failed")
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()

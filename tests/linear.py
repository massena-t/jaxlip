import os
import sys


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
from jaxlip.linear import SpectralLinear, OrthoLinear, l2_normalize
from jaxlip.newton_schulz import orthogonalize
import jax
from jax.numpy.linalg import norm
import jax.numpy as jnp
import flax.nnx as nnx
import optax


class TestSpectralLinearLayer(unittest.TestCase):
    upper_tol = 1e-3
    lower_tol = 1e-2

    def test_cache(self):
        vals = []
        spec = []
        for din in [2, 10, 50]:
            for dout in [2, 10, 50]:
                rng = nnx.Rngs(params=jax.random.key(0))
                lin_layer = SpectralLinear(din, dout, rngs=rng)
                lin_layer._cache_params()
                spec.append(float(norm(lin_layer.cache, ord=2)))
                vals.append(norm(lin_layer.cache, ord=2) > 1.0 + self.upper_tol)
                vals.append(norm(lin_layer.cache, ord=2) < 1.0 - self.lower_tol)
                lin_layer = None
        valid = any(vals)
        if is_verbose():
            print("Cached spectral weights l2 norm: \n", spec)
        self.assertFalse(valid)

    def test_weight(self):
        vals = []
        spec = []
        for din in [2, 10, 50]:
            for dout in [2, 10, 50]:
                rng = nnx.Rngs(params=jax.random.key(0))
                lin_layer = SpectralLinear(din, dout, rngs=rng)
                spec.append(float(norm(l2_normalize(lin_layer.w), ord=2)))
                vals.append(
                    norm(l2_normalize(lin_layer.w), ord=2) > 1.0 + self.upper_tol
                )
                vals.append(
                    norm(l2_normalize(lin_layer.w), ord=2) < 1.0 - self.lower_tol
                )
                lin_layer = None
        valid = any(vals)
        if is_verbose():
            print("Reparametrized spectral weights l2 norm: \n", spec)
        self.assertFalse(valid)


class TestOrthoLinearLayer(unittest.TestCase):
    upper_tol = 1e-3
    lower_tol = 1e-2

    def test_cache(self):
        vals = []
        spec = []
        for din in [2, 10, 50]:
            for dout in [2, 10, 50]:
                rng = nnx.Rngs(params=jax.random.key(0))
                lin_layer = OrthoLinear(din, dout, rngs=rng)
                lin_layer._cache_params()
                spec.append(float(norm(lin_layer.cache, ord=2)))
                vals.append(norm(lin_layer.cache, ord=2) > 1.0 + self.upper_tol)
                vals.append(norm(lin_layer.cache, ord=2) < 1.0 - self.lower_tol)
                lin_layer = None
        valid = any(vals)
        if is_verbose():
            print("Cached ortho weights l2 norm: \n", spec)
        self.assertFalse(valid)

    def test_weight(self):
        vals = []
        spec = []
        for din in [2, 10, 50]:
            for dout in [2, 10, 50]:
                rng = nnx.Rngs(params=jax.random.key(0))
                lin_layer = OrthoLinear(din, dout, rngs=rng)
                spec.append(float(norm(orthogonalize(lin_layer.w), ord=2)))
                vals.append(
                    norm(orthogonalize(lin_layer.w), ord=2) > 1.0 + self.upper_tol
                )
                vals.append(
                    norm(orthogonalize(lin_layer.w), ord=2) < 1.0 - self.lower_tol
                )
        valid = any(vals)
        if is_verbose():
            print("Reparametrized ortho weights l2 norm: \n", spec)
        self.assertFalse(valid)


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
            # Overfit random data
            optimizer = nnx.Optimizer(lin_layer, optax.sgd(1e-3), wrt=nnx.Param)
            params = nnx.state(lin_layer, nnx.Param)

            @nnx.jit
            def train_step(lin_layer, optimizer, x, y):
                def loss_fn(m):
                    y_pred = m(x)
                    return ((y_pred - y) ** 2).mean()

                loss, grads = nnx.value_and_grad(loss_fn)(lin_layer)
                optimizer.update(grads)
                return loss

            init_loss = ((b - lin_layer(a)) ** 2).mean()
            # Iterate over all samples
            for sample, label in zip(a, b):
                loss = train_step(lin_layer, optimizer, sample, label)
            post_loss = ((b - lin_layer(a)) ** 2).mean()
            trained.append(post_loss < init_loss)
            if is_verbose():
                print(f"Init loss: {init_loss}, after training loss: {post_loss}")
        trained = all(trained)
        self.assertTrue(trained)


class TestAttack(unittest.TestCase):
    def test_attack(self):
        rng = nnx.Rngs(params=jax.random.key(0))
        validity = []
        for lin_layer in [
            SpectralLinear(10, 2, rngs=rng),
            OrthoLinear(10, 2, rngs=rng),
        ]:
            key = jax.random.key(2025)
            origin = jax.random.uniform(key, 10)
            pert = 1e-6 * jax.random.uniform(key, 10)
            ref = lin_layer(origin)

            def loss_fn(p):
                out = lin_layer(origin + p)
                return jnp.linalg.norm(out - ref, ord=2)

            out = lin_layer(origin + pert)
            grad = jax.grad(loss_fn)(pert)  # same shape as pert: [1,H,W,C]
            pert = pert - 1e-2 * grad
            assert norm(grad, ord=2) > 0
            lip_estimate = loss_fn(pert) / jnp.linalg.norm(origin - pert, ord=2)

            if is_verbose():
                print("Lipschitz constant estimate is: ", lip_estimate)

            valid = lip_estimate < 1.0
            validity.append(valid)
        self.assertTrue(all(validity))


if __name__ == "main":
    unittest.main()

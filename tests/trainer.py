import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import unittest
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxlip.trainer import Trainer
from jaxlip.loss import LseHKRMulticlassLoss, LossXEnt, TauCCE
from models.mixer import MLPMixer
from jaxlip.linear import ParametrizedLinear
from jaxlip.batchop import BatchCentering


def is_verbose():
    return any(arg in ("-v", "--verbose") for arg in sys.argv)


class SimpleTestModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = ParametrizedLinear(
            4, 8, bias=True, parametrization="spectral", rngs=rngs
        )
        self.linear2 = ParametrizedLinear(
            8, 2, bias=True, parametrization="spectral", rngs=rngs
        )
        self.bc = BatchCentering(8)

    def __call__(self, x, ws=None):
        x = self.linear1(x, ws)
        x = jax.nn.relu(x)
        x = self.bc(x)
        x = self.linear2(x, ws)
        return x


class TestTrainerInitialization(unittest.TestCase):
    def setUp(self):
        self.rng = nnx.Rngs(params=jax.random.key(42))
        self.model = SimpleTestModel(self.rng)
        self.loss = TauCCE(temperature=1.0)

    def test_single_gpu_initialization(self):
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        self.assertFalse(trainer.distributed)
        self.assertFalse(trainer.vmap_reparametrizations)
        self.assertFalse(trainer.reparam_init)
        self.assertEqual(trainer.model, self.model)
        self.assertEqual(trainer.optimizer, optimizer)
        self.assertEqual(trainer.loss, self.loss)

    def test_multi_gpu_initialization(self):
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=True,
            vmap_reparametrizations=False,
        )

        self.assertTrue(trainer.distributed)
        self.assertFalse(trainer.vmap_reparametrizations)

    def test_vmap_reparametrizations_initialization(self):
        optimizer = optax.adam(1e-3)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=True,
            vmap_reparametrizations=True,
        )

        self.assertTrue(trainer.distributed)
        self.assertTrue(trainer.vmap_reparametrizations)
        self.assertFalse(trainer.reparam_init)


class TestTrainerModes(unittest.TestCase):
    def setUp(self):
        self.rng = nnx.Rngs(params=jax.random.key(42))
        self.model = SimpleTestModel(self.rng)
        self.loss = TauCCE(temperature=1.0)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))

    def test_train_mode_basic(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.train()
        self.assertFalse(self.model.bc.use_running_average)

    def test_eval_mode_basic(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.eval()
        self.assertTrue(self.model.bc.use_running_average)

    def test_vmap_train_initialization(self):
        optimizer = optax.adam(1e-3)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=True,
            vmap_reparametrizations=True,
        )

        self.assertFalse(hasattr(trainer, "buckets"))
        self.assertFalse(hasattr(trainer, "params"))
        self.assertFalse(hasattr(trainer, "opt_state"))
        self.assertFalse(hasattr(trainer, "_train_step_vmap"))
        self.assertFalse(hasattr(trainer, "_eval_step_vmap"))

        # First call to train() should initialize vmap structures
        trainer.train()

        self.assertTrue(trainer.reparam_init)
        self.assertTrue(hasattr(trainer, "buckets"))
        self.assertTrue(hasattr(trainer, "params"))
        self.assertTrue(hasattr(trainer, "opt_state"))
        self.assertTrue(hasattr(trainer, "_train_step_vmap"))
        self.assertTrue(hasattr(trainer, "_eval_step_vmap"))


class TestTrainerSteps(unittest.TestCase):
    def setUp(self):
        self.rng = nnx.Rngs(params=jax.random.key(42))
        self.model = SimpleTestModel(self.rng)
        self.loss = TauCCE(temperature=1.0)

        # Create simple test data
        self.batch_size = 4
        self.x = jax.random.normal(jax.random.key(0), (self.batch_size, 4))
        self.y = jax.random.randint(jax.random.key(1), (self.batch_size,), 0, 2)

    def test_single_gpu_train_step(self):
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.train()

        def snapshot_lin_params(model):
            w = {}
            b = {}
            for _, m in model.iter_modules():
                if isinstance(m, ParametrizedLinear):
                    # copy to detach from in-place updates
                    w[m._uid] = jnp.array(m.w.value).copy()
                    if m.bias:
                        b[m._uid] = jnp.array(m.b.value).copy()
            return w, b

        def total_l2_change(before, after):
            # sum L2 across all tensors in dict
            return float(
                sum(jnp.linalg.norm(after[k] - before[k]) for k in before.keys())
            )

        orig_w, orig_b = snapshot_lin_params(trainer.model)

        loss_value = trainer.train_step(self.x, self.y)

        new_w, new_b = snapshot_lin_params(trainer.model)

        w_delta = total_l2_change(orig_w, new_w)
        b_delta = total_l2_change(orig_b, new_b)

        self.assertGreater(w_delta, 0.0, "weights did not change")
        self.assertGreater(b_delta, 0.0, "biases did not change")

        # Check that loss is a scalar
        self.assertEqual(loss_value.shape, ())
        self.assertTrue(jnp.isfinite(loss_value))

    def test_single_gpu_eval_step(self):
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.eval()
        acc, cra = trainer.eval_step(self.x, self.y)

        # Check that metrics are scalars and in valid range
        self.assertEqual(acc.shape, ())
        self.assertEqual(cra.shape, ())
        self.assertTrue(0.0 <= acc <= 1.0)
        self.assertTrue(0.0 <= cra <= 1.0)

    def test_vmap_incompatible_with_single_gpu(self):
        optimizer = optax.adam(1e-3)
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=True,
        )

        trainer.train()

        # Should raise assertion error for single GPU + vmap
        with self.assertRaises(AssertionError):
            trainer.train_step(self.x, self.y)

        with self.assertRaises(AssertionError):
            trainer.eval_step(self.x, self.y)


class TestTrainerLossFunctions(unittest.TestCase):
    def setUp(self):
        self.rng = nnx.Rngs(params=jax.random.key(42))
        self.model = SimpleTestModel(self.rng)

        # Create test data
        self.batch_size = 4
        self.x = jax.random.normal(jax.random.key(0), (self.batch_size, 4))
        self.y = jax.random.randint(jax.random.key(1), (self.batch_size,), 0, 2)

    def test_xent_loss(self):
        loss = LossXEnt(offset=0.1, temperature=70.0)
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.train()
        loss_value = trainer.train_step(self.x, self.y)

        self.assertTrue(jnp.isfinite(loss_value))

    def test_tau_cce_loss(self):
        loss = TauCCE(temperature=1.0)
        optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))
        trainer = Trainer(
            model=self.model,
            optimizer=optimizer,
            loss=loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        trainer.train()
        loss_value = trainer.train_step(self.x, self.y)

        self.assertTrue(jnp.isfinite(loss_value))


class TestTrainerWorkflow(unittest.TestCase):
    def setUp(self):
        self.rng = nnx.Rngs(params=jax.random.key(42))
        self.model = SimpleTestModel(self.rng)
        self.loss = TauCCE(temperature=1.0)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(1e-3))

        # Create test data
        self.batch_size = 8
        self.x = jax.random.normal(jax.random.key(0), (self.batch_size, 4))
        self.y = jax.random.randint(jax.random.key(1), (self.batch_size,), 0, 2)

    def test_complete_training_workflow(self):
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss=self.loss,
            distributed=False,
            vmap_reparametrizations=False,
        )

        # Training phase
        trainer.train()
        self.assertFalse(self.model.bc.use_running_average)

        # Record initial accuracy
        trainer.eval()
        initial_acc, _ = trainer.eval_step(self.x, self.y)

        # Training steps
        trainer.train()
        losses = []
        for _ in range(5):
            loss_value = trainer.train_step(self.x, self.y)
            losses.append(float(loss_value))

        # Evaluation phase
        trainer.eval()
        self.assertTrue(self.model.bc.use_running_average)
        final_acc, final_cra = trainer.eval_step(self.x, self.y)

        # Verify metrics are computed correctly
        self.assertTrue(0.0 <= final_acc <= 1.0)
        self.assertTrue(0.0 <= final_cra <= 1.0)
        self.assertTrue(final_cra <= final_acc)  # CRA should be <= accuracy

        # Check that losses are finite
        self.assertTrue(all(jnp.isfinite(loss) for loss in losses))

        if is_verbose():
            print(f"Initial accuracy: {initial_acc:.3f}")
            print(f"Final accuracy: {final_acc:.3f}")
            print(f"Final CRA: {final_cra:.3f}")
            print(f"Loss progression: {losses}")


if __name__ == "__main__":
    unittest.main()

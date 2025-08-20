import jax
import jax.numpy as jnp
from flax import nnx
from typing import Union, Callable, Any
from jaxlip.reparametrizer import (
    collect_buckets,
    parametrize_vmapped_cached,
    parametrize_from_params_cached,
)
from jaxlip.utils import (
    cache_model_params,
    uncache_model_params,
    inject_biases,
    load_params_into_model,
)
from jaxlip.linear import ParametrizedLinear
import optax

state_axes = nnx.StateAxes(
    {
        nnx.Param: None,
        nnx.Cache: None,
        nnx.BatchStat: None,
    }
)


@nnx.jit
def train_step_single_gpu_base_reparam(
    model: nnx.Module,
    optimizer,
    loss: Callable,
    x: Any,
    y: Any,
):
    def loss_fn(m):
        logits = m(x)
        return loss(logits, y)

    loss_value, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss_value


@nnx.jit
def eval_step_single_gpu_base_reparam(
    model: nnx.Module,
    x: Any,
    y: Any,
):
    logits = model(x)
    sorted_logits = jnp.sort(logits, axis=-1)
    max1 = sorted_logits[..., -1]
    max2 = sorted_logits[..., -2]
    margin = max1 - max2
    correct = jnp.argmax(logits, axis=-1) == y
    is_wide = margin > (jnp.sqrt(2.0) * 36.0 / 255.0)
    robust = jnp.logical_and(correct, is_wide)
    return jnp.mean(correct), jnp.mean(robust)


@nnx.pmap(
    in_axes=(state_axes, None, None, 0, 0),
    out_axes=(state_axes, None, None),
    axis_name="device",
)
def train_step_multi_gpu_base_reparam(
    model,
    optimizer,
    loss: Callable,
    x: Any,
    y: Any,
):
    def loss_fn(m):
        logits = m(x)
        return loss(logits, y)

    loss_value, grads = nnx.value_and_grad(loss_fn)(model)
    loss_value = jax.lax.pmean(loss_value, axis_name="device")
    grads = jax.lax.pmean(grads, axis_name="device")
    optimizer.update(grads)
    return model, optimizer, loss_value


@nnx.pmap(in_axes=(None, 0, 0), out_axes=None, axis_name="device")
def eval_step_multi_gpu_base_reparam(model, x, y):
    logits = model(x)
    preds = jnp.argmax(logits, axis=-1)

    correct_mask = preds == y
    correct_sum = correct_mask.sum()

    total_sum = jnp.asarray(correct_mask.size, jnp.int32)
    correct_sum = jax.lax.psum(correct_sum, "device")
    total_sum = jax.lax.psum(total_sum, "device")

    sorted_logits = jnp.sort(logits, axis=-1)
    margin = sorted_logits[..., -1] - sorted_logits[..., -2]
    is_wide = margin > (jnp.sqrt(2.0) * 36.0 / 255.0)

    robust_sum = jnp.logical_and(is_wide, correct_mask).sum()
    robust_sum = jax.lax.psum(robust_sum, "device")

    acc = correct_sum / total_sum
    cra = robust_sum / total_sum
    return acc, cra


# @nnx.pmap(
#     in_axes=(state_axes, None, None, None, None, 0, 0),
#     out_axes=(state_axes, None, None, None),
#     axis_name="device",
# )
# def train_step_multi_gpu_vmap_reparam(model, params, opt_state, buckets, loss, x, y):
#     def loss_from_params(m, p, x, y):
#         ws = parametrize_from_params_cached(buckets, p)
#         inject_biases(m, p, ws)
#         logits = m(x, ws)
#         return loss(logits, y)
#
#     loss_value, grads = nnx.value_and_grad(loss_from_params, argnums=1)(
#         model, params, x, y
#     )
#     loss_value = jax.lax.pmean(loss_value, axis_name="device")
#     grads = jax.lax.pmean(grads, axis_name="device")
#
#     updates, opt_state = tx.update(grads, opt_state, params)
#     params = optax.apply_updates(params, updates)
#     return model, params, opt_state, loss_value
#
# @nnx.pmap(in_axes=(None, None, 0, 0), out_axes=None, axis_name="device")
# def eval_step_multi_gpu_base_reparam(model, params, x, y):
#     ws = parametrize_from_params_cached(buckets, params)
#     inject_biases(model, params, ws)
#     logits = model(x, ws)
#
#     preds = jnp.argmax(logits, axis=-1)
#     correct_mask = preds == y
#     correct_sum = correct_mask.sum()
#     total_sum = jnp.asarray(correct_mask.size, jnp.int32)
#     correct_sum = jax.lax.psum(correct_sum, "device")
#     total_sum = jax.lax.psum(total_sum, "device")
#
#     sorted_logits = jnp.sort(logits, axis=-1)
#     margin = sorted_logits[..., -1] - sorted_logits[..., -2]
#     is_wide = margin > (jnp.sqrt(2.0) * 36.0 / 255.0)
#     robust_sum = jnp.logical_and(is_wide, correct_mask).sum()
#     robust_sum = jax.lax.psum(robust_sum, "device")
#
#     acc = correct_sum / total_sum
#     cra = robust_sum / total_sum
#     return acc, cra


def make_vmap_steps(buckets, loss_fn, optimizer):
    @nnx.pmap(
        in_axes=(state_axes, None, None, 0, 0),  # model, params, opt_state, x, y
        out_axes=(state_axes, None, None, None),
        axis_name="device",
    )
    def train_step(model, params, opt_state, x, y):
        def loss_from_params(m, p, x, y):
            ws = parametrize_from_params_cached(buckets, p)
            inject_biases(m, p, ws)  # mutate in place, don’t reassign
            logits = m(x, ws)
            return loss_fn(logits, y)

        loss_val, grads = nnx.value_and_grad(loss_from_params, argnums=1)(
            model, params, x, y
        )
        loss_val = jax.lax.pmean(loss_val, "device")
        grads = jax.lax.pmean(grads, "device")

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return model, params, opt_state, loss_val

    @nnx.pmap(
        in_axes=(None, None, 0, 0),  # model, params, x, y
        out_axes=None,
        axis_name="device",
    )
    def eval_step(model, params, x, y):
        ws = parametrize_from_params_cached(buckets, params)
        inject_biases(model, params, ws)
        logits = model(x, ws)

        preds = jnp.argmax(logits, axis=-1)
        correct_mask = preds == y
        correct_sum = jax.lax.psum(correct_mask.sum(), "device")
        total_sum = jax.lax.psum(jnp.asarray(correct_mask.size, jnp.int32), "device")

        sorted_logits = jnp.sort(logits, axis=-1)
        margin = sorted_logits[..., -1] - sorted_logits[..., -2]
        is_wide = margin > (jnp.sqrt(2.0) * 36.0 / 255.0)
        robust_sum = jax.lax.psum(
            jnp.logical_and(is_wide, correct_mask).sum(), "device"
        )

        acc = correct_sum / total_sum
        cra = robust_sum / total_sum
        return acc, cra

    return train_step, eval_step


class Trainer:
    def __init__(
        self, model, optimizer, loss, distributed, vmap_reparametrizations=False
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.distributed = distributed
        self.vmap_reparametrizations = vmap_reparametrizations
        self.reparam_init = False

        # if self.vmap_reparametrizations:
        #     assert isinstance(optimizer, optax.Optimizer)

    def train(self):
        uncache_model_params(self.model)
        self.model.train()
        if self.vmap_reparametrizations and not self.reparam_init:
            self.buckets = collect_buckets(self.model)
            params_init = {
                **{
                    uid: w.value
                    for name, by_sig in self.buckets.items()
                    for _, items in by_sig.items()
                    for (uid, w, _h) in items
                },
                **{
                    f"b:{m._uid}": m.b.value
                    for _, m in self.model.iter_modules()
                    if isinstance(m, ParametrizedLinear) and m.bias
                },
            }
            self.params = params_init
            self.opt_state = self.optimizer.init(params_init)
            self._train_step_vmap, self._eval_step_vmap = make_vmap_steps(
                self.buckets, self.loss, self.optimizer
            )
            self.reparam_init = True
        pass

    def eval(self):
        self.model.eval()
        if self.vmap_reparametrizations:
            load_params_into_model(self.model, self.params)
        cache_model_params(self.model, verbose=False)
        pass

    def train_step(self, x, y):
        if not self.distributed:
            assert self.vmap_reparametrizations is False, (
                "vmap reparametrizations are not supported in single GPU mode"
            )
            loss_value = train_step_single_gpu_base_reparam(
                self.model, self.optimizer, self.loss, x, y
            )
        elif self.distributed:
            if not self.vmap_reparametrizations:
                self.model, self.optimizer, loss_value = (
                    train_step_multi_gpu_base_reparam(
                        self.model,
                        self.optimizer,
                        self.loss,
                        x,
                        y,
                    )
                )
            else:
                assert all(
                    hasattr(self, p) for p in ["buckets", "params", "opt_state"]
                ), (
                    "vmap reparametrizations require buckets, params and opt_state to be initialized, call `train` first"
                )
                self.model, self.params, self.opt_state, loss_value = (
                    self._train_step_vmap(self.model, self.params, self.opt_state, x, y)
                )
                return loss_value
        else:
            raise ValueError("Distributed training mode not set correctly")
        return loss_value

    def eval_step(self, x, y):
        if not self.distributed:
            assert self.vmap_reparametrizations is False, (
                "vmap reparametrizations are not supported in single GPU mode"
            )
            acc, cra = eval_step_single_gpu_base_reparam(self.model, x, y)
        else:
            if not self.vmap_reparametrizations:
                acc, cra = eval_step_multi_gpu_base_reparam(
                    self.model,
                    x,
                    y,
                )
            else:
                acc, cra = self._eval_step_vmap(self.model, self.params, x, y)
        return acc, cra

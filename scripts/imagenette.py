import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import nnx
from models.mixer import MLPMixer
import numpy as np
import albumentations as A
import cv2

from jaxlip.utils import cache_model_params, uncache_model_params
from utils.utils import get_model
from jax.tree_util import tree_map_with_path


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MLP-Mixer on Imagenette")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=75, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--temperature", type=float, default=70.0, help="Temperature for loss functions"
    )
    parser.add_argument("--alpha", type=float, default=0.985, help="Alpha for HKR loss")
    parser.add_argument(
        "--offset", type=float, default=0.1, help="Offset for cross-entropy loss"
    )
    parser.add_argument(
        "--detach_iter",
        type=int,
        default=20,
        help="Nb of TN iterations during training",
    )
    parser.add_argument(
        "--num_iters_train",
        type=int,
        default=50,
        help="Nb of TN iterations during training",
    )
    parser.add_argument(
        "--num_iters_eval",
        type=int,
        default=80,
        help="Nb of TN iterations during evaluation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        choices=["mixer", "convnet"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="hkr",
        choices=["hkr", "xent", "tau_cce"],
        help="Loss function to use",
    )
    args = parser.parse_args()
    return args


def preprocess(img, label):
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_crop(img, size=[128, 128, 3])  # same shape out
    return img, label


def get_datasets(bs):
    train_ds = (
        tfds.load("imagenette", split="train", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # deterministic
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # stochastic
        .shuffle(10_000)
        .batch(bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tfds.load("imagenette", split="validation", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # deterministic
        .shuffle(10_000)
        .batch(bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)


def main(args):
    rng = nnx.Rngs(params=jax.random.key(0))
    model = get_model(rng, args, dataset="imagenette")
    optimizer = nnx.Optimizer(model, optax.adam(args.lr))

    params = nnx.state(model, nnx.Param)
    total_params = (
        sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)) / 1e6
    )
    print(f"Nb params: {total_params:.2f}M")

    train_iter, test_iter = get_datasets(bs=args.batch_size)

    from jaxlip.loss import LseHKRMulticlassLoss, LossXEnt, TauCCE

    chosen_loss = {
        "hkr": LseHKRMulticlassLoss(alpha=args.alpha, temperature=args.temperature),
        "xent": LossXEnt(offset=args.offset, temperature=args.temperature),
        "tau_cce": TauCCE(temperature=args.temperature),
    }[args.loss_fn]

    @nnx.jit
    def train_step(model, optimizer, x, y, rescale_grads=True):
        def loss_fn(m):
            logits = m(x)
            return chosen_loss(logits, y)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def eval(model, x, y):
        logits = model(x)
        sorted_logits = jnp.sort(logits, axis=-1)
        max1 = sorted_logits[..., -1]
        max2 = sorted_logits[..., -2]
        margin = max1 - max2
        correct = jnp.argmax(logits, axis=-1) == y
        is_wide = margin > (jnp.sqrt(2.0) * 36.0 / 255.0)
        robust = jnp.logical_and(correct, is_wide)
        return jnp.mean(correct), jnp.mean(robust)

    model.train()
    for epoch in range(args.epochs):
        correct, total = 0, 0
        running_loss = 0
        for img, lbl in train_iter:
            x, y = jnp.array(img, jnp.float32), jnp.array(lbl, dtype=jnp.int32)
            running_loss += train_step(model, optimizer, x, y)

        cache_model_params(model, verbose=False)
        train_acc = train_cra = 0
        for img, lbl in train_iter:
            x, y = jnp.array(img, jnp.float32), jnp.array(lbl, dtype=jnp.int32)
            acc, rob = eval(model, x, y)
            train_acc += acc
            train_cra += rob
        train_acc /= len(train_iter)
        train_cra /= len(train_iter)

        val_acc = val_cra = 0
        for img, lbl in test_iter:
            x, y = jnp.array(img, jnp.float32), jnp.array(lbl, dtype=jnp.int32)
            acc, rob = eval(model, x, y)
            val_acc += acc
            val_cra += rob
        val_acc /= len(test_iter)
        val_cra /= len(test_iter)
        uncache_model_params(model)

        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_iter):.4f}")
        print(
            f"\t Train acc: {100 * train_acc:.3f}%, Train CRA: {100 * train_cra:.3f}%"
        )
        print(f"\t Val acc: {100 * val_acc:.3f}%, CRA: {100 * val_cra:.3f}%\n")
        print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)

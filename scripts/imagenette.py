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
        default=25,
        help="Nb of TN iterations during training",
    )
    parser.add_argument(
        "--num_iters_train",
        type=int,
        default=60,
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
        default="mixer",
        choices=["mixer", "convnet"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="tau_cce",
        choices=["hkr", "xent", "tau_cce"],
        help="Loss function to use",
    )
    parser.add_argument("--use_pmap", action="store_true")
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


def get_datasets(bs, use_pmap=False):
    num_devices = jax.local_device_count() if use_pmap else 1
    assert bs % num_devices == 0, (
        f"Batch size {bs} not divisible by {num_devices} devices"
    )
    per_device_bs = bs // num_devices

    train_ds = (
        tfds.load("imagenette", split="train", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # deterministic
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)  # stochastic
        .shuffle(10_000)
        .repeat()
        .batch(per_device_bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tfds.load("imagenette", split="validation", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)  # deterministic
        .shuffle(10_000)
        .repeat()
        .batch(per_device_bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)


def shard_batch(batch, num_devices):
    return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch)


def main(args):
    builder = tfds.builder("imagenette")
    info = builder.info
    num_train = info.splits["train"].num_examples
    num_val = info.splits["validation"].num_examples
    steps_per_epoch = num_train // args.batch_size
    val_steps = num_val // args.batch_size
    num_devices = jax.local_device_count() if args.use_pmap else 1
    print(f"Using {num_devices} device(s) for training")

    rng = nnx.Rngs(params=jax.random.key(0))
    model = get_model(rng, args, dataset="imagenette")
    optimizer = nnx.Optimizer(model, optax.adam(args.lr))

    params = nnx.state(model, nnx.Param)
    total_params = (
        sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)) / 1e6
    )
    print(f"Nb params: {total_params:.2f}M")

    train_loader, test_loader = get_datasets(bs=args.batch_size)

    from jaxlip.loss import LseHKRMulticlassLoss, LossXEnt, TauCCE

    chosen_loss = {
        "hkr": LseHKRMulticlassLoss(alpha=args.alpha, temperature=args.temperature),
        "xent": LossXEnt(offset=args.offset, temperature=args.temperature),
        "tau_cce": TauCCE(temperature=args.temperature),
    }[args.loss_fn]

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
            logits = m(x)
            return chosen_loss(logits, y)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        grads = jax.lax.pmean(grads, axis_name="device")
        optimizer.update(grads)
        return model, optimizer, loss

    @nnx.pmap(in_axes=(None, 0, 0), out_axes=(0, 0))
    def eval_step(model, x, y):
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
    for epoch in range(1, args.epochs + 1):
        test_acc = test_cra = train_loss = 0
        train_batches = 0

        train_iter = iter(train_loader)
        test_iter = iter(test_loader)

        for _ in range(steps_per_epoch):
            batch_images, batch_labels = [], []
            for _ in range(num_devices):
                images, labels = next(train_iter)
                batch_images.append(images)
                batch_labels.append(labels)

            x = jnp.array(np.concatenate(batch_images), dtype=jnp.float32)
            y = jnp.array(np.concatenate(batch_labels), dtype=jnp.int32)
            x = shard_batch(x, num_devices)
            y = shard_batch(y, num_devices)

            model, optimizer, loss = train_step(model, optimizer, x, y)
            train_loss += jnp.mean(loss)
            train_batches += 1

        cache_model_params(model, verbose=False)

        val_acc = val_cra = test_batches = 0

        for _ in range(val_steps):
            images, labels = next(test_iter)
            x = shard_batch(images, num_devices)
            y = shard_batch(labels, num_devices)
            acc, rob = eval_step(model, x, y)
            test_acc += float(acc.mean())
            test_cra += float(rob.mean())
            test_batches += 1
        test_acc /= test_batches
        test_cra /= test_batches

        uncache_model_params(model)
        print(f"Epoch {epoch}, loss: {train_loss / train_batches:.4f}")
        print(f"\t Test acc: {100 * test_acc:.3f}%, CRA: {100 * test_cra:.3f}%\n")
        print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)

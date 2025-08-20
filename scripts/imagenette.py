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
from jaxlip.linear import ParametrizedLinear


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
        "--model",
        type=str,
        default="mixer",
        choices=["mixer"],
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
    parser.add_argument("--vmap_reparams", action="store_true")
    args = parser.parse_args()
    return args


def preprocess(img, label):
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_crop(img, size=[128, 128, 3])
    return img, label


def get_datasets(bs, use_pmap=False):
    num_devices = jax.local_device_count() if use_pmap else 1
    assert bs % num_devices == 0, (
        f"Batch size {bs} not divisible by {num_devices} devices"
    )

    train_ds = (
        tfds.load("imagenette", split="train", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10_000)
        .batch(bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tfds.load("imagenette", split="validation", as_supervised=True)
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(10_000)
        .batch(bs, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)


def shard_batch(batch, num_devices):
    return jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch)


def main(args):
    builder = tfds.builder("imagenette")
    num_devices = jax.local_device_count() if args.use_pmap else 1
    print(f"Using {num_devices} device(s) for training")

    rng = nnx.Rngs(params=jax.random.key(0))
    model = get_model(rng, args, dataset="imagenette")
    params = nnx.state(model, nnx.Param)
    total_params = (
        sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)) / 1e6
    )
    print(f"Nb params: {total_params:.2f}M")

    train_loader, test_loader = get_datasets(bs=args.batch_size)

    from jaxlip.loss import LseHKRMulticlassLoss, LossXEnt, TauCCE

    loss = {
        "hkr": LseHKRMulticlassLoss(alpha=args.alpha, temperature=args.temperature),
        "xent": LossXEnt(offset=args.offset, temperature=args.temperature),
        "tau_cce": TauCCE(temperature=args.temperature),
    }[args.loss_fn]

    from jaxlip.trainer import Trainer

    if not args.vmap_reparams:
        optimizer = nnx.Optimizer(model, optax.adam(args.lr))
    else:
        optimizer = optax.adam(args.lr)

    trainer = Trainer(
        model,
        optimizer,
        loss,
        distributed=args.use_pmap,
        vmap_reparametrizations=args.vmap_reparams,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train()
        test_acc = test_cra = train_loss = 0
        train_batches = 0

        for x, y in train_loader:
            if args.use_pmap:
                x = shard_batch(x, num_devices)
                y = shard_batch(y, num_devices)
            loss_val = trainer.train_step(x, y)
            train_loss += float(loss_val)
            train_batches += 1

        trainer.eval()

        correct = robust = total = 0

        for x, y in test_loader:
            if args.use_pmap:
                x = shard_batch(x, num_devices)
                y = shard_batch(y, num_devices)
            acc, rob = trainer.eval_step(x, y)
            correct += float(acc)
            robust += float(rob)
            total += 1
        test_acc = correct / total
        test_cra = robust / total
        print(f"Epoch {epoch}, loss: {train_loss / train_batches:.4f}")
        print(f"\t Test acc: {100 * test_acc:.3f}%, CRA: {100 * test_cra:.3f}%\n")
        print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)

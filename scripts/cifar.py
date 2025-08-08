import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import jax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds
from flax import nnx
from models.mixer import MLPMixer
from models.convnet import ConvNet
import numpy as np
import albumentations as A
import cv2

from jaxlip.utils import cache_model_params, uncache_model_params
from utils.utils import get_model
import einops


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="MLP-Mixer on CIFAR-10")
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
        default="simple",
        choices=["simple", "mixer", "convnet"],
        help="Loss function to use",
    )
    parser.add_argument(
        "--loss_fn",
        type=str,
        default="tau_cce",
        choices=["hkr", "xent", "tau_cce"],
        help="Loss function to use",
    )
    args = parser.parse_args()
    return args


def get_datasets(bs):
    ds = tfds.load("cifar10", split=["train", "test"], as_supervised=True)
    train, test = ds
    train = train.shuffle(10_000).batch(bs, drop_remainder=True)
    test = test.batch(bs, drop_remainder=True)
    train_trfm = A.Compose(
        [
            A.RandomCrop(
                32, 32, pad_if_needed=True, border_mode=cv2.BORDER_REFLECT_101
            ),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )
    return tfds.as_numpy(train), tfds.as_numpy(test), train_trfm


def main(args):
    rng = nnx.Rngs(params=jax.random.key(0))
    model = get_model(rng, args, dataset="cifar10")
    optimizer = nnx.Optimizer(model, optax.adamw(args.lr))

    params = nnx.state(model, nnx.Param)
    total_params = (
        sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)) / 1e6
    )
    print(f"Nb params: {total_params:.2f}M")

    train_iter, test_iter, train_trfm = get_datasets(bs=args.batch_size)

    from jaxlip.loss import LseHKRMulticlassLoss, LossXEnt, TauCCE

    chosen_loss = {
        "hkr": LseHKRMulticlassLoss(alpha=args.alpha, temperature=args.temperature),
        "xent": LossXEnt(offset=args.offset, temperature=args.temperature),
        "tau_cce": TauCCE(temperature=args.temperature),
    }[args.loss_fn]

    @nnx.jit
    def train_step(model, optimizer, x, y):
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
        for images, labels in train_iter:
            augmented = train_trfm(images=images)
            images = augmented["images"]
            x = jnp.array(images, jnp.float32) / 255.0
            y = jnp.array(labels, jnp.int32)
            running_loss += train_step(model, optimizer, x, y)

        cache_model_params(model, verbose=False)
        train_acc = train_cra = 0
        for images, labels in train_iter:
            x = jnp.array(images, jnp.float32) / 255.0
            y = jnp.array(labels, jnp.int32)
            acc, rob = eval(model, x, y)
            train_acc += acc
            train_cra += rob
        train_acc /= len(train_iter)
        train_cra /= len(train_iter)

        # model.eval()
        val_acc = val_cra = 0
        for images, labels in test_iter:
            x = jnp.array(images, jnp.float32) / 255.0
            y = jnp.array(labels, jnp.int32)
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

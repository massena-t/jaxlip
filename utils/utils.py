import jax
import jax.numpy as jnp
from tqdm import tqdm

from models.simple import SimpleConvNet
from models.mixer import MLPMixer
from models.convnet import ConvNet


def get_model(rng, args, dataset):
    img_sz = {
        "cifar10": (32, 32, 3),
        "imagenette": (128, 128, 3),
    }[dataset]
    num_classes = {
        "cifar10": 10,
        "imagenette": 10,
    }[dataset]
    mean = {
        "cifar10": [0.4914, 0.4822, 0.4465],
        "imagenette": [0.5, 0.5, 0.5],
    }[dataset]
    std = {
        "cifar10": [0.2023, 0.1994, 0.2010],
        "imagenette": [0.5, 0.5, 0.5],
    }[dataset]
    if args.model == "simple" and dataset == "cifar10":
        model = SimpleConvNet(rng)
    elif args.model == "convnet":
        config = {
            "cifar10": [(64, 3, 2), (128, 3, 2), (256, 3, 1)],
            "imagenette": [(64, 3, 2), (128, 3, 2), (256, 3, 1)],
        }[dataset]
        stem_stride = {
            "cifar10": 1,
            "imagenette": 2,
        }[dataset]
        model = ConvNet(
            rngs=rng,
            input_shape=img_sz,
            dim_repeats_stride=config,
            stem_stride=stem_stride,
            num_classes=num_classes,
            num_iters_train=args.num_iters_train,
            num_iters_eval=args.num_iters_eval,
            detach_iter=args.detach_iter,
            mean=mean,
            std=std,
        )
    elif args.model == "mixer":
        assert img_sz[0] == img_sz[1], "H must equal W for (H,W,C) format image"
        patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim, num_layers = {
            "cifar10": (4, 512, 256, 1024, 8),
            "imagenette": (8, 512, 256, 1024, 8),
        }[dataset]
        model = MLPMixer(
            num_classes=num_classes,
            image_size=img_sz[0],
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            tokens_mlp_dim=tokens_mlp_dim,
            channels_mlp_dim=channels_mlp_dim,
            num_layers=num_layers,
            rngs=rng,
            mean=mean,
            std=std,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    return model


def l2_normalized_attack(
    model, img, label, step_size: float = 1e-2, max_steps: int = 50
):
    """
    Simple iterative ℓ₂‐normalized gradient attack.

    Args:
      model:      function(img_batch) → logits, expects shape [1, H, W, C]
      img:        jnp.ndarray of shape [1, H, W, C]
      label:      integer class label
      step_size:  float, size of each normalized‐gradient step
      max_steps:  int, max number of iterations

    Returns:
      pert:       final perturbation, same shape as img
      adv_img:    clipped adversarial image = img + pert
    """
    # initialize perturbation to zeros
    pert = jnp.zeros_like(img)
    success = False

    # one‐hot encode label for cross‐entropy
    num_classes = model(img).shape[-1]
    one_hot = jax.nn.one_hot(label, num_classes)

    # define loss as cross‐entropy of the perturbed image
    def loss_fn(p):
        logits = model(img + p)
        log_probs = jax.nn.log_softmax(logits)[0]
        return -jnp.dot(one_hot, log_probs).mean()

    for i in tqdm(range(max_steps)):
        # forward
        logits = model(img + pert)

        if i == 0:
            label = logits.argmax(-1)
        else:
            success = logits.argmax(-1) != label

        if success:
            break

        pred = jnp.argmax(logits, axis=-1)[0]

        # check success
        if pred != label:
            break

        # compute gradient of loss w.r.t. perturbation
        grad = jax.grad(loss_fn)(pert)  # same shape as pert: [1,H,W,C]

        # compute l₂‐norm over the single example
        grad_norm = jnp.linalg.norm(grad[0, ..., 0], ord=2)

        # step in direction of the gradient, normalized
        pert = pert + (grad / (grad_norm + 1e-12)) * step_size

    adv_img = jnp.clip(img + pert, 0.0, 1.0)
    return pert, adv_img

import os
import sys
import jax
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import einops
from jaxlip.conv import SpectralConv2d
from jaxlip.linear import OrthoLinear, SpectralLinear, ParametrizedLinear
from jaxlip.batchop import LipDyT, BatchCentering, LayerCentering
from jaxlip.activations import GroupSort2
from flax import nnx

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

lin_layer = OrthoLinear
# lin_layer = SpectralLinear


class PatchEmbedding(nnx.Module):
    """Convert an (H, W, C) image into a sequence of patch embeddings.

    Args:
        patch_size: Patch edge length *P* (pixels).
        in_channels: Number of input channels *C*.
        hidden_dim: Patch embedding size *D*.
        rngs: `nnx.Rngs` collection for parameter initialization.

    Input shape:  (B, H, W, C)
    Output shape: (B, N, D) where N = (H/P)·(W/P)
    """

    def __init__(
        self, *, patch_size: int, in_channels: int, hidden_dim: int, rngs: nnx.Rngs
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        flat_patch_dim = patch_size * patch_size * in_channels  # (P·P·C)
        # Explicitly specify in/out feature sizes (no shape inference)
        self.proj = lin_layer(
            din=flat_patch_dim,
            dout=hidden_dim,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, ws=None) -> jnp.ndarray:  # (B, H, W, C)
        batch, height, width, channels = x.shape
        assert channels == self.in_channels, (
            f"Expected {self.in_channels} channels, got {channels}"
        )
        assert height % self.patch_size == 0 and width % self.patch_size == 0, (
            "Image size must be divisible by patch size"
        )

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w  # N

        # (B, H, W, C) → (B, N, P·P·C)
        x = x.reshape(
            batch,
            num_patches_h,
            self.patch_size,
            num_patches_w,
            self.patch_size,
            channels,
        )
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))  # (B, Nh, Nw, P, P, C)
        x = x.reshape(batch, num_patches, self.patch_size * self.patch_size * channels)

        # Linear projection to hidden_dim D
        x = self.proj(x, ws)  # (B, N, D)
        return x


class MLP(nnx.Module):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        out_features: int,
        rngs: nnx.Rngs,
    ):
        self.fc1 = lin_layer(din=in_features, dout=hidden_features, rngs=rngs)
        self.act = GroupSort2()
        self.fc2 = lin_layer(din=hidden_features, dout=out_features, rngs=rngs)
        self.bc = BatchCentering(out_features)

    def __call__(self, x: jnp.ndarray, ws=None) -> jnp.ndarray:
        x = self.fc1(x, ws)
        x = self.act(x)
        x = self.fc2(x, ws)
        x = self.bc(x)
        return x


class MixerBlock(nnx.Module):
    """Single Mixer block combining token‑mixing and channel‑mixing MLPs.

    Args:
        num_patches: Token count *N*.
        hidden_dim: Hidden channel width *D*.
        tokens_mlp_dim: Hidden width for token‑mixing MLP.
        channels_mlp_dim: Hidden width for channel‑mixing MLP.
        rngs: `nnx.Rngs`.
    """

    def __init__(
        self,
        *,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        rngs: nnx.Rngs,
    ):
        self.norm1 = LayerCentering()
        self.token_mlp = MLP(
            in_features=num_patches,
            hidden_features=tokens_mlp_dim,
            out_features=num_patches,
            rngs=rngs,
        )

        self.norm2 = LayerCentering()
        self.channel_mlp = MLP(
            in_features=hidden_dim,
            hidden_features=channels_mlp_dim,
            out_features=hidden_dim,
            rngs=rngs,
        )
        self.dyt = LipDyT(hidden_dim, rngs=rngs)

    def __call__(self, x: jnp.ndarray, ws) -> jnp.ndarray:  # (B, N, D)
        # Token mixing ─ mix information across patches (N dimension)
        y = self.norm1(x)
        y = jnp.transpose(y, (0, 2, 1))  # (B, D, N) so N is last axis
        y = self.token_mlp(y, ws)  # (B, D, N)
        y = jnp.transpose(y, (0, 2, 1))  # (B, N, D)
        x = 0.5 * (x + y)  # Residual

        # Channel mixing ─ mix information across hidden channels (D dimension)
        y = self.norm2(x)
        y = self.channel_mlp(y, ws)  # (B, N, D)
        x = 0.5 * (x + y)  # Residual
        x = self.dyt(x)  # LipDyT layer
        return x

    def get_lipconstant(self):
        return self.dyt.get_lipconstant()


class MLPMixer(nnx.Module):
    """MLP‑Mixer vision model implemented with **Flax NNX** and explicit shapes.

    Args:
        num_classes: Output class count.
        image_size: Input image resolution (image_size × image_size).
        patch_size: Patch edge length *P*.
        hidden_dim: Channel width *D* after patch projection.
        tokens_mlp_dim: Hidden width of token‑mixing MLPs.
        channels_mlp_dim: Hidden width of channel‑mixing MLPs.
        num_layers: Number of Mixer blocks *L*.
        in_channels: Number of input channels *C* (default 3).
        rngs: `nnx.Rngs` collection.

    Input shape:  (B, image_size, image_size, in_channels)
    Output shape: (B, num_classes)
    """

    def __init__(
        self,
        *,
        num_classes: int,
        image_size: int,
        patch_size: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        num_layers: int,
        in_channels: int = 3,
        rngs: nnx.Rngs,
        mean=None,
        std=None,
    ):
        assert image_size % patch_size == 0, (
            "image_size must be divisible by patch_size"
        )
        num_patches = (image_size // patch_size) ** 2  # N
        self.mean = (
            nnx.Cache(jnp.array(mean))
            if mean is not None
            else nnx.Cache(jnp.array(0.0))
        )
        self.std = (
            nnx.Cache(jnp.array(std)) if std is not None else nnx.Cache(jnp.array(1.0))
        )

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        # Register Mixer blocks as a Python list attribute
        self.mixer_blocks = []
        for i in range(num_layers):
            block = MixerBlock(
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                tokens_mlp_dim=tokens_mlp_dim,
                channels_mlp_dim=channels_mlp_dim,
                rngs=rngs,
            )
            setattr(self, f"mixer_block_{i}", block)  # ensures registration
            self.mixer_blocks.append(block)

        self.norm = LayerCentering()
        self.head = lin_layer(din=hidden_dim, dout=num_classes, rngs=rngs)

    def __call__(self, x: jnp.ndarray, ws=None) -> jnp.ndarray:  # (B, H, W, C)
        lipconstant = 1.0
        x = x - jnp.expand_dims(self.mean.value, axis=(0, 1, 2))
        x = x / jnp.expand_dims(self.std.value, axis=(0, 1, 2))
        lipconstant = lipconstant * 1 / jnp.min(self.std)

        x = self.patch_embed(x, ws)  # (B, N, D)
        for block in self.mixer_blocks:
            x = block(x, ws)  # (B, N, D)
            lipconstant = lipconstant * block.get_lipconstant()

        x = self.norm(x)  # (B, N, D)
        D = x.shape[1]
        x = jnp.mean(x, axis=1) * jnp.sqrt(D)  # (B, D)
        logits = self.head(x, ws)  # (B, num_classes)
        return logits / lipconstant


if __name__ == "__main__":
    rng = nnx.Rngs(params=jax.random.key(0))
    model = MLPMixer(
        num_classes=10,
        image_size=128,
        patch_size=4,
        hidden_dim=64,
        tokens_mlp_dim=128,
        channels_mlp_dim=32,
        num_layers=2,
        rngs=rng,
    )

    from jaxlip.reparametrizer import (
        collect_buckets,
        parametrize_vmapped_cached,
        parametrize_from_params_cached,
        _sync,
        bench,
    )

    buckets = collect_buckets(model)
    reparam_weights = parametrize_vmapped_cached(buckets)

    x = jnp.ones((200, 128, 128, 3))

    model.eval()
    outs_normal = model(x)
    outs_vmap = model(x, reparam_weights)
    _sync((outs_normal, outs_vmap))

    diff = jnp.abs(outs_vmap - outs_normal)
    max_abs = float(jnp.max(diff))
    rel = float(max_abs / (float(jnp.max(jnp.abs(outs_normal))) + 1e-12))
    print(f"[compare] max_abs={max_abs:.3e}, rel_max={rel:.3e}")
    print(
        f"[compare] allclose (rtol=3e-3, atol=3e-3): {bool(jnp.allclose(outs_vmap, outs_normal, rtol=3e-3, atol=3e-3))}"
    )

    model.train()
    # Timing: eager and jit forwards
    f_normal = lambda x: model(x)
    f_vm = lambda x: model(x, reparam_weights)

    bench(f_normal, x, label="normal (eager)")
    bench(f_vm, x, label="vmap-weights (eager)")

    jit_normal = nnx.jit(lambda m, x: m(x))
    jit_vm = nnx.jit(lambda m, x, ws: m(x, ws))

    bench(lambda x: jit_normal(model, x), x, label="normal (jit)")
    bench(lambda x: jit_vm(model, x, reparam_weights), x, label="vmap-weights (jit)")

    # Adam gradient step (sanity)
    # Params tree = dict {uid -> raw W}; we optimize these, and compute ws from them.
    params_init = {
        **{
            uid: w.value
            for name, by_sig in buckets.items()
            for _, items in by_sig.items()
            for (uid, w, _h) in items
        },
        **{
            f"b:{m._uid}": m.b.value
            for _, m in model.iter_modules()
            if isinstance(m, ParametrizedLinear) and m.bias
        },
    }

    def inject_biases(model, params, ws):
        for _, m in model.iter_modules():
            if isinstance(m, ParametrizedLinear) and m.bias:
                key = f"b:{m._uid}"
                if key in params:
                    ws[key] = params[key]
        pass

    target = jnp.zeros((x.shape[0], 10), dtype=outs_normal.dtype)

    def loss_from_params(model, params, x):
        # Compute reparameterized weights from raw params, then forward
        ws = parametrize_from_params_cached(buckets, params)
        inject_biases(model, params, ws)
        y = model(x, ws)
        return jnp.mean((y - target) ** 2)

    import optax

    opt = optax.adam(1e-3)
    opt_state = opt.init(params_init)

    @nnx.jit
    def train_step(model, params, opt_state, x):
        loss, grads = nnx.value_and_grad(loss_from_params, argnums=1)(model, params, x)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return model, params, opt_state, loss

    model, params, opt_state, loss0 = train_step(model, params_init, opt_state, x)
    _sync(loss0)
    model, params, opt_state, loss1 = train_step(model, params, opt_state, x)
    _sync(loss1)

    print(
        f"[adam] step0 loss={float(loss0):.6f}  step1 loss={float(loss1):.6f}  finite={jnp.isfinite(loss1)}"
    )

    from jaxlip.utils import (
        cache_model_params,
        uncache_model_params,
        load_params_into_model,
    )

    load_params_into_model(model, params)
    model.eval()
    print(
        [
            m.use_running_average
            for _, m in model.iter_modules()
            if isinstance(m, BatchCentering)
        ]
    )
    cache_model_params(model)

    logits = model(x)
    loss = jnp.mean((model(x) - target) ** 2)
    print(f"{jnp.mean((model(x) - target) ** 2)=}")
    print(
        f"{jnp.mean((model(x, parametrize_from_params_cached(buckets, params)) - target) ** 2)=}"
    )
    print(
        f"{jnp.mean((model(x, parametrize_from_params_cached(buckets, params)) - target) ** 2)=}"
    )

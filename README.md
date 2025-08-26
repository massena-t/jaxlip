# JaxLip: Lipschitz Neural Networks in JAX

**Disclaimer:** This framework is still undergoing active development and some errors might be present.

A JAX implementation of Lipschitz-constrained neural networks using Flax NNX. This library provides layers and models that maintain Lipschitz continuity, which is useful for robust deep learning applications.

## Features

- **Efficient training**: The `Trainer` class allows for efficient parametrization computations via vmapping the computations (instead of applying them sequentially). Resulting in significant speedups. Allows for reparametrization caching for faster inference at eval time.
- **Convolutional layers**: `SpectralConv2d` layers with Tensor Norm normalization and `AOLConv2d` for approximately orthogonal convolutions.
- **Linear layers**: `OrthoLinear` layers using orthogonal weight matrices and Newton Schulz iterations and `SpectralLinear` with power iteration normalization.
- **Batch Operations**: `BatchCentering` and LayerCentering for improved training dynamics.
- **Activation Functions**: `GroupSort2` activation that preserves Lipschitz properties.
- **Model Utilities**: Caching mechanisms for efficient inference without having to recompute the weight parametrizations.

## Quick Start

```python
import jax
from flax import nnx
from jaxlip.models.convnet import ConvNet

# Initialize model
rng = nnx.Rngs(params=jax.random.key(0))
model = SimpleConvNet(rng, mean=0.5, std=0.5)

# Forward pass
x = jax.random.normal(jax.random.key(0), (1, 32, 32, 3))
output = model(x)
```

## Installation

Install from source:

```bash
git clone https://github.com/username/jaxlip.git
cd jaxlip
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Available Layers

### Linear Layers
- **SpectralLinear**: Linear layer with spectral normalization
- **OrthoLinear**: Linear layer with orthogonal weight constraints
- **ParametrizedLinear***: A customizeable linear layer where you can apply any weight transforming parametrization.

### Convolutional Layers  
- **SpectralConv2d**: 2D convolution with tensor normalization (Grishina et al, 2024).
- **AOLConv2d**: 2D convolution with AOL parametrization (Prach & Lampert, 2022).
- **ParametrizedLinear***: A customizeable convolutional layer where you can apply any weight transforming parametrization.

### Normalization Layers
- **BatchCentering**: Batch centering operation
- **BatchCentering2d**: 2D batch centering for convolutional layers
- **LayerCentering**: Layer-wise centering

### Activations
- **GroupSort2**: Lipschitz-preserving activation function
- **Abs**: Absolute value function.

## Using the model trainer

For basic single-gpu training:

```python

model = get_model(rng, args, dataset="imagenette")
optimizer = optax.adam(args.lr)
trainer = Trainer(
    model,
    optimizer,
    loss,
    distributed=False,
)

trainer.train()

for _ in range(10):
    x = ...
    y = ...
    loss_val = trainer.train_step(x, y)

trainer.eval()

for _ in range(5):
    x = ...
    y = ...
    acc, rob = trainer.eval_step(x, y)

```

For fast multi-gpu training:

```python

model = get_model(rng, args, dataset="imagenette")
optimizer = optax.adam(args.lr)
trainer = Trainer(
    model,
    optimizer,
    loss,
    distributed=True,
    vmap_reparametrizations=True,
)

trainer.train()

for _ in range(10):
    x = ...
    y = ...
    x = shard_batch(x, num_devices)
    y = shard_batch(y, num_devices)
    loss_val = trainer.train_step(x, y)

trainer.eval()

for _ in range(5):
    x = ...
    y = ...
    x = shard_batch(x, num_devices)
    y = shard_batch(y, num_devices)
    acc, rob = trainer.eval_step(x, y)

```

## Examples

See the `scripts/` directory for complete examples:
- `cifar.py`: CIFAR-10 classification
- `imagenette.py`: ImageNet subset classification

## Testing

Run tests using Python's unittest:

```bash
python -m unittest tests/*.py
```

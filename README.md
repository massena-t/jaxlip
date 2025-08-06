# JaxLip: Lipschitz Neural Networks in JAX

**Disclaimer:** This framework is still undergoing activate development and some errors might be present.

A JAX implementation of Lipschitz-constrained neural networks using Flax NNX. This library provides layers and models that maintain Lipschitz continuity, which is useful for robust deep learning applications.

## Features

- **Spectral Normalization**: SpectralConv2d and SpectralLinear layers with controllable Lipschitz constants
- **Orthogonal Constraints**: OrthoLinear layers using orthogonal weight matrices and AOLConv2d for approximately orthogonal convolutions
- **Batch Operations**: BatchCentering and LayerCentering for improved training dynamics
- **Activation Functions**: GroupSort2 activation that preserves Lipschitz properties
- **Model Utilities**: Caching mechanisms for efficient inference without having to recompute the weight parametrizations (use carefully)

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

## Available Layers

### Linear Layers
- **SpectralLinear**: Linear layer with spectral normalization
- **OrthoLinear**: Linear layer with orthogonal weight constraints

### Convolutional Layers  
- **SpectralConv2d**: 2D convolution with spectral normalization
- **AOLConv2d**: 2D convolution with AOL parametrization (Prach & Lampert, 2022)

### Normalization Layers
- **BatchCentering**: Batch centering operation
- **BatchCentering2d**: 2D batch centering for convolutional layers
- **LayerCentering**: Layer-wise centering

### Activations
- **GroupSort2**: Lipschitz-preserving activation function

## Model Caching

For efficient inference, use the model caching utilities:

```python
from jaxlip.utils import cache_model_params, uncache_model_params

# Cache parameters for inference
cache_model_params(model)

# Inference mode
output = model(x)

# IMPORTANT: Do not forget to uncache weights for training
uncache_model_params(model)
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


# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pallets is a project exploring pixel art generation using neural networks, with a focus on using various types of autoencoders. The key innovation is a novel one-hot encoded color mapping approach that ensures adherence to predefined color palettes, improving both dimensionality reduction and aesthetic quality in pixel art generation.

## Dependencies and Setup

The project requires the following dependencies (see requirements.txt):
- PyTorch and torchvision
- NumPy
- Matplotlib
- Jupyter
- Pillow
- nbformat
- ipympl

Setup instructions:
```bash
# Clone the required repositories
git clone https://github.com/tnn1t1s/cpunks-10k
git clone https://github.com/jmsdnns/pallets

# Setup environment
cd pallets
python -mvenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

- `pallets/`: Main package containing:
  - `models/`: Neural network architectures (AE, VAE, CVAE, Gumbel-Softmax)
  - `datasets/`: Dataset handling and color representation
  - `images.py`: Image processing utilities
  - `paths.py`: Path configuration for the project
  - `logging.py`: Logging utilities

- `nb/`: Jupyter notebooks organized by model type:
  - `ae/`: Autoencoder experiments
  - `vae/`: Variational Autoencoder experiments
  - `cvae/`: Conditional VAE experiments
  - `gumbel/`: Gumbel-Softmax VAE experiments
  - `dataset/`: Dataset exploration
  - `mathviz/`: Mathematical visualizations
  - `eval/`: Model evaluation

- `scripts/`: Utility scripts for training and data processing
  - `train_labeled_vae.py`: Script for training a labeled VAE

## Core Architecture

1. **Color Representation**:
   - The project uses a one-hot encoding approach for colors rather than standard RGB(A)
   - Each unique color in the dataset (222 colors) gets its own channel
   - This allows the model to strictly adhere to the predefined color palette

2. **Dataset Handling**:
   - The project uses a pixel art dataset (cpunks-10k)
   - `ColorOneHotMapper` converts between RGBA and one-hot representations
   - `OneHotCPunksDataset` and `FastOneHotCPunksDataset` handle data loading

3. **Model Architecture**:
   - Base autoencoder and variational autoencoder implementations
   - Convolutional variants that use Conv2D/ConvTranspose2D layers
   - Labeled variants that incorporate label information
   - Gumbel-Softmax variants for better handling of discrete representations

## Common Operations

### Running Notebooks

```bash
# Start Jupyter in the repository root
jupyter notebook
```

### Training Models

The project includes scripts for training various model types. For example, to train a labeled VAE:

```bash
cd pallets
python scripts/train_labeled_vae.py
```

Key parameters often used in training scripts:
- `USE_GPU`: Whether to use GPU acceleration (default: True)
- `EPOCHS`: Number of training epochs
- `LR`: Learning rate
- `BATCH_SIZE`: Batch size for training
- `TEST_SIZE`: Number of samples to use for testing

### Working with Trained Models

Models are saved using the `save` function in `models/base.py` and can be loaded with the `load` function:

```python
from pallets import models as M

# Load a model
device = M.get_device()
model, train_losses, test_losses = M.load('model_name', device)
```

## Implementation Details

1. **One-Hot Color Encoding**:
   - Each pixel is represented as a one-hot vector across 222 channels (one for each unique color)
   - Conversion between RGBA and one-hot is handled by the `ColorOneHotMapper`

2. **Model Configuration**:
   - Models are configured with input dimensions (222 for one-hot encoded colors)
   - Hidden dimensions for network layers
   - Latent dimensions for the compressed representation
   - Class dimensions for labeled variants (number of labels)

3. **GPU Acceleration**:
   - GPU detection and utilization is handled by `get_device()` from `models/base.py`
   - Supports both CUDA and MPS (for Apple Silicon)
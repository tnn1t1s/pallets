# Pallets

pallets is a project to explore pixel art generation in neural networks using toy model building blocks

## Abstract

We explore the application of convolutional autoencoders to generate pixel art. Our approach is twofold: first, we implement a standard RGB-based autoencoder model to understand baseline performance and highlight issues with current approaches - including those using very large generative models.  Second, we introduce a novel one-hot encoded color mapping autoencoder designed to adhere strictly to predefined color palettes, a critical aspect of pixel art.

We demonstrate a dimensionality reduction of the problem alongside order of magnitude performance improvement relative to the aesthetic of pixel art. We then suggest that this can be more broadly applied on larger images, more complex model architectures and ambitious generative pieces. 

## Setup

Get both the cpunks-10k repo and this one

```shell
git clone https://github.com/tnn1t1s/cpunks-10k
git clone https://github.com/tnn1t1s/pallets
```

Then setup a venv for pallets and install it

```
cd pallets
python -mvenv venv
source venv/bin/activate
pip install -e .
```

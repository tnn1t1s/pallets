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
pip install -r requirements.txt
```

## Notebooks

1. [Introduction](nb/Introduction.ipynb): Explore the dataset and how it is represented for experiments.
2. [RGBA vs One Hot](nb/RGBAvsOneHot.ipynb): Train and compare 4 autoencoders: 2 with RGBA colors and 2 with one hot encoded colors.
3. [Generation with VAE](nb/GenerationWithVAE.ipynb): Train a variational autoencoder to generate new images.
4. [Add Labels to VAE](nb/AddLabelsToVAE.ipynb): Train a VAE with images & labeled data to add ability to generate particular features.
5. [Find Errors in Earring Label](nb/FindEarrings.ipynb): Explore the 'earring' label in the dataset and improve its quality.
6. [Evaluate Improved Earring Label](nb/EvalNewEarrings.ipynb): This notebook introduces the use of Gumbel-Softmax in the VAE, and it uses a labels file with an improved earring label, to generate images with an accurate golden earring.

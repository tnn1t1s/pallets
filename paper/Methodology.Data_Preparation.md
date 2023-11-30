## 3.1 Dataset Description

The CPUNKS-10K dataset is a curated subset of the CryptoPunks image collection created by Larva Labs. The dataset has been specifically tailored for machine learning research. This dataset comprises 10,000 24x24x4 color images, each labeled with one of 5 unique types and 87 attributes. 

### Type Distribution

The five types in the dataset are mutually exclusive: Alien, Ape, Zombie, Male, Female.

### Attributes Distribution

The remaining 87 classes may overlap allowing for a rich and complex label structure that can be used for constrained autoencoder and decoder networks.

### Dataset Format

- `images/image{N}`: 10,000 (24,24,4) array of uint8s representing an image of ID N
- `labels.json`: A JSON format dictionary containing the set of types and attributues keyed by N.

We've made the decision to not using the 'types' classes for this work -- which are likely to infer assignment of gender. Using gender as a classification category in machine learning datasets can raise several important issues, particularly related to bias, ethics, and the complexity of gender identity. 

## 3.2 Data Preprocessing
For the work in this paper, we will use the images and attributes to build a set of models used to demonstrate the performance and phenomenological effects of using color palette aware neural networks. This section explains the preprocessing steps needed to build the features that will be used for model input

### 3.1 ImageDataLoader
(show all colors)
(show color mapper)
(show dataloader)
### 3.2 LabelDataLoader

### 3.3 Aligning Images and Labels



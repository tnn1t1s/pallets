## 3.1 Dataset Description

The CPUNKS-10K dataset is a curated subset of the CryptoPunks image collection created by Larva Labs. The dataset has been specifically tailored for machine learning research. This dataset comprises 10,000 24x24x4 color images, each labeled with one of 5 unique types and 87 attributes. 

### Type Distribution

The five types in the dataset are mutually exclusive: Alien, Ape, Zombie, Male, Female.

### Attibutes Distribution

The remaining 87 classes may overlap allowing for a rich and complex label structure that can be used for constrained autoencoder and decoder networks.

### Dataset Format

- `images/image{N}`: 10,000 (24,24,4) array of uint8s representing an image of ID N
- `labels.json`: A JSON format dictionary containing the set of types and attributues keyed by N.

## 3.2 Data Preprocessing

(explain color map)
(show all colors)
(show color mapper)
(show dataloader)

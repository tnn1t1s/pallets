import numpy as np
import torch

from .. import images
from .base import CPunksDataset


def make_one_hot_vector(index, length):
    one_hot_vector = np.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector


class ColorOneHotMapper:
    def __init__(self, unique_colors):
        self.color_to_one_hot = {}
        self.one_hot_to_color = {}

        for idx, color in enumerate(unique_colors):
            one_hot = make_one_hot_vector(idx, len(unique_colors))
            self.color_to_one_hot[tuple(color)] = one_hot
            self.one_hot_to_color[tuple(one_hot)] = color

    def to_one_hot(self, color):
        color_tuple = tuple(color)
        return self.color_to_one_hot.get(color_tuple, None)

    def to_color(self, one_hot):
        one_hot_tuple = tuple(one_hot)
        return self.one_hot_to_color.get(one_hot_tuple, None)


def one_hot_to_rgb(decoded_one_hot, mapper):
    # Choose the color with the highest probability for each pixel
    color_indices = np.argmax(decoded_one_hot, axis=-1)

    # Initialize an empty array for the RGB image
    rgb_image = np.zeros((color_indices.shape[0], color_indices.shape[1], 4))

    # Map each index back to an RGB color
    for i in range(color_indices.shape[0]):
        for j in range(color_indices.shape[1]):
            one_hot_vector = make_one_hot_vector(color_indices[i, j], 222)
            rgb_image[i, j] = mapper.to_color(one_hot_vector)

    return rgb_image


def rgb_to_one_hot(image, mapper):
    # Initialize an empty array
    one_hot_encoded_image = np.zeros(
        (image.shape[0], image.shape[1], len(mapper.color_to_one_hot))
    )

    # Iterate over each pixel and convert to one hot
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = image[i, j]
            one_hot = mapper.to_one_hot(color)

            if one_hot is None:
                continue

            one_hot_index = np.argmax(one_hot)
            one_hot_encoded_image[i, j, one_hot_index] = 1

    return one_hot_encoded_image


class OneHotEncodedImageDataset(CPunksDataset):
    def __init__(self, mapper, *args, **kwargs):
        super(OneHotEncodedImageDataset, self).__init__(*args, **kwargs)
        self.mapper = mapper

    def __getitem__(self, idx):
        image = images.get_punk(idx)
        one_hot_encoded_image = rgb_to_one_hot(image, self.mapper)
        return torch.tensor(one_hot_encoded_image, dtype=torch.float32)

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


def set_pixel(image, colors, x, y):
    (r, g, b, a) = colors
    image[0][x][y] = r
    image[1][x][y] = g
    image[2][x][y] = b
    image[3][x][y] = a
    return image


def one_hot_to_rgb(decoded_one_hot, mapper):
    # Choose the color with the highest probability for each pixel
    color_indices = np.argmax(decoded_one_hot, axis=0)

    # Initialize an empty array for the RGB image
    rgba_image = np.zeros((4, color_indices.shape[0], color_indices.shape[1]))

    # Map each index back to an RGB color
    for i in range(color_indices.shape[0]):
        for j in range(color_indices.shape[1]):
            one_hot_vector = make_one_hot_vector(color_indices[i, j], 222)
            colors = mapper.to_color(one_hot_vector)
            set_pixel(rgba_image, colors, i, j)

    return torch.tensor(rgba_image, dtype=torch.uint8)


def get_pixel(image, x, y):
    return (
        image[0][x][y].item(),  # R
        image[1][x][y].item(),  # G
        image[2][x][y].item(),  # B
        image[3][x][y].item()   # A
    )


def rgb_to_one_hot(image, mapper):
    one_hot_encoded_image = np.zeros(
        (len(mapper.color_to_one_hot), image.shape[1], image.shape[2])
    )
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            color = get_pixel(image, i, j)
            one_hot = mapper.to_one_hot(color)
            if one_hot is None:
                continue
            one_hot_index = np.argmax(one_hot)
            one_hot_encoded_image[one_hot_index, i, j] = 1
    return one_hot_encoded_image


def rgb_to_one_hot_old(image, mapper):
    # Initialize an empty array
    one_hot_encoded_image = np.zeros(
        (image.shape[0], image.shape[1], len(mapper.color_to_one_hot))
    )

    # Iterate over each pixel and convert to one hot
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            color = image[i, j]
            one_hot = mapper.to_one_hot(color)

            if one_hot is None:
                continue

            one_hot_index = np.argmax(one_hot)
            one_hot_encoded_image[one_hot_index, i, j] = 1

    return one_hot_encoded_image


class OneHotEncodedImageDataset(CPunksDataset):
    def __init__(self, mapper, *args, **kwargs):
        super(OneHotEncodedImageDataset, self).__init__(*args, **kwargs)
        self.mapper = mapper

    def __getitem__(self, idx):
        image = images.get_punk_tensor(idx)
        one_hot_encoded_image = rgb_to_one_hot(image, self.mapper)
        return torch.tensor(one_hot_encoded_image, dtype=torch.float32)

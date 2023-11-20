import numpy as np
import torch
from torch.utils.data import Dataset

from . import images


def make_one_hot(index, length):
    one_hot_vector = np.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector


class ColorOneHotMapper:
    def __init__(self, unique_colors):
        self.color_to_one_hot = {}
        self.one_hot_to_color = {}

        for idx, color in enumerate(unique_colors):
            one_hot = make_one_hot(idx, len(unique_colors))
            self.color_to_one_hot[tuple(color)] = one_hot
            self.one_hot_to_color[tuple(one_hot)] = color

    def get_one_hot(self, color):
        color_tuple = tuple(color)
        return self.color_to_one_hot.get(color_tuple, None)

    def get_color(self, one_hot):
        one_hot_tuple = tuple(one_hot)
        return self.one_hot_to_color.get(one_hot_tuple, None)


def decode_to_rgb(decoded_one_hot, mapper):
    # Choose the color with the highest probability for each pixel
    color_indices = np.argmax(decoded_one_hot, axis=-1)

    # Initialize an empty array for the RGB image
    rgb_image = np.zeros((color_indices.shape[0], color_indices.shape[1], 4))

    # Map each index back to an RGB color
    for i in range(color_indices.shape[0]):
        for j in range(color_indices.shape[1]):
            one_hot_vector = make_one_hot(color_indices[i, j], 222)
            rgb_image[i, j] = mapper.get_color(one_hot_vector)

    return rgb_image


def convert_image_to_one_hot(image, mapper):
    # Initialize an empty array
    one_hot_encoded_image = np.zeros(
        (image.shape[0], image.shape[1], len(mapper.color_to_one_hot))
    )

    # Iterate over each pixel and convert to one hot
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = image[i, j]
            one_hot = mapper.get_one_hot(color)

            if one_hot is None:
                continue

            one_hot_index = np.argmax(one_hot)
            one_hot_encoded_image[i, j, one_hot_index] = 1

    return one_hot_encoded_image


def split_dataset(data, test_size=0):
    """
    Splits a list of data into two randomized sets of indexes.
    """
    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)

    a_idx = data_indices[test_size:]
    b_idx = data_indices[:test_size]

    return a_idx, b_idx


class OneHotEncodedImageDataset(Dataset):
    def __init__(self, directory, mapper, test_size=0):
        self.directory = directory
        self.mapper = mapper
        self.image_files = [images.get_punk_path(i) for i in range(10000)]
        self.test_size = test_size
        split_idx = split_dataset(self.image_files, self.test_size)
        self.train_idx, self.test_idx = split_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = images.get_punk(idx)
        one_hot_encoded_image = convert_image_to_one_hot(image, self.mapper)
        return torch.tensor(one_hot_encoded_image, dtype=torch.float32)

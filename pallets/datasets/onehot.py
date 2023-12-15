import torch

from .base import CPunksDataset, FastCPunksDataset
from .. import images


def make_one_hot_vector(index, length):
    """
    Creates a one hot vector.
    """
    one_hot_vector = torch.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector


def set_color(image, colors, x, y):
    """
    Takes an 4 channel RGBA tensor and sets the colors properly for each
    channel at (x, y).
    """
    (r, g, b, a) = colors.tolist()[0:4]
    image[0][x][y] = r
    image[1][x][y] = g
    image[2][x][y] = b
    image[3][x][y] = a
    return image


def get_color(image, x, y):
    """
    Takes a 4 channel RGBA tensor and returns the RGBA colors for each channel
    at (x, y)
    """
    return torch.tensor([
        image[0][x][y].item(),  # R
        image[1][x][y].item(),  # G
        image[2][x][y].item(),  # B
        image[3][x][y].item()   # A
    ])


class ColorOneHotMapper:
    """
    Mapper from every color in dataset to its representation as a one-hot
    vector
    """
    def __init__(self, unique_colors):
        self.color_to_one_hot = {}
        self.one_hot_to_color = {}

        for idx, color in enumerate(unique_colors):
            one_hot = make_one_hot_vector(idx, len(unique_colors))
            self.color_to_one_hot[tuple(color.tolist())] = one_hot
            self.one_hot_to_color[tuple(one_hot.tolist())] = color

    def to_one_hot(self, color):
        return self.color_to_one_hot.get(tuple(color.tolist()), None)

    def to_color(self, one_hot):
        return self.one_hot_to_color.get(tuple(one_hot.tolist()), None)


def one_hot_to_rgba(image, mapper):
    """
    Converts an entire image from one hot vector channels to RGBA channels
    """
    # Choose the color with the highest probability for each pixel
    color_indices = torch.argmax(image, axis=0)

    # Initialize an empty array for the RGB image
    rgba_image = torch.zeros(
        (4, color_indices.shape[0], color_indices.shape[1])
    )

    # Map each index back to an RGB color
    for i in range(color_indices.shape[0]):
        for j in range(color_indices.shape[1]):
            one_hot_vector = make_one_hot_vector(color_indices[i, j], 222)
            colors = mapper.to_color(one_hot_vector)
            set_color(rgba_image, colors, i, j)

    return rgba_image


def rgba_to_one_hot(image, mapper):
    """
    Converts an entire image from RGBA channels to one hot vector channels
    """
    one_hot_encoded_image = torch.zeros(
        (len(mapper.color_to_one_hot), image.shape[1], image.shape[2])
    )
    for i in range(image.shape[1]):
        for j in range(image.shape[2]):
            color = get_color(image, i, j)
            one_hot = mapper.to_one_hot(color)
            if one_hot is None:
                continue
            one_hot_index = torch.argmax(one_hot)
            one_hot_encoded_image[one_hot_index, i, j] = 1
    return one_hot_encoded_image


class OneHotCPunksDataset(CPunksDataset):
    """
    Pytorch dataset that provides images in one-hot encoded form
    """
    def __init__(self, mapper, *args, **kwargs):
        # mapper before super().init
        self.mapper = mapper
        super(OneHotCPunksDataset, self).__init__(*args, **kwargs)

    def _load_punk(self, i):
        image = images.get_punk_tensor(i)
        return rgba_to_one_hot(image, self.mapper)
    

class FastOneHotCPunksDataset(OneHotCPunksDataset):
    """
    Same as OneHotCPunksDataset, but puts everything on the GPU
    """
    def __init__(self, device, *args, **kwargs):
        super(FastOneHotCPunksDataset, self).__init__(*args, **kwargs)
        # put data on GPU
        self.device = device
        self._images = torch.stack(self._images, 0).to(self.device)
        self._labels = torch.stack(self._labels, 0).to(self.device)

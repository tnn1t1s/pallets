import torch

from .base import CPunksDataset
from .. import images


def make_one_hot_vector(index, length):
    """
    Creates a one hot vector.
    """
    one_hot_vector = torch.zeros(length)
    one_hot_vector[index] = 1
    return one_hot_vector


# def set_color(image, colors, x, y):
#     """
#     Takes an 4 channel RGBA tensor and sets the colors properly for each
#     channel at (x, y).
#     """
#     (r, g, b, a) = colors.tolist()[0:4]
#     image[0][x][y] = r
#     image[1][x][y] = g
#     image[2][x][y] = b
#     image[3][x][y] = a
#     return image


# def get_color(image, x, y):
#     """
#     Takes a 4 channel RGBA tensor and returns the RGBA colors for each channel
#     at (x, y)
#     """
#     return torch.tensor([
#         image[0][x][y].item(),  # R
#         image[1][x][y].item(),  # G
#         image[2][x][y].item(),  # B
#         image[3][x][y].item()   # A
#     ])


class MPColorOneHotMapper:
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


def mp_one_hot_to_rgba(image, mapper):
    """
    Converts an entire image from one hot vector channels to RGBA channels
    """
    # Choose the color with the highest probability for each pixel
    color_indices = torch.argmax(image, axis=-1)

    # Initialize an empty array for the RGB image
    rgba_image = torch.zeros(
        (color_indices.shape[0], color_indices.shape[1], 4)
    )

    # Map each index back to an RGB color
    for i in range(color_indices.shape[0]):
        for j in range(color_indices.shape[1]):
            one_hot_vector = make_one_hot_vector(color_indices[i, j], 222)
            colors = mapper.to_color(one_hot_vector)
            # set_color(rgba_image, colors, i, j)
            rgba_image[i, j] = colors

    return rgba_image


def mp_rgba_to_one_hot(image, mapper):
    """
    Converts an entire image from RGBA channels to one hot vector channels
    """
    one_hot_encoded_image = torch.zeros(
        # (image.shape[1], image.shape[2], len(mapper.color_to_one_hot))
        (image.shape[0], image.shape[1], len(mapper.color_to_one_hot))
    )
    # for i in range(image.shape[1]):
    #     for j in range(image.shape[2]):
    #         color = get_color(image, i, j)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = image[i, j]
            one_hot = mapper.to_one_hot(color)
            if one_hot is None:
                continue
            one_hot_index = torch.argmax(one_hot)
            # one_hot_encoded_image[one_hot_index, i, j] = 1
            one_hot_encoded_image[i, j, one_hot_index] = 1
    # return one_hot_encoded_image
    return one_hot_encoded_image.permute((2, 0, 1))


class MPPunksDataset(CPunksDataset):
    """
    Loads tensor in mpimg format: (24, 24, 4)
    """
    def _load_punk(self, i):
        return images.get_punk_tensor(i).permute((1, 2, 0))


class MPOneHotEncodedImageDataset(MPPunksDataset):
    """
    Pytorch dataset that provides images in one-hot encoded form
    """
    def __init__(self, mapper, *args, **kwargs):
        # mapper before super().init
        self.mapper = mapper
        super(MPOneHotEncodedImageDataset, self).__init__(*args, **kwargs)

    def _load_punk(self, i):
        image = images.get_punk_tensor(i)
        return mp_rgba_to_one_hot(image, self.mapper)

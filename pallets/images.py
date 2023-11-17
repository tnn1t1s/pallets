import matplotlib.image as mpimg
import numpy as np


CPUNKS_DATA_DIR = "../../cpunks-10k/cpunks/data"
CPUNKS_IMAGE_DIR = "../../cpunks-10k/cpunks/images/training"


def get_punk_path(id):
    return f"{CPUNKS_IMAGE_DIR}/punk{id:#04d}.png"


def get_punk(id):
    """
    Returns a ndarray with loaded image
    """
    image_path = get_punk_path(id)
    return mpimg.imread(image_path)


#   unique_color_alpha_vectors
def one_image_colors(img_datum):
    """
    Returns an array of unique colors found in an image array
    """
    img_datum = img_datum.reshape(-1, img_datum.shape[2])
    uniques = np.unique(img_datum, axis=0)

    return uniques


#   find_unique_vectors_across_arrays
def list_image_colors(img_datas):
    """
    Returns an array of unique colors found in an array of images arrays
    """
    img_data = np.concatenate([i.reshape(-1, 4) for i in img_datas], axis=0)
    uniques = np.unique(img_data, axis=0)

    return uniques


def get_punk_colors():
    punks = [get_punk(i) for i in range(10000)]
    colors = list_image_colors(punks)
    return colors

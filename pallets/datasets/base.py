import numpy as np
import torch
from torch.utils.data import Dataset

from .. import images


def split_dataset(data, test_size=0):
    """
    Splits a list of data into two randomized sets of indexes.
    """
    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)

    a_idx = data_indices[test_size:]
    b_idx = data_indices[:test_size]

    return a_idx, b_idx


class CPunksDataset(Dataset):
    def __init__(self, test_size=0):
        self.image_files = [images.get_punk_path(i) for i in range(10000)]
        self.test_size = test_size

        split_idx = split_dataset(self.image_files, self.test_size)
        self.train_idx, self.test_idx = split_idx

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = images.get_punk(idx)
        return torch.tensor(image, dtype=torch.float32)

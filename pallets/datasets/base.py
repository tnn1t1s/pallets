import json
import torch
from torch.utils.data import Dataset

from .. import images

CPUNKS_LABELS = "../../cpunks-10k/cpunks/data/punks.json"


def split_dataset(ds_size, test_size=0):
    """
    Splits a list of data into two randomized sets of indexes.
    """
    # shuffle indexes
    data_indices = torch.randperm(ds_size).tolist()

    a_idx = data_indices[test_size:]
    b_idx = data_indices[:test_size]

    return a_idx, b_idx


class CPunksDataset(Dataset):
    """
    Pytorch dataset that provides all images from cpunks-10k as torch tensors
    """
    SIZE = 10000

    def __init__(self, test_size=0):
        self._images = [self._load_punk(i) for i in range(self.SIZE)]
        self._labels = self._load_labels()
        split_idx = split_dataset(self.SIZE, test_size)
        self.train_idx, self.test_idx = split_idx

    def _load_punk(self, i):
        return images.get_punk_tensor(i)
    
    def _load_labels(self):    
        all_labels = json.load(open(CPUNKS_LABELS))
        labels = []
        for _,label in all_labels.items():
            t = torch.tensor([v for _,v in label.items()])
            labels.append(t)
        return labels

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image = self._images[idx]
        labels = self._labels[idx]
        return image, labels


class FastCPunksDataset(CPunksDataset):
    """
    Same as CPunksDataset, but puts everything on the GPU
    """
    def __init__(self, device, *a, **kw):
        super(FastCPunksDataset, self).__init__(*a, **kw)
        # put data on GPU
        self.device = device
        self._images = torch.stack(self._images, 0).to(self.device)
        self._labels = torch.stack(self._labels, 0).to(self.device)

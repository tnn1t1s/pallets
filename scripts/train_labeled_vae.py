import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from pallets import images as I, datasets as DS, models as M


### Settings
USE_GPU = True
TEST_SIZE = 1000
EPOCHS = 50
LR = 1e-03
###


# To GPU, or not to GPU
print(f"GPU: {USE_GPU}", flush=True)
device = M.get_device(require_gpu=USE_GPU)


# Prep Data

print("Prep Data", flush=True)

all_colors = I.get_punk_colors()
mapper = DS.ColorOneHotMapper(all_colors)
dataset = DS.OneHotAndLabelsDataset(mapper, device=device, test_size=TEST_SIZE)
train_sampler = SubsetRandomSampler(dataset.train_idx)
test_sampler = SubsetRandomSampler(dataset.test_idx)

batch_size = 32
num_workers = 0

train_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
)
test_loader = DataLoader(
    dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers
)

num_labels = len(dataset._labels[0])


# Train Model

print("Train Model", flush=True)

model = M.vae.LabeledVAE(222, (64, 32), 20, num_labels).to(device)
criterion = M.vae.Loss().to(device)


train_losses, test_losses = M.vae.train(
    device, model, criterion, train_loader, test_loader,
    learn_rate=LR, epochs=EPOCHS
)


M.save(model, 'tlv.pkl')

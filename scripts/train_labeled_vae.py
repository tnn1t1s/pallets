import sys
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from pallets import images as I, datasets as DS, models as M, logging as L
from pallets.dependency import verify_cpunks_dependency


### Settings
USE_GPU = True
LOG_LEVEL = 'INFO'
TEST_SIZE = 1000
EPOCHS = 60
LR = 1e-03
BATCH_SIZE = 32
###


# Env
logger = L.init_logger(level=LOG_LEVEL, timestamp=True)
device = M.get_device(require_gpu=USE_GPU)

# Verify dependencies
logger.info("Verifying dependencies")
verify_cpunks_dependency(raise_error=True)

# Prep Data
logger.info("preparing data loaders")

if os.path.exists('onehot_ds.pt'):
    logger.info("- cached dataset")
    dataset = torch.load('onehot_ds.pt')
else:
    logger.info("- generating dataset")
    all_colors = I.get_punk_colors()
    mapper = DS.ColorOneHotMapper(all_colors)
    dataset = DS.FastOneHotCPunksDataset(device, mapper, test_size=TEST_SIZE)
    torch.save(dataset, 'onehot_ds.pt')

num_workers = 0
train_sampler = SubsetRandomSampler(dataset.train_idx)
train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=num_workers
)
test_sampler = SubsetRandomSampler(dataset.test_idx)
test_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=num_workers
)
num_labels = len(dataset._labels[0])


# Train Model

logger.info("starting model training")

model = M.vae.LabeledVAE(222, (64, 32), 20, num_labels)
criterion = M.vae.Loss()


train_losses, test_losses = M.vae.train(
    device, model, criterion, train_loader, test_loader,
    learn_rate=LR, epochs=EPOCHS
)


M.save('home_labeled_vae_1', model, train_losses, test_losses)

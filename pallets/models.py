import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# make 4 models:
# - naive linear + rgb
# - naive linear + one hot
# - simple conv  + rgb
# - simple conv  + one hot


class NaiveAutoencoder(nn.Module):
    """
    Base class for naive autoencoder.
    """
    DATA_SHAPE = None

    def __init__(self):
        super(NaiveAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.DATA_SHAPE), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(self.DATA_SHAPE)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), *self.DATA_SHAPE)
        return x


class NaiveRGBAAutoencoder(NaiveAutoencoder):
    """
    Naive autoencoder for RGBA images
    """
    DATA_SHAPE = (4, 24, 24)


class NaiveOneHotAutoencoder(NaiveAutoencoder):
    """
    Naive autoencoder for one-hot encoded images
    """
    DATA_SHAPE = (222, 24, 24)


class ConvRGBAutoencoder(nn.Module):
    def __init__(self):
        super(ConvRGBAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvOneHotAutoencoder(nn.Module):
    def __init__(self):
        super(ConvOneHotAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(222, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 222, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(
        device, model, criterion, train_loader, test_loader, learn_rate=1e-3,
        epochs=5
):
    """
    Training loop that tests the quality at each iteration while tracking
    everything necessary to make pretty graphs.
    """
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_losses = []
    test_losses = []

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        batch_losses = []

        for data in train_loader:
            inputs = data.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(np.array(loss.item()))

        batch_loss = np.array(batch_losses).mean(axis=0)
        print(f'epoch [{epoch+1}/{epochs}]')
        print(f'  - train loss: {batch_loss}')
        train_losses.append(batch_loss)

        model.eval()
        with torch.no_grad():
            batch_losses = []

            for data in test_loader:
                inputs = data.to(device)
                reconstruction = model(inputs)
                loss = criterion(reconstruction, inputs)

                batch_losses.append(np.array(loss.item()))

            batch_loss = np.array(batch_losses).mean(axis=0)
            print(f"  - test loss:  {batch_loss}")
            test_losses.append(batch_loss)

    return train_losses, test_losses


def _saved_path():
    """
    Helper function to save and load models from consistent location
    """
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(parent_dir, 'saved')
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    return models_dir


def save(model, filename):
    """
    Saves a model.
    """
    models_dir = _saved_path()
    filepath = os.path.join(models_dir, filename)
    torch.save(model, filepath)


def load(filename):
    """
    Loads a model.
    """
    models_dir = _saved_path()
    filepath = os.path.join(models_dir, filename)
    return torch.load(filepath)

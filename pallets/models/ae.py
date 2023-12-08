import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from .base import RGBA_CHANNELS, ONE_HOT_CHANNELS


class NaiveRGBAAutoencoder(nn.Module):
    """
    Naive autoencoder for 4 channel RGBA images
    """
    DATA_SHAPE = (RGBA_CHANNELS, 24, 24)

    def __init__(self):
        super(NaiveRGBAAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.DATA_SHAPE), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(self.DATA_SHAPE)),
            nn.Sigmoid(),
            nn.Unflatten(1, self.DATA_SHAPE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NaiveOneHotAutoencoder(nn.Module):
    """
    Naive autoencoder for one hot encoded color palette
    """
    DATA_SHAPE = (ONE_HOT_CHANNELS, 24, 24)

    def __init__(self):
        super(NaiveOneHotAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.DATA_SHAPE), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, np.prod(self.DATA_SHAPE)),
            nn.Unflatten(1, self.DATA_SHAPE)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvRGBAAutoencoder(nn.Module):
    """
    CNN autoencoder RGBA images
    """
    def __init__(self):
        super(ConvRGBAAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(RGBA_CHANNELS, 64, kernel_size=3, stride=1),
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
    """
    CNN autoencoder one hot images
    """
    def __init__(self):
        super(ConvOneHotAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(ONE_HOT_CHANNELS, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, ONE_HOT_CHANNELS, kernel_size=3, stride=1),
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
            data = data.to(device)
            reconstruction = model(data)
            loss = criterion(reconstruction, data)

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
                data = data.to(device)
                reconstruction = model(data)
                loss = criterion(reconstruction, data)

                batch_losses.append(np.array(loss.item()))

            batch_loss = np.array(batch_losses).mean(axis=0)
            print(f"  - test loss:  {batch_loss}")
            test_losses.append(batch_loss)

    return train_losses, test_losses

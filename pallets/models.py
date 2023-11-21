import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# make 4 models:
# - naive linear + rgb
# - naive linear + one hot
# - simple conv  + rgb
# - simple conv  + one hot

class OneHotAutoencoder(nn.Module):
    def __init__(self):
        super(OneHotAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(24*24*222, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 24*24*222),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 222, 24, 24)
        return x


class SimpleConvAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(222, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 222, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x.permute(0, 3, 1, 2)
        x = self.encoder(x)
        x = self.decoder(x)
        # x.permute(0, 2, 3, 1)
        return x


def train(
        model, criterion, train_loader, test_loader, learn_rate=1e-3, epochs=5
):
    # Special case for Conv2D
    if isinstance(model, SimpleConvAutoencoder):
        # PyTorch's nn.Conv2d expects input in the format:
        #   (batch_size, channels, height, width)
        #
        # Our dataloader uses mpimg, which uses this format:
        #   (batch_size, height, width, channels)
        #
        # Change from: [32, 24, 24, 222]
        # Change to:   [32, 222, 24, 24]
        def prepare_inputs(x): return x.permute(0, 3, 1, 2)
    else:
        def prepare_inputs(x): return x

    # Main loop
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        batch_losses = []

        for data in train_loader:
            inputs = prepare_inputs(data)
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
                inputs = prepare_inputs(data)
                reconstruction = model(inputs)
                loss = criterion(reconstruction, inputs)

                batch_losses.append(np.array(loss.item()))

            batch_loss = np.array(batch_losses).mean(axis=0)
            print(f"  - test loss:  {batch_loss}")
            test_losses.append(batch_loss)

    return train_losses, test_losses

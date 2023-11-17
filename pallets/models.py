import torch.nn as nn
import torch.optim as optim


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
        x = x.view(x.size(0), 24, 24, 222)
        return x


def train_onehot(model, dataloader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for data in dataloader:
            inputs = data
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')


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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train_simple_conv(model, dataloader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for data in dataloader:
            # PyTorch's nn.Conv2d expects input in the format:
            # (batch_size, channels, height, width)
            # Our dataloader used the format
            # (batch_size, height, width, channels).
            #
            # Change from [32, 24, 24, 222] to [32, 222, 24, 24]
            inputs = data.permute(0, 3, 1, 2)

            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

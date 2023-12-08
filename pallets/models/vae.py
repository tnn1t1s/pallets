import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.fc_dim = 128

        self.fc_mean = nn.Linear(self.fc_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.fc_dim, self.latent_dim)

        self.encode = nn.Sequential(
            nn.Conv2d(
                self.input_dim, self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dims[0], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_dims[1] * 24 * 24, self.fc_dim),
            nn.ReLU(),
        )

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dim, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.hidden_dims[1] * 24 * 24),
            nn.ReLU(),
            nn.Unflatten(-1, (self.hidden_dims[1], 24, 24)),
            nn.ConvTranspose2d(
                self.hidden_dims[1], self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.hidden_dims[0], self.input_dim,
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encode(x)

        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.decode(z)
        return z, mu, logvar


class LabeledVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, n_labels):
        super(LabeledVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.fc_dim = 128

        self.fc_mean = nn.Linear(
            self.fc_dim + self.n_labels, self.latent_dim
        )
        self.fc_logvar = nn.Linear(
            self.fc_dim + self.n_labels, self.latent_dim
        )

        self.encode = nn.Sequential(
            nn.Conv2d(
                self.input_dim, self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_dims[0], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.hidden_dims[1] * 24 * 24, self.fc_dim),
            nn.ReLU(),
        )

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dim + self.n_labels, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.hidden_dims[1] * 24 * 24),
            nn.ReLU(),
            nn.Unflatten(-1, (self.hidden_dims[1], 24, 24)),
            nn.ConvTranspose2d(
                self.hidden_dims[1], self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.hidden_dims[0], self.input_dim,
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        x = self.encode(x)

        x = torch.cat([x, labels], dim=1)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z = torch.cat([z, labels], dim=1)

        z = self.decode(z)
        return z, mu, logvar



class Loss(nn.Module):
    """
    Implementing the loss function this way lets us put it on the GPU
    """
    def forward(self, reconstructed_x, x, mean, logvar):
        recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + kl_div


def train(
        device, model, criterion, train_loader, test_loader, learn_rate=1e-3,
        epochs=5
):
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_losses = []
    test_losses = []

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        batch_losses = []
        train_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(inputs, labels)
            loss = criterion(recon_batch, inputs, mu, logvar)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            batch_losses.append(np.array(loss.item()))

            if batch_idx % 100 == 0:
                print('epoch {} ({:.0f}%)\t loss: {:.6f}'.format(
                    epoch+1,
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(batch_data)))

        batch_loss = np.array(batch_losses).mean(axis=0)
        print('epoch {} (100%)\t loss: {:.6f}'.format(
            epoch+1,
            train_loss / len(train_loader.dataset.train_idx)))
        train_losses.append(batch_loss)

        model.eval()
        with torch.no_grad():
            batch_losses = []

            test_loss = 0
            for data in test_loader:
                inputs, labels = data
                recon, mu, logvar = model(inputs, labels)
                loss = criterion(recon, inputs, mu, logvar)
                test_loss += loss.item()

                batch_losses.append(np.array(loss.item()))

            batch_loss = np.array(batch_losses).mean(axis=0)
            print("epoch {} test\t loss: {:.6f}".format(
                epoch+1,
                test_loss / len(test_loader.dataset.test_idx)))
            test_losses.append(batch_loss)

    return train_losses, test_losses


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..logging import logger, log_train_config


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encode = nn.Sequential(
            nn.Linear(
                self.input_dim, self.hidden_dim
            ),
            nn.LeakyReLU(),
        )

        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decode = nn.Sequential(
            nn.Linear(
                self.latent_dim, self.hidden_dim
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.hidden_dim, self.input_dim
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.encode(x)

        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.decode(z)
        z = z.view(-1, 222, 24, 24)
        return z, mu, logvar


class LabeledVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, classes_dim):
        super(LabeledVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim

        self.encode = nn.Sequential(
            nn.Linear(
                self.input_dim + self.classes_dim, self.hidden_dim
            ),
            nn.LeakyReLU(),
        )

        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decode = nn.Sequential(
            nn.Linear(
                self.latent_dim, self.hidden_dim
            ),
            nn.LeakyReLU(),
            nn.Linear(
                self.hidden_dim, self.input_dim
            ),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, labels), dim=1)
        x = self.encode(x)

        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.decode(z)
        z = z.view(-1, 222, 24, 24)
        return z, mu, logvar


class ConvVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(ConvVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.encode = nn.Sequential(
            nn.Conv2d(
                self.input_dim, self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_dims[0], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_dims[1], self.hidden_dims[2],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.fc_mean = nn.Linear(
            self.hidden_dims[2] * 24 * 24, self.latent_dim
        )
        self.fc_logvar = nn.Linear(
            self.hidden_dims[2] * 24 * 24, self.latent_dim
        )

        self.decode = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dims[2] * 24 * 24),
            nn.Unflatten(-1, (self.hidden_dims[2], 24, 24)),
            nn.ConvTranspose2d(
                self.hidden_dims[2], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                self.hidden_dims[1], self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.LeakyReLU(),
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


class LabeledConvVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, classes_dim):
        super(LabeledConvVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim

        self.encode = nn.Sequential(
            nn.Conv2d(
                self.input_dim + self.classes_dim, self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_dims[0], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                self.hidden_dims[1], self.hidden_dims[2],
                kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        self.fc_mean = nn.Linear(
            self.hidden_dims[2] * 24 * 24, self.latent_dim
        )
        self.fc_logvar = nn.Linear(
            self.hidden_dims[2] * 24 * 24, self.latent_dim
        )

        self.decode = nn.Sequential(
            nn.Linear(
                self.latent_dim, self.hidden_dims[2] * 24 * 24
            ),
            nn.Unflatten(-1, (self.hidden_dims[2], 24, 24)),
            nn.ConvTranspose2d(
                self.hidden_dims[2], self.hidden_dims[1],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                self.hidden_dims[1], self.hidden_dims[0],
                kernel_size=3, stride=1, padding=1, output_padding=0
            ),
            nn.LeakyReLU(),
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

    def encode_labels(self, x, labels):
        labels = labels.view(x.shape[0], labels.shape[1], 1, 1)
        ones = torch.ones(
            x.shape[0], labels.shape[1], x.shape[2], x.shape[3]
        )
        ones = ones.to(labels.device)
        ones = ones * labels
        x = torch.cat((x, ones), dim=1).to(x.device)
        return x

    def forward(self, x, labels):
        x = self.encode_labels(x, labels)
        x = self.encode(x)

        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        z = self.decode(z)
        return z, mu, logvar


class Loss(nn.Module):
    def forward(self, recon_x, x, mean, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return recon_loss + kl_div


def train(
    device, model, criterion, train_loader, test_loader, learn_rate=1e-3,
    epochs=5, with_labels=False
):
    """
    Variational Autoencoder focused training loop. Returns the loss information
    collected across epochs.
    """
    log_train_config(model, criterion, epochs, learn_rate)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_losses = []
    test_losses = []

    model.to(device)
    criterion.to(device)

    # Helper function to manage whether or not labels are used
    def handle_params(batch_data):
        if with_labels:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            model_args = (inputs, labels)
        else:
            inputs, _ = batch_data
            inputs = inputs.to(device)
            model_args = (inputs,)
        return model_args

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_loader):
            model_args = handle_params(batch_data)
            inputs = model_args[0]  # after device is mapped
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(*model_args)

            loss = criterion(reconstruction, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_losses.append(np.array(loss.item()))

            if batch_idx % 100 == 0:
                logger.info('epoch {} ({}%) loss: {:.6f}'.format(
                    epoch+1,
                    str(100 * batch_idx // len(train_loader)).rjust(3),
                    np.array(epoch_losses).mean(axis=0) / len(batch_data)
                ))

        epoch_loss = np.array(epoch_losses).mean(axis=0)
        train_losses.append(epoch_loss)
        logger.info('epoch {} (100%) loss: {:.6f}'.format(
            epoch+1,
            epoch_loss / len(batch_data)
        ))

        model.eval()
        with torch.no_grad():
            epoch_losses = []

            test_loss = 0
            for batch_data in test_loader:
                model_args = handle_params(batch_data)
                inputs = model_args[0]  # after device is mapped
                reconstruction, mu, logvar = model(*model_args)

                loss = criterion(reconstruction, inputs, mu, logvar)
                test_loss += loss.item()
                epoch_losses.append(np.array(loss.item()))

            epoch_loss = np.array(epoch_losses).mean(axis=0)
            test_losses.append(epoch_loss)
            logger.info("epoch {} (test) loss: {:.6f}".format(
                epoch+1,
                epoch_loss / len(batch_data)
            ))

    return train_losses, test_losses

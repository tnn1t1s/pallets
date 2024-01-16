import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..logging import logger, log_train_config


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, classes_dim):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim

        self.encode_1 = nn.Linear(
            self.input_dim + self.classes_dim, self.hidden_dim
        )

        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decode_1 = nn.Linear(
            self.latent_dim + self.classes_dim, self.hidden_dim
        )
        self.decode_2 = nn.Linear(
            self.hidden_dim, self.input_dim  # + self.classes_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x, targets):
        # encode targets into inputs
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, targets), dim=1)

        x = self.encode_1(x)
        x = F.leaky_relu(x)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decoder(self, z, targets):
        # encode targets into latent space
        z = torch.cat((z, targets), dim=1)

        z = self.decode_1(z)
        z = F.leaky_relu(z)
        z = self.decode_2(z)
        z = torch.sigmoid(z)

        z = z.view((z.shape[0], 222, 24, 24))

        return z

    def forward(self, x, targets):
        z, mu, logvar = self.encoder(x, targets)
        z = self.decoder(z, targets)
        return z, mu, logvar


class LabeledCVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, classes_dim):
        super(LabeledCVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim

        self.encode_1 = nn.Linear(
            self.input_dim + self.classes_dim, self.hidden_dim
        )

        self.fc_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        self.decode_1 = nn.Linear(
            self.latent_dim + self.classes_dim, self.hidden_dim
        )
        self.decode_2 = nn.Linear(
            self.hidden_dim, self.input_dim + self.classes_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_targets(self, x, targets):
        """encode targets into inputs"""
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, targets), dim=1)
        return x

    def encoder(self, x, targets):
        x = self.encode_targets(x, targets)

        x = self.encode_1(x)
        x = F.leaky_relu(x)
        mu = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decoder(self, z, targets):
        # encode targets into latent space
        z = torch.cat((z, targets), dim=1)

        z = self.decode_1(z)
        z = F.leaky_relu(z)
        z = self.decode_2(z)
        z = torch.sigmoid(z)

        return z

    def forward(self, x, targets):
        z, mu, logvar = self.encoder(x, targets)
        z = self.decoder(z, targets)
        return z, mu, logvar


class ConvCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, classes_dim):
        super(ConvCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim
        self.last_hidden_dim = 32

        self.encode_1 = nn.Conv2d(
            self.input_dim + self.classes_dim, 128,
            kernel_size=3, stride=1, padding=1
        )
        self.encode_2 = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1
        )
        self.encode_3 = nn.Conv2d(
            64, self.last_hidden_dim, kernel_size=3, stride=1, padding=1
        )

        self.fc_mean = nn.Linear(self.last_hidden_dim*24*24, self.latent_dim)
        self.fc_logvar = nn.Linear(self.last_hidden_dim*24*24, self.latent_dim)

        self.decode_1 = nn.Linear(
            self.latent_dim + self.classes_dim, self.last_hidden_dim*24*24
        )
        self.decode_2 = nn.ConvTranspose2d(
            self.last_hidden_dim, 64, kernel_size=3, stride=1, padding=1
        )
        self.decode_3 = nn.ConvTranspose2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )
        self.decode_4 = nn.ConvTranspose2d(
            128, self.input_dim,  # + self.classes_dim,
            kernel_size=3, stride=1, padding=1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encoder(self, x, targets):
        # encode targets into inputs
        targets = targets.view(x.shape[0], targets.shape[1], 1, 1)
        ones = torch.ones(x.shape[0], targets.shape[1], x.shape[2], x.shape[3])
        ones = ones.to(targets.device)
        ones = ones * targets
        x = torch.cat((x, ones), dim=1).to(x.device)

        x = self.encode_1(x)
        x = F.leaky_relu(x)
        x = self.encode_2(x)
        x = F.leaky_relu(x)
        x = self.encode_3(x)
        x = F.leaky_relu(x)

        mu = self.fc_mean(x.view(-1, self.last_hidden_dim*24*24))
        logvar = self.fc_logvar(x.view(-1, self.last_hidden_dim*24*24))
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decoder(self, z, targets):
        # encode targets into latent space
        ones = torch.ones(z.shape[0], targets.shape[1])
        ones = ones.to(targets.device)
        ones = ones * targets
        z = torch.cat((z, targets), dim=1).to(z.device)

        z = self.decode_1(z)
        z = z.view(-1, self.last_hidden_dim, 24, 24)

        z = self.decode_2(z)
        z = F.leaky_relu(z)
        z = self.decode_3(z)
        z = F.leaky_relu(z)
        z = self.decode_4(z)
        z = torch.sigmoid(z)

        return z

    def forward(self, x, targets):
        z, mu, logvar = self.encoder(x, targets)
        z = self.decoder(z, targets)
        return z, mu, logvar


class LabeledConvCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, classes_dim):
        super(LabeledConvCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim
        self.last_hidden_dim = 32

        self.encode_1 = nn.Conv2d(
            self.input_dim + self.classes_dim, 128,
            kernel_size=3, stride=1, padding=1
        )
        self.encode_2 = nn.Conv2d(
            128, 64, kernel_size=3, stride=1, padding=1
        )
        self.encode_3 = nn.Conv2d(
            64, self.last_hidden_dim, kernel_size=3, stride=1, padding=1
        )

        self.fc_mean = nn.Linear(self.last_hidden_dim*24*24, self.latent_dim)
        self.fc_logvar = nn.Linear(self.last_hidden_dim*24*24, self.latent_dim)

        self.decode_1 = nn.Linear(
            self.latent_dim + self.classes_dim, self.last_hidden_dim*24*24
        )
        self.decode_2 = nn.ConvTranspose2d(
            self.last_hidden_dim, 64, kernel_size=3, stride=1, padding=1
        )
        self.decode_3 = nn.ConvTranspose2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )
        self.decode_4 = nn.ConvTranspose2d(
            128, self.input_dim + self.classes_dim,
            kernel_size=3, stride=1, padding=1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_targets(self, x, targets):
        """Encode targets into inputs"""
        targets = targets.view(x.shape[0], targets.shape[1], 1, 1)
        ones = torch.ones(
            x.shape[0], targets.shape[1], x.shape[2], x.shape[3]
        )
        ones = ones.to(targets.device)
        ones = ones * targets
        x = torch.cat((x, ones), dim=1).to(x.device)
        return x

    def encoder(self, x, targets):
        x = self.encode_targets(x, targets)

        x = self.encode_1(x)
        x = F.leaky_relu(x)
        x = self.encode_2(x)
        x = F.leaky_relu(x)
        x = self.encode_3(x)
        x = F.leaky_relu(x)

        mu = self.fc_mean(x.view(-1, 32*24*24))
        logvar = self.fc_logvar(x.view(-1, 32*24*24))
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def decoder(self, z, targets):
        # encode targets into latent space
        ones = torch.ones(z.shape[0], targets.shape[1])
        ones = ones.to(targets.device)
        ones = ones * targets
        z = torch.cat((z, targets), dim=1).to(z.device)

        z = self.decode_1(z)
        z = z.view(-1, self.last_hidden_dim, 24, 24)

        z = self.decode_2(z)
        z = F.leaky_relu(z)
        z = self.decode_3(z)
        z = F.leaky_relu(z)
        z = self.decode_4(z)
        z = torch.sigmoid(z)

        return z

    def forward(self, x, targets):
        z, mu, logvar = self.encoder(x, targets)
        z = self.decoder(z, targets)
        return z, mu, logvar


class Loss(nn.Module):
    def forward(self, recon_x, x, mean, logvar):
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return recon_loss + kl_div


def train(
    device, model, criterion, train_loader, test_loader, learn_rate=1e-3,
    epochs=5, conditional_loss=False
):
    """
    Conditional VAE focused training loop. Returns the loss information
    collected across epochs.
    """
    log_train_config(model, criterion, epochs, learn_rate)

    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    train_losses = []
    test_losses = []

    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            reconstruction, mu, logvar = model(inputs, labels)

            if conditional_loss and hasattr(model, 'encode_targets'):
                inputs = model.encode_targets(inputs, labels)
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
                inputs, labels = batch_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                reconstruction, mu, logvar = model(inputs, labels)

                if hasattr(model, 'encode_targets'):
                    inputs = model.encode_targets(inputs, labels)
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

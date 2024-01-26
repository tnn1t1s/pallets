import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..logging import logger, log_train_config


class GSVAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, classes_dim, temperature
    ):
        super(GSVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim
        self.temperature = temperature

        self.encode_1 = nn.Linear(
            self.input_dim, self.hidden_dim
        )
        self.encode_2 = nn.Linear(
            self.hidden_dim, self.latent_dim * self.classes_dim
        )

        self.decode_1 = nn.Linear(
            self.latent_dim * self.classes_dim, self.hidden_dim
        )
        self.decode_2 = nn.Linear(
            self.hidden_dim, self.input_dim
        )

    def encoder(self, x):
        x = self.encode_1(x)
        x = F.leaky_relu(x)
        x = self.encode_2(x)
        x = F.leaky_relu(x)
        return x

    def decoder(self, z):
        z = self.decode_1(z)
        z = F.leaky_relu(z)
        z = self.decode_2(z)
        z = torch.sigmoid(z)
        return z

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        z = self.encoder(x)

        z_y = z.view(z.size(0), self.latent_dim, self.classes_dim)
        z = F.gumbel_softmax(z_y, tau=self.temperature, hard=True)
        z_y = F.softmax(z_y, dim=-1).reshape(*z.size())

        z = self.decoder(z.view(z.shape[0], -1))

        z = z.view(z.shape[0], 222, 24, 24)
        return z, z_y


class LabeledGSVAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, classes_dim, temperature
    ):
        super(LabeledGSVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.classes_dim = classes_dim
        self.temperature = temperature

        self.encode_1 = nn.Linear(
            self.input_dim + self.classes_dim, self.hidden_dim
        )
        self.encode_2 = nn.Linear(
            self.hidden_dim, self.latent_dim * self.classes_dim
        )

        self.decode_1 = nn.Linear(
            self.latent_dim * self.classes_dim, self.hidden_dim
        )
        self.decode_2 = nn.Linear(
            self.hidden_dim, self.input_dim
        )

    def encode_targets(self, x, targets):
        """encode targets into inputs"""
        x = x.view(x.shape[0], -1)
        x = torch.cat((x, targets), dim=1)
        return x

    def encoder(self, x, labels):
        x = self.encode_targets(x, labels)
        x = self.encode_1(x)
        x = F.leaky_relu(x)
        x = self.encode_2(x)
        x = F.leaky_relu(x)
        return x

    def decoder(self, z):
        z = self.decode_1(z)
        z = F.leaky_relu(z)
        z = self.decode_2(z)
        z = torch.sigmoid(z)
        return z

    def forward(self, x, labels):
        z = self.encoder(x, labels)

        z_y = z.view(z.size(0), self.latent_dim, self.classes_dim)
        z = F.gumbel_softmax(z_y, tau=self.temperature, hard=True)
        z_y = F.softmax(z_y, dim=-1).reshape(*z.size())

        z = self.decoder(z.view(z.shape[0], -1))

        z = z.view(z.shape[0], 222, 24, 24)
        return z, z_y


class Loss(nn.Module):
    def forward(self, recon_x, x, mean, classes_dim):
        log_ratio = torch.log(mean * classes_dim + 1e-20)
        kl_div = torch.sum(mean * log_ratio, dim=-1).mean()
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        return recon_loss + kl_div


def train(
    device, model, criterion, train_loader, test_loader, learn_rate=1e-3,
    epochs=5, with_labels=False
):
    """
    VAE with Gumbel Softmax focused training loop. Returns the loss information
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
            reconstruction, mu = model(*model_args)

            loss = criterion(reconstruction, inputs, mu, model.classes_dim)
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
                reconstruction, mu = model(*model_args)

                loss = criterion(reconstruction, inputs, mu, model.classes_dim)
                test_loss += loss.item()
                epoch_losses.append(np.array(loss.item()))

            epoch_loss = np.array(epoch_losses).mean(axis=0)
            test_losses.append(epoch_loss)
            logger.info("epoch {} (test) loss: {:.6f}".format(
                epoch+1,
                epoch_loss / len(batch_data)
            ))

    return train_losses, test_losses

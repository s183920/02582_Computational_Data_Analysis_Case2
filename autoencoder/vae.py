
"""
The following is an import of PyTorch libraries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
"""
A Convolutional Variational Autoencoder
"""
class VAE(nn.Module):
    def __init__(self, imgChannels=1, img_shape = (200,200), zDim=256):
        super(VAE, self).__init__()

        self.feature_dim = 64*21*21
        self.zDim = zDim
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5)
        self.encConv2 = nn.Conv2d(16, 32, 5,stride=2)
        self.encConv3 = nn.Conv2d(32, 64, 5,stride=2)
        self.encConv4 = nn.Conv2d(64, 64, 5,stride=2)

        self.encFC1 = nn.Linear(self.feature_dim, zDim)
        self.encFC2 = nn.Linear(self.feature_dim, zDim)

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.decFC1 = nn.Linear(zDim, self.feature_dim)
        self.decConv1 = nn.ConvTranspose2d(64, 64, 5,stride=2,output_padding=1)
        self.decConv2 = nn.ConvTranspose2d(64, 32, 5,stride=2,output_padding=1)
        self.decConv3 = nn.ConvTranspose2d(32, 16, 5,stride=2,output_padding=1)
        self.decConv4 = nn.ConvTranspose2d(16, imgChannels, 5)

    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))

        x = x.view(-1, self.feature_dim)
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 64, 21, 21)
        x = F.relu(self.decConv1(x))
        x = F.relu(self.decConv2(x))
        x = F.relu(self.decConv3(x))
        x = torch.sigmoid(self.decConv4(x))
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        if self.training:
            z = self.reparameterize(mu, logVar)
        else:
            z = mu
        out = self.decoder(z)
        return out, mu, logVar

if __name__ == "__main__":
    model = VAE(zDim=2,imgChannels=3).to("cuda")
    x = torch.ones((2,3,200,200)).float().to("cuda")
    out,_,_ = model(x)
    print(out.shape)
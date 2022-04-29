import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random

class AutoEncoderNet(torch.nn.Module):
    def __init__(self, n_channels, dim_last_layer, latent_features):
        super(AutoEncoderNet, self).__init__()
        
        n_flatten = torch.prod(torch.tensor(dim_last_layer))
        
        self.encoder = nn.Sequential(
                    nn.Conv2d(n_channels, 16, 5),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 5,stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 5,stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 5,stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1),
                    nn.Linear(n_flatten, 512),
                    nn.ReLU(),
                    nn.Linear(512, latent_features)
                )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, n_flatten),
            nn.ReLU(),
            nn.Unflatten(1,dim_last_layer),
            nn.ConvTranspose2d(64, 64, 5,stride=2,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 5,stride=2,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 5,stride=2,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, n_channels, 5),
            nn.Sigmoid()
        )


    def forward(self, x):
        latent_space = self.encoder(x)
        x_reconstruction = self.decoder(latent_space)

        return latent_space, x_reconstruction
    
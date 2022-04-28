
"""
The following is an import of PyTorch libraries.
"""


print("Loading packages")
import torch
import torch.nn.functional as F
from vae import VAE
#from data import train_loader, test_loader
from dataloader import UTKFaceDataset
"""
Initialize Hyperparameters
"""
print("Hyperparameters")
learning_rate = 5e-4
num_epochs = 100
device = "cuda"
"""
Initialize the network and the Adam optimizer
"""
batch_size = 128

print("DATA")
dataset = UTKFaceDataset()
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

print("MODEL")
net = VAE(zDim=32).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""

print("TRAIN")
for epoch in range(num_epochs):
    for idx, data in enumerate(train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

        # Backpropagation based on the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))


import time
current_time = time.time()
torch.save(net.state_dict(), f"autoencoder/models/{current_time}.pt")

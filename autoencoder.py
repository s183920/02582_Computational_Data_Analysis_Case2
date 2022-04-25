import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import wandb
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 15, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=8, stride=2),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 15, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 9, stride=2, padding=0),
            nn.ReLU(True),
  
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 4 * 4 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 4, 4))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 16, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 16, stride=3, padding = 0, output_padding = 2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 4, kernel_size = 16, stride=2,  padding = 0, output_padding = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 4, out_channels = 3, kernel_size = 18, stride=1,  padding = 0, output_padding = 0),
            nn.ReLU(True),
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x



def train(encoder, decoder, train_loader):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_func = nn.MSELoss()
    encoder.train()
    decoder.train()

    logger = wandb.init(project="autoencoder", name="autoencoder", entity='petergroenning')
    for epoch in range(100):
        for step, x in enumerate(train_loader):
            
            output = decoder(encoder(x))
            loss = loss_func(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.log({'Loss': loss.cpu().item()})
            # if step % 100 == 0:
                # logge
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())


class UTKFaceDataset(Dataset):
    def __init__(self, test_size = .2, data_type = "train"):
        """
        Create data loader for UTKFace dataset
        """
        
        self.data_dir = 'data/UTKFace/'
        # label_cols = ["age", "gender", "race"]

        # split files and labels in train and test
        self.all_files = os.listdir(self.data_dir)

        if data_type == "train":
            self.data = []
            for file in tqdm(self.all_files[:256]):
                img = Image.open(self.data_dir + file)
                self.data.append(np.asanyarray(img).reshape(3,200,200))
                img.close()
            self.X = torch.tensor(np.stack(self.data), dtype = torch.float32)
     


    def get_data(self):
        return self.X.shape, self.y.shape

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        img = self.X[idx]
        return img
 
if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    from tqdm import tqdm
    datasets = UTKFaceDataset()
    train_loader = DataLoader(datasets, batch_size=32, shuffle=True)
    
    encoder = Encoder(encoded_space_dim=128, fc2_input_dim=512)
    decoder = Decoder(encoded_space_dim=128, fc2_input_dim=512)

    train(encoder, decoder, train_loader)
    

  
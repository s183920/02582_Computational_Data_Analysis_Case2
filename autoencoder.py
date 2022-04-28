import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import wandb
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        

        ### Convolutional section

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 9, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 9, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 9, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 9, stride=2, padding=1),
            nn.ReLU(True),
  
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(8 * 8 * 64, encoded_space_dim),
            nn.ReLU(True),
        )
        



    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 8 * 8 * 64),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(64, 8, 8))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 9, stride=2,  padding = 1, output_padding= 1,),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 9, stride=2, padding = 2, output_padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 9, stride=2,  padding = 3, output_padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 3, kernel_size = 9, stride=2,  padding = 2, output_padding = 1),
        )
        

        self.post_net =nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        
        y = self.post_net(x)
        
        y = torch.sigmoid(x + y)
        return x, y



def train(encoder, decoder, train_loader, device = 'cpu', logger = None):
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001, weight_decay=1e-05)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_func = nn.MSELoss()
    encoder.to(device).train()
    decoder.to(device).train()

    for epoch in range(100):
        for step, x in enumerate(train_loader):
            x = torch.tensor(x.clone() , dtype = torch.float32, device = device) / 255.0
            pre_out, post_out = decoder(encoder(x))
            loss =  0.8 * loss_func(pre_out, x) + loss_func(post_out, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.log({'Loss': loss.cpu().item()})
            
        input = wandb.Image(x[0].cpu().detach().numpy().reshape(200,200,3))
        output = wandb.Image(post_out[0].cpu().detach().numpy().reshape(200,200,3))
        logger.log({"Input": input,
                    "Output": output})

        scheduler.step()
class UTKFaceDataset(Dataset):
    def __init__(self, test_size = .2, data_type = "train"):
        """
        Create data loader for UTKFace dataset
        """
        
        self.data_dir = 'data/UTKFace/'
    
        self.all_files = os.listdir(self.data_dir)

        if data_type == "train":
            self.data = []
            for file in tqdm(self.all_files[:]):
                img = Image.open(self.data_dir + file)
                self.data.append(np.asanyarray(img).reshape(3,200,200))
                img.close()
            self.X = torch.tensor(np.stack(self.data)) 


    def get_data(self):
        return self.X.shape, self.y.shape

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        img = self.X[idx]
        return img
 
if __name__ == '__main__':
    dim = 256
    # torch.manual_seed(1000)

    dataset = UTKFaceDataset()
    # train_loader = DataLoader(datasets, batch_size=32, shuffle=True)
    # logger = wandb.init(project="autoencoder", name=f"AE {dim} TEST", entity='petergroenning')
    encoder_weights = torch.load('models/encoder_256.pt', map_location=torch.device('cpu'))
    decoder_weights = torch.load('models/decoder_256.pt', map_location=torch.device('cpu'))

    encoder = Encoder(encoded_space_dim=dim)
    decoder = Decoder(encoded_space_dim=dim)
    encoder.load_state_dict(encoder_weights)
    decoder.load_state_dict(decoder_weights)
    encoder.eval()
    decoder.eval()
    print(dataset.X.shape)
    
    # decoder = Decoder(encoded_space_dim=dim, fc2_input_dim=512)
    # torch.manual_seed(1000)
    # train(encoder, decoder, train_loader, device = 'cuda', logger = logger)
    # torch.save(encoder.state_dict(), f"encoder_{dim}.pth")
    # torch.save(decoder.state_dict(), f"decoder{dim}.pth")
    

  
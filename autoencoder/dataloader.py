import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from torch import tensor
import dotenv
from pathlib import Path

dotenv.load_dotenv()
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT"))


class UTKFaceDataset(Dataset):
    def __init__(self, test_size = .2, data_type = "train"):
        """
        Create data loader for UTKFace dataset
        """
        self.df = pd.read_csv(PROJECT_ROOT / "data/info.csv")
        self.data_dir = PROJECT_ROOT / 'data/UTKFace/'
        label_cols = ["age", "gender", "race"]

        # split files and labels in train and test
        all_files = os.listdir(self.data_dir)
        train_files, test_files, train_labels, test_labels = train_test_split(all_files, self.df, test_size = test_size, random_state=1)

        if data_type == "train":
            self.data = [ImageOps.grayscale(Image.open(self.data_dir / img)) for img in train_files]
            self.X = np.stack(self.data)
            self.y = train_labels[label_cols]
            self.transforms = transforms.Compose([
                                    transforms.ToPILImage(),
                                    # transforms.RandomRotation(30),
                                    # transforms.RandomResizedCrop(224),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])

        elif data_type == "test":
            self.data = [ImageOps.grayscale(Image.open(self.data_dir / img)) for img in test_files]
            self.X = np.stack(self.data)
            self.y = test_labels[label_cols]
            self.transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        # transforms.Resize(255),
                                        # transforms.CenterCrop(224),
                                        transforms.ToTensor()])

    def get_data(self):
        return self.X.shape, self.y.shape

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        img = self.transforms(self.X[idx])
        label = tensor(self.y.iloc[idx].values)
        return img, label


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, labels in dl:
        show_images(images, nmax)
        break


if __name__ == "__main__":
    dataset = UTKFaceDataset()
    batch_size = 64
    train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    show_batch(train_dl)

    plt.show()
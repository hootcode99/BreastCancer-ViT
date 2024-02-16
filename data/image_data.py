import torch
from torch.utils.data import random_split, DataLoader
import lightning as L
from ImageDataset import ImageDataset
from torchvision import transforms


class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_dir, label_dir, batch_size):
        super().__init__()
        self.data_path = data_dir
        self.label_path = label_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.train_data = None
        self.test_data = None

    def setup(self, mode: str):
        dataset = ImageDataset(data_path=self.data_path, label_path=self.label_path, transforms=self.transforms)
        data_length = dataset.__len__()
        self.train_data, self.test_data = random_split(
            dataset,
            [data_length * 0.8, data_length * 0.2],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

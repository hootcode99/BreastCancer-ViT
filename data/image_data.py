import torch
from torch.utils.data import random_split, DataLoader
import lightning as L
from ImageDataset import ImageDataset
from torchvision import transforms


class ImageDataModule(L.LightningDataModule):
    def __init__(self, data_path, label_path, batch_size):
        super().__init__()
        self.data_dir = data_path
        self.label_dir = label_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])
        self.train = None
        self.validation = None
        self.test = None
        self.predict = None


    def prepare_data(self):
        # TODO: Get data from directory
        pass

    def setup(self, mode: str):
        # TODO: Define Data splits for each mode

        if mode == "train":
            dataset = ImageDataset(data_path=self.data_dir, label_path=self.label_dir, transforms=)
            self.train, self.validation = random_split(
                dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        if mode == "test":
            self.test =

        if mode == "predict":
            self.predict =

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size)

import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_path, label_path, transforms):
        if os.path.exists(data_path) and os.path.exists(label_path):
            self.dataset_dir = Path(data_path)
            self.labels_dir = Path(label_path)
            self.all_filenames = os.listdir(self.dataset_dir)
        else:
            self.dataset_dir = None
            self.labels_dir = None
            self.all_filenames = []

        self.transforms = transforms

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        image_name = self.all_filenames[idx]
        image_path = os.path.join(self.dataset_dir, image_name)
        image_pil = Image.open(image_path).convert('RGB')
        sample = self.transforms(image_pil)



        return sample, image_name
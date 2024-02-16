import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, data_path, label_path, transforms):

        if os.path.exists(data_path) and os.path.exists(label_path):
            self.dataset_dir = Path(data_path)
            self.labels_dir = Path(label_path)
            self.labels_df = pd.read_csv(self.labels_dir)
            # self.all_filenames = os.listdir(self.dataset_dir)
            self.all_filenames = self.labels_df["Image Index"].tolist()
        else:

            self.dataset_dir = None
            self.labels_dir = None
            self.all_filenames = []

        self.transforms = transforms

    def __len__(self):
        return len(self.all_filenames)

    def __getitem__(self, idx):
        img_name = self.all_filenames[idx]

        img_path = os.path.join(self.dataset_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        # img_pil.show()
        img_sample = self.transforms(img_pil)

        img_labels_entry = (self.labels_df.loc[self.labels_df["Image Index"] == img_name])
        img_labels_tensor = torch.tensor(img_labels_entry.iloc[:, 2:16].values.tolist()[0])
        # print(img_labels_tensor)

        return img_sample, img_labels_tensor

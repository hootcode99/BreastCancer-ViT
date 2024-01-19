import torch
from torch import nn
import torch.optim as optim
import lightning as L


class CNN(L.LightningModule):

    def __init__(self, img_chnls, num_classes):
        super(CNN, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            # IN: 3 x 256 x 256

            nn.Conv2d(img_chnls, 64, kernel_size=3, stride=2, padding=1, bias=False),
            # OUT: 64 x 128 x 128

            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 64 x 64

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            # OUT: 128 x 32 x 32

            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # OUT: 128 x 16 x 16

            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1, bias=False),
            # OUT: 256 x 14 x 14

            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # OUT: 256 x 7 x 7

            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1, bias=False),
            # OUT: 512 x 5 x 5

            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # OUT: 512 x 2 x 2

            nn.Flatten(),
            # OUT: 2048

            nn.Linear(in_features=2048, out_features=512),
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=128),
            nn.Dropout(0.3),
            nn.Linear(in_features=128, out_features=num_classes)

        )

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.forward(data)
        loss = self.criterion(logits, labels)
        return loss

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.model_step(batch, batch_idx)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

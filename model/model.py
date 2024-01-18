import torch
from torch import nn

import lightning as L


class DetectorNet(L.LightningModule):

    def __init__(self):
        super(DetectorNet, self).__init__()

    def forward(self, x):
        return x

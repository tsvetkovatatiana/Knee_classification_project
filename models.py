from args import *
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(MyModel, self).__init__()

        if backbone == "resnet18":
            self.model = models.resnet18(num_classes=5)
        elif backbone == "resnet34":
            self.model = models.resnet34(num_classes=5)
        else:
            self.model = models.resnet50(num_classes=5)

        #  self.model.fc = nn

    def forward(self, x):
        return 0




from args import *
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class MyModel(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(MyModel, self).__init__()

        if backbone == "resnet18":
            self.model = models.resnet18(weights=ResNet18_Weights)
        elif backbone == "resnet34":
            self.model = models.resnet34(num_classes=5)
        else:
            self.model = models.resnet50(num_classes=5)

        self.model.fc = nn.Linear(self.model.fc.in_features,5)

    def forward(self, x):
        return self.model(x)

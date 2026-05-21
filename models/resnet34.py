import torch.nn as nn
from torchvision.models import resnet34

class ResNet34(nn.Module):
    def __init__(self, num_classes=200, in_channels=3, image_size=None, **kwargs):
        super().__init__()
        self.model = resnet34(weights=None)

        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity() # Remove intial maxpool
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    
import torch.nn as nn
from torchvision.models import wide_resnet50_2

class WideResNet(nn.Module):
    def __init__(self, num_classes=200, in_channels=3, image_size=None, **kwargs):
        super().__init__()
        self.model = wide_resnet50_2(weights=None)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
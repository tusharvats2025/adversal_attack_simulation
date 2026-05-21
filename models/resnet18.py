import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, image_size=None, **kwargs):
        super().__init__()
        self.model = resnet18(weights=None)

        # Adjust conv1 for small images (kernel 7 -> 3)
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=3, bias=False
        )
        self.model.maxpool = nn.Identity() # Remove inital maxpool for small images.
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
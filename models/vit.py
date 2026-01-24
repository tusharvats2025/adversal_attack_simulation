import torch.nn as nn
from torchvision.models import vit_b_16

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=200, in_channels=64):
        super().__init__()
        self.model = vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.heads.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
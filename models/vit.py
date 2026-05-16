import torch.nn as nn
from torchvision.models import vit_b_16

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=200, in_channels=3, image_size=224, **kwargs):
        super().__init__()
        self.model = vit_b_16(weights=None)

        #optional: adapt input channels if needed
        if in_channels != 3:
            self.model.conv_proj = nn.Conv2d(
                in_channels,
                self.model.conv_proj.out_channels,
                kernel_size=self.model.conv_proj.kernel_size,
                stride=self.model.conv_proj.stride,
                padding=self.model.conv_proj.padding,
                bias=False,
            )
        
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
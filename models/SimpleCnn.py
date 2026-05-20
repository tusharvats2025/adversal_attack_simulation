import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels = 1, image_size=28, **kwargs):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # Output: (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )

        self._to_linear = None
        self._get_flatten_size(in_channels, image_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # fixed: now uses num_classes
        )
    def _get_flatten_size(self, in_channels, image_size):
        with torch.no_grad():
            x = torch.zeros(1, in_channels, image_size, image_size)  # Dummy MNIST-like input
            x = self.features(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # Output: (32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2) 
        )

        self._to_linear = None
        self._get_flatten_size()
        
        self.classifier = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # Dummy MNIST-like input
            x = self.features(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x
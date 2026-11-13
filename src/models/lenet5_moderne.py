import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, use_pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LeNetModernized(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, padding=1, use_pool=True),
            ConvBlock(32, 64, kernel_size=3, padding=1, use_pool=True),
            ConvBlock(64, 128, kernel_size=3, padding=1, use_pool=True),
            
            # Réduit chaque canal à 1x1 pixel
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               # Sortie : (Batch, 384)
            nn.Linear(128, 256),        # Exactement 384 en entrée
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
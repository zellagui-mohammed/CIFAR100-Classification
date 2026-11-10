
import torch
import torch.nn as nn


class LeNetModernized(nn.Module):

    def __init__(self, num_classes=100):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 400),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(400, 300),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(300, 200),
            nn.ReLU(inplace=True),

            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

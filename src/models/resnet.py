
import torch
import torch.nn as nn


# ==========================================
# 1. BASIC BLOCK
# ==========================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Première convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Deuxième convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Shortcut
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels * self.expansion:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out


# ==========================================
# 2. RESNET-18
# ==========================================

class ResNet(nn.Module):

    def __init__(self, num_classes=1000, in_channels=3):
        super().__init__()

        self.in_channels = 64

        # ----------------------------------
        # STEM
        # ----------------------------------

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )

        # ----------------------------------
        # RESNET STAGES
        # ----------------------------------

        self.layer1 = self._make_layer(
            out_channels=64,
            num_blocks=2,
            stride=1
        )

        self.layer2 = self._make_layer(
            out_channels=128,
            num_blocks=2,
            stride=2
        )

        self.layer3 = self._make_layer(
            out_channels=256,
            num_blocks=2,
            stride=2
        )

        self.layer4 = self._make_layer(
            out_channels=512,
            num_blocks=2,
            stride=2
        )

        # ----------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            512 * BasicBlock.expansion,
            num_classes
        )

    # ======================================
    # 3. MAKE LAYER
    # ======================================

    def _make_layer(self, out_channels, num_blocks, stride):

        layers = []

        # Premier BasicBlock : éventuellement downsampling
        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride
            )
        )

        # Mise à jour des canaux
        self.in_channels = out_channels * BasicBlock.expansion

        # BasicBlocks restants
        for _ in range(1, num_blocks):

            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    # ======================================
    # 4. FORWARD
    # ======================================

    def forward(self, x):

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # Stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

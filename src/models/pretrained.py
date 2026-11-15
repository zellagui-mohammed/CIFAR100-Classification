import torch.nn as nn
from torchvision import models

def build_pretrained(model_name: str = "resnet18", num_classes: int = 100, freeze_backbone: bool = True):
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        else:
            # Dégeler layer3 et layer4
            for param in model.layer3.parameters():
                param.requires_grad = True

            for param in model.layer4.parameters():
                param.requires_grad = True

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        return model


    elif model_name == "vgg11_bn":
        model = models.vgg11_bn(weights=models.VGG11_BN_Weights.DEFAULT)
        
        if freeze_backbone:
            # 1. On fige tout le backbone par défaut
            for param in model.parameters():
                param.requires_grad = False
                
            # 2. On dégèle UNIQUEMENT les blocs 4 et 5 (à partir de l'index 15 dans model.features)
            # Cela représente ~75% des paramètres convolutifs mais évite de recalculer le début du réseau !
            for param in model.features[15:].parameters():
                param.requires_grad = True

        model.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        return model

    else:
        raise ValueError(f"Modèle non supporté : {model_name}")
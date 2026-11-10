from .lenet import LeNet
from .vgg import VGG
from .resnet import ResNet
from .lenet5_moderne import LeNetModernized
# Dictionnaire des configurations VGG standards
MODELS = (
    "lenet5",
    "vgg16",
    "resnet18",
    "lenet5_moderne"
)

def build_model(model_name, in_channels=3, num_classes=10):
    """
    Fonction fabrique pour instancier un modèle .
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Modèle {model_name} inconnu. Choisissez parmi {MODELS}")
    
    if model_name=="lenet5":
        return LeNet(nb_out=num_classes)
    if model_name=="vgg16":
        return VGG(in_channels=in_channels, num_classes=num_classes, config=[2, 2, 3, 3, 3])
    if model_name=="lenet5_moderne":
        return LeNetModernized(num_classes=num_classes)
    elif model_name=="resnet18" :
        return ResNet(num_classes=num_classes,in_channels=in_channels)
    
__all__ = ["LeNet", "VGG", "ResNet", "build_model", "MODELS","LeNetModernized"]
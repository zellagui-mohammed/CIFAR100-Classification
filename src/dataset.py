"""Chargement de CIFAR-100 : transformations, split train/val/test et DataLoaders.

Utilisation :
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128, augment=True)
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

# projet/data, calculé depuis la position de ce fichier (indépendant du dossier de lancement)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Moyenne et écart-type de CIFAR-100 par canal (R, G, B)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


# --------------------------------------------------------------------------- #
# 1. Transformations
# --------------------------------------------------------------------------- #

def get_transforms(augment: bool) -> transforms.Compose:
    """Retourne la chaîne de transformations appliquée aux images.

    augment=True  : entraînement -> recadrage aléatoire + miroir, puis normalisation.
    augment=False : validation / test -> uniquement la normalisation.
    """
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # décale l'image de quelques pixels
            transforms.RandomHorizontalFlip(),      # miroir gauche/droite une fois sur deux
            transforms.ToTensor(),
            normalize,
        ])

    return transforms.Compose([transforms.ToTensor(), normalize])


# --------------------------------------------------------------------------- #
# 2. Chargement des jeux de données bruts
# --------------------------------------------------------------------------- #

def load_datasets(data_dir: Path = DATA_DIR, augment: bool = True):
    """Charge CIFAR-100 et retourne (train_augmenté, train_sans_augmentation, test).

    Le jeu d'entraînement est chargé deux fois : une version avec augmentation
    (pour entraîner) et une version sans (pour la validation, qui doit rester
    "propre"). Les deux contiennent les mêmes 50 000 images dans le même ordre.
    """
    train_aug = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=get_transforms(augment)
    )
    train_plain = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=get_transforms(False)
    )
    test_set = datasets.CIFAR100(
        data_dir, train=False, download=True, transform=get_transforms(False)
    )
    return train_aug, train_plain, test_set


# --------------------------------------------------------------------------- #
# 3. Split train / validation
# --------------------------------------------------------------------------- #

def split_indices(n_samples: int, val_size: int, seed: int):
    """Mélange les indices de façon reproductible et les sépare en (train, val)."""
    if not 0 < val_size < n_samples:
        raise ValueError(f"val_size doit être entre 1 et {n_samples - 1}, reçu {val_size}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator).tolist()
    return indices[val_size:], indices[:val_size]   # (train_idx, val_idx)


# --------------------------------------------------------------------------- #
# 4. Fonction principale
# --------------------------------------------------------------------------- #

def get_dataloaders(
    batch_size: int = 128,
    val_size: int = 5000,
    augment: bool = True,
    num_workers: int = 2,
    seed: int = 42,
):
    """Retourne (train_loader, val_loader, test_loader).

    - train : 50 000 - val_size images, avec augmentation si augment=True
    - val   : val_size images tirées du train, sans augmentation
    - test  : 10 000 images officielles (à n'utiliser qu'à la fin)
    """
    train_aug, train_plain, test_set = load_datasets(augment=augment)
    train_idx, val_idx = split_indices(len(train_aug), val_size, seed)

    # Subset = "vue" sur une partie du dataset, sans copier les images
    train_set = Subset(train_aug, train_idx)
    val_set = Subset(train_plain, val_idx)

    # Options communes aux trois loaders
    options = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),   # accélère le transfert vers le GPU
    )

    train_loader = DataLoader(train_set, shuffle=True, **options)   # mélangé à chaque epoch
    val_loader = DataLoader(val_set, shuffle=False, **options)
    test_loader = DataLoader(test_set, shuffle=False, **options)

    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# Test rapide : python -m src.dataset
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Dossier des données :", DATA_DIR)

    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"train: {len(train_loader.dataset)} | "
          f"val: {len(val_loader.dataset)} | "
          f"test: {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print("batch d'images :", tuple(images.shape))   # (128, 3, 32, 32)
    print("batch de labels:", tuple(labels.shape))   # (128,)
# Classification CIFAR-100

Comparaison de plusieurs architectures de réseaux de neurones (LeNet-5, VGG, ResNet)
sur le jeu de données CIFAR-100, avec PyTorch.

## Le jeu de données

CIFAR-100 contient 60 000 images couleur de 32x32 pixels réparties en 100 classes :
50 000 pour l'entraînement et 10 000 pour le test. Dans ce projet, 5 000 images du
jeu d'entraînement sont mises de côté pour la validation.

## Structure du projet

```
cifar100-classification/
├── config/            # Configurations YAML (base.yaml + un fichier par modèle)
├── src/
│   ├── models/        # Architectures : LeNet, VGG, ResNet
│   ├── dataset.py     # Chargement des données et split train/val/test
│   ├── train.py       # Boucle d'entraînement, validation, early stopping
│   ├── evaluate.py    # Métriques : accuracy, top-5, F1, paramètres, vitesse
│   └── utils.py       # Config, seed, device, dossiers de run
├── data/              # Données téléchargées (non versionné)
├── runs/              # Résultats de chaque expérience (non versionné)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/VOTRE-PSEUDO/cifar100-classification.git
cd cifar100-classification

python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation

Tester le chargement des données (téléchargement automatique au premier lancement) :

```bash
python -m src.dataset
```

Chaque expérience crée un dossier `runs/<modele>_<date>/` contenant :

- `config.yaml` : la configuration utilisée
- `metrics.csv` : loss et accuracy à chaque epoch
- `best.pth` : le meilleur modèle (sur la validation)
- `results.json` : les résultats finaux sur le test

## Configuration

Les paramètres communs sont dans `config/base.yaml`. Chaque modèle
(`lenet.yaml`, `resnet.yaml`, `vgg.yaml`) ne définit que ce qui change.

## Résultats

| Modèle    | Accuracy | Top-5 | F1 macro | Paramètres |
|-----------|----------|-------|----------|------------|
| LeNet-5   |          |       |          |            |
| VGG-16    |          |       |          |            |
| ResNet-18 |          |       |          |            |

## Licence

Ce projet est distribué sous licence MIT (voir le fichier `LICENSE`).
"""Point d'entrée : entraîne un modèle pré-entraîné, puis l'évalue sur le test.

Utilisation :
    python main.py                              # utilise la config par défaut (DEFAULT_CONFIG)
    python main.py --config resnet18_pretrained.yaml
"""

import argparse
import time

import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.evaluate import evaluate
from src.models.pretrained import build_pretrained
from src.train import fit
from src.utils import create_run_dir, get_device, load_config, save_results, set_seed

DEFAULT_CONFIG = "vgg11_bn.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement sur CIFAR-100")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="fichier de config du modèle (dans config/)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configuration, seed, device, dossier de run
    config = load_config(args.config)
    set_seed(config["seed"])
    device = get_device()
    run_dir = create_run_dir(config)
    print(f"Config : {args.config} | device : {device} | run : {run_dir.name}")

    # 2. Modèle
    model = build_pretrained(
        config["model"]["name"],
        num_classes=config["model"]["num_classes"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )

    # 3. Données (tous les paramètres viennent de la config)
    data_cfg = config["data"]
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=data_cfg["batch_size"],
        val_size=data_cfg["val_size"],
        augment=data_cfg["augment"],
        num_workers=data_cfg["num_workers"],
        seed=config["seed"],
        image_size=data_cfg.get("image_size", 32),
        normalization=data_cfg.get("normalization", "cifar"),
    )

    # 4. Entraînement
    print("Début de l'entraînement")
    start = time.perf_counter()
    best_val_acc = fit(model, train_loader, val_loader, config, run_dir, device)
    minutes = (time.perf_counter() - start) / 60
    print(f"Durée de l'entraînement : {minutes:.1f} minutes")

    # 5. Test final : on recharge le meilleur modèle (meilleure validation)
    model.load_state_dict(torch.load(run_dir / "best.pth", map_location=device))
    results = evaluate(model, test_loader, nn.CrossEntropyLoss(), device, full=True)
    results["best_val_acc"] = best_val_acc
    results["train_minutes"] = round(minutes, 1)

    save_results(run_dir, results)
    print("Résultats sur le test :")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
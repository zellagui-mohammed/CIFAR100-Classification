"""Entraînement d'un modèle : boucle d'apprentissage, validation, early stopping.

Utilisation :
    from src.train import fit
    best_acc = fit(model, train_loader, val_loader, config, run_dir, device)
"""

import torch
import torch.nn as nn

from src.evaluate import evaluate
from src.utils import log_epoch


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Crée l'optimiseur à partir de la section `training` de la config."""
    name = cfg["optimizer"].lower()

    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-2),
        )
    raise ValueError(f"Optimiseur inconnu : {name}")
    raise ValueError(f"Optimiseur inconnu : {name} (choix : sgd, adam)")


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    """Une epoch d'entraînement : retourne la loss et l'accuracy moyennes."""
    model.train()   # active dropout / batch norm en mode entraînement

    total_loss, correct, n_samples = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()                 # remet les gradients à zéro
        outputs = model(images)               # 1. prédiction
        loss = criterion(outputs, targets)    # 2. erreur
        loss.backward()                       # 3. calcul des gradients
        optimizer.step()                      # 4. mise à jour des poids

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        n_samples += images.size(0)

    return {"loss": total_loss / n_samples, "acc": correct / n_samples}


def fit(model, train_loader, val_loader, config: dict, run_dir, device) -> float:
    """Entraîne le modèle et retourne la meilleure accuracy de validation.

    - Sauvegarde le meilleur modèle dans run_dir/best.pth
    - Enregistre les métriques de chaque epoch dans run_dir/metrics.csv
    - S'arrête tôt si la validation ne s'améliore plus (early stopping)
    """
    cfg = config["training"]
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    # Réduit progressivement le learning rate au fil des epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_acc = 0.0
    epochs_without_progress = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log_epoch(run_dir, {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["acc"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["acc"], 4),
            "val_top5": round(val_metrics["top5"], 4),
        })

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train loss {train_metrics['loss']:.3f} acc {train_metrics['acc']:.3f} | "
              f"val loss {val_metrics['loss']:.3f} acc {val_metrics['acc']:.3f}")

        # Meilleur modèle : on le sauvegarde
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            epochs_without_progress = 0
            torch.save(model.state_dict(), run_dir / "best.pth")
        else:
            epochs_without_progress += 1

        # Early stopping
        if epochs_without_progress >= cfg["early_stopping_patience"]:
            print(f"Arrêt anticipé : pas d'amélioration depuis "
                  f"{cfg['early_stopping_patience']} epochs.")
            break

    return best_acc
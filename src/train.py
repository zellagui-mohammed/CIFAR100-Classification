"""Entraînement d'un modèle : boucle d'apprentissage, validation, early stopping.

Utilisation :
    from src.train import fit
    best_acc = fit(model, train_loader, val_loader, config, run_dir, device)
"""

import torch
import torch.nn as nn
from torchvision.transforms import v2

from src.evaluate import evaluate
from src.utils import log_epoch


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Crée l'optimiseur avec learning rate différentiel si backbone_lr est défini."""
    name = cfg["optimizer"].lower()
    base_model = model.module if hasattr(model, "module") else model

    # Sépare le backbone (features) du classifieur si le modèle possède ces attributs
    if "backbone_lr" in cfg and hasattr(base_model, "features") and hasattr(base_model, "classifier"):
        params = [
            {"params": base_model.features.parameters(), "lr": cfg["backbone_lr"]},
            {"params": base_model.classifier.parameters(), "lr": cfg["lr"]},
        ]
    else:
        params = base_model.parameters()

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.get("lr", 3e-4),
            weight_decay=cfg.get("weight_decay", 5e-2),
        )
    raise ValueError(f"Optimiseur inconnu : {name} (choix : sgd, adam, adamw)")


def train_one_epoch(model, loader, criterion, optimizer, device, cutmix_or_mixup) -> dict:
    """Une epoch d'entraînement : retourne la loss et l'accuracy moyennes."""
    model.train()

    total_loss, correct, n_samples = 0.0, 0, 0

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        images, targets = cutmix_or_mixup(images, targets)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        # targets est en format one-hot/probabilités après MixUp/CutMix
        correct += (outputs.argmax(dim=1) == targets.argmax(dim=1)).sum().item()
        n_samples += images.size(0)

    return {"loss": total_loss / n_samples, "acc": correct / n_samples}


def fit(model, train_loader, val_loader, config: dict, run_dir, device) -> float:
    """Entraîne le modèle et retourne la meilleure accuracy de validation."""
    cfg = config["training"]
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.1))
    optimizer = build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    cutmix_or_mixup = v2.RandomChoice([
        v2.MixUp(alpha=0.2, num_classes=config["model"]["num_classes"]),
        v2.CutMix(alpha=1.0, num_classes=config["model"]["num_classes"]),
    ])

    if torch.cuda.device_count() > 1:
        print(f"Utilisation de {torch.cuda.device_count()} GPUs !")
        model = nn.DataParallel(model)

    best_acc = 0.0
    epochs_without_progress = 0

    for epoch in range(1, cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cutmix_or_mixup
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        log_epoch(run_dir, {
            "epoch": epoch,
            "lr": optimizer.param_groups[-1]["lr"],
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["acc"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["acc"], 4),
            "val_top5": round(val_metrics["top5"], 4),
        })

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | "
              f"train loss {train_metrics['loss']:.3f} acc {train_metrics['acc']:.3f} | "
              f"val loss {val_metrics['loss']:.3f} acc {val_metrics['acc']:.3f}")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            epochs_without_progress = 0
            # Sauvegarde propre compatible mono-GPU et multi-GPU
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, run_dir / "best.pth")
        else:
            epochs_without_progress += 1

        if epochs_without_progress >= cfg["early_stopping_patience"]:
            print(f"Arrêt anticipé : pas d'amélioration depuis "
                  f"{cfg['early_stopping_patience']} epochs.")
            break

    return best_acc
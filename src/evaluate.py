"""Évaluation d'un modèle : loss, accuracy, top-5 et (optionnel) F1, paramètres, vitesse.

Utilisation :
    from src.evaluate import evaluate
    metrics = evaluate(model, val_loader, criterion, device)               # rapide (validation)
    report  = evaluate(model, test_loader, criterion, device, full=True)   # complet (test final)
"""

import time

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """F1 macro : F1 calculé pour chaque classe puis moyenné (toutes les classes comptent pareil)."""
    true_pos = torch.bincount(targets[preds == targets], minlength=num_classes).float()
    pred_count = torch.bincount(preds, minlength=num_classes).float()
    true_count = torch.bincount(targets, minlength=num_classes).float()

    precision = true_pos / pred_count.clamp(min=1)
    recall = true_pos / true_count.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    return f1.mean().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device, full: bool = False, num_classes: int = 100) -> dict:
    """Évalue le modèle sur un DataLoader.

    full=False : retourne loss, acc (top-1), top5.
    full=True  : ajoute f1_macro, params et inference_ms_per_image.
    """
    model.eval()   # désactive dropout / fige batch norm

    total_loss, correct_top1, correct_top5, n_samples = 0.0, 0, 0, 0
    all_preds, all_targets = [], []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)   # pondéré par la taille du batch
        n_samples += images.size(0)

        # Top-1 : la classe prédite est la bonne
        preds = outputs.argmax(dim=1)
        correct_top1 += (preds == targets).sum().item()

        # Top-5 : la bonne classe est parmi les 5 meilleures prédictions
        top5 = outputs.topk(5, dim=1).indices
        correct_top5 += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        if full:
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    metrics = {
        "loss": total_loss / n_samples,
        "acc": correct_top1 / n_samples,
        "top5": correct_top5 / n_samples,
    }

    if full:
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        metrics["f1_macro"] = macro_f1(preds, targets, num_classes)
        metrics["params"] = count_parameters(model)
        metrics["inference_ms_per_image"] = 1000 * elapsed / n_samples

    return metrics
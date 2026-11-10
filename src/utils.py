"""Fonctions utilitaires : config, reproductibilité et dossiers de run."""

import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _merge(base: dict, override: dict) -> dict:
    """Fusionne deux dictionnaires : les valeurs de `override` remplacent celles de `base`."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)   # fusion des sous-sections
        else:
            result[key] = value
    return result


def load_config(model_config: str) -> dict:
    """Charge base.yaml puis y applique les surcharges du fichier du modèle.

    Exemple : load_config("lenet.yaml") ou load_config("config/lenet.yaml")
    """
    with open(CONFIG_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)

    with open(CONFIG_DIR / Path(model_config).name) as f:
        config = _merge(config, yaml.safe_load(f))

    return config


# --------------------------------------------------------------------------- #
# Reproductibilité
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Fixe toutes les seeds pour obtenir des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Dossier de run : runs/<modele>_<date>/
# --------------------------------------------------------------------------- #

def create_run_dir(config: dict) -> Path:
    """Crée runs/<modele>_<AAAAMMJJ_HHMMSS>/ et y copie la config utilisée.

    Retourne le chemin du dossier, à passer au trainer.
    """
    name = config["model"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = RUNS_DIR / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True)

    # Copie de la config finale (base + surcharges) : on saura toujours
    # avec quels paramètres ce run a été lancé.
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return run_dir


def log_epoch(run_dir: Path, metrics: dict) -> None:
    """Ajoute une ligne dans metrics.csv (une ligne par epoch).

    Exemple : log_epoch(run_dir, {"epoch": 1, "train_loss": 4.2, "val_acc": 0.12})
    """
    path = run_dir / "metrics.csv"
    is_new = not path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if is_new:
            writer.writeheader()   # l'en-tête n'est écrit qu'une fois
        writer.writerow(metrics)


def save_results(run_dir: Path, results: dict) -> None:
    """Enregistre les résultats finaux (accuracy test, top-5, ...) dans results.json."""
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

def get_device() -> torch.device:
    """Retourne le GPU s'il est disponible, sinon le CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
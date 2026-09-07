"""Zentrales Pfad Register und Verzeichnis Anker des Projekts"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "resisc45"
POISONED_DIR = PROCESSED_DIR / "poisoned"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RQ3_DIR = EXPERIMENTS_DIR / "RQ3"
CACHE_DIR = PROJECT_ROOT / ".cache"

RQ_CHOICES = ("RQ1", "RQ2", "RQ2.2-sim", "RQ2.2-dissim", "RQ2.4", "RQ3")
ATTACK_CHOICES = ("clean", "RQ1", "RQ2")
RQS_WITHOUT_RATE = ("RQ2.4",)


def validate_rq(rq: str):
    """Validiert die übergebene Forschungsfrage"""
    if rq not in RQ_CHOICES:
        raise ValueError(f"Falsche rq '{rq}'. Richtig: {', '.join(RQ_CHOICES)}")
    return rq


def validate_attack(attack: str):
    """Validiert die übergebene Angriffsart"""
    if attack not in ATTACK_CHOICES:
        raise ValueError(f"Falscher attack '{attack}'. Richtig: {', '.join(ATTACK_CHOICES)}")
    return attack


def hdf5_path(split: str):
    """Liefert den absoluten Pfad zu einer der HDF5-Dateien
    
    Args:
        split (str): Name des Datensplits (z.B. "train", "val", "test")
    """
    return PROCESSED_DIR / f"{split}.h5"


def poisoned_labels_path(rq: str, seed: int, rate: float):
    """Ermittelt den Zielpfad für manipulierte/vergiftete Label-Tensoren

    Args:
        rq (str): Die Forschungsfrage
        seed (int): Der verwendete Zufalls-Seed
        rate (float): Die Vergiftungsrate

    Returns:
        Path: Absoluter Pfad zur generierten PyTorch (.pt) Tensor-Datei
    """
    if rq in RQS_WITHOUT_RATE:
        filename = "labels.pt"
    else:
        filename = f"labels_{rate:.2f}.pt"
    return POISONED_DIR / validate_rq(rq) / f"seed_{seed}" / filename


def experiment_dir(rq: str, model: str, defense: str | None = None, attack: str | None = None):
    """Intelligenter Router für das Erstellen von Experiment-Ausgabeverzeichnissen.

    1. Baseline-Experimente (RQ1, RQ2):
       Hier werden ungeschützte Modelle trainiert. Als Unterordner nur die Netzwerk-Architektur
        Layout: experiments/<rq>/<model>/
       
    2. Verteidigungs-Experimente (RQ3):
       Verteidigungen mit verschiedenen Modellen unter spezifischen Angriffsszenarien
        Layout: experiments/RQ3/<defense>/<model>/<attack>/

    Args:
        rq (str): Aktuelle Forschungsfrage
        model (str): Verwendete Architektur
        defense (str | None): Name der Verteidigung
        attack (str | None): Name des simulierten Angriffs

    Raises:
        ValueError: Wenn eine `defense` übergeben wird, aber keine `attack`

    Returns:
        Path: Das absolute Verzeichnis für diesen Lauf
    """
    base = EXPERIMENTS_DIR / validate_rq(rq)
    if defense is None:
        return base / model
    if attack is None:
        raise ValueError("experiment_dir(): mit 'defense' muss auch 'attack' gesetzt sein.")
    return base / defense / model / validate_attack(attack)


def training_config_path():
    """Absoluter Pfad zur Trainingskonfiguration"""
    return PACKAGE_ROOT / "config" / "training.yaml"


def data_config_path():
    """Absoluter Pfad zur Datenkonfiguration"""
    return PACKAGE_ROOT / "data_preparation" / "config.yaml"

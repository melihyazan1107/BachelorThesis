"""Namensregister für alle Verteidigungsstrategien.

Softwarearchitektur:
Dieses Modul fungiert als "Registry" (Verzeichnis). 
Wenn das Trainingsskript startet, liest es aus der YAML-Konfiguration lediglich 
einen Text-String (z.B. "label_smoothing"). Dieses Modul ist dafür zuständig, 
diesen String zu validieren und in das entsprechende, einsatzbereite Python-Objekt 
zu übersetzen.
"""
from typing import Any

import torch

from implementation.shared.defenses.base import Defense, NullDefense
from implementation.shared.defenses.preventive import (ELRDefense,GCEDefense,LabelSmoothingDefense,MixupDefense)
from implementation.shared.defenses.reactive import GMMFilterDefense, SmallLossDefense

DEFENSE_CHOICES = ("none","label_smoothing","mixup","gce","elr","small_loss","gmm_filter")

DEFENSE_GROUPS = {
    "none": "none",
    "label_smoothing": "preventive",
    "mixup": "preventive",
    "gce": "preventive",
    "elr": "preventive",
    "small_loss": "reactive",
    "gmm_filter": "reactive"}

DEFAULT_WARMUP_EPOCHS = 25


def validate_defense(name: str):
    """Prüft, ob der übergebene String einer Verteidigung entspricht

    Args:
        name (str): Der zu prüfende Name der Verteidigung

    Raises:
        ValueError: Wenn der Name nicht in DEFENSE_CHOICES existiert

    Returns:
        str: Der Name
    """
    if name not in DEFENSE_CHOICES:
        raise ValueError(f"Falsche Defense Art: '{name}'. Richtig: {', '.join(DEFENSE_CHOICES)}")
    return name


def defense_name_from_config(cfg):
    """Extrahiert den Namen der Verteidigung sicher aus der Trainingskonfiguration

    Args:
        cfg (dict): Das geladene Dictionary der training.yaml

    Returns:
        str: Der validierte Name
    """
    defense_cfg = cfg.get("defense") or {}
    return validate_defense(str(defense_cfg.get("type", "none")))


def build_defense(cfg, *, seed, rate, num_samples,num_classes, device: torch.device):
    """Baut und konfiguriert das passende Objekt

    Args:
        cfg (dict): Die komplette Trainings-Konfigurationsdatei
        seed (int): Der Random-Seed
        rate (float): Die Poisoning Rate der Daten.
        num_samples (int): Gesamtanzahl der Bilder im Trainings-Split
        num_classes (int): Gesamtanzahl der existierenden Klassen
        device (torch.device): CPU oder GPU

    Returns:
        Defense: Eine einsatzbereite Kindklasse von `base.Defense`.
    """
    defense_cfg = cfg.get("defense") or {}
    name = defense_name_from_config(cfg)
    warmup_epochs = int(defense_cfg.get("warmup_epochs", DEFAULT_WARMUP_EPOCHS))

    def hyperparameters(key):
        return dict(defense_cfg.get(key) or {})

    if name == "none":
        return NullDefense()

    if name == "label_smoothing":
        return LabelSmoothingDefense(epsilon=hyperparameters("label_smoothing").get("epsilon", 0.1))

    if name == "mixup":
        return MixupDefense(alpha=hyperparameters("mixup").get("alpha", 0.2), seed=seed)

    if name == "gce":
        return GCEDefense(q=hyperparameters("gce").get("q", 0.7))

    if name == "elr":
        elr = hyperparameters("elr")
        return ELRDefense(num_samples=num_samples,num_classes=num_classes,lambda_weight=elr.get("lambda", 3.0),beta=elr.get("beta", 0.7),device=device)

    if name == "small_loss":
        small_loss = hyperparameters("small_loss")
        return SmallLossDefense(rate=rate,warmup_epochs=warmup_epochs,forget_rate=small_loss.get("forget_rate"),ramp_epochs=small_loss.get("ramp_epochs", 10))

    if name == "gmm_filter":
        return GMMFilterDefense( num_samples=num_samples,warmup_epochs=warmup_epochs,p_threshold=hyperparameters("gmm_filter").get("p_threshold", 0.5),seed=seed,device=device)

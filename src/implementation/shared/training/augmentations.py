"""Baut die Pipelines (Data Augmentation)

Die Architektur der Datenverarbeitung ist konfigurationsgetrieben. 
1. Training: Enthält Zufall, um das Modell robuster 
   zu machen und Overfitting zu verhindern.
2. Validierung/Test: Ist komplett deterministisch. Jedes Bild wird exakt so gezeigt, 
   wie es ist, um reproduzierbare Metriken zu berechnen
"""
from torchvision import transforms


def _normalize(config):
    """Erstellt die Normalisierungs-Transformation für die Farbkanäle

    Args:
        config (dict): Das Konfigurations-Dictionary

    Returns:
        transforms.Normalize: Die fertige PyTorch-Transformation
    """
    return transforms.Normalize(mean=config["normalize_mean"], std=config["normalize_std"])


def get_train_transforms(config):
    """Baut die dynamische Pipeline für das Modell-Training

    Args:
        config (dict): Konfigurations-Dictionary für Augmentierungen

    Returns:
        transforms.Compose: Die Pipeline
    """
    pipeline = [transforms.ToPILImage()]

    if config.get("random_crop"):
        pipeline.append(transforms.RandomResizedCrop(config["crop_size"], scale=tuple(config["crop_scale"])))
    if config.get("random_horizontal_flip"):
        pipeline.append(transforms.RandomHorizontalFlip(p=config["horizontal_flip_p"]))

    pipeline.append(transforms.ToTensor())
    pipeline.append(_normalize(config))
    return transforms.Compose(pipeline)


def get_val_transforms(config):
    """Baut die statische Pipeline für Validierung, Test und Inferenz

    Args:
        config (dict): Das Konfigurations-Dictionary

    Returns:
        transforms.Compose: Transformations-Pipeline
    """
    return transforms.Compose([transforms.ToTensor(), _normalize(config)])

"""Einheitliche Schnittstelle für verschiedene Netzwerk-Architekturen"""
import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, mobilenet_v2, resnet18


class UnifiedClassifier(nn.Module):
    """Trennt den Feature-Extraktor vom Klassifikations-Kopf.

    1. `forward_features`: Liefert den hochdimensionalen Vektor
    2. `forward`: Liefert die finale Klassifikation
    """

    def __init__(self, feature_extractor: nn.Module, classifier: nn.Module):
        """Initialisiert den Wrapper mit den zwei Bausteinen

        Args:
            feature_extractor (nn.Module): Der Backbone des Netzwerks
            classifier (nn.Module): Der finale Linear-Layer
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self._classifier = classifier

    def forward_features(self, x: torch.Tensor):
        """Schickt das Bild nur durch den Backbone, um das Embedding zu extrahieren"""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor):
        """Der Forward-Pass"""
        return self._classifier(self.forward_features(x))

    @property
    def classifier(self):
        """Erlaubt den direkten Zugriff auf den Linear-Layer"""
        return self._classifier


def get_model(model_name: str, num_classes: int = 45):
    """Baut das gewünschte Modell und zerlegt es in seine Bausteine

    Args:
        model_name (str): Die gewünschte Architektur ("resnet18", "mobilenetv2", "convnext_tiny")
        num_classes (int): Anzahl der Ausgangs-Neuronen (Default: 45 für RESISC45)

    Returns:
        UnifiedClassifier: Das fertige Modell
        
    Raises:
        ValueError: Wenn ein unbekannter Modellname übergeben wird
    """
    if model_name == "resnet18":
        base = resnet18(weights=None, num_classes=num_classes)
        features = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
            base.avgpool, nn.Flatten(1))
        classifier = base.fc

    elif model_name == "mobilenetv2":
        base = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=0.5)
        features = nn.Sequential(
            base.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1))
        classifier = base.classifier

    elif model_name == "convnext_tiny":
        base = convnext_tiny(weights=None, num_classes=num_classes)
        features = nn.Sequential(
            base.features,
            base.avgpool,
            base.classifier[0],
            base.classifier[1])
        classifier = base.classifier[2]

    return UnifiedClassifier(features, classifier)


def load_state_dict_without_compile_prefix(model: nn.Module, checkpoint_path, device):
    """Lädt Modellgewichte

    Args:
        model (nn.Module): Das leere Modell
        checkpoint_path (Path): Pfad zur gespeicherten `.pt` oder `.pth` Datei
        device (torch.device): Ziel-Hardware

    Returns:
        nn.Module: Das Modell mit den geladenen Gewichten
    """
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

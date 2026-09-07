"""Zentrale Konfiguration für alle Pipeline-Skripte

Architektur-Hinweis:
Jeder Lauf in diesem Projekt wird von exakt zwei separaten Konfigurationsdateien gesteuert:
1. Daten-Config (`data_config`): Definiert, welche Daten genutzt werden (Splits, 
   Poisoning-Raten, Zielpaare für Angriffe).
2. Trainings-Config (`training_config`): Definiert, wie trainiert wird (Architektur, 
   Lernrate, Epochen, Augmentierungen).
"""
from pathlib import Path

import yaml

from implementation.shared import paths


def load_yaml(path):
    """Liest eine YAML-Datei sicher ein und wandelt sie in ein Dictionary um

    Args:
        path (Path): Pfad zur YAML-Datei

    Returns:
        dict: Das Dictionary der Konfiguration
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_config_paths(training_config=None, data_config=None):
    """Löst die Pfade auf und wendet Projekt-Defaults an, falls keine übergeben wurden

    Args:
        training_config (Path, optional): Übergebener Pfad zur Trainings-YAML
        data_config (Path, optional): Übergebener Pfad zur Daten-YAML

    Returns:
        tuple: (training_path, data_path) als Path-Objekte
    """
    training_path = training_config or paths.training_config_path()
    data_path = data_config or paths.data_config_path()
    return training_path, data_path


def load_configs(training_config=None, data_config=None):
    """Haupt-Einstiegspunkt für alle Skripte zum Laden beider Konfigurationen

    Args:
        training_config (Path, optional): Pfad zur Trainings-Konfiguration
        data_config (Path, optional): Pfad zur Daten-Konfiguration

    Returns:
        tuple: Ein Tupel aus zwei Dictionaries: (data_config_dict, training_config_dict)
    """
    training_path, data_path = resolve_config_paths(training_config, data_config)
    return load_yaml(data_path), load_yaml(training_path)


def add_config_arguments(parser):
    """Erweitert einen ArgumentParser um die beiden Standard-Konfigurations-Flags

    Args:
        parser (argparse.ArgumentParser): Der Argument-Parser des aufrufenden Skripts

    Returns:
        argparse.ArgumentParser: Der `--config` und `--data-config` erweiterte Parser
    """
    parser.add_argument("--config", type=Path, default=None,help="Trainings-YAML (Default: implementation/config/training.yaml).")
    parser.add_argument("--data-config", type=Path, default=None,help="Daten-/Poisoning-YAML (Default: data_preparation/config.yaml).")
    return parser

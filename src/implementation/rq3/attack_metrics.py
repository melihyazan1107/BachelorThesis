"""Berechnet Angriffsmetriken und Kollateralschäden (RQ3)

Aufruf:
    cd src && python -m implementation.rq3.attack_metrics [--data-config PATH]
"""

import implementation.shared.env

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from implementation.rq3.common import GROUP_KEYS, iter_rq3_runs, write_csv
from implementation.shared import config, paths


def load_target_pairs(data_config_path=None):
    """Lädt die Definition der manipulierten Klassenpaare aus der Konfiguration

    Args:
        data_config_path (Path): Pfad zur config.yaml

    Returns:
        list: Liste von Tupeln, die die angegriffenen Klassen repräsentieren
    """
    data_config = config.load_yaml(data_config_path or paths.data_config_path())
    target_pairs = []
    for class_a, class_b in data_config["poisoning"]["target_pairs"]:
        target_pairs.append((int(class_a), int(class_b)))
    return target_pairs


def divide_or_nan(numerator, denominator):
    """Führt eine sichere Division durch, um ZeroDivisionErrors zu vermeiden"""
    if denominator > 0:
        return numerator / denominator
    return np.nan


def compute_attack_metrics(confusion, target_pairs):
    """Analysiert die Confusion-Matrix auf gezielte Angriffs-Fehlklassifikationen.

    Args:
        confusion (np.ndarray): Confusion-Matrix [num_classes, num_classes]
        target_pairs (list): Liste der manipulierten Klassenpaare

    Returns:
        dict: Wörterbuch mit den berechneten Metriken:
              - attack_success_rate: Wie oft wurde A als B und B als A vorhergesagt?
              - attacked_accuracy: Wie oft wurden Bilder von A/B korrekt erkannt?
              - collateral_accuracy: Wie präzise war das Modell bei allen anderen Klassen?
    """
    target_partner = {}
    for class_a, class_b in target_pairs:
        target_partner[class_a] = class_b
        target_partner[class_b] = class_a
    attacked_classes = sorted(target_partner)
    unattacked_classes = [c for c in range(confusion.shape[0]) if c not in target_partner]

    samples_per_class = confusion.sum(axis=1)
    attacked_support = samples_per_class[attacked_classes].sum()
    unattacked_support = samples_per_class[unattacked_classes].sum()

    success_count = sum(confusion[c, target_partner[c]] for c in attacked_classes)
    attacked_correct = sum(confusion[c, c] for c in attacked_classes)
    collateral_correct = sum(confusion[c, c] for c in unattacked_classes)

    per_class_success = []
    for class_id in attacked_classes:
        if samples_per_class[class_id] > 0:
            partner = target_partner[class_id]
            per_class_success.append(confusion[class_id, partner] / samples_per_class[class_id])

    max_class_success = np.nan
    if per_class_success:
        max_class_success = np.nanmax(per_class_success)

    return {
        "n_attacked_classes": len(attacked_classes),
        "attack_success_rate": divide_or_nan(success_count, attacked_support),
        "attacked_accuracy": divide_or_nan(attacked_correct, attacked_support),
        "collateral_accuracy": divide_or_nan(collateral_correct, unattacked_support),
        "overall_accuracy": divide_or_nan(np.trace(confusion), confusion.sum()),
        "max_class_success": max_class_success}


def collect(root, target_pairs):
    """Iteriert durch alle Experiment-Ordner und berechnet die Metriken

    Args:
        root (Path): Das Verzeichnis mit den RQ3-Ergebnissen
        target_pairs (list): Die Zielpaare für die Metrik-Berechnung

    Returns:
        pd.DataFrame: Master-Tabelle (Zeilen = einzelne Experiment-Runs)
    """
    rows = []
    for defense, group, model, attack, seed, rate, run_dir in iter_rq3_runs(root):
        confusion_path = run_dir / "confusion_matrix.npy"
        if not confusion_path.exists():
            continue
        metrics = compute_attack_metrics(np.load(confusion_path), target_pairs)
        rows.append({
            "defense": defense,
            "group": group,
            "model": model,
            "attack": attack,
            "seed": seed,
            "rate": rate,
            **metrics})
        
    return pd.DataFrame(rows)


def parse_args():
    """Parst die Eingabeargumente aus der Kommandozeile"""
    parser = argparse.ArgumentParser(description="Attack-Success-Rate und Kollateralschaden (RQ3).")
    parser.add_argument("--data-config", type=Path, default=None, help="Daten-YAML mit poisoning.target_pairs (Default: data_preparation/config.yaml).")
    parser.add_argument("--root", type=Path, default=None, help="Wurzel der RQ3-Ergebnisse (Default: experiments/RQ3).")
    parser.add_argument("--out", type=Path, default=None, help="Ausgabeverzeichnis (Default: --root).")
    return parser.parse_args()


def main():
    """Durchsucht die Ergebnisse, aggregiert sie und schreibt CSVs"""
    args = parse_args()
    root = args.root or paths.RQ3_DIR
    output_dir = args.out or root
    target_pairs = load_target_pairs(args.data_config)

    attacked_classes = set()
    for pair in target_pairs:
        attacked_classes.update(pair)
    print(f"ATTACK METRICS (RQ3): {len(target_pairs)} Zielpaare, {len(attacked_classes)} Klassen betroffen")

    master_df = collect(root, target_pairs)
    if master_df.empty:
        print(f"Keine 'confusion_matrix.npy' Dateien unter {root} gefunden.", flush=True)
        return

    aggregated_df = master_df.groupby(GROUP_KEYS, as_index=False).agg(
        attack_success_rate=("attack_success_rate", "mean"),
        collateral_accuracy=("collateral_accuracy", "mean"),
        attacked_accuracy=("attacked_accuracy", "mean"),
        seeds=("seed", "nunique"))
    
    write_csv(master_df, output_dir / "rq3_attack_metrics.csv", "rq3_attack_metrics")
    write_csv(aggregated_df, output_dir / "rq3_attack_cells.csv", "rq3_attack_cells")


if __name__ == "__main__":
    main()

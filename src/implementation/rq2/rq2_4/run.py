"""Erstellung der Datensätze RQ2.4

Pro Seed wird Folgendes generiert:
1. Ein PyTorch-Tensor (`.pt`) mit den neuen, verfälschten Labels
2. Eine Übergangsmatrix (Transition Matrix) als JSON
3. Eine Heatmap (PDF) zur visuellen Analyse der Verwechslungen
4. Ein Summary (JSON) mit der absolut erreichten Vergiftungsrate
"""
import json

import pandas as pd
import torch

from implementation.rq2.common import find_baseline_telemetry_csv
from implementation.rq2.rq2_4.poisoner import build_poisoned_labels
from implementation.shared import paths
from implementation.shared.poisoning.metrics import PoisoningMetrics, plot_transition_heatmap_pdf
from implementation.shared.training.dataset import read_labels


def generate(data_config, train_config):
    """Führt den Vergiftungsprozess für alle konfigurierten Seeds durch

    Args:
        data_config (dict): Das geladene Dictionary der Daten-Konfiguration
        train_config (dict): Das geladene Dictionary der Trainings-Konfiguration
    """
    num_classes = data_config["poisoning"]["num_classes"]
    rq2_4_cfg = data_config["rq2_4"]
    window_start = rq2_4_cfg["window_start"]
    window_end = rq2_4_cfg["window_end"]
    confusion_threshold = rq2_4_cfg["confusion_threshold"]
    model_name = train_config["experiment"]["model"]
    baseline_rq = train_config["experiment"]["baseline_rq"]

    clean_labels = read_labels(paths.hdf5_path("train"))

    for seed in data_config["poisoning"]["seeds"]:
        telemetry_csv = find_baseline_telemetry_csv(baseline_rq, model_name, seed)
        telemetry = pd.read_csv(telemetry_csv, usecols=["epoch", "sample_index", "predicted_class"])
        poisoned_labels = build_poisoned_labels(telemetry, clean_labels, window_start, window_end, confusion_threshold)

        output_path = paths.poisoned_labels_path("RQ2.4", seed, rate=0.0)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(poisoned_labels), output_path)

        metrics = PoisoningMetrics(clean_labels, poisoned_labels, num_classes)
        metrics.save_transition_matrix_json(output_dir / "transition_matrix.json")
        plot_transition_heatmap_pdf(metrics.transition_matrix, output_dir / "heatmap.pdf",title=f"RQ2.4 Uncertainty (seed {seed})")

        summary = {
            "seed": seed,
            "window": [window_start, window_end],
            "confusion_threshold": confusion_threshold,
            "poisoned_count": metrics.get_poisoned_count(),
            "empirical_rate": metrics.compute_empirical_rate()}
        
        with open(output_dir / "poisoning_summary.json", "w", encoding="utf-8") as json_file:
            json.dump(summary, json_file, indent=2)

        print(f"RQ2.4 seed={seed}: {metrics.get_poisoned_count()} Flips (rate {metrics.compute_empirical_rate():.4f})")

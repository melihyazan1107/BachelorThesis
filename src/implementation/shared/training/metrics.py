"""Sammelt, berechnet und speichert die Telemetrie pro Epoche

Architektur-Design:
Dieses Modul unterstützt zwei Betriebsmodi:
1. Standard-Modus (RQ1/RQ2): Zeichnet nur die Basis-Metriken auf (Accuracy, Loss).
2. Diagnostik-Modus (RQ3): Wenn `diagnostics=True` gesetzt ist, wird das Schema 
   um komplexe Analysekennzahlen (Separabilität, Memorization, Filtergüte, Hardware-Kosten) 
   erweitert
"""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

DIAGNOSTIC_COLUMNS = ("memorization_rate","loss_auroc","loss_overlap","filter_precision","filter_recall","filter_f1","kept_fraction","epoch_seconds","peak_vram_mb")


def _telemetry_arrays(telemetry_dict, keys):
    """Extrahiert Listen aus dem Dictionary und wandelt sie in NumPy-Arrays um"""
    return tuple(np.array(telemetry_dict[key]) for key in keys)


def loss_overlap(clean_losses, poison_losses, bins=100):
    """Berechnet den Overlap der beiden Loss-Verteilungen

    Mathematischer Hintergrund:
    Erstellt zwei Histogramme für den Loss der sauberen und den 
    Loss der vergifteten Bilder. Anschließend wird die Schnittmenge (Fläche unter 
    beiden Kurven) berechnet
    - Wert 0.0: Perfekte Trennung. Es gibt einen Schwellenwert, der beide Gruppen fehlerfrei trennt.
    - Wert 1.0: Komplette Überschneidung. Das Modell kann saubere und vergiftete 
                Bilder anhand des Losses absolut nicht unterscheiden.

    Args:
        clean_losses (np.ndarray): Loss-Werte der echten Bilder
        poison_losses (np.ndarray): Loss-Werte der Bilder mit falschen Labels
        bins (int): Auflösung der Histogramme

    Returns:
        float: Die Überschneidungsfläche (0.0 bis 1.0).
    """
    if clean_losses.size == 0 or poison_losses.size == 0:
        return float("nan")

    min_loss = float(min(clean_losses.min(), poison_losses.min()))
    max_loss = float(max(clean_losses.max(), poison_losses.max()))
    if not np.isfinite(min_loss) or not np.isfinite(max_loss) or max_loss - min_loss < 1e-12:
        return 1.0

    edges = np.linspace(min_loss, max_loss, bins + 1)
    density_clean, _ = np.histogram(clean_losses, bins=edges, density=True)
    density_poison, _ = np.histogram(poison_losses, bins=edges, density=True)
    return float(np.minimum(density_clean, density_poison).sum() * (edges[1] - edges[0]))


def separability(losses, poison_mask):
    """Berechnet AUROC und Overlap"""
    if not poison_mask.any() or poison_mask.all():
        return float("nan"), float("nan")
    auroc = float(roc_auc_score(poison_mask, losses))
    overlap = loss_overlap(losses[~poison_mask], losses[poison_mask])
    return auroc, overlap


def _blank_if_nan(value):
    """Formatierungshilfe: Schreibt leere Strings in die CSV statt 'nan', was Speicher spart"""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


class MetricsTracker:
    """Verwaltet und aggregiert alle Telemetrie-Daten über den Verlauf des Trainings"""
    def __init__(self, num_classes=45, diagnostics=False):
        self.num_classes = num_classes
        self.diagnostics = diagnostics
        self.epoch_data = []

    def compute_epoch_metrics(self, telemetry_dict, epoch_stats=None, keep_threshold=0.5):
        """Kondensiert die Rohdaten zehntausender Bilder zu einem einzigen Epochen-Report

        Args:
            telemetry_dict (dict): Dictionary mit Listen für Loss, Vorhersagen
            epoch_stats (dict, optional): Hardware-Metriken (Dauer, VRAM)
            keep_threshold (float): Ab welchem Gewicht ein Bild als "behalten" gilt

        Returns:
            dict: Eine Row mit allen berechneten Metriken für diese Epoche
        """
        if not telemetry_dict:
            return {}

        true_labels, noisy_labels, predictions, losses = _telemetry_arrays(
            telemetry_dict, ["true_label", "noisy_label", "predicted_class", "loss"])
        epoch = telemetry_dict["epoch"][0]

        acc_noisy = accuracy_score(noisy_labels, predictions)
        acc_clean = accuracy_score(true_labels, predictions)

        clean_mask = true_labels == noisy_labels
        poison_mask = ~clean_mask

        clean_loss = np.mean(losses[clean_mask]) if np.any(clean_mask) else 0.0
        poison_loss = np.mean(losses[poison_mask]) if np.any(poison_mask) else 0.0

        metrics = {
            "epoch": epoch,
            "phase": "train",
            "acc_clean": float(acc_clean),
            "acc_noisy": float(acc_noisy),
            "clean_loss": float(clean_loss),
            "poison_loss": float(poison_loss)}

        if self.diagnostics:
            metrics.update(self._diagnostics(
                telemetry_dict, noisy_labels, predictions, losses, poison_mask,
                epoch_stats or {}, keep_threshold))

        self.epoch_data.append(metrics)
        return metrics

    def _diagnostics(self, telemetry_dict, noisy_labels, predictions, losses, poison_mask, epoch_stats, keep_threshold):
        """Berechnet die Diagnose-Metriken für reaktive Verteidigungen"""
        auroc, overlap = separability(losses, poison_mask)

        if poison_mask.any():
            memorization = float(np.mean(predictions[poison_mask] == noisy_labels[poison_mask]))
        else:
            memorization = float("nan")

        filter_precision = filter_recall = filter_f1 = kept_fraction = float("nan")
        weights = telemetry_dict.get("sample_weight")
        if weights is not None:
            weights = np.asarray(weights)
            kept = weights >= keep_threshold
            kept_fraction = float(np.mean(kept))
            dropped = ~kept

            if dropped.any():
                filter_precision = float(np.mean(poison_mask[dropped]))
            if poison_mask.any():
                filter_recall = float(np.mean(dropped[poison_mask]))

            both_defined = np.isfinite(filter_precision) and np.isfinite(filter_recall)
            if both_defined and (filter_precision + filter_recall) > 0:
                precision_recall_sum = filter_precision + filter_recall
                filter_f1 = 2 * filter_precision * filter_recall / precision_recall_sum

        values = {
            "memorization_rate": memorization,
            "loss_auroc": auroc,
            "loss_overlap": overlap,
            "filter_precision": filter_precision,
            "filter_recall": filter_recall,
            "filter_f1": filter_f1,
            "kept_fraction": kept_fraction,
            "epoch_seconds": epoch_stats.get("epoch_seconds"),
            "peak_vram_mb": epoch_stats.get("peak_vram_mb")}
        
        return {key: _blank_if_nan(values[key]) for key in DIAGNOSTIC_COLUMNS}


    def save_epoch_csv(self, path: Path):
        """Schreibt die Epochen-Daten in eine CSV-Datei"""
        if not self.epoch_data:
            return
        columns = self.epoch_data[0].keys()
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.epoch_data)

    def save_confusion_matrix_json(self, telemetry_dict, path: Path):
        """Speichert die Konfusionsmatrix für die aktuelle Epoche"""
        if not telemetry_dict:
            return
        
        true_labels, predictions = _telemetry_arrays(
            telemetry_dict, ["true_label", "predicted_class"])
        
        class_ids = np.arange(self.num_classes)
        matrix = confusion_matrix(true_labels, predictions, labels=class_ids)
        with open(path, "w") as file:
            json.dump(matrix.tolist(), file)


    def save_sample_telemetry(self, telemetry_dict, path: Path):
        """Speichert die Rohdaten für JEDES Bild"""
        if not telemetry_dict:
            return

        columns = {
            "epoch": telemetry_dict["epoch"],
            "sample_index": telemetry_dict["sample_index"],
            "true_label": telemetry_dict["true_label"],
            "noisy_label": telemetry_dict["noisy_label"],
            "predicted_class": telemetry_dict["predicted_class"],
            "confidence": telemetry_dict["confidence"],
            "loss": telemetry_dict["loss"]}
        
        if "sample_weight" in telemetry_dict:
            columns["sample_weight"] = telemetry_dict["sample_weight"]

        write_header = not path.exists()
        pd.DataFrame(columns).to_csv(path, mode="a", header=write_header, index=False)

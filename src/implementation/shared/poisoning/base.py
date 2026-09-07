"""Zentrale Base Class für alle Label-Poisoning-Angriffe

Generierte Artefakte pro (Seed, Rate) Kombination unter `<output_dir>/<rq_name>/seed_<seed>/`:
    1. labels_<rate>.pt: Der finale PyTorch-Tensor (int64) mit den falschen Labels
    2. transition_matrix_<rate>.json: 45x45 Matrix
    3. poisoning_summary_<rate>.json: Meta-Report
    4. heatmap_<rate>.pdf: Heatmap der Transition-Matrix
    5. retention_bar_<rate>.pdf: Balkendiagramm
"""
from pathlib import Path

import numpy as np
import torch

from implementation.shared.poisoning.metrics import (PoisoningMetrics,plot_class_retention_bar_pdf,plot_transition_heatmap_pdf)
from implementation.shared.training.dataset import read_labels


class BasePoisoner:
    """Abstract Class für alle Vergiftungen

    Attributes:
        rq_name (str): Name der Forschungsfrage
        erasure_limit (float): Sicherheit damit die Klassen nicht ausgelöscht werden
    """
    rq_name = ""
    erasure_limit = 0.5

    def __init__(self, train_h5_path: Path, output_dir: Path, seeds, rates, num_classes):
        """Initialisiert das Poisoning

        Args:
            train_h5_path (Path): Pfad zum HDF5-Archiv mit den sauberen Trainingsdaten
            output_dir (Path): Basis-Zielordner für die generierten Artefakte
            seeds (list): Liste der Random-Seeds
            rates (list): Liste der Vergiftungsraten
            num_classes (int): Gesamtanzahl der Klassen im Datensatz
        """
        self.train_h5_path = train_h5_path
        self.output_dir = output_dir
        self.seeds = seeds
        self.rates = rates
        self.num_classes = num_classes

    def _flip(self, labels, rate, seed):
        """Logik des Angriffes

        Args:
            labels (np.ndarray): Das 1D-Array der sauberen Labels
            rate (float): Der Anteil der zu verfälschenden Labels
            seed (int): Der Seed für reproduzierbare Angriffe

        Returns:
            np.ndarray: Ein NEUES Array mit den poisoned Labels
        
        Raises:
            NotImplementedError: Wenn eine Kind Klasse diese Methode nicht implementiert
        """
        raise NotImplementedError

    def run(self):
        """Führt die komplette Pipeline aus

        Lädt die sauberen Labels exakt einmal in den RAM.
        Iteriert dann über alle Kombinationen aus Seed und Rate, ruft die 
        spezifische `_flip` auf und triggert den Artefakt-Export
        """
        clean_labels = read_labels(self.train_h5_path)

        for seed in self.seeds:
            for rate in self.rates:
                poisoned_labels = self._flip(clean_labels, rate, seed)
                self._write_artifacts(clean_labels, poisoned_labels, seed, rate)

    def _write_artifacts(self, clean_labels, poisoned_labels, seed, rate):
        """Speichert den generierten Tensor und alle Analyse-Artefakte ab

        Args:
            clean_labels (np.ndarray): Originale Ground-Truth-Labels
            poisoned_labels (np.ndarray): Die modifizierten Labels aus `_flip`
            seed (int): Der aktuell verwendete Seed
            rate (float): Die aktuell verwendete Rate
        """
        out_dir = self.output_dir / self.rq_name / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{rate:.2f}"

        labels_tensor = torch.from_numpy(poisoned_labels.astype(np.int64))
        torch.save(labels_tensor, out_dir / f"labels_{tag}.pt")

        metrics = PoisoningMetrics(clean_labels, poisoned_labels, self.num_classes)
        metrics.verify_erasure_limit(limit=self.erasure_limit)
        metrics.save_transition_matrix_json(out_dir / f"transition_matrix_{tag}.json")
        metrics.save_poisoning_summary_json(out_dir / f"poisoning_summary_{tag}.json",{"seed": seed, "rate": rate, "erasure_limit": self.erasure_limit})

        plot_transition_heatmap_pdf(metrics.transition_matrix,out_dir / f"heatmap_{tag}.pdf", title=f"{self.rq_name} Transition Heatmap (Rate: {rate})")
        plot_class_retention_bar_pdf(metrics.transition_matrix, out_dir / f"retention_bar_{tag}.pdf")
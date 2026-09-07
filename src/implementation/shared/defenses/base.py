"""Verteidigungsstrategien

Dieses Modul ist das Interface für alle Defenses
Es nutzt das Strategy Pattern, um tiefgreifende Veränderungen am 
Trainingsprozess vorzunehmen, ohne die eigentliche 
Trainingsschleife für jede neue Methode umschreiben zu müssen

Eine Verteidigung kann an drei exakt definierten Stellen eingreifen:
1. Vor dem Forward-Pass: `transform_batch` (z.B. für Mixup, Daten-Augmentierung)
2. Bei der Fehlerberechnung: `compute_loss` (z.B. für Label Smoothing, GCE, ELR)
3. Bei der Gewichtung/Filterung: `sample_weights` (z.B. Small-Loss, GMM Filter)

Die Standard-Implementierung (`NullDefense`) reproduziert das ungeschützte 
Training aus RQ1 und RQ2
"""
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


class Defense:
    """Abstract Class für alle Verteidigungen"""

    name = "none"
    group = "none"

    needs_clean_forward = False
    loss_is_plain_ce = True
    needs_idx = False

    keep_threshold = 0.5

    def on_epoch_start(self, epoch: int):
        """Wird exakt einmal zu Beginn jeder neuen Epoche aufgerufen.

        Dient der Initialisierung epochenbezogener Variablen, wie zum 
        Beispiel dem schrittweisen Erhöhen von Schedules

        Args:
            epoch (int): Die aktuelle Epochen-Nummer
        """
        pass
        pass

    def observe(self, sample_indices: Tensor, plain_loss: Tensor):
        """Ermöglicht es reaktiven Filtern, den Zustand des Netzes zu beobachten

        Args:
            sample_indices (Tensor): Globale IDs der Bilder im aktuellen Batch
            plain_loss (Tensor): Der unmodifizierte Standard-Loss für diese Bilder
        """
        pass

    def transform_batch(self, images: Tensor, targets: Tensor):
        """Modifiziert Eingabedaten vor dem Modell-Forward-Pass

        Args:
            images (Tensor): Die rohen Bild-Tensoren vom DataLoader
            targets (Tensor): Die Labels

        Returns:
            tuple: (Modifizierte Bilder, Context-Dictionary)
        """
        return images, {}

    def compute_loss(self, logits: Tensor, targets: Tensor, sample_indices: Tensor, context: dict[str, Any]):
        """Berechnet den Loss für jedes Bild

        Args:
            logits (Tensor): Die rohen Netzwerk-Ausgaben
            targets (Tensor): Die Ziel-Labels
            sample_indices (Tensor): Die Bild-IDs
            context (dict): Der von `transform_batch` übergebene Kontext

        Returns:
            Tensor: Ein 1D-Tensor mit dem spezifischen Loss für jedes Bild im Batch
        """
        return F.cross_entropy(logits, targets, reduction="none")

    def sample_weights(self, per_sample_loss: Tensor, targets: Tensor, sample_indices: Tensor):
        """Vergibt individuelle Gewichte für jedes Bild vor dem Gradienten-Update.

        Args:
            per_sample_loss (Tensor): Der individuelle Fehler jedes Bildes
            targets (Tensor): Die Ziel-Labels
            sample_indices (Tensor): Die globalen Bild-IDs

        Returns:
            Tensor | None: Gewichts-Tensor der Form [Batch_Size]. Wenn None 
                           zurückgegeben wird, zählt jedes Bild gleich (Standard)
        """
        return None

    def state(self):
        """Liefert Metadaten der Verteidigung"""
        return {"name": self.name, "group": self.group}


class NullDefense(Defense):
    """Die Standard-Baseline"""
    pass


def aggregate_loss(per_sample_loss: Tensor, weights: Tensor | None):
    """Fasst den individuellen Loss zu einem finalen Skalar zusammen

    Args:
        per_sample_loss (Tensor): Unreduzierter Loss [Batch_Size]
        weights (Tensor | None): Optionale Gewichte pro Bild

    Returns:
        Tensor: Skalarer Loss
    """
    if weights is None:
        return per_sample_loss.mean()
    return (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)


def derive_generator(seed: int, salt: int):
    """Erstellt einen deterministischen, aber unabhängigen Zufallsgenerator

    Args:
        seed (int): Der globale Experiment-Seed
        salt (int): Eine spezifische Zahl, damit der Generator nicht in jeder 
        Epoche dieselben Zahlen liefert

    Returns:
        torch.Generator: Ein Zufallsgenerator
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed) * 1_000_003 + int(salt))
    return generator


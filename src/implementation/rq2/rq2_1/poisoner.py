"""Implementiert Targeted Label-Poisoning für RQ2.1

Hier wird ein gezielter Angriff simuliert. Das Skript nutzt die zuvor ermittelten Paare 
von extrem ähnlichen Klassen und tauscht deren Labels untereinander aus.

Zwei wichtige Entscheidungen:
1. Symmetrischer Tausch: Es werden immer genau 'n' Samples von Klasse A zu B und 
   'n' Samples von Klasse B zu A geflippt. Das garantiert, dass die Gesamtgröße 
    jeder Klasse absolut identisch bleibt.
2. Erasure-Limit: Verhindert, dass eine Klasse komplett durch den Tausch ausgelöscht 
   wird.
"""
from pathlib import Path

import numpy as np

from implementation.shared.poisoning.base import BasePoisoner


class TargetedPoisoner(BasePoisoner):
    """Gezielte Label-Manipulationen zwischen definierten Klassen-Paaren
    
    Attributes:
        rq_name (str): Forschungsfrage identifizieren
        target_pairs (list of tuples): Eine Liste von Klassen-ID-Paaren,
                                       zwischen denen die Labels getauscht werden
        erasure_limit (float): Der maximale Anteil einer Klasse, der getauscht werden darf
                               Schützt die Klassen
    """

    rq_name = "RQ2"

    def __init__(self, train_h5_path: Path, output_dir: Path, seeds, rates, num_classes, target_pairs, erasure_limit):
        """Initialisiert den Targeted Poisoner und validiert die Zielpaare

        Args:
            train_h5_path (Path): Pfad zu den sauberen Trainingsdaten
            output_dir (Path): Zielverzeichnis für die vergifteten Daten
            seeds (list): Liste von Random Seeds für Reproduzierbarkeit
            rates (list): Liste der Vergiftungsraten
            num_classes (int): Gesamtanzahl der Klassen im Datensatz
            target_pairs (list of tuples): Die zu tauschenden Klassenpaare
            erasure_limit (float): Maximale Tausch-Quote pro Klasse

        Raises:
            ValueError: Wenn eine Klasse in mehr als einem target_pair existiert
        """
        super().__init__(train_h5_path, output_dir, seeds, rates, num_classes)
        self.target_pairs = target_pairs
        self.erasure_limit = erasure_limit

        all_class_ids = []
        for pair in target_pairs:
            all_class_ids.extend(pair)

        if len(all_class_ids) != len(set(all_class_ids)):
            duplicates = set()

            for class_id in all_class_ids:
                if all_class_ids.count(class_id) > 1:
                    duplicates.add(class_id)
            raise ValueError(f"target_pairs enthalten doppelte Klassen-IDs: {duplicates}")


    def _flip(self, labels, rate, seed):
        """Verteilt das Tausch-Budget gleichmäßig und flippt die Labels

        Ablauf:
        1. Berechnet die absolute Anzahl an Bildern, die verändert werden sollen
        2. Rundet auf eine gerade Zahl ab
        3. Verteilt die Tausch-Operationen gleichmäßig auf alle verfügbaren Klassenpaare
        4. Deckelt die Operationen pro Paar am `erasure_limit`
        5. Führt den eigentlichen Tausch aus

        Args:
            labels (np.ndarray): Das Array der sauberen Klassen-Labels
            rate (float): Der zu verfälschende Gesamtanteil des Datensatzes
            seed (int): Seed für den Zufallsgenerator

        Raises:
            ValueError: Wenn die Tausch-Rate wegen des Erasure-Limits 
                        mathematisch nicht erreicht werden kann

        Returns:
            np.ndarray: Array der manipulierten Labels
        """
        num_samples = len(labels)
        total_flips = int(rate * num_samples)
        total_flips -= total_flips % 2

        total_swaps = total_flips // 2
        base_swaps, leftover_swaps = divmod(total_swaps, len(self.target_pairs))

        swaps_per_pair = []

        for pair_index, (class_a, class_b) in enumerate(self.target_pairs):
            swap_limit = min(int(self.erasure_limit * np.sum(labels == class_a)),int(self.erasure_limit * np.sum(labels == class_b)))
            desired_swaps = base_swaps + (1 if pair_index < leftover_swaps else 0)
            swaps_per_pair.append(min(desired_swaps, swap_limit))

        total_achievable = sum(swaps * 2 for swaps in swaps_per_pair)

        if total_achievable < total_flips:
            raise ValueError(f"Requested {total_flips} flips but only {total_achievable} achievable")

        poisoned_labels = labels.copy()
        random_state = np.random.default_rng(seed)

        for (class_a, class_b), num_swaps in zip(self.target_pairs, swaps_per_pair):
            indices_a = np.flatnonzero(labels == class_a)
            selected_a = random_state.choice(indices_a, size=num_swaps, replace=False)
            poisoned_labels[selected_a] = class_b

            indices_b = np.flatnonzero(labels == class_b)
            selected_b = random_state.choice(indices_b, size=num_swaps, replace=False)
            poisoned_labels[selected_b] = class_a

        return poisoned_labels

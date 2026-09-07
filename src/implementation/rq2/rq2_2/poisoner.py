"""Implementiert isoliertes Label-Poisoning für RQ2.2

Wichtiger Unterschied:
Die Rate bezieht sich hier NICHT auf den gesamten Datensatz!
Eine Rate von 0.2 bedeutet hier, dass 20% der Bilder dieser zwei Klassen 
getauscht werden, nicht 20% des ganzen Datensatzes
"""

import numpy as np

from implementation.shared.poisoning.base import BasePoisoner


class SinglePairPoisoner(BasePoisoner):
    """Führt einen symmetrischen Label-Flip zwischen zwei Klassen durch

    Attributes:
        erasure_limit (float): Wert für die Base-Klasse, da die Rate hier 
                               ohnehin relativ zur Klassengröße ist. Wird auf 0 gesetzt
        target_pair (tuple): Ein Tupel aus zwei Klassen (class_a, class_b).
        rq_name (str): Name ("RQ2.2-similar" oder "RQ2.2-dissimilar"), um die generierten 
                       HDF5-Dateien sauber im Dateisystem zu trennen.
    """

    erasure_limit = 0.0

    def __init__(self, train_h5_path, output_dir, seeds, rates, num_classes, pair, tag):
        """Initialisiert den Poisoner für ein spezifisches Klassenpaar

        Args:
            train_h5_path (Path): Pfad zum sauberen Trainingsdatensatz
            output_dir (Path): Basis-Ausgabeverzeichnis
            seeds (list): Liste der zu verwendenden Random Seeds
            rates (list): Liste der Vergiftungsraten (z.B. 0.1, 0.2). Bezieht sich hier 
                          auf die lokale Klassengröße
            num_classes (int): Gesamtanzahl der Klassen
            pair (tuple/list): Die zwei Klassen, zwischen denen getauscht wird
            tag (str): Tag für das Paar (z.B. "similar" oder "dissimilar")
        """
        super().__init__(train_h5_path, output_dir, seeds, rates, num_classes)
        self.target_pair = (int(pair[0]), int(pair[1]))
        self.rq_name = f"RQ2.2-{tag}"

    def _flip(self, labels, rate, seed):
        """Tauscht Labels zwischen dem spezifizierten Zielpaar symmetrisch aus
        
        Ablauf:
        1. Findet alle Indizes, die zur Klasse A oder Klasse B gehören
        2. Ermittelt die Obergrenze für den Tausch basierend auf der kleineren 
           der beiden Klassen
        3. Berechnet die Anzahl der Flips als Prozentsatz dieser Obergrenze
        4. Zieht zufällig Indizes aus A und weist ihnen Label B zu und umgekehrt

        Args:
            labels (np.ndarray): Das 1D-Array der ursprünglichen, sauberen Klassen-Labels
            rate (float): Der Anteil der Klassengröße, der getauscht werden soll (z.B. 0.3 = 30%)
            seed (int): Random Seed für deterministische, reproduzierbare Zufallszüge

        Returns:
            np.ndarray: Eine Kopie des Label-Arrays mit den lokal verfälschten Paaren
        """
        random_state = np.random.default_rng(seed)
        class_a, class_b = self.target_pair
        indices_a = np.flatnonzero(labels == class_a)
        indices_b = np.flatnonzero(labels == class_b)
        num_swaps = int(rate * min(len(indices_a), len(indices_b)))

        poisoned_labels = labels.copy()
        if num_swaps > 0:
            selected_a = random_state.choice(indices_a, size=num_swaps, replace=False)
            selected_b = random_state.choice(indices_b, size=num_swaps, replace=False)
            poisoned_labels[selected_a] = class_b
            poisoned_labels[selected_b] = class_a

        return poisoned_labels

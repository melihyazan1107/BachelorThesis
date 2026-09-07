"""Implementiert symmetrisches Label-Rauschen für RQ1

"""
import numpy as np

from implementation.shared.poisoning.base import BasePoisoner


class SymmetricPoisoner(BasePoisoner):
    """Klasse zur Injektion von symmetrischem Label-Rauschen
    
    Beim symmetrischen Rauschen wird ein vorher definierter Prozentsatz der 
    Labels zufällig geflippt.
    
    Attributes:
        rq_name (str): die Forschungsfrage 
    """

    rq_name = "RQ1"

    def _flip(self, labels, rate, seed):
        """Verfälscht einen Teil der Labels durch zufällige Klassenwechsel

        Der Algorithmus arbeitet in drei Schritten:
        1. Wählt zufällig die Indizes aus, deren Labels verändert werden sollen
        2. Generiert für jeden gewählten Index einen zufälligen Offset zwischen 
           1 und `self.num_classes - 1`. Die 0 ist ausgeschlossen, was garantiert, 
           dass sich das Label definitiv ändert und nicht zufällig gleich bleibt
        3. Addiert den Offset zum ursprünglichen Label und wendet eine Modulo-Operation an

        Args:
            labels (np.ndarray): Das ursprüngliche 1D-Array mit den sauberen Klassen-Labels
            rate (float): Der Anteil der Labels, der korrumpiert werden soll
            seed (int): Random Seed für den Generator, für die Reproduzierbarkeit

        Returns:
            np.ndarray: Eine Kopie, bei dem ein Teil der Labels verfälscht wurde
        """
        random_state = np.random.RandomState(seed)
        num_samples = len(labels)
        num_poisoned = int(rate * num_samples)

        poison_indices = random_state.choice(num_samples, size=num_poisoned, replace=False)
        offsets = random_state.randint(1, self.num_classes, size=num_poisoned)

        poisoned_labels = labels.copy()
        poisoned_labels[poison_indices] = (labels[poison_indices] + offsets) % self.num_classes
        return poisoned_labels

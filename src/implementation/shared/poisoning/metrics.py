"""Analysiert und visualisiert die Auswirkungen von Label-Poisoning-Angriffen

Dieses Modul übernimmt die Qualitätssicherung nach einem Angriff:
1. Es erstellt eine Transition-Matrix
2. Es prüft hart, ob das "Erasure-Limit" verletzt wurde
3. Es berechnet die tatsächliche Vergiftungsrate
4. Es generiert zwei PDFs für die menschliche Prüfung
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PoisoningMetrics:
    """Berechnet und speichert alle Metriken eines durchgeführten Label-Angriffs

    - Y-Achse entsprechen der ECHTEN, originalen Klasse
    - X-Achse entsprechen der NEUEN, vergifteten Klasse
    - Die Hauptdiagonale zeigt, wie viele Bilder ihr originales Label behalten haben
    """

    def __init__(self, clean_labels, poisoned_labels, num_classes):
        """Initialisiert die Metriken und baut die Transition-Matrix auf

        Args:
            clean_labels (np.ndarray): Array der originalen Ground-Truth-Labels
            poisoned_labels (np.ndarray): Array der manipulierten Labels
            num_classes (int): Gesamtanzahl der Klassen
        """
        self.clean_labels = clean_labels
        self.poisoned_labels = poisoned_labels
        self.num_classes = num_classes
        self.total = len(clean_labels)

        self.transition_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for clean_label, poisoned_label in zip(self.clean_labels, self.poisoned_labels):
            self.transition_matrix[clean_label, poisoned_label] += 1

    def verify_erasure_limit(self, limit=0.5):
        """Prüft, ob eine Klasse durch den Angriff versehentlich zerstört wurde

        Args:
            limit (float): Der Mindestanteil an Bildern, die das originale 
                           Label behalten müssen

        Raises:
            ValueError: Wenn auch nur eine einzige Klasse dieses Limit unterschreitet
        """
        for class_id in range(self.num_classes):
            class_total = np.sum(self.clean_labels == class_id)
            if class_total == 0:
                continue
            retention_rate = self.transition_matrix[class_id, class_id] / class_total
            if retention_rate < limit:
                raise ValueError(f"Erasure-Limit verletzt")

    def compute_empirical_rate(self):
        """Berechnet die tatsächlich Vergiftungsrate"""
        if self.total == 0:
            return 0.0
        return self.get_poisoned_count() / self.total

    def get_poisoned_count(self):
        """Zählt die absolute Anzahl der Bilder, deren Label sich verändert hat"""
        return int(np.sum(self.clean_labels != self.poisoned_labels))

    def save_transition_matrix_json(self, path: Path):
        """Speichert die 2D-Matrix als JSON"""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.transition_matrix.tolist(), file)

    def save_poisoning_summary_json(self, path: Path, config_metadata):
        """Erstellt einen Meta-Report des Angriffs"""
        summary = {"total_images": self.total,"poisoned_count": self.get_poisoned_count(),"empirical_rate": self.compute_empirical_rate(),**config_metadata}

        with open(path, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)


def plot_transition_heatmap_pdf(transition_matrix, out_pdf: Path, title: str = "Poisoning Transition Matrix"):
    """Erstellt eine Heatmap der Label-Verschiebungen

    Args:
        transition_matrix (np.ndarray): Die quadratische Matrix [Original, Vergiftet]
        out_pdf (Path): Zielpfad für die PDF-Datei
        title (str): Titel des Diagramms
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log1p(transition_matrix), cmap="Reds", cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.xlabel("Poisoned Class")
    plt.ylabel("Original Class")
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()


def plot_class_retention_bar_pdf(transition_matrix, out_pdf: Path):
    """Erstellt ein Balkendiagramm

    Dieses Diagramm visualisiert pro Klasse, wie viel Prozent der Bilder grün (behalten) 
    und wie viel rot (vergiftet) sind

    Args:
        transition_matrix (np.ndarray): Die quadratische Matrix [Original, Vergiftet]
        out_pdf (Path): Zielpfad für die PDF-Datei
    """
    num_classes = transition_matrix.shape[0]

    kept = np.diag(transition_matrix)
    totals = np.maximum(np.sum(transition_matrix, axis=1), 1)
    kept_percent = (kept / totals) * 100
    flipped_percent = 100 - kept_percent

    classes = np.arange(num_classes)
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(classes, kept_percent, bar_width, label="Original (Retained)", color="#4CAF50")
    ax.bar(classes, flipped_percent, bar_width, bottom=kept_percent, label="Flipped (Poisoned)", color="#F44336")
    ax.axhline(y=50, color="black", linestyle="--", label="Erasure Limit (50%)")

    ax.set_xlabel("Class ID")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Original vs. Flipped Labels per Class (Erasure Limit Verification)")
    ax.set_xlim(-1, num_classes)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf")
    plt.close(fig)

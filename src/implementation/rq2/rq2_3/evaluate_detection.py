"""Wertet die Ergebnisse der Blindstudie RQ2.3 aus

Berechnet werden zwei Kernmetriken:
1. Detection Rate (True Positive Rate): Wie viel Prozent der tatsächlich vergifteten Bilder wurden gefunden?
2. Fehlalarmquote (False-Positive-Rate): Wie viel Prozent der sauberen Bilder wurden fälschlicherweise als vergiftet markiert?

Aufruf:
    cd src && python -m implementation.rq2.rq2_3.evaluate_detection
"""
import implementation.shared.env

from pathlib import Path

import pandas as pd

from implementation.shared import paths


def evaluate_detection(key_csv: Path, annotations_csv: Path):
    """Berechnet die Erfolgs- und Fehlerraten der menschlichen Evaluation

    Args:
        key_csv (Path): Pfad zur Lösungsdatei
        annotations_csv (Path): Pfad zur annotierten Datei des Gutachters 

    Returns:
        dict: Ein Dictionary mit den absoluten Zählern und den berechneten Raten:
              - n_poisoned (int): Anzahl der vergifteten Bilder im Test
              - n_clean (int): Anzahl der sauberen Bilder im Test
              - detection_rate (float): Anteil der gefundenen vergifteten Bilder [0.0 - 1.0]
              - false_positive_rate (float): Anteil der Fehlalarme bei sauberen Bildern [0.0 - 1.0]
    """
    ground_truth = pd.read_csv(key_csv).set_index("id")
    annotations = pd.read_csv(annotations_csv).set_index("id")
    merged = ground_truth.join(annotations["flagged"])

    poisoned_samples = merged[merged["is_poisoned"] == 1]
    clean_samples = merged[merged["is_poisoned"] == 0]

    detection_rate = 0.0
    if not poisoned_samples.empty:
        detection_rate = float(poisoned_samples["flagged"].mean())

    false_positive_rate = 0.0
    if not clean_samples.empty:
        false_positive_rate = float(clean_samples["flagged"].mean())

    return {
        "n_poisoned": len(poisoned_samples),
        "n_clean": len(clean_samples),
        "detection_rate": detection_rate,
        "false_positive_rate": false_positive_rate}


def main():
    """Auswertung und Formatierung der Ergebnisse für die Konsole"""
    review_dir = paths.PROJECT_ROOT / "human_review"
    result = evaluate_detection(review_dir / "key.csv", review_dir / "annotations.csv")
    print(f"Poisoned: {result['n_poisoned']}, Clean: {result['n_clean']}")
    print(f"Detection Rate: {result['detection_rate']:.2%}")
    print(f"False-Positive-Rate:  {result['false_positive_rate']:.2%}")


if __name__ == "__main__":
    main()

"""Implementiert Label-Poisoning basierend auf Modell-Unsicherheit für RQ2.4

Hier wird das Prinzip des Active Learnings zunutze gemacht und damit eine Art 
Reverse Active Learning umgesetzt
"""
import numpy as np


def build_poisoned_labels(telemetry, clean_labels, window_start, window_end, threshold):
    """Generiert vergiftete Labels basierend auf Fehlklassifikationen

    Ein Label wird genau dann geflippt, wenn zwei Bedingungen erfüllt sind:
    1. Die meist-vorhergesagte Klasse weicht vom echten, sauberen Label ab
    2. Die relative Häufigkeit dieser falschen Vorhersage innerhalb des Epochen-Fensters 
       liegt strikt ÜBER dem Schwellenwert

    Args:
        telemetry (pd.DataFrame): DataFrame mit den Vorhersagen des Baseline-Modells
        clean_labels (np.ndarray): Das 1D-Array der ursprünglichen, korrekten Klassen-IDs
        window_start (int): Erste Epoche des Auswertungsfensters
        window_end (int): Letzte Epoche des Auswertungsfensters
        threshold (float): Der relative Schwellenwert (0.0 bis 1.0)

    Returns:
        np.ndarray: Ein Array, das die modifizierten Labels enthält
    """
    in_window = (telemetry["epoch"] >= window_start) & (telemetry["epoch"] <= window_end)
    telemetry_window = telemetry[in_window]
    num_epochs = window_end - window_start + 1

    poisoned_labels = clean_labels.astype(np.int64).copy()

    for sample_index, predictions in telemetry_window.groupby("sample_index")["predicted_class"]:
        classes, class_counts = np.unique(predictions.to_numpy(), return_counts=True)
        most_frequent = int(class_counts.argmax())
        majority_class = int(classes[most_frequent])
        prediction_frequency = class_counts[most_frequent] / num_epochs

        is_confusion = majority_class != int(clean_labels[sample_index])
        if is_confusion and prediction_frequency > threshold:
            poisoned_labels[int(sample_index)] = majority_class

    return poisoned_labels

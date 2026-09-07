"""Gemeinsame Werkzeuge und Funktionen für die RQ2-Experimente

1. Das saubere Baseline-Modell (Rate 0.0) aus RQ1 laden
2. Bilder durch das Modell zu schicken, um deren Embeddings zu extrahieren
3. Die Klassen-Prototypen berechnen.
4. Ähnlichkeiten zwischen Klassen bewerten und Paare bilden
"""
from pathlib import Path

import numpy as np
import torch

from implementation.shared import paths
from implementation.shared.training.models import get_model, load_state_dict_without_compile_prefix

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def baseline_run_dir(rq, model_name, seed):
    """Ermittelt das Verzeichnis des Trainingslaufs
    
    Args:
        rq (str): Die Base-RQ.
        model_name (str): Architektur.
        seed (int): Seed Zufallswert
        
    Returns:
        Path: Pfad zum Ausgabeordner des sauberen Runs
    """
    return paths.experiment_dir(rq, model_name) / f"seed_{seed}" / "rate_0.0"


def find_baseline_checkpoint(rq, model_name, seed):
    """Sucht die Datei mit den besten Modell-Gewichten des sauberen Runs
    
    Raises:
        FileNotFoundError: Wenn der Run noch nicht ausgeführt wurde
    """
    run_dir = baseline_run_dir(rq, model_name, seed)
    candidates = sorted(run_dir.glob("best_model_weights_rate_0.00_epoch_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"Kein Baseline-Checkpoint unter {run_dir}")
    return candidates[-1]


def find_baseline_telemetry_csv(rq, model_name, seed):
    """Findet die Telemetrie-CSV des Baseline-Runs"""
    path = baseline_run_dir(rq, model_name, seed) / "telemetry" / "sample_telemetry.csv"
    if not path.exists():
        raise FileNotFoundError(f"Keine Baseline-Telemetrie: {path}")
    return path


def load_baseline_model(data_config, train_config, device):
    """Lädt das Baseline-Modell in die Auswertung

    Args:
        data_config (dict): Datenkonfiguration (liefert die Klassenanzahl)
        train_config (dict): Trainingskonfiguration (liefert Architektur und Baseline-RQ)
        device (torch.device): CPU oder GPU für das Modell

    Returns:
        torch.nn.Module: Das fertig geladene Modell, bereits im `.eval()` Modus
    """
    num_classes = data_config["poisoning"]["num_classes"]
    model_name = train_config["experiment"]["model"]
    baseline_rq = train_config["experiment"]["baseline_rq"]
    seed = train_config["experiment"]["seeds"][0]

    checkpoint_path = find_baseline_checkpoint(baseline_rq, model_name, seed)
    print(f"Baseline checkpoint: {checkpoint_path}")

    model = get_model(model_name, num_classes=num_classes).to(device)
    load_state_dict_without_compile_prefix(model, checkpoint_path, device)
    model.eval()
    return model


def compute_features(model, raw_images, indices, device, transform, batch_size):
    """Extrahiert die hochdimensionalen Features (Embeddings) für bestimmte Bilder.

    Args:
        model: Das PyTorch Modell
        raw_images: Komplettes Array der rohen Bilder (RAM)
        indices: Die Indizes der spezifischen Bilder, die verarbeitet werden sollen
        device: CPU oder GPU
        transform: Torchvision Transformations-Pipeline
        batch_size: Größe des Batches

    Returns:
        np.ndarray: Matrix der Form [Anzahl_Bilder, Feature_Dimension].
    """
    embeddings = []
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start:batch_start + batch_size]
        batch = torch.stack([transform(raw_images[index]) for index in batch_indices]).to(device)
        with torch.no_grad():
            features = model.forward_features(batch)
        embeddings.append(features.cpu())
    return torch.cat(embeddings).numpy()


def compute_class_prototypes(model, images, labels, num_classes, device, transform, batch_size):
    """Berechnet den Prototyp für jede Klasse im Feature-Raum

    Returns:
        np.ndarray: Array [num_classes, Feature_Dimension]
    """
    per_class_mean = []
    for class_id in range(num_classes):
        class_indices = np.flatnonzero(labels == class_id)
        embeddings = compute_features(model, images, class_indices, device, transform, batch_size)
        per_class_mean.append(embeddings.mean(axis=0))
    return np.stack(per_class_mean, axis=0)


def cosine_similarity_matrix(prototypes):
    """Berechnet die paarweise Kosinus-Ähnlichkeit aller Prototypen

    Args:
        prototypes (np.ndarray): Matrix der Prototypen

    Returns:
        np.ndarray: Quadratische Distanzmatrix [num_classes, num_classes] mit Werten von -1.0 bis 1.0
    """
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    normalized = prototypes / np.clip(norms, 1e-12, None)
    return normalized @ normalized.T


def compute_similarity_threshold(similarity_matrix, percentile):
    """Berechnet den Schwellenwert basierend auf einem Perzentil aller möglichen Paare"""
    num_classes = similarity_matrix.shape[0]
    pairwise_similarities = similarity_matrix[np.triu_indices(num_classes, k=1)]
    return float(np.percentile(pairwise_similarities, percentile))


def greedy_max_similarity_matching(similarity_matrix, min_similarity):
    """Ordnet Klassen basierend auf maximaler Ähnlichkeit exklusiven Paaren zu (Greedy Algorithmus).

    Args:
        similarity_matrix (np.ndarray): Quadratische Matrix mit Ähnlichkeitswerten
        min_similarity (float): Hard-Limit. Paare mit geringerem Score werden nicht gebildet

    Returns:
        tuple: (pairs, unmatched)
               - pairs: Liste von Tupeln (class_a, class_b, float_score)
               - unmatched: Liste der Klassen-IDs, die nicht mehr verpartnert werden konnten
    """
    num_classes = similarity_matrix.shape[0]
    similarity_matrix = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix, -np.inf)

    remaining = set(range(num_classes))
    pairs = []

    while len(remaining) > 1:
        best_score, best_pair = -np.inf, None
        for class_a in remaining:
            for class_b in remaining:
                if class_b <= class_a:
                    continue
                if similarity_matrix[class_a, class_b] > best_score:
                    best_score = similarity_matrix[class_a, class_b]
                    best_pair = (class_a, class_b)
        if best_score < min_similarity:
            break
        class_a, class_b = best_pair
        pairs.append((class_a, class_b, float(best_score)))
        remaining.discard(class_a)
        remaining.discard(class_b)

    return pairs, sorted(remaining)


def most_dissimilar_pair(similarity_matrix):
    """Sucht das unähnlichste Klassenpaar

    Returns:
        tuple: (class_a, class_b, score)
    """
    class_a, class_b = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)
    return int(class_a), int(class_b), float(similarity_matrix[class_a, class_b])

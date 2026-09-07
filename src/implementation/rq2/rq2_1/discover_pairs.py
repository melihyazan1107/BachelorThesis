"""Findet ähnliche Zielklassen-Paare für RQ2.1

Dieses Skript analysiert den Feature-Raum eines bereits trainierten, 
sauberen Baseline-Modells, um Klassen zu finden, die sich visuell bzw. semantisch 
sehr ähnlich sind. Diese Paare werden später für gezielte Poisoning-Angriffe genutzt

Der Algorithmus arbeitet in 5 Schritten:
1. Lädt das saubere Baseline-Modell
2. Berechnet pro Klasse den "Prototyp"
3. Berechnet die Kosinus-Ähnlichkeit zwischen allen möglichen Klassen-Prototypen
4. Definiert einen dynamischen Schwellenwert basierend auf einem Perzentil aller Ähnlichkeiten
5. Führt ein Greedy-Matching durch, um einzigartige Paare zu bilden

HINWEIS:
Das generierte JSON-Ergebnis wird von KEINEM anderen Skript automatisch eingelesen! 
Es dient rein als Bericht. Die ermittelten Paare müssen per Hand in die Datei 
`data_preparation/config.yaml` (unter target_pairs) kopiert werden.

Aufruf:
    cd src && python -m implementation.rq2.rq2_1.discover_pairs --config PATH --data-config PATH
"""
import implementation.shared.env

import argparse
import json

import h5py
import torch

from implementation.rq2 import common
from implementation.shared import config, paths
from implementation.shared.training.augmentations import get_val_transforms
from implementation.shared.training.dataset import read_class_names


def parse_args():
    """Parst die Kommandozeilenargumente
    
    Erwartet Pfade zu den Configs
    
    Returns:
        argparse.Namespace: Die geparsten Argumente
    """
    parser = argparse.ArgumentParser(description="Aehnliche Klassenpaare fuer RQ2.1 finden.")
    config.add_config_arguments(parser)
    return parser.parse_args()


def load_train_images_and_labels(train_h5_path):
    """Lädt den kompletten Trainingsdatensatz in den Arbeitsspeicher

    Args:
        train_h5_path (str): Pfad zur train.h5 Datei

    Returns:
        tuple: (images, labels) als vollständige NumPy-Arrays
    """
    with h5py.File(train_h5_path, "r") as h5_file:
        return h5_file["images"][:], h5_file["labels"][:]


def save_results(out_path, percentile_threshold, similarity_threshold, pairs, dissimilar, unmatched, class_names):
    """Speichert die gefundenen Paare und Statistiken als formatiertes JSON ab.
    
    WICHTIG: Die Einträge unter "pairs" müssen manuell vom Entwickler in die 
    `config.yaml` übertragen werden!

    Args:
        out_path (Path): Zielpfad für die JSON-Datei
        percentile_threshold (int/float): Genutztes Perzentil aus der Config
        similarity_threshold (float): Der daraus resultierende absolute Cosine-Score
        pairs (list): Liste von gematchten Tupeln (class_a, class_b, score)
        dissimilar (tuple): Das unähnlichste Paar (class_a, class_b, lowest_score)
        unmatched (list): Liste von Klassen-IDs, für die kein Partner gefunden wurde
        class_names (list): Mapping-Liste von Klassen-ID zu Klassen-String
    """
    dissimilar_a, dissimilar_b, lowest_similarity = dissimilar
    results = {
        "percentile_threshold": percentile_threshold,
        "similarity_threshold": similarity_threshold,
        "pairs": [],
        "most_dissimilar_pair": {
            "class_a": dissimilar_a,
            "class_b": dissimilar_b,
            "name_a": class_names[dissimilar_a],
            "name_b": class_names[dissimilar_b],
            "cosine_similarity": lowest_similarity},
        "unmatched": [{"class": class_id, "name": class_names[class_id]} for class_id in unmatched]}
    
    for class_a, class_b, score in pairs:
        results["pairs"].append({
            "class_a": class_a,
            "class_b": class_b,
            "name_a": class_names[class_a],
            "name_b": class_names[class_b],
            "cosine_similarity": score})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"Gespeichert unter {out_path}.")


def main():
    """Hauptablauf zur Identifikation ähnlicher Klassen.
    
    Der Ablauf in Detail:
    1. Berechnet Prototypen: Das Modell wird ohne den finalen Klassifikator genutzt,
       um Feature-Vektoren zu generieren. Alle Vektoren einer Klasse 
       werden gemittelt. Dieser Durchschnittsvektor ist der Prototyp der Klasse.
    2. Die Kosinus-Ähnlichkeit wird zwischen allen Prototypen 
       berechnet (Werte von -1 bis 1, wobei 1 identisch bedeutet).
    3. Es wird ein Schwellenwert berechnet. Wenn das Perzentil z.B. 95 ist, 
       müssen Paare zu den Top 5% der ähnlichsten Paare gehören, um berücksichtigt zu werden.
    4. "Greedy Matching": Das Paar mit dem absolut höchsten Ähnlichkeitswert wird 
       verbunden und aus dem Pool entfernt. Dann folgt das zweithöchste, usw.
    """
    args = parse_args()
    data_cfg, train_cfg = config.load_configs(args.config, args.data_config)
    num_classes = data_cfg["poisoning"]["num_classes"]
    percentile_threshold = data_cfg["rq2_1"]["percentile_threshold"]
    batch_size = data_cfg["rq2_1"]["inference_batch_size"]

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = common.load_baseline_model(data_cfg, train_cfg, device)

    train_h5_path = paths.hdf5_path("train")
    images, labels = load_train_images_and_labels(train_h5_path)
    class_names = read_class_names(train_h5_path)
    transform = get_val_transforms(train_cfg["augmentations"])

    prototypes = common.compute_class_prototypes(model, images, labels, num_classes, device, transform, batch_size)
    similarity_matrix = common.cosine_similarity_matrix(prototypes)
    similarity_threshold = common.compute_similarity_threshold(similarity_matrix, percentile_threshold)
    pairs, unmatched = common.greedy_max_similarity_matching(similarity_matrix, similarity_threshold)
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    dissimilar = common.most_dissimilar_pair(similarity_matrix)

    num_possible_pairs = num_classes * (num_classes - 1) // 2

    print(f"Mindest Ähnlichkeit ({percentile_threshold}. Perzentil aller {num_possible_pairs} moeglichen Klassenpaare): {similarity_threshold:.4f}")
    print(f"{len(pairs)} Paare oberhalb der Schwelle, {len(unmatched)} Klassen ohne ausreichend aehnlichen Partner:")
    
    for class_a, class_b, score in pairs:
        print(f"{class_names[class_a]:>25} ähnlich {class_names[class_b]:<25}  cos_sim={score:.4f}")
    if unmatched:
        print(f"Kein Paar gefunden: {[class_names[class_id] for class_id in unmatched]}")

    dissimilar_a, dissimilar_b, lowest_similarity = dissimilar
    print(f"Unaehnlichstes Paar: {class_names[dissimilar_a]} zu {class_names[dissimilar_b]} cos_sim={lowest_similarity:.4f}")

    save_results(common.ARTIFACTS_DIR / "discovered_target_pairs.json", percentile_threshold, similarity_threshold, pairs, dissimilar, unmatched, class_names)


if __name__ == "__main__":
    main()

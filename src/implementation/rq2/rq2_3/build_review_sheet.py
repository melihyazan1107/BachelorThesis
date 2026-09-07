"""Erstellt eine blinde Stichprobe für die menschliche Evaluation RQ2.3

Dieses Skript generiert ein Dataset für menschliche Gutachter, um zu testen, 
ob das eingefügte Label-Rauschen für das menschliche Auge erkennbar ist

Ablauf:
1. Zieht zufällig `n` vergiftete und `m` saubere Bilder aus dem Datensatz
2. Mischt diese Bilder komplett durch
3. Speichert die Bilder unter fortlaufenden Nummern (z.B. 0000.png, 0001.png).
4. Erstellt zwei CSV-Dateien:
   - review.csv: Für den Gutachter. Enthält nur die ID und das (potenziell falsche) Label
   - key.csv: Der "Lösungsschlüssel" zur späteren Auswertung der Gutachter-Leistung

Aufruf:
    cd src && python -m implementation.rq2.rq2_3.build_review_sheet --config PATH --data-config PATH
"""
import implementation.shared.env

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from implementation.shared import config, paths
from implementation.shared.training.dataset import read_class_names, read_labels


def draw_blind_sample(poisoned_mask, n_poisoned, n_clean, sample_seed):
    """Zieht eine zufällige Stichprobe aus sauberen und vergifteten Indizes

    Args:
        poisoned_mask (np.ndarray): Ein Boolean-Array (True = Bild wurde vergiftet, False = sauber)
        n_poisoned (int): Gewünschte Anzahl vergifteter Bilder für die Stichprobe
        n_clean (int): Gewünschte Anzahl sauberer Bilder
        sample_seed (int): Seed für Reproduzierbarkeit der Studie

    Returns:
        np.ndarray: Ein 1D-Array mit den zufällig ausgewählten Original-Indizes 
                    des HDF5-Datensatzes, in komplett randomisierter Reihenfolge.
    """
    rng = np.random.default_rng(sample_seed)
    num_poisoned = min(n_poisoned, int(poisoned_mask.sum()))
    num_clean = min(n_clean, int((~poisoned_mask).sum()))
    selected_poisoned = rng.choice(np.flatnonzero(poisoned_mask), size=num_poisoned, replace=False)
    selected_clean = rng.choice(np.flatnonzero(~poisoned_mask), size=num_clean, replace=False)

    sample_indices = np.concatenate([selected_poisoned, selected_clean])
    return sample_indices[rng.permutation(len(sample_indices))]


def read_images_in_sample_order(train_h5_path, sample_indices):
    """Lädt Bilder aus der HDF5-Datei und bewahrt die gemischte Reihenfolge

    Args:
        train_h5_path (Path): Pfad zur HDF5-Trainingsdatei
        sample_indices (np.ndarray): Die unsortierten/gemischten Ziel-Indizes

    Returns:
        np.ndarray: Die geladenen Bilder exakt in der Reihenfolge
    """
    sort_order = np.argsort(sample_indices)
    with h5py.File(train_h5_path, "r") as h5_file:
        sorted_images = h5_file["images"][sample_indices[sort_order]]
    sampled_images = np.empty_like(sorted_images)
    sampled_images[sort_order] = sorted_images
    return sampled_images


def build_review_sheet(rq, seed, rate, n_poisoned, n_clean, output_dir, sample_seed=0):
    """Macht die Erstellung der Blindstudie

    Args:
        rq (str): Die Forschungsfrage
        seed (int): Der Seed
        rate (float): Die Vergiftungsrate
        n_poisoned (int): Anzahl der vergifteten Bilder
        n_clean (int): Anzahl der sauberen Bilder
        output_dir (Path): Basisverzeichnis
        sample_seed (int): Seed für die Auswahl der Stichprobe

    Returns:
        Path: Pfad zum Ausgabeverzeichnis
    """
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    train_h5_path = paths.hdf5_path("train")
    poisoned_labels = torch.load(paths.poisoned_labels_path(rq, seed, rate), weights_only=True)
    poisoned_labels = poisoned_labels.numpy()
    clean_labels = read_labels(train_h5_path)
    class_names = read_class_names(train_h5_path)

    sample_indices = draw_blind_sample(poisoned_labels != clean_labels, n_poisoned, n_clean, sample_seed)
    sampled_images = read_images_in_sample_order(train_h5_path, sample_indices)

    review_rows, key_rows = [], []

    for review_id, original_index in enumerate(sample_indices):
        assigned_label = int(poisoned_labels[original_index])
        true_label = int(clean_labels[original_index])
        image_path = output_dir / "images" / f"{review_id:04d}.png"
        Image.fromarray(sampled_images[review_id]).save(image_path)
        review_rows.append(f"{review_id},{class_names[assigned_label]}")
        key_rows.append(f"{review_id},{int(assigned_label != true_label)},{true_label},{assigned_label}, {class_names[true_label]},{class_names[assigned_label]}")

    (output_dir / "review.csv").write_text("id,assigned_label\n" + "\n".join(review_rows) + "\n", encoding="utf-8")
    (output_dir / "key.csv").write_text("id,is_poisoned,true_class,assigned_class,true_name,assigned_name\n" + "\n".join(key_rows) + "\n", encoding="utf-8")

    return output_dir


def parse_args():
    """Für die Config-Pfade, welche geparsed werden"""
    parser = argparse.ArgumentParser(description="Review-Sheet fuer die Human-Review (RQ2.3)")
    config.add_config_arguments(parser)
    return parser.parse_args()


def main():
    """Lädt die Konfigurationen und triggert die Sheet-Generierung"""
    args = parse_args()
    data_cfg, train_cfg = config.load_configs(args.config, args.data_config)
    rq2_3_cfg = data_cfg["rq2_3"]
    rq = paths.validate_rq(train_cfg["experiment"]["rq"])
    seed = data_cfg["poisoning"]["seeds"][0]

    output_dir = build_review_sheet(
        rq=rq,
        seed=seed,
        rate=rq2_3_cfg["rate"],
        n_poisoned=rq2_3_cfg["n_poisoned"],
        n_clean=rq2_3_cfg["n_clean"],
        output_dir=paths.PROJECT_ROOT / "human_review",
        sample_seed=rq2_3_cfg["sample_seed"])
    
    print(f"Review-Sheet fuer {rq} (seed={seed}, rate={rq2_3_cfg['rate']}) erstellt: {output_dir}")

if __name__ == "__main__":
    main()

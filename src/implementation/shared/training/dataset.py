"""PyTorch Dataset-Implementierung"""
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 45
IMAGE_SIZE = 224


def read_labels(h5_path):
    """Lädt ausschließlich die sauberen Labels"""
    with h5py.File(h5_path, "r") as h5_file:
        return h5_file["labels"][:]


def read_class_names(h5_path):
    """Liest die menschenlesbaren Klassennamen aus der HDF5-Datei"""
    with h5py.File(h5_path, "r") as h5_file:
        return [name.decode("utf-8") for name in h5_file["class_names"][:]]


def _load_dataset_tensors(path, poisoned_labels_file):
    """Lädt Bilder und beide Label-Varianten vollständig in den RAM

    Args:
        path (Path): Pfad zur HDF5-Basisdatei
        poisoned_labels_file (Path): Pfad zur `.pt`-Datei mit den manipulierten Labels

    Returns:
        tuple: (clean_labels als int64, poisoned_labels als int64, images)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"HDF5 dataset file not found: {path}")

    with h5py.File(path, "r") as h5_file:
        clean_labels = torch.from_numpy(h5_file["labels"][:].astype("int64"))
        images = h5_file["images"][:]

    num_samples = len(clean_labels)
    if len(images) != num_samples:
        raise ValueError(f"Mismatch: {len(images)} images vs {num_samples} clean labels")
    if len(images) > 0 and IMAGE_SIZE not in images.shape:
        raise ValueError(f"Expected {IMAGE_SIZE}x{IMAGE_SIZE} image dimensions, got shape: {images.shape}")

    if poisoned_labels_file:
        if not Path(poisoned_labels_file).exists():
            raise FileNotFoundError(f"Label map file not found: {poisoned_labels_file}")
        poisoned_labels = torch.load(poisoned_labels_file, weights_only=True)
        if len(poisoned_labels) != num_samples:
            raise ValueError(f"Mismatch: {len(poisoned_labels)} poisoned labels vs {num_samples} clean labels")
    else:
        poisoned_labels = clean_labels

    if clean_labels.max() >= NUM_CLASSES or clean_labels.min() < 0:
        raise ValueError(f"Clean labels must be in the range [0, {NUM_CLASSES - 1}].")
    if poisoned_labels.max() >= NUM_CLASSES or poisoned_labels.min() < 0:
        raise ValueError(f"Poisoned labels must be in the range [0, {NUM_CLASSES - 1}].")

    return clean_labels, poisoned_labels, images


class HDF5Dataset(Dataset):
    """PyTorch Dataset-Klasse für Training"""
    def __init__(self, path, transform=None, poisoned_labels_file=None):
        """Initialisiert das Dataset und lädt alle Daten in den RAM

        Args:
            path (Path): Pfad zur train.h5, val.h5 oder test.h5
            transform (callable, optional): Pipeline zur Datenaugmentierung
            poisoned_labels_file (Path): Pfad zu den poisoned Labels
        """
        self.transform = transform
        self.clean_labels, self.poisoned_labels, self.images = _load_dataset_tensors(
            path, poisoned_labels_file)

    def __len__(self):
        """Gibt die Gesamtanzahl der Bilder im Datensatz zurück"""
        return len(self.clean_labels)

    def __getitem__(self, index):
        """Liefert ein einzelnes Sample anhand seines Indexes

        Args:
            index (int): Die ID des abgerufenen Bildes

        Returns:
            tuple: (Bild-Tensor, echtes_Label_int, falsches_Label_int, Index_int)
        """
        image = self.images[index]
        if self.transform:
            image = self.transform(image)

        true_label = int(self.clean_labels[index])
        noisy_label = int(self.poisoned_labels[index])
        return image, true_label, noisy_label, index

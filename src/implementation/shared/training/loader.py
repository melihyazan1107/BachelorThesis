"""PyTorch DataLoader, steuert das Daten-Routing

Es übernimmt zwei Hauptaufgaben:
1. Performance-Tuning: Konfiguration von Multi-Processing (Workers), Memory-Pinning 
   und Prefetching, um die GPU konstant mit Daten zu füttern
2. Experiment-Routing: Die Funktion `get_train_loader` entscheidet anhand der 
   Experiment-Parameter (Rate, Angriffs-Name), ob die echte Ground-Truth oder 
   die manipulierten Labels geladen werden
"""
from torch.utils.data import DataLoader

from implementation.shared import paths
from implementation.shared.training.augmentations import get_train_transforms, get_val_transforms
from implementation.shared.training.dataset import HDF5Dataset


def create_hdf5_dataloader(dataset_path, transform, cfg, shuffle, poisoned_labels_file=None,split_name="data"):
    """Funktion zur Erstellung eines DataLoaders

    Liest die Hardware-Konfiguration aus der `training.yaml` und setzt PyTorch-
    spezifische Optimierungen ein:
    - pin_memory: Reserviert fixen Arbeitsspeicher, was den Transfer von CPU zu GPU beschleunigt
    - num_workers: Startet parallele Sub-Prozesse, die die Daten-Augmentierung 
      auf der CPU berechnen, während die GPU trainiert
    - persistent_workers: Hält die Sub-Prozesse zwischen den Epochen am Leben, 
      um teuren Initialisierungs-Overhead zu vermeiden

    Args:
        dataset_path (str): Pfad zur HDF5-Datei
        transform (object): Die Pipeline
        cfg (dict): Das komplett geladene Trainings-Dictionary
        shuffle (bool): True für Training, False für Auswertung
        poisoned_labels_file (str, optional): Pfad zum poisoned Label-Tensor
        split_name (str): Für die Konsolenausgabe

    Returns:
        DataLoader: Der PyTorch DataLoader
    """
    hardware = cfg["hardware"]
    batch_size = cfg["experiment"]["batch_size"]
    num_workers = hardware["num_workers"]

    if num_workers > 0:
        persistent_workers = hardware["persistent_workers"]
        prefetch_factor = hardware["prefetch_factor"]
    else:
        persistent_workers = False
        prefetch_factor = None

    mapping_note = f" with mapping {poisoned_labels_file}" if poisoned_labels_file else ""
    print(f"Loading {split_name} from {dataset_path}{mapping_note}, (workers={num_workers})", flush=True)

    dataset = HDF5Dataset(dataset_path, transform, poisoned_labels_file=poisoned_labels_file)

    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=hardware["pin_memory"],persistent_workers=persistent_workers,prefetch_factor=prefetch_factor)


def get_base_loaders(cfg):
    """Erstellt die deterministischen DataLoader für Validierung und Test

    Args:
        cfg (dict): Trainings-Konfiguration

    Returns:
        tuple: (val_loader, test_loader)
    """
    val_transforms = get_val_transforms(cfg["augmentations"])
    val_loader = create_hdf5_dataloader(paths.hdf5_path("val"), val_transforms, cfg, shuffle=False, split_name="validation data")
    test_loader = create_hdf5_dataloader(paths.hdf5_path("test"), val_transforms, cfg, shuffle=False, split_name="test data")
    return val_loader, test_loader


def get_train_loader(cfg, seed, rate, attack=None):
    """Erstellt den Trainings-Loader und wendet das Label-Poisoning an

    Args:
        cfg (dict): Trainings-Konfiguration
        seed (int): Der aktuelle Experiment-Seed
        rate (float): Die aktuelle Vergiftungsrate
        attack (str, optional): Der Name des Angriffs

    Returns:
        DataLoader: Der DataLoader für die Trainings-Schleife
    """
    if rate == 0.0 or attack == "clean":
        poisoned_labels_file = None
    else:
        label_source = attack or cfg["experiment"]["rq"]
        poisoned_labels_file = paths.poisoned_labels_path(label_source, seed, rate)

    return create_hdf5_dataloader(paths.hdf5_path("train"),get_train_transforms(cfg["augmentations"]),cfg,shuffle=True,poisoned_labels_file=poisoned_labels_file,split_name="train data")

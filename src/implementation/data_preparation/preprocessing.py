"""Konvertiert den NWPU-RESISC45 Datensatz in HDF5-Container (Train/Val/Test).

Dieses Skript liest die rohen Bilddateien ein und speichert sie als unnormalisierte, 
nicht augmentierte HWC/uint8-Tensoren. Die Entscheidung über Normalisierung und 
Augmentierung wird somit bewusst in die Trainings-Pipeline verschoben.

Der Train/Val-Split erfolgt stratifiziert und ist durch einen festen Seed 
vollständig reproduzierbar.
"""
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class RESISC45Preprocessor:
    """Kapselt die Logik zur Vorverarbeitung und Konvertierung des Datensatzes.
    
    Diese Klasse liest die Ordnerstruktur des NWPU-RESISC45 Datensatzes, 
    führt einen reproduzierbaren Train-/Validation-Split durch und speichert 
    die Bilder speichereffizient in HDF5-Dateien ab.
    """


    def __init__(self, raw_dir: Path, output_dir: Path, val_split_ratio, seed, target_size):
        """Initialisiert den Preprocessor.

        Args:
            raw_dir (Path): Pfad zum Wurzelverzeichnis der Rohdaten
            output_dir (Path): Zielverzeichnis für die generierten .h5 Dateien
            val_split_ratio (float): Anteil der Trainingsdaten, der für Validierung abgezweigt wird
            seed (int): Random Seed für den stratifizierten Split
            target_size (list): Zielauflösung der Bilder als [Höhe, Breite].
        """
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.target_size = tuple(target_size)


    def run(self):
        """Führt die komplette Verarbeitungs-Pipeline aus
        
        Ablauf:
        1. Erstellt das Zielverzeichnis
        2. Ermittelt dynamisch alle Klassennamen anhand der Ordnerstruktur
        3. Verarbeitet die Testdaten und schreibt `test.h5`
        4. Führt einen stratifizierten Split der Trainingsdaten in Train/Val durch
        5. Verarbeitet und schreibt `train.h5` sowie `val.h5`
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_dir = self.raw_dir / "train" / "train"
        test_dir = self.raw_dir / "test" / "test"
        class_names = sorted([entry.name for entry in train_dir.iterdir() if entry.is_dir()])

        test_paths, test_labels = self._get_paths_and_labels(test_dir, class_names)
        self._write_hdf5(self.output_dir / "test.h5", test_paths, test_labels, class_names,"Test Set")

        all_train_paths, all_train_labels = self._get_paths_and_labels(train_dir, class_names)
        train_paths, val_paths, train_labels, val_labels = train_test_split(all_train_paths,all_train_labels,test_size=self.val_split_ratio,random_state=self.seed,stratify=all_train_labels)

        self._write_hdf5(self.output_dir / "train.h5", train_paths, train_labels, class_names,"Train Set")
        self._write_hdf5(self.output_dir / "val.h5", val_paths, val_labels, class_names,"Validation Set")


    def _get_paths_and_labels(self, base_dir: Path, class_names):
        """Sammelt alle Bildpfade und generiert die zugehörigen Labels.

        Iteriert durch die Klassenordner und weist jedem Bild anhand des
        Ordnernamens eine eindeutige Klassen-ID (0 bis N-1) zu.

        Args:
            base_dir (Path): Basisverzeichnis
            class_names (list): Sortierte Liste der Klassennamen

        Returns:
            tuple: (Liste mit Bildpfaden, NumPy Array mit Integer-Labels)
        """
        paths = []
        labels = []
        for class_id, class_name in enumerate(class_names):
            for image_path in sorted((base_dir / class_name).glob("*.jpg")):
                paths.append(image_path)
                labels.append(class_id)
        return paths, np.array(labels, dtype=np.int32)

    def _write_hdf5(self, out_path: Path, image_paths, labels, class_names, split_name):
        """Schreibt Bilddaten in ein HDF5-Archiv.

        Erstellt HDF5-Dataset zuerst leer und dann Bild für Bild laden, direkt auf die 
        HDF5-File schreiben
        
        Es nutzt LZF-Kompression sowie Chunking, für Training schnelle Zugriffe 
        auf einzelne Batches.

        Args:
            out_path (Path): Dateipfad für die Ausgabe-HDF5
            image_paths (list): Liste der Dateipfade der Bilder
            labels (np.ndarray): Array der zugehörigen Klassen-IDs
            class_names (list): Liste der als Strings lesbaren Klassennamen
            split_name (str): Anzeigename für die tqdm-Fortschrittsanzeige
        """
        num_samples = len(image_paths)
        height, width = self.target_size

        with h5py.File(out_path, "w") as h5_file:
            images = h5_file.create_dataset("images",shape=(num_samples, height, width, 3),dtype="uint8",chunks=(32, height, width, 3),compression="lzf")
            h5_file.create_dataset("labels", data=labels)
            ascii_names = [name.encode("ascii", "ignore") for name in class_names]
            h5_file.create_dataset("class_names", data=ascii_names)

            for index, path in enumerate(tqdm(image_paths, desc=f"Writing {split_name} to HDF5")):
                image = Image.open(path).convert("RGB")
                image = image.resize(self.target_size, Image.LANCZOS)
                images[index] = np.array(image, dtype=np.uint8)

        print(f"Finished {out_path.name}: {num_samples} images.")

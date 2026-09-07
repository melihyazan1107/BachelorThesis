"""Preprocessing und Label-Poisoning

Softwarearchitektur (Dispatcher-Muster):
Anstatt für jede Forschungsfrage ein eigenes Datenskript aufzurufen, startet 
immer dieses eine Skript. Es liest `experiment.rq` aus der 
Trainings-Konfiguration, validiert die Eingabe und ruft dynamisch den exakt 
passenden Poisoner auf

Aufruf:
    cd src && python -m implementation.run_pipeline --config PATH --data-config PATH
"""
import implementation.shared.env

import argparse

from implementation.data_preparation.preprocessing import RESISC45Preprocessor
from implementation.rq1.poisoner import SymmetricPoisoner
from implementation.rq2.rq2_1.poisoner import TargetedPoisoner
from implementation.rq2.rq2_2 import run as rq2_2_run
from implementation.rq2.rq2_4 import run as rq2_4_run
from implementation.shared import config, paths


def banner(title):
    print("=" * 100)
    print(title)
    print("=" * 100)


def parse_args():
    """Parst die Kommandozeilenargumente über das zentrale Config-Modul"""
    parser = argparse.ArgumentParser(description="Preprocessing + Poisoning fuer die in der YAML gewaehlte RQ.")
    config.add_config_arguments(parser)
    return parser.parse_args()


def run_preprocessing(data_cfg, processed_dir):
    """Konvertiert die rohen JPEG-Bilder einmalig in das HDF5-Format

    Args:
        data_cfg (dict): Die Daten-Konfiguration
        processed_dir (Path): Zielordner
    """
    preprocessing_cfg = data_cfg["preprocessing"]
    splits = [processed_dir / f"{split}.h5" for split in ("train", "val", "test")]

    if preprocessing_cfg.get("skip_if_exists", True) and all(split.exists() for split in splits):
        print("Alle HDF5-Splits vorhanden -- Preprocessing uebersprungen (preprocessing.skip_if_exists: false erzwingt den Lauf).")
        return

    RESISC45Preprocessor(raw_dir=paths.PROJECT_ROOT / data_cfg["paths"]["raw_data_dir"], output_dir=processed_dir, val_split_ratio=preprocessing_cfg["val_split_ratio"], seed=preprocessing_cfg["global_seed"], target_size=preprocessing_cfg["target_size"]).run()


def run_poisoning(rq, data_cfg, train_cfg, processed_dir):
    """Dispatcher für die Poisoning-Mechanismen

    Nimmt die gewünschte Forschungsfrage (RQ) und delegiert die 
    Ausführung an die entsprechende Poisoner-Klasse.

    Args:
        rq (str): Die validierte Forschungsfrage
        data_cfg (dict): Definition von Seeds, Raten und Zielpaaren
        train_cfg (dict): Wird nur für RQ2.4 weitergereicht
        processed_dir (Path): Verzeichnis der HDF5-Dateien
    """
    poisoning_cfg = data_cfg["poisoning"]
    shared_arguments = dict(train_h5_path=processed_dir / "train.h5", output_dir=processed_dir / "poisoned", seeds=poisoning_cfg["seeds"], rates=poisoning_cfg["rates"], num_classes=poisoning_cfg["num_classes"])

    if rq == "RQ1":
        SymmetricPoisoner(**shared_arguments).run()
    elif rq == "RQ2":
        TargetedPoisoner(**shared_arguments,target_pairs=poisoning_cfg["target_pairs"],erasure_limit=poisoning_cfg["class_erasure_limit"]).run()
    elif rq.startswith("RQ2.2-"):
        rq2_2_run.generate(data_cfg, rq)
    elif rq == "RQ2.4":
        rq2_4_run.generate(data_cfg, train_cfg)
    else:
        raise ValueError(f"Kein Poisoner fuer rq '{rq}' hinterlegt (RQ3 erzeugt keine Labels).")


def main():
    """Steuert den Ablauf des Skripts (Config-Laden zu Preprocess zu Poison)"""
    args = parse_args()
    data_cfg, train_cfg = config.load_configs(args.config, args.data_config)

    rq = paths.validate_rq(train_cfg["experiment"]["rq"])
    processed_dir = paths.PROJECT_ROOT / data_cfg["paths"]["processed_dir"]

    banner("Phase 1: Preprocessing & HDF5 conversion")
    run_preprocessing(data_cfg, processed_dir)

    print()
    banner(f"Phase 2: Label poisoning fuer {rq}")
    run_poisoning(rq, data_cfg, train_cfg, processed_dir)

    print()
    banner("Pipeline complete.")


if __name__ == "__main__":
    main()

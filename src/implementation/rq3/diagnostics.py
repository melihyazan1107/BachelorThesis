"""Wertet die Telemetriedaten aus, um die Kernhypothesen der Verteidigungen zu prüfen (RQ3).

- AUROC & Overlap: Wie gut lassen sich saubere/vergiftete Bilder anhand des Losses trennen?
- Filtergüte: Wie gut arbeitet der Filter der Verteidigung (Precision/Recall/F1)?
- Memorization: Ab wann beginnt das Modell, die falschen Labels auswendig zu lernen?

Aufruf:
    cd src && python -m implementation.rq3.diagnostics [--root DIR] [--out DIR] [--legacy RQ1]
"""
import implementation.shared.env

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from implementation.rq3.common import GROUP_KEYS, iter_rq3_runs, seed_and_rate, write_csv
from implementation.shared import paths
from implementation.shared.training.metrics import separability

SEPARABILITY_COLUMNS = ["defense", "group", "model", "attack", "seed", "rate", "epoch", "loss_auroc", "loss_overlap", "clean_loss", "poison_loss"]
TELEMETRY_CHUNK_ROWS = 500_000

def collect_epoch_metrics(root):
    """Sammelt die bereits pro Epoche vorberechneten Metriken aller RQ3-Läufe

    Args:
        root (Path): Das Wurzelverzeichnis der RQ3-Experimente.

    Returns:
        pd.DataFrame: Ein zusammengefügter DataFrame mit den Epochen-Metriken aller Läufe.
    """
    frames = []
    for defense, group, model, attack, seed, rate, run_dir in iter_rq3_runs(root):
        metrics_df = pd.read_csv(run_dir / "telemetry" / "epoch_metrics.csv")
        metrics_df.insert(0, "defense", defense)
        metrics_df.insert(1, "group", group)
        metrics_df.insert(2, "model", model)
        metrics_df.insert(3, "attack", attack)
        metrics_df.insert(4, "seed", seed)
        metrics_df.insert(5, "rate", rate)
        frames.append(metrics_df)

    if not frames:
        return pd.DataFrame(columns=SEPARABILITY_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def separability_from_sample_telemetry(telemetry_csv):
    """Rekonstruiert die Separabilitäts-Metrik nachträglich aus alten Rohdaten

    Args:
        telemetry_csv (Path): Pfad zur massiven sample_telemetry.csv Datei

    Returns:
        pd.DataFrame: Epochen-Metriken (AUROC, Overlap, Clean/Poison Loss)
    """
    epoch_buckets = {}
    columns = ["epoch", "true_label", "noisy_label", "loss"]
    chunks = pd.read_csv(telemetry_csv, usecols=columns, chunksize=TELEMETRY_CHUNK_ROWS)
    for chunk in chunks:
        is_poisoned = (chunk["true_label"] != chunk["noisy_label"]).to_numpy()
        losses = chunk["loss"].to_numpy(dtype=np.float64)
        for epoch, chunk_indices in chunk.groupby("epoch").indices.items():
            bucket = epoch_buckets.setdefault(int(epoch), [])
            bucket.append((losses[chunk_indices], is_poisoned[chunk_indices]))

    rows = []

    for epoch in sorted(epoch_buckets):
        losses = np.concatenate([part_losses for part_losses, _ in epoch_buckets[epoch]])
        is_poisoned = np.concatenate([part_mask for _, part_mask in epoch_buckets[epoch]])
        auroc, overlap = separability(losses, is_poisoned)

        clean_loss = losses[~is_poisoned].mean() if (~is_poisoned).any() else np.nan
        poison_loss = losses[is_poisoned].mean() if is_poisoned.any() else np.nan
        rows.append({
            "epoch": epoch,
            "loss_auroc": auroc,
            "loss_overlap": overlap,
            "clean_loss": clean_loss,
            "poison_loss": poison_loss})
        
    return pd.DataFrame(rows)


def collect_legacy_separability(research_questions):
    """Steuert die nachträgliche Rekonstruktion für alte Forschungsfragen"""
    frames = []

    for rq in research_questions:
        base_dir = paths.EXPERIMENTS_DIR / paths.validate_rq(rq)
        telemetry_csvs = sorted(base_dir.glob("*/seed_*/rate_*/telemetry/sample_telemetry.csv"))

        for telemetry_csv in telemetry_csvs:
            run_dir = telemetry_csv.parents[1]
            params = seed_and_rate(run_dir)
            if params is None:
                continue
            seed, rate = params
            model_name = run_dir.parts[-3]
            print(f"  - {rq} | {model_name} | Seed {seed} | Rate {rate} ...", flush=True)

            separability_df = separability_from_sample_telemetry(telemetry_csv)
            separability_df.insert(0, "defense", "none")
            separability_df.insert(1, "group", "none")
            separability_df.insert(2, "model", model_name)
            separability_df.insert(3, "attack", rq)
            separability_df.insert(4, "seed", seed)
            separability_df.insert(5, "rate", rate)

            frames.append(separability_df)

    if not frames:
        return pd.DataFrame(columns=SEPARABILITY_COLUMNS)
    
    return pd.concat(frames, ignore_index=True)


def separability_table(metrics_df):
    """Aggregiert die Trennbarkeits-Metriken über alle Random-Seeds.
    
    AUROC:
    - 1.0: Perfekte Trennung (Alle sauberen Bilder haben einen niedrigeren Loss als vergiftete)
    - 0.5: Raten (Losses von sauberen und vergifteten Bildern sind identisch verteilt)
    """
    if metrics_df.empty:
        return metrics_df

    numeric_df = metrics_df.copy()

    for column in ("loss_auroc", "loss_overlap"):
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    return numeric_df.dropna(subset=["loss_auroc"]).groupby(GROUP_KEYS, as_index=False).agg(
        auroc_mean=("loss_auroc", "mean"),
        auroc_std=("loss_auroc", "std"),
        auroc_last=("loss_auroc", "last"),
        overlap_mean=("loss_overlap", "mean"),
        overlap_std=("loss_overlap", "std"),
        epochs=("epoch", "nunique"),
        seeds=("seed", "nunique"))


def filter_quality_table(metrics_df):
    """Bewertet, wie präzise eine Verteidigung vergiftete Daten entfernt hat"""
    empty = pd.DataFrame(columns=["defense", "model", "attack", "rate"])
    if metrics_df.empty or "filter_precision" not in metrics_df.columns:
        return empty

    numeric_df = metrics_df.copy()
    for column in ("filter_precision", "filter_recall", "filter_f1", "kept_fraction"):
        numeric_df[column] = pd.to_numeric(numeric_df[column], errors="coerce")

    valid_df = numeric_df.dropna(subset=["filter_precision"])
    if valid_df.empty:
        return empty

    table = valid_df.groupby(GROUP_KEYS, as_index=False).agg(
        precision_mean=("filter_precision", "mean"),
        recall_mean=("filter_recall", "mean"),
        f1_mean=("filter_f1", "mean"),
        kept_fraction_mean=("kept_fraction", "mean"),
        epochs=("epoch", "nunique"))

    table["precision_lift"] = table["precision_mean"] - table["rate"]
    return table


def memorization_table(metrics_df):
    """Verfolgt die Memorization-Rate über die Epochen hinweg"""
    if metrics_df.empty or "memorization_rate" not in metrics_df.columns:
        return pd.DataFrame(columns=["defense", "model", "attack", "rate", "epoch"])

    numeric_df = metrics_df.copy()
    numeric_df["memorization_rate"] = pd.to_numeric(
        numeric_df["memorization_rate"], errors="coerce")

    valid_df = numeric_df.dropna(subset=["memorization_rate"])
    return valid_df.groupby(GROUP_KEYS + ["epoch"], as_index=False).agg(
        memorization_mean=("memorization_rate", "mean"),
        memorization_std=("memorization_rate", "std"),
        seeds=("seed", "nunique"))


def parse_args():
    """Parst die CLI-Argumente"""
    parser = argparse.ArgumentParser(description="Separabilitaets-Diagnostik fuer RQ3.")
    parser.add_argument("--root", type=Path, default=None,help="Wurzel der RQ3-Ergebnisse (Default: experiments/RQ3).")
    parser.add_argument("--out", type=Path, default=None,help="Ausgabeverzeichnis (Default: <root>/diagnostics).")
    parser.add_argument("--legacy", nargs="*", default=None, metavar="RQ",help="Zusaetzlich AUROC/Overlap aus bestehenden RQ1/RQ2-Laeufen rekonstruieren (z. B. --legacy RQ1 RQ2).")
    return parser.parse_args()


def main():
    """Sammelt Metriken und generiert die Statistik-Tabellen"""
    args = parse_args()
    root = args.root or paths.RQ3_DIR
    output_dir = args.out or (root / "diagnostics")

    print("TELEMETRIE (RQ3)")
    metrics_df = collect_epoch_metrics(root)
    if metrics_df.empty:
        print(f"Keine RQ3-Laeufe unter {root} gefunden.", flush=True)
    else:
        write_csv(metrics_df, output_dir / "epoch_metrics_all.csv", "Rohdaten")
        write_csv(separability_table(metrics_df), output_dir / "separability.csv", "Separabilitaet")
        write_csv(filter_quality_table(metrics_df), output_dir / "filter_quality.csv", "Filterguete")
        write_csv(memorization_table(metrics_df), output_dir / "memorization.csv", "Memorization")

    if args.legacy is not None:
        research_questions = args.legacy or ["RQ1", "RQ2"]
        print(f"REKONSTRUKTION LAEUFE: {', '.join(research_questions)}")
        legacy_df = collect_legacy_separability(research_questions)
        if legacy_df.empty:
            print("Keine 'sample_telemetry.csv' gefunden.", flush=True)
        else:
            write_csv(legacy_df, output_dir / "separability_legacy_raw.csv", "Legacy-Rohdaten")
            write_csv(separability_table(legacy_df), output_dir / "separability_legacy.csv", "Legacy-Separabilitaet")


if __name__ == "__main__":
    main()

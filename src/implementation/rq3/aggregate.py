"""Evaluiert die Ergebnisse der Verteidigungen (RQ3).

1. Wie viel Prozentpunkte Genauigkeit rettet eine Verteidigung 
   im Vergleich zum ungeschützten Baseline-Modell
2. Clean Cost: Wie viel Leistung kostet die Verteidigung
3. Wilcoxon-Test: Sind die Verbesserungen der Verteidigungen signifikant

Aufruf:
    cd src && python -m implementation.rq3.aggregate [--root DIR] [--out DIR]
"""
import implementation.shared.env

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from implementation.rq3.common import GROUP_KEYS, attack_dir_metadata, defense_group, write_csv
from implementation.shared import paths

BASELINE_DEFENSE = "none"
MIN_PAIRS_FOR_WILCOXON = 6
MASTER_COLUMNS = ["defense", "group", "model", "attack", "seed", "rate","accuracy", "precision", "recall", "f1","baseline_accuracy", "robustness_gap", "clean_cost","seconds_per_epoch", "peak_vram_mb", "epochs_run"]


def run_costs(run_dir):
    """Extrahiert Laufzeit- und Hardwarekosten eines spezifischen Trainings

    Args:
        run_dir (Path): Pfad zum Verzeichnis

    Returns:
        dict: Enthält 'seconds_per_epoch' (float), 'peak_vram_mb' (float) und 'epochs_run' (int).
    """
    epoch_csv = run_dir / "telemetry" / "epoch_metrics.csv"
    if not epoch_csv.exists():
        return {"seconds_per_epoch": np.nan, "peak_vram_mb": np.nan, "epochs_run": np.nan}

    epoch_df = pd.read_csv(epoch_csv)
    epoch_seconds = pd.to_numeric(epoch_df.get("epoch_seconds"), errors="coerce")
    peak_vram = pd.to_numeric(epoch_df.get("peak_vram_mb"), errors="coerce")

    if epoch_seconds.notna().sum() > 1:
        mean_seconds = epoch_seconds.iloc[1:].mean()
    elif epoch_seconds.notna().any():
        mean_seconds = epoch_seconds.mean()
    else:
        mean_seconds = np.nan

    return {"seconds_per_epoch": mean_seconds,"peak_vram_mb": peak_vram.max(),"epochs_run": len(epoch_df),}


def collect_summaries(root):
    """Sucht und aggregiert alle `summary_metrics.csv` Dateien

    Args:
        root (Path): Das Verzeichnis der RQ3-Ergebnisse.

    Returns:
        pd.DataFrame: Ein unbereinigtes DataFrame mit allen Rohdaten
    """
    frames = []
    for summary_csv in sorted(root.glob("*/*/*/summary_metrics.csv")):
        defense, group, model, attack = attack_dir_metadata(summary_csv.parent)
        frame = pd.read_csv(summary_csv)
        frame["defense"] = defense
        frame["group"] = group
        frame["model"] = model
        frame["attack"] = attack

        costs = []
        for row in frame.itertuples():
            run_dir = summary_csv.parent / f"seed_{int(row.seed)}" / f"rate_{row.rate}"
            costs.append(run_costs(run_dir))
        frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(costs)], axis=1)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def duplicate_clean_data(data):
    """Dupliziert die sauberen Trainingsläufe für alle Angriffsarten.

    Args:
        data (pd.DataFrame): Der aggregierte Datensatz

    Returns:
        pd.DataFrame: Der erweiterte Datensatz
    """
    clean = data[data["attack"] == "clean"]
    attacked = data[data["attack"] != "clean"]
    if clean.empty or attacked.empty:
        return data

    copies = []
    for attack in sorted(attacked["attack"].unique()):
        copy = clean.copy()
        copy["attack"] = attack
        copies.append(copy)
    return pd.concat([attacked, *copies], ignore_index=True)


def add_comparisons(data):
    """Berechnet die relativen Performance-Metriken (Robustness Gap & Clean Cost).

    Beantwortet die Fragen: 
    - Wie viel besser ist die Verteidigung als Nichtstun?
    - Wie sehr schadet die Verteidigung der sauberen Performance?

    Args:
        data (pd.DataFrame): Der DataFrame mit den Clean-Daten

    Returns:
        pd.DataFrame: Der DataFrame erweitert um `baseline_accuracy`, 
                      `robustness_gap` und `clean_cost`.
    """
    baseline = (data[data["defense"] == BASELINE_DEFENSE].set_index(["model", "attack", "seed", "rate"])["accuracy"])
    keys = list(zip(data["model"], data["attack"], data["seed"], data["rate"]))
    baseline_accuracy = np.array([baseline.get(key, np.nan) for key in keys], dtype=float)

    data = data.copy()
    data["baseline_accuracy"] = baseline_accuracy
    data["robustness_gap"] = data["accuracy"] - data["baseline_accuracy"]

    clean_gap = (data[data["rate"] == 0.0].drop_duplicates(subset=["defense", "model", "seed"]).set_index(["defense", "model", "seed"])["robustness_gap"])
    clean_keys = list(zip(data["defense"], data["model"], data["seed"]))
    data["clean_cost"] = np.array([clean_gap.get(key, np.nan) for key in clean_keys], dtype=float)

    return data


def wilcoxon_table(master_df):
    """Prüft per Wilcoxon-Test, ob eine Verteidigung signifikant hilft

    Args:
        master_df (pd.DataFrame): Der vollständig berechnete DataFrame

    Returns:
        pd.DataFrame: Eine Tabelle, die für jede Defense pro Modell den p-Wert 
                      und die Anzahl der Success/ Failure gegen die Baseline zeigt.
    """
    table_rows = []
    attacked_df = master_df[(master_df["defense"] != BASELINE_DEFENSE) & (master_df["rate"] > 0.0)]

    for (defense, model), group_df in attacked_df.groupby(["defense", "model"]):
        valid_pairs = group_df.dropna(subset=["accuracy", "baseline_accuracy"])
        differences = (valid_pairs["accuracy"] - valid_pairs["baseline_accuracy"]).to_numpy()

        if len(differences) < MIN_PAIRS_FOR_WILCOXON or np.allclose(differences, 0.0):
            statistic, p_value = np.nan, np.nan
        else:
            statistic, p_value = wilcoxon(differences, zero_method="wilcox")

        if len(differences):
            mean_gap, median_gap = np.mean(differences), np.median(differences)
        else:
            mean_gap, median_gap = np.nan, np.nan

        table_rows.append({
            "defense": defense,
            "group": defense_group(defense),
            "model": model,
            "n_pairs": len(differences),
            "mean_gap": mean_gap,
            "median_gap": median_gap,
            "wins": np.sum(differences > 0),
            "losses": np.sum(differences < 0),
            "wilcoxon_statistic": statistic,
            "p_value": p_value})

    return pd.DataFrame(table_rows).sort_values(["model", "group", "mean_gap"], ascending=[True, True, False])


def cell_table(data):
    """Aggregiert die Daten auf Zellen-Ebene"""
    return data.groupby(GROUP_KEYS, as_index=False).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        robustness_gap_mean=("robustness_gap", "mean"),
        robustness_gap_std=("robustness_gap", "std"),
        clean_cost_mean=("clean_cost", "mean"),
        seconds_per_epoch=("seconds_per_epoch", "mean"),
        peak_vram_mb=("peak_vram_mb", "max"),
        seeds=("seed", "nunique"))


def parse_args():
    """Parst Eingabepfade aus der Kommandozeile"""
    parser = argparse.ArgumentParser(description="Aggregiert die RQ3-Laeufe zu rq3_master.csv.")
    parser.add_argument("--root", type=Path, default=None,
                        help="Wurzel der RQ3-Ergebnisse (Default: experiments/RQ3).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Ausgabeverzeichnis (Default: --root).")
    return parser.parse_args()


def main():
    """Hauptablauf zur Generierung der Auswertungs-CSVs"""
    args = parse_args()
    root = args.root or paths.RQ3_DIR
    output_dir = args.out or root

    data = collect_summaries(root)
    if data.empty:
        print(f"Keine summary_metrics.csv unter {root} gefunden.", flush=True)
        return

    data = add_comparisons(duplicate_clean_data(data))
    data = data[[column for column in MASTER_COLUMNS if column in data.columns]]

    write_csv(data, output_dir / "rq3_master.csv", "rq3_master")
    write_csv(cell_table(data), output_dir / "rq3_cells.csv", "rq3_cells")
    write_csv(wilcoxon_table(data), output_dir / "rq3_wilcoxon.csv", "rq3_wilcoxon")

    print("Robustness Gap (Mittel ueber Seeds x Raten x Angriffe, nur rate > 0):", flush=True)
    print(wilcoxon_table(data).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

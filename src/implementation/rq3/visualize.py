"""Generiert die finalen Plots für Forschungsfrage 3

Dieses Skript liest die CSV-Dateien der drei vorherigen Auswertungsskripte 
(aggregate, attack_metrics, diagnostics) und erzeugt daraus 5 Grafiken

Visuelles Design-Konzept:
Um die Plots verständlich zu halten, gilt über alle Abbildungen hinweg:
- Farbe = Methodengruppe (z.B. Blau = ungeschützt, Orange = präventiv, Grün = reaktiv).
- Linienstil & Marker = Die spezifische Verteidigung (z.B. durchgezogene Linie mit Kreis).

Aufruf:
    cd src && python -m implementation.rq3.visualize [--experiments DIR] [--out DIR] [--rate R]
"""
import implementation.shared.env

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from implementation.shared import paths
from implementation.shared.defenses import DEFENSE_CHOICES, DEFENSE_GROUPS

GROUP_COLORS = {"none": "#2a78d6", "preventive": "#eb6834", "reactive": "#1baf7a"}
ATTACK_COLORS = {"RQ1": "#2a78d6", "RQ2": "#eb6834", "clean": "#898781"}

DEFENSE_STYLE = {
    "none": ("-", "o"),
    "label_smoothing": ("-", "s"),
    "mixup": ("--", "^"),
    "gce": ("-.", "D"),
    "elr": (":", "v"),
    "small_loss": ("-", "P"),
    "gmm_filter": ("--", "X")}

INK_COLOR = "#0b0b0b"
MUTED_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"


def apply_style():
    """Wendet saubere Design-Regeln auf Matplotlib an"""
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.edgecolor": AXIS_COLOR, "axes.labelcolor": INK_COLOR, "axes.titlesize": 10,
        "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
        "xtick.color": MUTED_COLOR, "ytick.color": MUTED_COLOR,
        "text.color": INK_COLOR, "font.size": 9,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5})


def defense_color(defense):
    """Gibt die Farbe der Methodengruppe einer Verteidigung zurück"""
    return GROUP_COLORS.get(DEFENSE_GROUPS.get(defense, "none"), MUTED_COLOR)


def defense_style(defense):
    """Gibt das Tupel für eine Verteidigung zurück"""
    return DEFENSE_STYLE.get(defense, ("-", "o"))


def ordered_defenses(values):
    """Filtert eine Liste von Verteidigungen"""
    present = set(values)
    return [defense for defense in DEFENSE_CHOICES if defense in present]


def read_csv_or_none(path: Path, label):
    """Lädt eine CSV-Datei"""
    if not path.exists():
        print(f"SKIP {label}: {path} fehlt", flush=True)
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"SKIP {label}: {path} ist leer", flush=True)
        return None
    return df


def subplot_grid(df, row_key, col_key, size=(3.6, 2.9), sharey=True):
    """Erstellt ein Grid von Subplots

    Args:
        df (pd.DataFrame): Daten
        row_key (str): Spaltenname
        col_key (str): Spaltenname
        size (tuple): Größe (Breite, Höhe)
        sharey (bool): Skalierung

    Returns:
        tuple: (fig, axes, rows, cols)
    """
    rows = sorted(df[row_key].unique())
    cols = sorted(df[col_key].unique())
    fig, axes = plt.subplots(
        len(rows), len(cols),
        squeeze=False, sharex=True, sharey=sharey,
        figsize=(size[0] * len(cols), size[1] * len(rows)))
    return fig, axes, rows, cols


def label_scatter_points(ax, points, minimum_gap=0.075):
    """Platziert Text-Labels in Scatter-Plots so, dass sie sich nicht überlappen

    Args:
        ax (plt.Axes): Das aktuelle Matplot
        points (list): Liste von Tupeln (x, y, label_text).
        minimum_gap (float): Der relative Mindestabstand (0.0 bis 1.0) zwischen zwei Texten
    """
    ax.margins(x=0.22, y=0.14)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    span_x, span_y = (x1 - x0) or 1.0, (y1 - y0) or 1.0

    previous_fraction = None
    for x, y, text in sorted(points, key=lambda point: -point[1]):
        fraction = (y - y0) / span_y
        if previous_fraction is not None and previous_fraction - fraction < minimum_gap:
            fraction = previous_fraction - minimum_gap
        previous_fraction = fraction

        label_y = y0 + fraction * span_y
        to_the_right = (x - x0) / span_x < 0.70
        if to_the_right:
            sign, horizontal_alignment = 1, "left"
        else:
            sign, horizontal_alignment = -1, "right"

        displaced = abs(label_y - y) > 0.015 * span_y
        if displaced:
            arrow = dict(arrowstyle="-", color=AXIS_COLOR, linewidth=0.7, shrinkA=0, shrinkB=6)
        else:
            arrow = None

        ax.annotate(text, xy=(x, y), xytext=(x + sign * 0.045 * span_x, label_y), textcoords="data", ha=horizontal_alignment, va="center", fontsize=8, color=INK_COLOR, arrowprops=arrow)


def save_figure(fig, output_dir: Path, filename):
    """Speichert die Figur als PNG"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename)
    plt.close(fig)


def figure_accuracy_grid(cells_df, output_dir):
    """Liniendiagramme der Accuracy bei steigender Vergiftungsrate"""
    attack_df = cells_df[cells_df["attack"] != "clean"]
    if attack_df.empty:
        print("Keine Angriffs-Daten", flush=True)
        return

    fig, axes, models, attacks = subplot_grid(attack_df, "model", "attack")
    for row, model in enumerate(models):
        for col, attack in enumerate(attacks):
            ax = axes[row][col]
            cell_df = attack_df[(attack_df["model"] == model) & (attack_df["attack"] == attack)]

            for defense in ordered_defenses(cell_df["defense"]):
                curve = cell_df[cell_df["defense"] == defense].sort_values("rate")
                linestyle, marker = defense_style(defense)
                ax.errorbar(
                    curve["rate"], curve["accuracy_mean"],
                    yerr=curve["accuracy_std"].fillna(0.0),
                    color=defense_color(defense), linestyle=linestyle, marker=marker,
                    markeredgecolor="white", markeredgewidth=0.6,
                    capsize=2, elinewidth=0.8, label=defense)

            ax.set_title(f"{model} · {attack}")
            if row == len(models) - 1:
                ax.set_xlabel("Label-Flipping-Rate")
            if col == 0:
                ax.set_ylabel("Test-Accuracy")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)),bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Accuracy je Verteidigung und Rate  (Mittel ± Std über 3 Seeds)", y=1.0)
    save_figure(fig, output_dir, "fig1_accuracy_grid.png")


def figure_robustness_gap(master_df, output_dir):
    """Horizontales Balkendiagramm, das den Robustness Gap darstellt"""
    defended_df = master_df[(master_df["rate"] > 0.0) & (master_df["defense"] != "none")]
    if defended_df.empty:
        print("Keine Verteidigungs-Daten mit Rate > 0", flush=True)
        return

    grouped = defended_df.groupby(["model", "attack", "defense"], as_index=False)
    summary_df = grouped["robustness_gap"].agg(["mean", "std"]).reset_index()

    fig, axes, models, attacks = subplot_grid(summary_df, "model", "attack", size=(4.4, 3.6),sharey=False)
    for row, model in enumerate(models):
        for col, attack in enumerate(attacks):
            ax = axes[row][col]
            cell_df = summary_df[(summary_df["model"] == model) & (summary_df["attack"] == attack)]

            defenses = ordered_defenses(cell_df["defense"])
            per_defense = cell_df.set_index("defense")
            positions = list(range(len(defenses)))

            ax.barh(
                positions,
                [per_defense["mean"].get(defense, 0.0) for defense in defenses],
                height=0.68, xerr=[per_defense["std"].get(defense, 0.0) for defense in defenses],
                color=[defense_color(defense) for defense in defenses],
                edgecolor="white", linewidth=1.0,
                error_kw={"ecolor": MUTED_COLOR, "elinewidth": 0.8, "capsize": 2})

            ax.axvline(0.0, color=AXIS_COLOR, linewidth=1.2)
            ax.set_yticks(positions)
            if col == 0:
                ax.set_yticklabels(defenses)
            else:
                ax.set_yticklabels([])
            ax.set_title(f"{model} · {attack}")
            if row == len(models) - 1:
                ax.set_xlabel("Robustness Gap vs. none  (Δ Accuracy)")

    groups = sorted({group for group in DEFENSE_GROUPS.values() if group != "none"})
    handles = []
    for group in groups:
        handles.append(Patch(facecolor=GROUP_COLORS[group], edgecolor="white", label=group))
    fig.legend(handles=handles, title="Methodengruppe", loc="lower center", ncol=2,bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Übertreffen die Verteidigungen die unverteidigte Baseline?", y=1.0)
    save_figure(fig, output_dir, "fig2_robustness_gap.png")


def figure_loss_auroc(separability_df, output_dir):
    """Liniendiagramm für die AUROC über die Rate"""
    attack_df = separability_df[separability_df["attack"] != "clean"]
    if attack_df.empty:
        print("Keine Separabilitäts-Daten", flush=True)
        return

    baseline_df = attack_df[attack_df["defense"] == "none"]

    if baseline_df.empty:
        plot_df = attack_df
    else:
        plot_df = baseline_df

    models = sorted(plot_df["model"].unique())
    fig, axes = plt.subplots(1, len(models), squeeze=False, sharey=True, figsize=(4.6 * len(models), 3.8))

    for col, model in enumerate(models):
        ax = axes[0][col]
        model_df = plot_df[plot_df["model"] == model]

        for attack in sorted(model_df["attack"].unique()):
            curve = model_df[model_df["attack"] == attack].sort_values("rate")
            ax.errorbar(
                curve["rate"], curve["auroc_mean"],
                yerr=curve["auroc_std"].fillna(0.0),
                color=ATTACK_COLORS.get(attack, MUTED_COLOR), marker="o",
                markeredgecolor="white", markeredgewidth=0.6,
                capsize=2, elinewidth=0.8, label=attack)

        ax.axhline(0.5, color=AXIS_COLOR, linestyle=(0, (4, 4)), linewidth=1.2)
        ax.annotate(
            "0.5 = keine Trennung", xy=(1.0, 0.5), xycoords=("axes fraction", "data"),
            xytext=(-3, -11), textcoords="offset points",
            ha="right", color=MUTED_COLOR, fontsize=8)

        ax.set_title(model)
        ax.set_xlabel("Label-Flipping-Rate")
        if col == 0:
            ax.set_ylabel("Loss-AUROC  (clean vs. poisoned)")
            ax.legend(title="Angriff", loc="center right")

    fig.suptitle("Trennbarkeit der Loss-Verteilungen — die Voraussetzung der reaktiven Gruppe",y=1.03)
    save_figure(fig, output_dir, "fig3_loss_auroc.png")


def figure_cost_benefit(master_df, output_dir):
    """Scatter Plot, das Zeitkosten gegen Genauigkeitsgewinn plottet"""
    costed_df = master_df[master_df["rate"] > 0.0]
    costed_df = costed_df.dropna(subset=["seconds_per_epoch", "robustness_gap"])
    if costed_df.empty:
        print("Keine Kostenspalten", flush=True)
        return

    summary_df = costed_df.groupby(["model", "defense"], as_index=False).agg(
        gap=("robustness_gap", "mean"),
        seconds=("seconds_per_epoch", "mean"),
    )

    models = sorted(summary_df["model"].unique())
    fig, axes = plt.subplots(1, len(models), squeeze=False, figsize=(4.6 * len(models), 4.0))

    for col, model in enumerate(models):
        ax = axes[0][col]
        model_df = summary_df[summary_df["model"] == model]

        ax.scatter(
            model_df["seconds"], model_df["gap"], s=90,
            color=[defense_color(defense) for defense in model_df["defense"]],
            edgecolor="white", linewidth=1.2, zorder=3)
        
        ax.axhline(0.0, color=AXIS_COLOR, linewidth=1.2)
        points = list(zip(model_df["seconds"], model_df["gap"], model_df["defense"]))
        label_scatter_points(ax, points)

        ax.set_title(model)
        ax.set_xlabel("Sekunden pro Epoche")
        if col == 0:
            ax.set_ylabel("Mittlerer Robustness Gap")

    fig.suptitle("Kosten-Nutzen: Robustheitsgewinn gegen Rechenzeit", y=1.01)
    save_figure(fig, output_dir, "fig4_cost_benefit.png")


def figure_memorization(memorization_df, output_dir, rate=0.20):
    """Zeigt, ab welcher Epoche das Modell anfängt, die falschen Labels auswendig zu lernen"""
    at_rate_df = memorization_df[(memorization_df["rate"] == rate) & (memorization_df["attack"] != "clean")]

    if at_rate_df.empty:
        print(f"Keine Memorization-Daten bei Rate {rate}", flush=True)
        return

    fig, axes, models, attacks = subplot_grid(at_rate_df, "model", "attack")
    for row, model in enumerate(models):
        for col, attack in enumerate(attacks):
            ax = axes[row][col]
            cell_df = at_rate_df[(at_rate_df["model"] == model) & (at_rate_df["attack"] == attack)]

            for defense in ordered_defenses(cell_df["defense"]):
                curve = cell_df[cell_df["defense"] == defense].sort_values("epoch")
                linestyle, marker = defense_style(defense)
                ax.plot(
                    curve["epoch"], curve["memorization_mean"],
                    color=defense_color(defense), linestyle=linestyle, label=defense,
                    marker=marker, markevery=10, markeredgecolor="white", markeredgewidth=0.6)

            ax.set_title(f"{model} · {attack}")
            if row == len(models) - 1:
                ax.set_xlabel("Epoche")
            if col == 0:
                ax.set_ylabel("Memorization Rate")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)),bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Memorization der vergifteten Labels über die Epochen  (Rate {rate:.0%})", y=1.0)
    save_figure(fig, output_dir, "fig5_memorization.png")


def parse_args():
    """Parst Eingabeparameter aus der Kommandozeile"""
    parser = argparse.ArgumentParser(description="Die fuenf RQ3-Abbildungen generieren.")
    parser.add_argument("--experiments", type=Path, default=None,help="Wurzel der RQ3-Ergebnisse (Default: experiments/RQ3).")
    parser.add_argument("--out", type=Path, default=None,help="Ausgabeverzeichnis (Default: <experiments>/figures).")
    parser.add_argument("--rate", type=float, default=0.20,help="Flipping-Rate fuer die Memorization-Kurven (Default: 0.20).")
    return parser.parse_args()


def main():
    """Liest alle CSVs ein und macht die 5 Grafiken"""
    args = parse_args()
    base_dir = args.experiments or paths.RQ3_DIR
    output_dir = args.out or (base_dir / "figures")
    apply_style()

    master_df = read_csv_or_none(base_dir / "rq3_master.csv", "rq3_master.csv")
    cells_df = read_csv_or_none(base_dir / "rq3_cells.csv", "rq3_cells.csv")
    separability_df = read_csv_or_none(base_dir / "diagnostics" / "separability.csv","separability.csv")
    if separability_df is None:
        separability_df = read_csv_or_none(base_dir / "diagnostics" / "separability_legacy.csv","separability_legacy.csv")
    memorization_df = read_csv_or_none(base_dir / "diagnostics" / "memorization.csv","memorization.csv")

    if cells_df is not None:
        figure_accuracy_grid(cells_df, output_dir)
    if master_df is not None:
        figure_robustness_gap(master_df, output_dir)
        figure_cost_benefit(master_df, output_dir)
    if separability_df is not None:
        figure_loss_auroc(separability_df, output_dir)
    if memorization_df is not None:
        figure_memorization(memorization_df, output_dir, rate=args.rate)


if __name__ == "__main__":
    main()

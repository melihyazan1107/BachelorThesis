"""Visualisierungs-Werkzeuge für einen einzelnen Trainingslauf

1. Lernkurven
2. Degradationskurve
3. Embedding-Plots
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _scale_and_embed(features, method: str, n_components):
    """Reduziert die Bild-Features Dimensionen.

    Args:
        features (np.ndarray): Matrix der extrahierten Modell-Embeddings [N, Feature_Dim]
        method (str): "tsne" oder "pca"
        n_components (int): Ziel-Dimension

    Returns:
        np.ndarray: Die projizierte Matrix der Form [N, n_components]
    """
    scaled = StandardScaler().fit_transform(features)
    if method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=40, early_exaggeration=12)
    else:
        reducer = PCA(n_components=n_components)
    return reducer.fit_transform(scaled)


def _save_and_close(out_path: Path, filename):
    """Speichert einen Plot"""
    plt.tight_layout()
    plt.savefig(out_path / filename)
    plt.close()


def plot_degradation_curve(metrics_df, out_path: Path):
    """Plottet die Degradation Curve

    Args:
        metrics_df (pd.DataFrame): Die aggregierten Metriken aller bisherigen Läufe
        out_path (Path): Zielordner für das Diagramm
    """
    plt.figure(figsize=(10, 6), dpi=300)
    sns.lineplot(data=metrics_df, x="rate", y="accuracy", hue="seed", marker="o", alpha=0.7)

    mean_df = metrics_df.groupby("rate")["accuracy"].mean().reset_index()
    plt.plot(mean_df["rate"], mean_df["accuracy"], color="black", linewidth=3, linestyle="--",marker="s", label="Mean")

    rates = metrics_df["rate"].unique()
    plt.xticks(rates, [f"{int(rate * 100)}%" for rate in rates])
    plt.ylabel("Test Accuracy")
    plt.xlabel("Noise Rate")
    plt.legend()
    _save_and_close(out_path, "degradation_curve.png")


def plot_learning_curves(history, out_path: Path, noise_rate):
    """Erstellt den Loss- und Accuracy-Plot über die Epochen

    Args:
        history (list): Liste von Dictionaries
        out_path (Path): Zielverzeichnis für den Plot
        noise_rate (float): Die Vergiftungsrate dieses speziellen Laufs
    """
    history_df = pd.DataFrame(history)
    if noise_rate is not None:
        title_suffix = f"Noise Rate {int(noise_rate * 100)}%"
    else:
        title_suffix = ""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    fig.suptitle(f"Learning Curves{title_suffix}")

    sns.lineplot(data=history_df, x="epoch", y="train_loss", ax=axes[0], label="Train")
    sns.lineplot(data=history_df, x="epoch", y="val_loss", ax=axes[0], label="Val")
    axes[0].set_ylabel("Loss")

    accuracy_column = "train_acc_poison" if noise_rate else "train_acc_clean"
    sns.lineplot(data=history_df, x="epoch", y=accuracy_column, ax=axes[1], label="Train")
    sns.lineplot(data=history_df, x="epoch", y="val_acc", ax=axes[1], label="Val")
    axes[1].set_ylabel("Accuracy")

    _save_and_close(out_path, "learning_curves.png")


def plot_embeddings(features, labels, out_path: Path, method="tsne"):
    """Projiziert die Ausgaben in einen 2D-Raum

    Args:
        features (np.ndarray): Matrix der Bild-Embeddings
        labels (np.ndarray): Vektor der zugehörigen Ground-Truth-Labels
        out_path (Path): Zielverzeichnis
        method (str): "tsne" oder "pca"
    """
    embedded = _scale_and_embed(features, method, n_components=2)

    plt.figure(figsize=(16, 12), dpi=300)
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=labels, cmap="tab10", alpha=0.6, s=10)
    plt.colorbar(scatter)
    _save_and_close(out_path, f"{method}_embeddings.png")


def plot_embeddings_3d(features, labels, out_path: Path, method="tsne"):
    """Projiziert die Ausgaben in einen 3D-Raum"""
    embedded = _scale_and_embed(features, method, n_components=3)

    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], c=labels, cmap="tab10",alpha=0.6, s=10)
    fig.colorbar(scatter)
    _save_and_close(out_path, f"{method}_3d_embeddings.png")

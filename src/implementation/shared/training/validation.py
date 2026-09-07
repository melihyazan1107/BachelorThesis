"""Validierungs- und Test-Routinen für das Modell-Training

1. Epochen-Validierung (`validate`): Wird nach JEDER Epoche aufgerufen. Sie muss 
   extrem schnell sein und berechnet nur Loss und Accuracy, um zu entscheiden, ob 
   das aktuelle Modell als neuer "Best Checkpoint" gespeichert werden soll.
2. Abschluss-Evaluierung (`evaluate_and_export`): Läuft exakt einmal ganz am Ende 
   des Trainings. Hier wird das in Phase 1 ermittelte beste Modell geladen. 
   Es führt eine Analyse durch (Konfusionsmatrix, Scikit-Learn Report), 
   extrahiert die Bild-Features für PCA/t-SNE-Plots
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from implementation.shared.training.models import load_state_dict_without_compile_prefix
from implementation.shared.training.visualize import plot_embeddings, plot_embeddings_3d


def _run_inference(model, dataloader, device, criterion=None, extract_features=False):
    """Die Inferenz-Schleife.

    Nutzt Automatic Mixed Precision (AMP), um den Forward-Pass massiv zu beschleunigen

    Args:
        model (torch.nn.Module): Das zu evaluierende Netzwerk
        dataloader (DataLoader): Der Validierungs- oder Test-Datenlader
        device (torch.device): CPU oder GPU
        criterion (callable, optional): Loss-Funktion
        extract_features (bool): Wenn True, wird das Modell in Backbone und Classifier aufgetrennt

    Returns:
        dict: Wörterbuch mit 'loss', 'accuracy' und optional 'predictions', 'truths', 'features'.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    predictions, truths, feature_batches = [], [], []

    with torch.no_grad():
        for images, true_labels, _, _ in dataloader:
            images = images.to(device, non_blocking=True)
            true_labels_on_device = true_labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if extract_features:
                    features = model.forward_features(images)
                    logits = model.classifier(features)
                else:
                    logits = model(images)

                if criterion is not None:
                    loss = criterion(logits, true_labels_on_device).mean()
                    total_loss += loss.item() * images.size(0)

            total_correct += (logits.argmax(1) == true_labels_on_device).sum().item()
            total_samples += images.size(0)

            if extract_features:
                predictions.extend(logits.argmax(1).cpu().numpy())
                truths.extend(true_labels.numpy())
                feature_batches.append(features.detach().cpu())

    results = {"loss": total_loss / total_samples,"accuracy": total_correct / total_samples}

    if extract_features:
        results["predictions"] = predictions
        results["truths"] = truths
        results["features"] = torch.cat(feature_batches).view(len(truths), -1).numpy()
    return results


def validate(model, val_loader, criterion, device):
    """Wrapper für die Validierung am Ende jeder Epoche"""
    results = _run_inference(model, val_loader, device, criterion=criterion)
    return results["loss"], results["accuracy"]


def evaluate_and_export(model, test_loader, out_dir, device, seed, rate, best_model_filename, export_traced=True, export_embeddings=True):
    """Umfassende Abschluss-Analyse und Modell-Export

    Args:
        model (torch.nn.Module): Das aktuelle, fertig trainierte Netzwerk
        test_loader (DataLoader): Der Datensatz mit absolut ungesehenen, sauberen Bildern
        out_dir (Path): Das Zielverzeichnis für diesen Lauf
        device (torch.device): CPU oder GPU
        seed (int): Der Experiment-Seed
        rate (float): Die Vergiftungsrate
        best_model_filename (str): Name der Datei mit den besten Validierungs-Gewichten
        export_traced (bool): Wenn True, wird das Modell exportiert
        export_embeddings (bool): Wenn True, werden PCA/t-SNE Plots generiert

    Returns:
        dict: Die finalen Metriken (Accuracy, Precision, Recall, F1)
    """
    if hasattr(model, "_orig_mod"):
        raw_model = model._orig_mod
    else:
        raw_model = model
    load_state_dict_without_compile_prefix(raw_model, out_dir / best_model_filename, device)

    inference = _run_inference(raw_model, test_loader, device, extract_features=True)
    test_predictions = inference["predictions"]
    test_truths = inference["truths"]
    features = inference["features"]

    report = classification_report(test_truths, test_predictions, output_dict=True, zero_division=0)
    metrics = {
        "seed": seed,
        "rate": rate,
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]}

    matrix = confusion_matrix(test_truths, test_predictions)
    np.save(out_dir / "confusion_matrix.npy", matrix)
    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    ConfusionMatrixDisplay(matrix).plot(ax=ax, cmap="viridis", include_values=False)
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    if export_embeddings:
        plot_embeddings(features, test_truths, out_dir, "pca")
        plot_embeddings(features, test_truths, out_dir, "tsne")
        plot_embeddings_3d(features, test_truths, out_dir, "pca")
        plot_embeddings_3d(features, test_truths, out_dir, "tsne")

    raw_model.cpu()

    if export_traced:
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.jit.save(torch.jit.trace(raw_model, dummy_input), out_dir / "model_traced.pt")

    return metrics

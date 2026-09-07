"""Haupteinstiegspunkt für das Modell-Training

Dieses Skript steuert die kompletten Trainings-Sweeps. Es iteriert systematisch über 
alle konfigurierten Random-Seeds und Vergiftungsraten

Aufruf:
    cd src && python -m implementation.shared.training.run --config PATH --data-config PATH
                                                           [--seed S] [--defense NAME]
"""
import implementation.shared.env

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from implementation.shared import config, paths
from implementation.shared.defenses import DEFENSE_CHOICES, validate_defense
from implementation.shared.defenses.registry import defense_name_from_config
from implementation.shared.training.loader import get_base_loaders, get_train_loader
from implementation.shared.training.train import run_training_loop, setup_training
from implementation.shared.training.validation import evaluate_and_export
from implementation.shared.training.visualize import plot_degradation_curve, plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Training gemaess Trainings-YAML.")
    """Parst die Kommandozeilenargumente"""
    config.add_config_arguments(parser)
    parser.add_argument("--seed", type=int, default=None,help="Ueberschreibt experiment.seeds durch genau diesen Seed (ein SLURM-Array-Task pro Seed).")
    parser.add_argument("--defense", type=str, default=None, choices=list(DEFENSE_CHOICES),help="Ueberschreibt defense.type. damit reicht EIN YAML pro Modell fuer alle sieben Arme.")
    return parser.parse_args()


def update_summary_metrics(metrics, summary_dir: Path):
    """Fügt die Metriken eines einzelnen Runs an die Master-Tabelle an"""
    summary_csv = summary_dir / "summary_metrics.csv"
    current_df = pd.DataFrame([metrics])

    if summary_csv.exists():
        existing_df = pd.read_csv(summary_csv)
        metrics_df = pd.concat([existing_df, current_df], ignore_index=True)
        metrics_df = metrics_df.drop_duplicates(subset=["seed", "rate"], keep="last")
    else:
        metrics_df = current_df

    metrics_df.to_csv(summary_csv, index=False)

    if len(metrics_df) > 1:
        plot_degradation_curve(metrics_df, summary_dir)


def export_flags_for_rate(cfg, rate):
    """Bestimmt, welche rechenintensiven Artefakte gespeichert werden sollen"""
    export_cfg = cfg.get("export") or {}
    export_traced = bool(export_cfg.get("traced_model", True))

    embedding_setting = export_cfg.get("embedding_plots", True)
    if isinstance(embedding_setting, (list, tuple)):
        export_embeddings = any(abs(float(r) - float(rate)) < 1e-9 for r in embedding_setting)
    else:
        export_embeddings = bool(embedding_setting)
    return export_traced, export_embeddings


def execute_single_run(cfg, num_classes, device, val_loader, test_loader, seed, rate, defense_name=None, attack=None):
    """Führt einen vollständig isolierten Trainingslauf für EINE Parameter-Kombination aus

    1. Zufallszahlen fixieren
    2. Ordnerstruktur anlegen
    3. Daten (und potenziell vergiftete Labels) laden
    4. Modell und Verteidigung initialisieren
    5. Epochen trainieren (run_training_loop)
    6. Evaluieren, Visualisieren und RAM aufräumen
    """
    torch.manual_seed(seed)
    out_dir = paths.experiment_dir(cfg["experiment"]["rq"], cfg["experiment"]["model"], defense=defense_name, attack=attack)
    out_dir = out_dir / f"seed_{seed}" / f"rate_{rate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_train_loader(cfg, seed, rate, attack=attack)
    components = setup_training(cfg, device, num_classes, seed=seed, rate=rate, num_samples=len(train_loader.dataset))

    history, best_model_filename = run_training_loop(cfg, components, train_loader, val_loader, device, out_dir, seed, rate)

    defense_state = {**components.defense.state(),"attack": attack or cfg["experiment"]["rq"],"seed": seed,"rate": rate}
    with open(out_dir / "defense_state.json", "w", encoding="utf-8") as file:
        json.dump(defense_state, file, indent=2)

    plot_learning_curves(history, out_dir, noise_rate=rate)

    if best_model_filename is None:
        print("No checkpoint was saved. Skipping evaluation.", flush=True)
        metrics = None
    else:
        export_traced, export_embeddings = export_flags_for_rate(cfg, rate)
        metrics = evaluate_and_export(components.model, test_loader, out_dir, device, seed, rate, best_model_filename,export_traced=export_traced, export_embeddings=export_embeddings)

    del train_loader, components
    gc.collect()
    return metrics


def run_experiment(cfg, num_classes, device: torch.device, val_loader, test_loader):
    """Die Experiment-Schleife (Iteriert über Attacks, Seeds und Rates)"""
    rq = cfg["experiment"]["rq"]
    model_name = cfg["experiment"]["model"]

    attacks = cfg["experiment"].get("attacks")
    if attacks:
        defense_name = defense_name_from_config(cfg)
        arms = [(defense_name, paths.validate_attack(attack)) for attack in attacks]
    else:
        arms = [(None, None)]

    for defense_name, attack in arms:
        summary_dir = paths.experiment_dir(rq, model_name, defense=defense_name, attack=attack)
        summary_dir.mkdir(parents=True, exist_ok=True)

        for seed in cfg["experiment"]["seeds"]:
            for rate in cfg["experiment"]["rates"]:
                if attack is not None and (attack == "clean") != (rate == 0.0):
                    continue
                if attack is not None:
                    print(f"-- defense={defense_name} attack={attack} seed={seed} rate={rate} --",
                          flush=True)

                metrics = execute_single_run(
                    cfg, num_classes, device, val_loader, test_loader, seed, rate,
                    defense_name=defense_name, attack=attack)
                if metrics is not None:
                    update_summary_metrics(metrics, summary_dir)


def main():
    """Startet das Trainings-Framework und konfiguriert die Hardware"""
    print("Starting main script", flush=True)
    args = parse_args()
    training_path, data_path = config.resolve_config_paths(args.config, args.data_config)
    cfg = config.load_yaml(training_path)
    num_classes = config.load_yaml(data_path)["poisoning"]["num_classes"]

    if args.seed is not None:
        cfg["experiment"]["seeds"] = [args.seed]
    if args.defense is not None:
        cfg.setdefault("defense", {})["type"] = validate_defense(args.defense)

    rq = paths.validate_rq(cfg["experiment"]["rq"])
    print(f"Config: {training_path}", flush=True)
    print(f"Experiment: rq={rq} model={cfg['experiment']['model']} rates={cfg['experiment']['rates']}", flush=True)
    print(f"Defense: {defense_name_from_config(cfg)} | Seeds: {cfg['experiment']['seeds']} | Attacks: {cfg['experiment'].get('attacks') or '(Einzelarm)'}", flush=True)

    hardware = cfg["hardware"]
    torch.set_float32_matmul_precision(hardware["matmul_precision"])
    print(f"Set float32 matmul precision to: {hardware['matmul_precision']}", flush=True)

    print("Initializing PyTorch and checking CUDA", flush=True)
    torch.backends.cudnn.benchmark = hardware["cudnn_benchmark"]
    print(f"Set cudnn.benchmark to: {hardware['cudnn_benchmark']}", flush=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device}", flush=True)

    print("Initializing base loaders", flush=True)
    val_loader, test_loader = get_base_loaders(cfg)
    print("Starting experiment run", flush=True)
    run_experiment(cfg, num_classes, device, val_loader, test_loader)


if __name__ == "__main__":
    main()

"""Der zentrale Trainingscode: Setup, Epochen-Schleife und Telemetrie"""
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from implementation.shared.defenses import NullDefense, build_defense
from implementation.shared.defenses.base import Defense, aggregate_loss
from implementation.shared.training.metrics import MetricsTracker
from implementation.shared.training.models import get_model
from implementation.shared.training.validation import validate


@dataclass
class TrainingComponents:
    """Datencontainer für alle Zustandsobjekte eines Trainingslaufs"""
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    defense: Defense
    num_classes: int


def _build_model(cfg, device, num_classes):
    """Instanziiert das Modell"""
    model = get_model(cfg["experiment"]["model"], num_classes=num_classes).to(device)
    model = model.to(memory_format=torch.channels_last)

    if cfg["hardware"]["compile"] and hasattr(torch, "compile"):
        try:
            print("Testing torch.compile support")
            torch.compile(torch.nn.Linear(1, 1).to(device))(torch.randn(1, 1).to(device))
            model = torch.compile(model)
            print("Successfully compiled model using torch.compile.")
        except Exception as error:
            print(f"Skipping torch.compile due to error: {error}")

    return model


def _build_optimizer(cfg, model):
    """Konfiguriert den Optimierer"""
    optimizer_cfg = cfg["optimizer"]
    if optimizer_cfg["type"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=optimizer_cfg["lr"],
            weight_decay=optimizer_cfg["weight_decay"],
            momentum=optimizer_cfg["momentum"],
            nesterov=optimizer_cfg["nesterov"],
            foreach=True)
    
    return torch.optim.AdamW(model.parameters(),lr=optimizer_cfg["lr"],weight_decay=optimizer_cfg["weight_decay"],fused=True)


def _build_scheduler(cfg, optimizer):
    """Konfiguriert die Anpassung der Lernrate"""
    scheduler_cfg = cfg["scheduler"]
    total_epochs = cfg["experiment"]["epochs"]

    if scheduler_cfg["type"] != "cosine":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg["milestones"],
            gamma=scheduler_cfg["gamma"])

    warmup_epochs = int(scheduler_cfg.get("warmup_epochs") or 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=scheduler_cfg["warmup_start_factor"],total_iters=warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, scheduler], milestones=[warmup_epochs])
        print(f"LR-Warmup aktiv: {warmup_epochs}", flush=True)
    return scheduler


def setup_training(cfg, device, num_classes, seed=0, rate=0.0, num_samples=0):
    """Zentrale Funktion, die alle Objekte für den Trainingslauf baut"""
    model = _build_model(cfg, device, num_classes)
    criterion = nn.CrossEntropyLoss(reduction="none")
    defense = build_defense(cfg, seed=seed, rate=rate, num_samples=num_samples, num_classes=num_classes, device=device)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler("cuda")
    return TrainingComponents(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, scaler=scaler, defense=defense, num_classes=num_classes)


def _cat_to_numpy(tensor_list):
    """Funktion für das Telemetrie-Logging"""
    return torch.cat(tensor_list).cpu().numpy()


@contextmanager
def _frozen_batchnorm_stats(model):
    """Kontext-Manager zum temporären Einfrieren der BatchNorm-Statistiken"""
    batchnorm_layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    saved = []
    for layer in batchnorm_layers:
        if layer.num_batches_tracked is None:
            tracked = None
        else:
            tracked = layer.num_batches_tracked.clone()
        saved.append((layer, layer.momentum, tracked))

    for layer, _, _ in saved:
        layer.momentum = 0.0
    try:
        yield
    finally:
        for layer, momentum, tracked in saved:
            layer.momentum = momentum
            if tracked is not None:
                layer.num_batches_tracked.copy_(tracked)


class _EpochTelemetry:
    """Sammler für die Telemetriedaten jedes einzelnen Bildes"""

    def __init__(self, epoch):
        self.epoch = epoch
        self.sample_indices = []
        self.true_labels = []
        self.noisy_labels = []
        self.predictions = []
        self.losses = []
        self.confidences = []
        self.weights = []
        self.has_sample_weights = False

    def add_batch(self, *, sample_indices, true_labels, noisy_labels, predictions, loss,confidence, weights):
        """Hängt einen verarbeiteten Mini-Batch an den Speicher an"""
        self.sample_indices.append(sample_indices)
        self.true_labels.append(true_labels)
        self.noisy_labels.append(noisy_labels)
        self.predictions.append(predictions.detach())
        self.losses.append(loss.detach().float().cpu())
        self.confidences.append(confidence.detach().cpu())
        if weights is None:
            self.weights.append(torch.ones(len(sample_indices)))
        else:
            self.weights.append(weights.detach().float().cpu())
            self.has_sample_weights = True

    def as_dict(self):
        """Kompiliert in große, flache NumPy-Arrays"""
        sample_indices = _cat_to_numpy(self.sample_indices)
        columns = {
            "epoch": [self.epoch] * len(sample_indices),
            "sample_index": sample_indices,
            "true_label": _cat_to_numpy(self.true_labels),
            "noisy_label": _cat_to_numpy(self.noisy_labels),
            "predicted_class": _cat_to_numpy(self.predictions),
            "confidence": _cat_to_numpy(self.confidences),
            "loss": _cat_to_numpy(self.losses)}
        if self.has_sample_weights:
            columns["sample_weight"] = _cat_to_numpy(self.weights)
        return columns


def _telemetry_view(model, defense, images, logits, noisy_labels, loss_per_sample):
    """Generiert für Loss und Logits für die Telemetrie"""
    if defense.needs_clean_forward:
        with torch.no_grad(), _frozen_batchnorm_stats(model):
            clean_logits = model(images)
    else:
        clean_logits = logits

    if defense.loss_is_plain_ce and not defense.needs_clean_forward:
        plain_loss = loss_per_sample
    else:
        with torch.no_grad():
            plain_loss = F.cross_entropy(clean_logits, noisy_labels, reduction="none")

    return clean_logits, plain_loss


def _move_batch_to_device(batch, device, defense):
    """Verschiebt Tensoren auf die GPU"""
    images, true_labels, noisy_labels, sample_indices = batch
    images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
    true_labels = true_labels.to(device, non_blocking=True)
    noisy_labels = noisy_labels.to(device, non_blocking=True)
    if defense.needs_idx:
        defense_indices = sample_indices.to(device, non_blocking=True)
    else:
        defense_indices = sample_indices
    return images, true_labels, noisy_labels, sample_indices, defense_indices


def _epoch_stats(epoch_start, device):
    """Erfasst Hardware Metriken"""
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_vram_mb = None
    return {"epoch_seconds": time.time() - epoch_start, "peak_vram_mb": peak_vram_mb}


def train_epoch(epoch, model, train_loader, optimizer, scaler, device, defense=None):
    """Führt das Training für eine einzelne Epoche über den kompletten Datensatz aus"""
    defense = defense or NullDefense()
    defense.on_epoch_start(epoch)

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    epoch_start = time.time()

    train_loss = 0.0
    train_correct_clean = 0
    train_correct_poison = 0
    train_total = 0
    telemetry = _EpochTelemetry(epoch)

    for batch in train_loader:
        images, true_labels, noisy_labels, sample_indices, defense_indices = (_move_batch_to_device(batch, device, defense))

        optimizer.zero_grad(set_to_none=True)
        batch_images, context = defense.transform_batch(images, noisy_labels)

        with torch.amp.autocast("cuda"):
            logits = model(batch_images)
            loss_per_sample = defense.compute_loss(logits, noisy_labels, defense_indices, context)
            weights = defense.sample_weights(
                loss_per_sample.detach(), noisy_labels, defense_indices)
            loss_scalar = aggregate_loss(loss_per_sample, weights)

            clean_logits, plain_loss = _telemetry_view(model, defense, images, logits, noisy_labels, loss_per_sample)
            confidences = torch.softmax(clean_logits, dim=1).max(dim=1)[0]

        scaler.scale(loss_scalar).backward()
        scaler.step(optimizer)
        scaler.update()

        defense.observe(defense_indices, plain_loss)

        predictions = clean_logits.argmax(1)
        train_loss += loss_scalar.item() * images.size(0)
        train_correct_clean += (predictions == true_labels).sum().item()
        train_correct_poison += (predictions == noisy_labels).sum().item()
        train_total += images.size(0)

        telemetry.add_batch(sample_indices=sample_indices,true_labels=true_labels,noisy_labels=noisy_labels,predictions=predictions,loss=plain_loss,confidence=confidences,weights=weights)

    telemetry_dict = telemetry.as_dict()
    epoch_stats = _epoch_stats(epoch_start, device)

    return (train_loss / train_total,train_correct_clean / train_total,train_correct_poison / train_total,telemetry_dict,epoch_stats)


def _should_log_samples(sample_level, epoch, total_epochs):
    """Entscheidet ob die sample_telemetry geschrieben werden soll"""
    if sample_level == "window":
        return epoch <= 5 or epoch % 5 == 0 or epoch > total_epochs - 10
    return True


def _rotate_best_checkpoint(out_dir, previous_filename, model, rate, epoch):
    """Speichert das aktuell beste Modell und löscht das alte"""
    if previous_filename is not None:
        old_file = out_dir / previous_filename
        if old_file.exists():
            try:
                old_file.unlink()
            except OSError:
                pass

    filename = f"best_model_weights_rate_{rate:.2f}_epoch_{epoch:03d}.pt"
    torch.save(model.state_dict(), out_dir / filename)
    return filename


def _write_epoch_telemetry(tracker, telemetry, telemetry_dir, epoch, epoch_stats, defense,sample_level, total_epochs):
    """Speichert alle generierten Artefakte einer Epoche ab"""
    tracker.compute_epoch_metrics(telemetry, epoch_stats=epoch_stats, keep_threshold=defense.keep_threshold)
    tracker.save_confusion_matrix_json(telemetry, telemetry_dir / f"confusion_matrix_ep{epoch:03d}.json")
    tracker.save_epoch_csv(telemetry_dir / "epoch_metrics.csv")
    if _should_log_samples(sample_level, epoch, total_epochs):
        tracker.save_sample_telemetry(telemetry, telemetry_dir / "sample_telemetry.csv")


def run_training_loop(cfg, components: TrainingComponents, train_loader, val_loader, device: torch.device, out_dir: Path, seed, rate):
    """Epochen-Iteration"""
    model = components.model
    defense = components.defense
    best_val_loss = float("inf")
    best_model_filename = None
    history = []

    telemetry_cfg = cfg.get("telemetry") or {}
    sample_level = str(telemetry_cfg.get("sample_level", "full"))
    diagnostics = bool(telemetry_cfg.get("diagnostics", False))
    total_epochs = cfg["experiment"]["epochs"]

    telemetry_dir = out_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    metrics_tracker = MetricsTracker(num_classes=components.num_classes, diagnostics=diagnostics)

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc_clean, train_acc_poison, telemetry, epoch_stats = train_epoch(epoch, model, train_loader, components.optimizer, components.scaler, device,defense=defense)
        components.scheduler.step()

        val_loss, val_acc = validate(model, val_loader, components.criterion, device)
        history.append({"epoch": epoch,"train_loss": train_loss,"val_loss": val_loss,"train_acc_clean": train_acc_clean,"train_acc_poison": train_acc_poison,"val_acc": val_acc})

        _write_epoch_telemetry(metrics_tracker, telemetry, telemetry_dir, epoch, epoch_stats,defense, sample_level, total_epochs)
        del telemetry

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_filename = _rotate_best_checkpoint(out_dir, best_model_filename, model, rate, epoch)

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_start))
        print(f"Epoch {epoch:03d} | Seed {seed} | Rate {rate:.2f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc (clean): {train_acc_clean:.4f} | Acc (poison): {train_acc_poison:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed}",flush=True)

    return history, best_model_filename

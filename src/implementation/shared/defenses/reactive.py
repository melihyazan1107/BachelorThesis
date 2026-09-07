"""Reaktive Verteidigungen gegen Label-Rauschen

Eingriffe in die Trainingsschleife:
An der Stelle `sample_weights`, um auffällige Bilder 
vor dem Gradienten-Update auszublenden.

WICHTIGER UNTERSCHIED (Orakel vs. Realität):
- SmallLossDefense arbeitet als "Orakel". Es bekommt die echte Vergiftungsrate 
  als Parameter übergeben und schneidet einfach exakt diesen Prozentsatz ab
- GMMFilterDefense ist Unsupervised. Es kennt die Rate nicht und 
  versucht, die sauberen/vergifteten Gruppen durch Clustering selbst zu schätzen
"""
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from implementation.shared.defenses.base import Defense


class SmallLossDefense(Defense):
    """Filtert Bilder basierend auf dem Small-Loss-Kriterium hart heraus"""
    name = "small_loss"
    group = "reactive"
    loss_is_plain_ce = True

    def __init__(self, rate: float, warmup_epochs: int = 25, forget_rate: float | None = None, ramp_epochs: int = 10):
        """Initialisiert die Small-Loss Verteidigung mit einem linearen Ramp-Up

        Args:
            rate (float): Die echte Vergiftungsrate des Datensatzes
            warmup_epochs (int): Epochen ohne Filterung
            forget_rate (float | None): Ziel-Verwerfungsquote. Standardmäßig gleich `rate`
            ramp_epochs (int): Anzahl der Epochen nach dem Warmup
        """
        self.rate = float(rate)
        self.warmup_epochs = int(warmup_epochs)

        if forget_rate is None:
            self.forget_rate = float(rate)
        else:
            self.forget_rate = float(forget_rate)
        self.ramp_epochs = max(1, int(ramp_epochs))
        self.active_forget_rate = 0.0
        self.epoch = 0


    def on_epoch_start(self, epoch):
        """Aktualisiert die aktuelle Verwerfungsquote basierend auf dem Schedule"""
        self.epoch = epoch
        if epoch <= self.warmup_epochs or self.forget_rate <= 0.0:
            self.active_forget_rate = 0.0
            return

        progress = min(1.0, (epoch - self.warmup_epochs) / self.ramp_epochs)
        self.active_forget_rate = self.forget_rate * progress

    def sample_weights(self, per_sample_loss, targets, sample_indices):
        """Wendet den Filter auf den aktuellen Batch an

        Sortiert die Loss-Werte aufsteigend. Die `num_to_keep` kleinsten Losses 
        erhalten ein Gewicht von 1.0 (Behalten), der Rest 0.0 (Verwerfen)
        """
        if self.active_forget_rate <= 0.0:
            return None

        batch_size = per_sample_loss.numel()
        num_to_keep = max(1, int(round((1.0 - self.active_forget_rate) * batch_size)))
        if num_to_keep >= batch_size:
            return None

        kept_indices = torch.argsort(per_sample_loss)[:num_to_keep]
        weights = torch.zeros_like(per_sample_loss)
        weights[kept_indices] = 1.0
        return weights

    def state(self):
        return {"name": self.name,"group": self.group,"rate": self.rate,"forget_rate": self.forget_rate,"warmup_epochs": self.warmup_epochs,"ramp_epochs": self.ramp_epochs,"oracle": True}


class GMMFilterDefense(Defense):
    """Unsupervised Filtering via Gaussian Mixture Models
    
    Diese Methode braucht kein Orakel-Wissen. Sie merkt sich die Loss-Werte aller 
    Bilder aus der vorherigen Epoche. Zu Beginn jeder neuen Epoche legt sie zwei 
    Gauß-Kurven (GMM) über diese Verteilung: 
    - Kurve A (niedriger Mittelwert) repräsentiert die sauberen Daten
    - Kurve B (hoher Mittelwert) repräsentiert die vergifteten Daten
    """
    name = "gmm_filter"
    group = "reactive"
    loss_is_plain_ce = True
    needs_idx = True


    def __init__(self, num_samples: int, warmup_epochs: int = 25, p_threshold: float = 0.5,seed: int = 0, device: torch.device | str = "cpu"):
        """Initialisiert den GMM-Filter

        Args:
            num_samples (int): Gesamtgröße des Datensatzes
            warmup_epochs (int): Epochen ohne Filterung
            p_threshold (float): Metrik-Schwellenwert
            seed (int): Zufallswert Initialisiert
            device (torch.device | str): CPU oder GPU
        """
        self.num_samples = int(num_samples)
        self.warmup_epochs = int(warmup_epochs)
        self.p_threshold = float(p_threshold)
        self.keep_threshold = float(p_threshold)
        self.seed = int(seed)
        self.device = device

        self._previous_losses = np.zeros(self.num_samples, dtype=np.float64)
        self._has_been_seen = np.zeros(self.num_samples, dtype=bool)
        self._clean_probabilities
        self.fitted_epoch
        self.last_clean_mean
        self.last_noisy_mean

    def observe(self, sample_indices, plain_loss):
        """Speichert den Loss des aktuellen Batches für die nächste Epoche ab"""
        cpu_indices = sample_indices.detach().cpu().numpy()
        self._previous_losses[cpu_indices] = plain_loss.detach().float().cpu().numpy()
        self._has_been_seen[cpu_indices] = True

    def on_epoch_start(self, epoch):
        """Passt das GMM an die Loss-Verteilung der letzten Epoche an"""
        if epoch <= self.warmup_epochs or not self._has_been_seen.any():
            self._clean_probabilities = None
            return

        seen_losses = self._previous_losses[self._has_been_seen]
        min_loss, max_loss = float(seen_losses.min()), float(seen_losses.max())
        if max_loss - min_loss < 1e-12:
            self._clean_probabilities = None
            return

        normalized_losses = (self._previous_losses - min_loss) / (max_loss - min_loss)
        normalized_losses = normalized_losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4,random_state=self.seed)
        gmm.fit(normalized_losses[self._has_been_seen])

        clean_component = int(np.argmin(gmm.means_.ravel()))
        clean_probabilities = gmm.predict_proba(normalized_losses)[:, clean_component]
        clean_probabilities[~self._has_been_seen] = 1.0

        self.last_clean_mean = float(gmm.means_.ravel()[clean_component])
        self.last_noisy_mean = float(gmm.means_.ravel()[1 - clean_component])
        self.fitted_epoch = epoch
        self._clean_probabilities = torch.as_tensor(clean_probabilities, dtype=torch.float32, device=self.device)

    def sample_weights(self, per_sample_loss, targets, sample_indices):
        """Gibt die Wahrscheinlichkeiten als Soft-Weights für den Batch zurück"""
        if self._clean_probabilities is None:
            return None
        return self._clean_probabilities[sample_indices].to(per_sample_loss.dtype)

    def state(self):
        return {"name": self.name,"group": self.group,"warmup_epochs": self.warmup_epochs,"p_threshold": self.p_threshold,"seed": self.seed,"oracle": False,"fitted_epoch": self.fitted_epoch,"clean_component_mean": self.last_clean_mean,"noisy_component_mean": self.last_noisy_mean}

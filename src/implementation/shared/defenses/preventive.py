"""Präventive Verteidigungen gegen Label-Rauschen

Eingriffe in die Trainingsschleife:
- Label Smoothing, GCE, ELR -> Verändern die Loss-Berechnung (`compute_loss`)
- Mixup                     -> Verändert Bilder UND Loss (`transform_batch` + `compute_loss`)
"""
import numpy as np
import torch
import torch.nn.functional as F

from implementation.shared.defenses.base import Defense, derive_generator


class LabelSmoothingDefense(Defense):
    """Verhindert Overconfidence des Modells"""
    name = "label_smoothing"
    group = "preventive"
    loss_is_plain_ce = False

    def __init__(self, epsilon: float = 0.1):
        """Initialisiert Label Smoothing.
        
        Args:
            epsilon (float): Die Glättungsrate. 0.1 bedeutet, dass 10% der Wahrscheinlichkeits-
                             masse von der Zielklasse abgezogen und verteilt werden
        """
        self.epsilon = float(epsilon)

    def compute_loss(self, logits, targets, sample_indices, context):
        """Berechnet die Cross-Entropy mit integriertem Smoothing"""
        return F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.epsilon)

    def state(self):
        return {"name": self.name, "group": self.group, "epsilon": self.epsilon}


class MixupDefense(Defense):
    """Erzeugt Trainingsdaten durch Interpolation von Bildern"""
    name = "mixup"
    group = "preventive"
    loss_is_plain_ce = False
    needs_clean_forward = True


    def __init__(self, alpha: float = 0.2, seed: int = 0):
        """Initialisiert Mixup
        
        Args:
            alpha (float): Parameter der Verteilung
            seed (int): Globaler Seed
        """
        self.alpha = float(alpha)
        self.seed = int(seed)
        self._mix_ratio_rng = np.random.default_rng(self.seed + 90_000)
        self._permutation_generator = derive_generator(self.seed, salt=1)


    def transform_batch(self, images, targets):
        """Mischt die Bilder des aktuellen Batches untereinander"""
        if self.alpha > 0.0:
            mix_ratio = float(self._mix_ratio_rng.beta(self.alpha, self.alpha))
        else:
            mix_ratio = 1.0
        shuffled_indices = torch.randperm(images.size(0), generator=self._permutation_generator)
        shuffled_indices = shuffled_indices.to(targets.device)

        mixed_images = mix_ratio * images + (1.0 - mix_ratio) * images[shuffled_indices]
        context = {
            "targets_a": targets,
            "targets_b": targets[shuffled_indices],
            "mix_ratio": mix_ratio}
        
        return mixed_images.contiguous(memory_format=torch.channels_last), context


    def compute_loss(self, logits, targets, sample_indices, context):
        mix_ratio = context["mix_ratio"]
        loss_a = F.cross_entropy(logits, context["targets_a"], reduction="none")
        loss_b = F.cross_entropy(logits, context["targets_b"], reduction="none")
        return mix_ratio * loss_a + (1.0 - mix_ratio) * loss_b


    def state(self):
        return {"name": self.name, "group": self.group, "alpha": self.alpha, "seed": self.seed}


class GCEDefense(Defense):
    """Generalized Cross Entropy
    
    Der Parameter `q` steuert das Verhalten:
    q -> 0.0: Entspricht der strikten Cross-Entropy
    q -> 1.0: Entspricht dem Mean Absolute Error (MAE)
    q = 0.7:  Der Sweetspot aus dem Original-Paper
    """
    name = "gce"
    group = "preventive"
    loss_is_plain_ce = False


    def __init__(self, q: float = 0.7):
        self.q = float(q)


    def compute_loss(self, logits, targets, sample_indices, context):
        probabilities = F.softmax(logits.float(), dim=1)
        target_probabilities = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probabilities = target_probabilities.clamp(1e-7, 1.0)
        return (1.0 - target_probabilities.pow(self.q)) / self.q


    def state(self):
        return {"name": self.name, "group": self.group, "q": self.q}


class ELRDefense(Defense):
    """Early-Learning Regularization (ELR)"""
    name = "elr"
    group = "preventive"
    loss_is_plain_ce = False
    needs_idx = True


    def __init__(self, num_samples: int, num_classes: int, lambda_weight: float = 3.0, beta: float = 0.7, device: torch.device | str = "cpu"):
        """Initialisiert ELR
        
        Args:
            num_samples: Gesamtanzahl aller Bilder im Datensatz
            num_classes: Anzahl der möglichen Klassen
            lambda_weight: Gewichtung, wie stark Abweichungen bestraft werden
            beta: Trägheit des Momentum. 0.7 bedeutet, das Wissen wird zu 70% 
            behalten und 30% durch die neueste Epoche aktualisiert
            device: CPU oder GPU
        """
        self.lambda_weight = float(lambda_weight)
        self.beta = float(beta)
        self.num_samples = int(num_samples)
        self.num_classes = int(num_classes)
        self.target = torch.full((self.num_samples, self.num_classes), 1.0 / self.num_classes, dtype=torch.float32, device=device)


    def compute_loss(self, logits, targets, sample_indices, context):
        """Berechnet die Cross-Entropy"""
        probabilities = F.softmax(logits.float(), dim=1).clamp(1e-4, 1.0 - 1e-4)
        normalized_probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

        self.target[sample_indices] = (self.beta * self.target[sample_indices] + (1.0 - self.beta) * normalized_probabilities.detach())

        cross_entropy_loss = F.cross_entropy(logits, targets, reduction="none")
        target_agreement = (self.target[sample_indices] * probabilities).sum(dim=1)
        regularization = torch.log((1.0 - target_agreement).clamp_min(1e-8))
        return cross_entropy_loss + self.lambda_weight * regularization


    def state(self):
        return {"name": self.name, "group": self.group, "lambda": self.lambda_weight, "beta": self.beta, "buffer_shape": [self.num_samples, self.num_classes]}

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SoftCrossEntropyLoss(nn.Module):
    """
    Soft Cross Entropy Loss supporting class weights and multi-task outputs.
    """
    def __init__(self, class_weights: Dict[str, torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights or {}
        self.loss_values = {}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        self.loss_values = {}

        for task, logits in outputs.items():
            probs = F.log_softmax(logits, dim=-1)
            target_probs = targets[task]

            if task in self.class_weights:
                weights = self.class_weights[task].to(logits.device)
                target_probs = target_probs * weights.unsqueeze(0)

            loss = -(target_probs * probs).sum(dim=-1).mean()
            self.loss_values[task] = loss
            total_loss += loss

        return total_loss
    

class EmotionNLLLoss(nn.Module):
    """
    Negative log-likelihood loss supporting soft labels and uncertainty modeling.
    Switch between hard and soft targets using `use_soft_labels`.
    """
    def __init__(self, class_weights: Dict[str, torch.Tensor] = None, use_soft_labels: bool = True):
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.loss_values = {}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        self.loss_values = {}

        mu = outputs["mu"]
        logvar = outputs["logvar"]
        var = torch.exp(logvar)

        for task, target in targets.items():
            if task in self.class_weights:
                weights = self.class_weights[task].to(mu.device)
                target = target * weights.unsqueeze(0)

            if self.use_soft_labels:
                # Soft labels version
                loss_per_class = ((mu - target) ** 2) / var + logvar
                loss = (loss_per_class * target).sum(dim=-1).mean()
            else:
                # Hard labels version
                hard_targets = target.argmax(dim=-1)  # B
                loss = (((mu - F.one_hot(hard_targets, num_classes=mu.size(-1)).float()) ** 2) / var + logvar)
                loss = loss.sum(dim=-1).mean()

            self.loss_values[task] = loss
            total_loss += loss

        return total_loss

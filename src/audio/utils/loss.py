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
    Negative log-likelihood loss with uncertainty modeling.
    """
    def __init__(self, class_weights: dict = None):
        super().__init__()
        self.class_weights = class_weights or {}
        self.loss_values = {}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        self.loss_values = {}

        for task in targets:
            mu = outputs["mu"]
            logvar = outputs["logvar"]

            target = targets[task]
            var = torch.exp(logvar)

            if task in self.class_weights:
                weights = self.class_weights[task].to(mu.device)
                target = target * weights.unsqueeze(0)

            loss = ((mu - target) ** 2) / var + logvar
            loss = loss.sum(dim=-1).mean()
            self.loss_values[task] = loss
            total_loss += loss

        return total_loss
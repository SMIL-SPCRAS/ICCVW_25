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
    

class EmotionFocalLoss(nn.Module):
    """
    Focal Loss with support for multi-task outputs, soft targets, class weights, and label smoothing.
    """
    def __init__(self,
                 class_weights: Dict[str, torch.Tensor] = None,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.class_weights = class_weights or {}
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.loss_values = {}

    def forward(self,
                outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        self.loss_values = {}

        for task, logits in outputs.items():
            probs = F.softmax(logits, dim=-1)
            targets_task = targets[task].to(logits.device)

            if self.label_smoothing > 0:
                targets_task = targets_task * (1 - self.label_smoothing) + self.label_smoothing / targets_task.size(-1)

            pt = (probs * targets_task).sum(dim=-1)
            log_pt = torch.log(pt.clamp(min=1e-8))
            focal_term = (1 - pt).clamp(min=1e-4) ** self.gamma
            loss = -focal_term * log_pt

            if task in self.class_weights:
                weights = self.class_weights[task].to(logits.device)
                classwise_weights = (targets_task * weights.unsqueeze(0)).sum(dim=-1)
                loss = loss * classwise_weights

            task_loss = loss.mean()
            self.loss_values[task] = task_loss
            total_loss += task_loss

        return total_loss
    

class EmotionNLLLoss(nn.Module):
    """
    Negative log-likelihood loss supporting soft labels and uncertainty modeling.
    Switch between hard and soft targets using `use_soft_labels`.
    """
    def __init__(self, 
                 class_weights: Dict[str, torch.Tensor] = None, 
                 use_soft_labels: bool = True,
                 clamp_logvar: bool = True,
                 logvar_clamp_range: tuple = (-3.0, 3.0),
                 logvar_reg_weight: float = 0.01):
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.loss_values = {}

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        self.loss_values = {}

        mu = outputs["mu"]
        logvar = outputs["logvar"]
        if self.clamp_logvar:
            logvar = logvar.clamp(min=self.logvar_clamp_range[0], max=self.logvar_clamp_range[1])

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
            
            # Regularization on logvar to avoid collapse to -∞
        logvar_penalty = torch.mean(logvar ** 2)
        total_loss += self.logvar_reg_weight * logvar_penalty

        return total_loss


class EmotionNLLLossStable(nn.Module):
    """
    Numerically stable loss function for emotion classification using soft labels and uncertainty modeling.
    Avoids collapse due to extreme logvars and includes optional kl-penalty and label smoothing.
    """
    def __init__(self,
                 class_weights: dict = None,
                 use_soft_labels: bool = True,
                 clamp_logvar: bool = True,
                 logvar_clamp_range: tuple = (-3.0, 3.0),
                 logvar_reg_weight: float = 0.01,
                 kl_weight: float = 0.5,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing

        self.loss_values = {}

    def set_warmup_mode(self, is_warmup: bool):
        self.use_soft_labels = is_warmup
        self.label_smoothing = 0.2 if is_warmup else 0.0
        self.kl_weight = 0.01 if is_warmup else 0.5

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        mu = outputs["mu"]                      # shape: [B, C]
        logvar = outputs["logvar"]              # shape: [B, C]
        probs = outputs["emo"]                  # softmax(mu / temperature)

        if self.clamp_logvar:
            logvar = logvar.clamp(*self.logvar_clamp_range)
        
        var = torch.exp(logvar)

        total_loss = 0.0

        for task, target in targets.items():
            if task in self.class_weights:
                weights = self.class_weights[task].to(mu.device)
                target = target * weights.unsqueeze(0)

            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / target.size(-1)

            if self.use_soft_labels:
                loss_per_class = ((mu - target) ** 2) / var + logvar
                loss_nll = (loss_per_class * target).sum(dim=-1).mean()
            else:
                hard_target = target.argmax(dim=-1)
                onehot = F.one_hot(hard_target, num_classes=mu.size(-1)).float()
                loss_per_class = ((mu - onehot) ** 2) / var + logvar
                loss_nll = loss_per_class.sum(dim=-1).mean()

            task_loss = loss_nll
            loss_kl = 0.0

            if self.kl_weight > 0:
                target_probs = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                loss_kl = F.kl_div(probs.log(), target_probs, reduction='batchmean')
                task_loss += self.kl_weight * loss_kl

            logvar_penalty = torch.mean(logvar ** 2)
            task_loss += self.logvar_reg_weight * logvar_penalty
            total_loss += task_loss
            self.loss_values[task] = task_loss

        return total_loss


class EmotionFocalLossStable(nn.Module):
    def __init__(
        self,
        class_weights: dict = None,
        gamma: float = 2.0,
        use_soft_labels: bool = True,
        clamp_logvar: bool = True,
        logvar_clamp_range: tuple = (-3.0, 3.0),
        logvar_reg_weight: float = 0.1,
        kl_weight: float = 0.5,
        label_smoothing: float = 0.1,
        confidence_penalty_weight: float = 0.05,
        logvar_min: float = -1.5
    ):
        super().__init__()
        self.class_weights = class_weights or {}
        self.gamma = gamma
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing
        self.confidence_penalty_weight = confidence_penalty_weight
        self.logvar_min = logvar_min
        self.loss_values = {}

    def set_warmup_mode(self, is_warmup: bool):
        self.use_soft_labels = is_warmup
        self.label_smoothing = 0.2 if is_warmup else 0.0
        self.kl_weight = 0.01 if is_warmup else 0.5

    def forward(self, outputs: dict, targets: dict) -> torch.Tensor:
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        probs = outputs["emo"]

        if self.clamp_logvar:
            logvar = logvar.clamp(min=self.logvar_min, max=self.logvar_clamp_range[1])

        var = torch.exp(logvar)
        total_loss = 0.0

        for task, target in targets.items():
            target = target.to(mu.device)

            if task in self.class_weights:
                weights = self.class_weights[task].to(mu.device)
                target = target * weights.unsqueeze(0)

            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / target.size(-1)

            if self.use_soft_labels:
                loss_per_class = ((mu - target) ** 2) / var + logvar
                pt = (probs * target).sum(dim=-1)
                focal_weight = (1 - pt).clamp(min=1e-4) ** self.gamma
                loss_nll = focal_weight * (loss_per_class * target).sum(dim=-1)
            else:
                hard_target = target.argmax(dim=-1)
                onehot = F.one_hot(hard_target, num_classes=mu.size(-1)).float()
                loss_per_class = ((mu - onehot) ** 2) / var + logvar
                pt = probs.gather(1, hard_target.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - pt).clamp(min=1e-4) ** self.gamma
                loss_nll = focal_weight * loss_per_class.sum(dim=-1)

            loss_nll = loss_nll.mean()
            task_loss = loss_nll

            misfit = (mu - target).abs()
            overconfidence_penalty = (torch.exp(-logvar) * misfit).mean()
            task_loss += 0.05 * overconfidence_penalty

            if self.kl_weight > 0:
                target_probs = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                loss_kl = F.kl_div(probs.log(), target_probs, reduction='batchmean')
                task_loss += self.kl_weight * loss_kl

            logvar_penalty = torch.mean(logvar ** 2)
            confidence_penalty = torch.mean(torch.exp(-logvar))

            task_loss += self.logvar_reg_weight * logvar_penalty
            task_loss += self.confidence_penalty_weight * confidence_penalty

            total_loss += task_loss
            self.loss_values[task] = task_loss

        return total_loss
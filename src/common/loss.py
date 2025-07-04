import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    """Soft Cross Entropy Loss supporting class weights and multi-task outputs."""
    def __init__(self, 
                 class_weights: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.class_weights = class_weights or {}
        self.loss_values = {}

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
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
    

class SoftFocalLoss(nn.Module):
    def __init__(self,
                 gamma: float = 2.0,
                 class_weights: dict[str, torch.Tensor] | None = None,
                 label_smoothing: float = 0.0,
                 eps: float = 1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights or {}
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.loss_values = {}

    def forward(self,
                outputs: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        self.loss_values = {}

        for task, logits in outputs.items():
            log_p = F.log_softmax(logits, dim=-1) # log p_i
            p = log_p.exp() # p_i
            t = targets[task].to(logits.device) # t_i

            if self.label_smoothing > 0:
                t = t * (1 - self.label_smoothing) + self.label_smoothing / t.size(-1)

            # class-weights ≡ αᵢ
            if task in self.class_weights:
                alpha = self.class_weights[task].to(logits.device)  # shape (C,)
                t = t * alpha.unsqueeze(0)

            focal = (1.0 - p).clamp(min=self.eps).pow(self.gamma)  # (1-p)^{γ}
            loss = -(t * focal * log_p).sum(dim=-1).mean()  # sum_i …
            self.loss_values[task] = loss.detach()
            total += loss

        return total
    

class EmotionFocalLoss(nn.Module):
    """Focal Loss with support for multi-task outputs, soft targets, class weights, and label smoothing."""
    def __init__(self,
                 class_weights: dict[str, torch.Tensor] | None = None,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.class_weights = class_weights or {}
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.loss_values = {}

    def forward(self,
                outputs: dict[str, torch.Tensor],
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
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
                 class_weights: dict[str, torch.Tensor] | None = None, 
                 use_soft_labels: bool = True,
                 clamp_logvar: bool = True,
                 logvar_clamp_range: tuple[float, float] = (-3.0, 3.0),
                 logvar_reg_weight: float = 0.01) -> None:
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.loss_values = {}

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
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
                 class_weights: dict[str, torch.Tensor] | None = None,
                 use_soft_labels: bool = True,
                 clamp_logvar: bool = True,
                 logvar_clamp_range: tuple[float, float] = (-3.0, 3.0),
                 logvar_reg_weight: float = 0.01,
                 kl_weight: float = 0.5,
                 label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing

        self.loss_values = {}

    def set_warmup_mode(self,
                        is_warmup: bool):
        self.use_soft_labels = is_warmup
        self.label_smoothing = 0.2 if is_warmup else 0.0
        self.kl_weight = 0.01 if is_warmup else 0.5

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
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
    

class VAEEmotionNLLLossStable(nn.Module):
    """
    Numerically stable loss function for emotion classification using soft labels and uncertainty modeling.
    Avoids collapse due to extreme logvars and includes optional kl-penalty and label smoothing.
    """
    def __init__(self,
                 class_weights: dict[str, torch.Tensor] | None = None,
                 use_soft_labels: bool = True,
                 clamp_logvar: bool = True,
                 logvar_clamp_range: tuple[float, float] = (-3.0, 3.0),
                 logvar_reg_weight: float = 0.01,
                 kl_weight: float = 0.5,
                 label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_soft_labels = use_soft_labels
        self.clamp_logvar = clamp_logvar
        self.logvar_clamp_range = logvar_clamp_range
        self.logvar_reg_weight = logvar_reg_weight
        self.kl_weight = kl_weight
        self.label_smoothing = label_smoothing

        self.loss_values = {}

    def set_warmup_mode(self,
                        is_warmup: bool):
        self.use_soft_labels = is_warmup
        self.label_smoothing = 0.2 if is_warmup else 0.0
        self.kl_weight = 0.01 if is_warmup else 0.5

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
        
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        z = outputs["z"]
        probs = F.softmax(z / 1.5, dim=-1)

        if self.clamp_logvar:
            logvar = logvar.clamp(*self.logvar_clamp_range)

        total_loss = 0.0

        for task, target in targets.items():
            if task in self.class_weights:
                weights = self.class_weights[task].to(mu.device)
                target = target * weights.unsqueeze(0)

            if self.label_smoothing > 0:
                target = target * (1 - self.label_smoothing) + self.label_smoothing / target.size(-1)

            if self.use_soft_labels:
                # Use z (sampled) instead of mu
                loss_nll = F.kl_div(probs.log(), target, reduction="batchmean")
            else:
                hard_target = target.argmax(dim=-1)
                loss_nll = F.cross_entropy(z, hard_target)

            # KL divergence to N(0, I)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

            # Regularization on logvar itself to prevent exploding uncertainty
            logvar_penalty = torch.mean(logvar ** 2)

            task_loss = loss_nll + self.kl_weight * kl_div + self.logvar_reg_weight * logvar_penalty
            total_loss += task_loss
            self.loss_values[task] = task_loss

        return total_loss


class EmotionFocalLossStable(nn.Module):
    def __init__(
        self,
        class_weights: dict[str, torch.Tensor] | None = None,
        gamma: float = 2.0,
        use_soft_labels: bool = True,
        clamp_logvar: bool = True,
        logvar_clamp_range: tuple[float, float] = (-3.0, 3.0),
        logvar_reg_weight: float = 0.1,
        kl_weight: float = 0.5,
        label_smoothing: float = 0.1, confidence_penalty_weight: float = 0.05, logvar_min: float = -1.5) -> None:
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

    def set_warmup_mode(self, 
                        is_warmup: bool) -> None:
        self.use_soft_labels = is_warmup
        self.label_smoothing = 0.2 if is_warmup else 0.0
        self.kl_weight = 0.01 if is_warmup else 0.5

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
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
    

class MultiHeadFocalLoss(nn.Module):
    def __init__(self, 
                 gamma: float = 2.0, 
                 class_weights: dict[str, torch.Tensor] | None = None,
                 bootstrap_ratio: float = 0.1) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.bootstrap_ratio = bootstrap_ratio
        self.loss_values = {}

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
        logits_list = outputs["emo_heads"]
        target = targets["emo"]
        total_loss = 0.0

        if self.class_weights is not None:
            cw = self.class_weights.to(target.device)
            target = target * cw.unsqueeze(0)

        # Bootstrap
        K = target.size(-1)
        uniform = torch.full_like(target, 1.0 / K)
        target_bs = (1 - self.bootstrap_ratio) * target + self.bootstrap_ratio * uniform

        for idx, logits in enumerate(logits_list):
            prob = F.softmax(logits, dim=-1)
            logp = F.log_softmax(logits, dim=-1)

            pt = (prob * target_bs).sum(dim=-1).clamp(min=1e-6)
            focal_w = (1 - pt) ** self.gamma

            loss = -(target_bs * logp).sum(dim=-1) * focal_w
            loss = loss.mean()

            self.loss_values[f"emo_head_{idx}"] = loss
            total_loss += loss

        total_loss = total_loss / len(logits_list)
        self.loss_values = {"emo": total_loss}
        return total_loss


class MultiHeadSoftCrossEntropyLoss(nn.Module):
    def __init__(self, 
                 class_weights: dict[str, torch.Tensor] | None = None,
                 bootstrap_ratio: float = 0.1) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.bootstrap_ratio = bootstrap_ratio
        self.loss_values = {}

    def forward(self, 
                outputs: dict[str, torch.Tensor], 
                targets: dict[str, torch.Tensor]) -> torch.Tensor:
        logits_list = outputs["emo_heads"]
        target = targets["emo"]
        total_loss = 0.0

        if self.class_weights is not None:
            cw = self.class_weights.to(target.device)
            target = target * cw.unsqueeze(0)

        # Bootstrap
        K = target.size(-1)
        uniform = torch.full_like(target, 1.0 / K)
        target_bs = (1 - self.bootstrap_ratio) * target + self.bootstrap_ratio * uniform

        for idx, logits in enumerate(logits_list):
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(target_bs * log_probs).sum(dim=-1).mean()
            self.loss_values[f"emo_head_{idx}"] = loss
            total_loss += loss
            
        total_loss = total_loss / len(logits_list)
        self.loss_values = {"emo": total_loss}
        return total_loss

import torch

from common.metrics import MacroF1, UAR


def create_scheduler(
        cfg: dict[str, any],
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
    """Factory method to create a learning rate scheduler."""
    scheduler_type = cfg["scheduler_type"]
    scheduler_params = cfg["scheduler_params"]

    schedulers = {
        "ReduceLROnPlateau": lambda opt, params: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **params),
        "CosineAnnealingLR": lambda opt, params: torch.optim.lr_scheduler.CosineAnnealingLR(opt, **params),
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type](optimizer, scheduler_params)


def create_metrics(cfg: dict[str, any], task: str = "emo") -> list[any]:
    """Create a list of metrics based on config."""
    metric_names = cfg.get("metrics", ["UAR", "MacroF1"])
    available = {
        "UAR": UAR,
        "MacroF1": MacroF1,
    }
    
    return [available[name](task=task) for name in metric_names if name in available]



import torch.nn as nn


class UnfreezeScheduler:
    def __init__(self, model: nn.Module, logger: any, schedule: dict[int, list[str]]) -> None:
        self.model = model
        self.logger = logger
        self.schedule = schedule
        self.unfroze = set()

    def apply(self, epoch: int) -> None:
        if epoch in self.schedule:
            for name, module in self.model.named_modules():
                for target in self.schedule[epoch]:
                    if target in name and name not in self.unfroze:
                        for param in module.parameters():
                            param.requires_grad = True
                        
                        self.logger.info(f"Unfroze module: {name}")
                        self.unfroze.add(name)
    
    def is_warming_up(self, epoch: int) -> bool:
        return any(epoch < key for key in self.schedule)

    def get_total_warmup_epochs(self) -> int:
        return max(self.schedule.keys(), default=0)
    

class GradientBasedUnfreezeScheduler:
    def __init__(
        self,
        model: nn.Module,
        logger: any,
        layers_to_track: list[str],
        threshold: float = 1e-4,
        max_unfreeze_per_epoch: int = 1,
        temperature_schedule: dict[int, float] = None,
    ) -> None:
        self.model = model
        self.logger = logger
        self.layers_to_track = layers_to_track
        self.threshold = threshold
        self.max_unfreeze = max_unfreeze_per_epoch
        self.unfrozen_layers = set()
        self._is_warming_up = True
        self.temperature_schedule = temperature_schedule or {}

    def is_warming_up(self, epoch: int) -> bool:
        """Return whether the model is in the warmup phase."""
        return self._is_warming_up

    def step(self, epoch: int) -> None:
        """
        Called during training. Computes gradient norms for tracked layers
        and unfreezes the most promising ones.
        """
        grad_norms = {}

        for name, param in self.model.named_parameters():
            if any(layer in name for layer in self.layers_to_track) and param.grad is not None:
                if name in self.unfrozen_layers:
                    continue
                norm = param.grad.norm().item()
                grad_norms[name] = norm

        if not grad_norms:
            return

        sorted_layers = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
        to_unfreeze = sorted_layers[:self.max_unfreeze]

        for name, _ in to_unfreeze:
            for param_name, param in self.model.named_parameters():
                if name in param_name:
                    param.requires_grad = True
            
            self.unfrozen_layers.add(name)
            self.logger.info(f"🔓 Unfroze layer: {name}")

        # Stop warmup after sufficient layers have been unfrozen
        if len(self.unfrozen_layers) >= len(self.layers_to_track):
            self._is_warming_up = False

        # Apply temperature schedule
        if epoch in self.temperature_schedule:
            new_temp = self.temperature_schedule[epoch]
            if hasattr(self.model.uncertainty_head, 'temperature'):
                self.model.uncertainty_head.temperature = new_temp
                self.logger.info(f"🔥 Temperature set to {new_temp} at epoch {epoch}")

    def apply(self, epoch: int) -> None:
        self.step(epoch)

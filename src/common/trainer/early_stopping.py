from common.trainer.trainer import Trainer


class EarlyStopping:
    """
    Early stopping to terminate training when monitored metric stops improving.
    Now delegates model saving to Trainer.
    """
    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        monitor: str = 'val_loss',
        trace_func: callable = print
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.monitor = monitor
        self.trace_func = trace_func

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val = float('inf')

    def __call__(self, metrics: dict[str, float], trainer: Trainer, epoch: int = None) -> bool:
        val = metrics.get(self.monitor)
        if val is None:
            raise ValueError(f"Metric '{self.monitor}' not found in metrics dict.")

        score = -val

        if self.best_score is None:
            self.best_score = score
            trainer.save_checkpoint(epoch=metrics.get("epoch", -1), val_metric=val, is_best=True)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            trainer.save_checkpoint(epoch=metrics.get("epoch", -1), val_metric=val, is_best=True)
            self.counter = 0

        if epoch:
            trainer.save_checkpoint(epoch=epoch, val_metric=val)

        return self.early_stop
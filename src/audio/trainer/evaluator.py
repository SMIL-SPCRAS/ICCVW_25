import os
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any

from sklearn.metrics import confusion_matrix

from audio.trainer.visualize import plot_conf_matrix
from audio.trainer.metric_manager import MetricManager

try:
    import wandb
except ImportError:
    wandb = None


class Evaluator:
    """
    Evaluator for computing metrics and plotting confusion matrices.
    """
    def __init__(
        self,
        metric_manager: MetricManager,
        label_names: Dict[str, list],
        logger: Any,
        plot_dir: str,
        use_wandb: bool = False
    ):
        self.metric_manager = metric_manager
        self.label_names = label_names
        self.logger = logger
        self.plot_dir = plot_dir
        self.use_wandb = use_wandb and wandb is not None

    def evaluate(
        self,
        targets: Dict[str, Any],
        predicts: Dict[str, Any],
        epoch: int,
        phase: str
    ) -> Dict[str, float]:
        results = self.metric_manager.calculate_all(targets, predicts)
        results_str = " | ".join([f"{k.split('_')[-1]}={v*100:.2f}" for k, v in results.items()])
        self.logger.info(f"[Epoch {epoch}] 📊 {phase.upper()}: {results_str}")

        if self.use_wandb:
            for name, value in results.items():
                wandb.log({f"{phase}_{name}": value * 100, "epoch": epoch})

        return results

    def draw_confusion_matrix(
        self,
        targets: Dict[str, Any],
        predicts: Dict[str, Any],
        task: str,
        epoch: int,
        phase: str,
        is_best: bool = False
    ):
        y_true = np.array(targets[task])
        y_pred = np.array(predicts[task])

        if y_true.ndim > 1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim > 1:
            y_pred = y_pred.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.label_names[task]))))
        title = f"{task.capitalize()} {phase.capitalize()} Epoch {epoch}"
        if is_best:
            save_path = os.path.join(self.plot_dir, f"cm_{task}_{phase}_best.svg")
        else:
            save_path = os.path.join(self.plot_dir, f"cm_{task}_{phase}_epoch_{epoch}.svg")
            
        fig = plot_conf_matrix(cm, 
                               labels=self.label_names[task], 
                               title=title, normalize=True, save_path=save_path)
        
        if self.use_wandb:
            wandb.log({f"conf_matrix_{task}_{phase}": wandb.Image(save_path)}, step=epoch)

        return fig

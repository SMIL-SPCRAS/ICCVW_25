import os
import numpy as np

from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix

from audio.trainer.visualize import plot_conf_matrix
from audio.trainer.metric_manager import MetricManager
from audio.utils.mlflow_logger import MLflowLogger


class Evaluator:
    """Evaluator for computing metrics and plotting confusion matrices."""
    def __init__(
        self,
        metric_manager: MetricManager,
        label_names: dict[str, list],
        logger: any,
        plot_dir: str,
        ml_logger: MLflowLogger = None
    ) -> None:
        self.metric_manager = metric_manager
        self.label_names = label_names
        self.logger = logger
        self.plot_dir = plot_dir
        self.ml_logger = ml_logger

    def evaluate(
        self,
        targets: dict[str, any],
        predicts: dict[str, any],
        epoch: int,
        phase: str
    ) -> dict[str, float]:
        results = self.metric_manager.calculate_all(targets, predicts)
        results_str = " | ".join([f"{k} = {v*100:.2f}%" for k, v in results.items()])
        self.logger.info(f"[Epoch {epoch}] 📊 {phase.upper()}: {results_str}")

        if self.ml_logger:
            metrics = {f"{phase}_{name}": value * 100 for name, value in results.items()}
            metrics["epoch"] = epoch
            self.ml_logger.log_metrics(metrics, step=epoch)

        return results

    def draw_confusion_matrix(
        self,
        targets: dict[str, any],
        predicts: dict[str, any],
        db: str,
        task: str,
        epoch: int,
        phase: str,
        is_best: bool = False
    ) -> Figure:
        y_true = np.array(targets[task])
        y_pred = np.array(predicts[task])

        if y_true.ndim > 1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim > 1:
            y_pred = y_pred.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(self.label_names[task]))))
        title = f"{task.capitalize()} {phase.capitalize()} Epoch {epoch}"
        if is_best:
            save_path = os.path.join(self.plot_dir, f"cm_{db}_{task}_{phase}_best.png")
        else:
            save_path = os.path.join(self.plot_dir, f"cm_{db}_{task}_{phase}_epoch_{epoch}.png")
            
        fig = plot_conf_matrix(cm, 
                               labels=self.label_names[task], 
                               title=title, normalize=True, save_path=save_path)
        
        if self.ml_logger:
            self.ml_logger.log_artifact(save_path, artifact_path=f"plots/{phase}/cm_epoch_{epoch}")

        return fig

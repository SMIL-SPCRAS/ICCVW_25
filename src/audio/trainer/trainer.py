import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Any, Optional

from audio.trainer.loss_manager import LossManager
from audio.trainer.evaluator import Evaluator
from audio.utils.mlflow_logger import MLflowLogger
from audio.utils.utils import log_logvar, log_mu_statistics
from audio.utils.schedulers import UnfreezeScheduler

class Trainer:
    """
    Trainer class for multi-task models with logging, evaluation and checkpointing.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: Any,
        device: torch.device,
        metrics: List[Any],
        logger: Any,
        log_dir: str,
        plot_dir: str,
        checkpoint_dir: str,
        final_activations: Dict[str, Any],
        ml_logger: MLflowLogger = None,
        unfreeze_scheduler: UnfreezeScheduler = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics
        self.logger = logger
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.checkpoint_dir = checkpoint_dir
        self.final_activations = final_activations
        self.ml_logger = ml_logger

        self.unfreeze_scheduler = unfreeze_scheduler

        self.history = defaultdict(dict)

    def _apply_unfreeze_schedule(self, epoch: int):
        if epoch in self.unfreeze_schedule:
            for name, module in self.model.named_modules():
                for target in self.unfreeze_schedule[epoch]:
                    if target in name and name not in self.frozen_modules:
                        for param in module.parameters():
                            param.requires_grad = True
                        self.logger.info(f"Unfroze module: {name}")
                        self.frozen_modules.add(name)

    def _run_epoch(self, dataloader, epoch: int, phase: str) -> Dict[str, Any]:
        is_train = phase == 'train'
        self.model.train() if is_train else self.model.eval()
        if self.unfreeze_scheduler and is_train:
            self.unfreeze_scheduler.apply(epoch)

        if self.unfreeze_scheduler and hasattr(self.loss_fn, 'set_warmup_mode'):
            self.loss_fn.set_warmup_mode(self.unfreeze_scheduler.is_warming_up(epoch))

        if is_train and epoch == 10:
            if hasattr(self.model.uncertainty_head, 'temperature'):
                self.model.uncertainty_head.temperature = 1.0
                self.logger.info("Temperature reduced to 1.0")

        loss_tracker = LossManager()
        for task in self.final_activations:
            loss_tracker.track_task(task)

        all_predicts = defaultdict(lambda: defaultdict(list))
        all_targets = defaultdict(lambda: defaultdict(list))

        loop = tqdm(dataloader, desc=f"{phase} epoch {epoch}", leave=False)
        for batch in loop:
            inputs, labels, metas = batch
            inputs = inputs.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}

            with torch.set_grad_enabled(is_train):
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()

            batch_size = inputs.size(0)
            for task, val in self.loss_fn.loss_values.items():
                loss_tracker.update({task: val.item()}, batch_size)

            for i, db in enumerate(metas["db"]):
                for task in self.final_activations:
                    pred = self.final_activations[task](outputs[task][i].unsqueeze(0))
                    all_predicts[db][task].append(pred.detach().cpu().numpy()[0])
                    all_targets[db][task].append(labels[task][i].detach().cpu().numpy())

            loop.set_postfix(loss=loss.item())

        mean_losses = loss_tracker.compute()
        total_loss = sum(mean_losses.values())
        if self.ml_logger:
            self.ml_logger.log_metrics({f"{phase}_loss": total_loss}, step=epoch)

        for task, value in mean_losses.items():
            if self.ml_logger:
                self.ml_logger.log_metrics({f"{phase}_loss_{task}": value}, step=epoch)

        lr = self.optimizer.param_groups[0]['lr']
        if self.ml_logger:
            self.ml_logger.log_metrics({f"{phase}_lr": lr}, step=epoch)

        self.logger.info("=" * 80)
        loss_str = " | ".join([f"{task} Loss = {mean_losses[task]:.3f}" for task in mean_losses])
        self.logger.info(f"[Epoch {epoch}] {'🔁' if is_train else '🔍'} {phase.upper()}: Total Loss = {total_loss:.3f} | {loss_str}")

        log_logvar(self.logger, self.ml_logger, self.plot_dir, outputs, epoch, phase)
        if epoch % 5 == 0 and "mu" in outputs:
            log_mu_statistics(
                outputs["mu"], labels["emo"],
                epoch=epoch, phase=phase,
                logger=self.logger,
                ml_logger=self.ml_logger,
                plot_dir=self.plot_dir
            )
        
        self.history[epoch][f"{phase}_loss"] = total_loss
        self.history[epoch][f"{phase}_lr"] = lr

        return {"targets": all_targets, "predicts": all_predicts, "loss": total_loss}

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, Any]:
        return self._run_epoch(dataloader, epoch, 'train')

    def validate_epoch(self, dataloader, epoch: int, evaluator: Optional[Evaluator] = None, phase_name: Optional[str] = 'val') -> Dict[str, Any]:
        result = self._run_epoch(dataloader, epoch, phase_name)
        result["metrics"] = {}
        
        if evaluator is not None:
            for db in result["predicts"]:
                targets_db = {task: result["targets"][db][task] for task in result["targets"][db]}
                predicts_db = {task: result["predicts"][db][task] for task in result["predicts"][db]}
                metrics_result = evaluator.evaluate(targets_db, predicts_db, epoch, phase=f"{phase_name}_{db}")
                
                result["metrics"][db] = metrics_result
                
                for task in predicts_db:
                    evaluator.draw_confusion_matrix(targets_db, predicts_db, db, task, epoch, phase_name)
                
                for k, v in metrics_result.items():
                    self.history[epoch][f"{phase_name}_{db}_{k}"] = v
                
        return result

    def predict(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_predicts = {task: [] for task in self.final_activations}
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs)
                for task in outputs:
                    predicts = self.final_activations[task](outputs[task])
                    all_predicts[task].extend(predicts.cpu().numpy())

        return {task: np.array(all_predicts[task]) for task in all_predicts}

    def save_metrics(self, path: str = "metrics_history.csv") -> None:
        df = pd.DataFrame.from_dict(self.history, orient="index").sort_index()
        df.index.name = "epoch"
        df.reset_index(inplace=True)
        df.to_csv(path, index=False)
        self.logger.info(f"Metrics saved to {path}")

    def save_checkpoint(self, epoch: int, val_metric: float, is_best: bool = False) -> str:
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'val_metric': val_metric
        }

        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pt"))
        else:
            torch.save(state, os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))

        if self.ml_logger:
            self.ml_logger.log_artifact(os.path.join(self.checkpoint_dir, "best_model.pt"))

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {"epoch": checkpoint['epoch'], "val_metric": checkpoint['val_metric']}

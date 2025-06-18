import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
from typing import Dict, List, Any, Optional

from audio.trainer.loss_manager import LossManager
from audio.trainer.evaluator import Evaluator

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
        checkpoint_dir: str,
        final_activations: Dict[str, Any],
        use_wandb: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics
        self.logger = logger
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.final_activations = final_activations
        self.writer = SummaryWriter(log_dir=log_dir)
        self.use_wandb = use_wandb and wandb is not None
        if self.use_wandb:
            wandb.watch(self.model, log='all')

        self.history = defaultdict(dict)

    def _run_epoch(self, dataloader, epoch: int, phase: str) -> Dict[str, Any]:
        is_train = phase == 'train'
        self.model.train() if is_train else self.model.eval()
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
                for task in outputs:
                    pred = self.final_activations[task](outputs[task][i].unsqueeze(0))
                    all_predicts[db][task].append(pred.detach().cpu().numpy()[0])
                    all_targets[db][task].append(labels[task][i].detach().cpu().numpy())

            loop.set_postfix(loss=loss.item())

        mean_losses = loss_tracker.compute()
        total_loss = sum(mean_losses.values())
        self.writer.add_scalar(f'{phase}/loss', total_loss, epoch)
        if self.use_wandb:
            wandb.log({f"{phase}_loss": total_loss, "epoch": epoch})

        for task, value in mean_losses.items():
            self.writer.add_scalar(f"{phase}/loss_{task}", value, epoch)
            if self.use_wandb:
                wandb.log({f"{phase}_loss_{task}": value, "epoch": epoch})

        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar(f'{phase}/learning_rate', lr, epoch)
        if self.use_wandb:
            wandb.log({f"{phase}_lr": lr, "epoch": epoch})

        self.logger.info("=" * 80)
        loss_str = " | ".join([f"{task} Loss={mean_losses[task]:.3f}" for task in mean_losses])
        self.logger.info(f"[Epoch {epoch}] {'🔁' if is_train else '🔍'} {phase.upper()}: Total Loss={total_loss:.3f} | {loss_str}")

        self.history[epoch][f"{phase}_loss"] = total_loss
        self.history[epoch][f"{phase}_lr"] = lr

        return {"targets": all_targets, "predicts": all_predicts, "loss": total_loss}

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, Any]:
        return self._run_epoch(dataloader, epoch, 'train')

    def validate_epoch(self, dataloader, epoch: int, evaluator: Optional[Evaluator] = None) -> Dict[str, Any]:
        result = self._run_epoch(dataloader, epoch, 'val')

        if evaluator is not None:
            for db in result["predicts"]:
                targets_db = {task: result["targets"][db][task] for task in result["targets"][db]}
                predicts_db = {task: result["predicts"][db][task] for task in result["predicts"][db]}
                metrics_result = evaluator.evaluate(targets_db, predicts_db, epoch, phase=f"val_{db}")
                evaluator.draw_confusion_matrix(targets_db, predicts_db, "emo", epoch, "val", is_best=True)
                for k, v in metrics_result.items():
                    self.history[epoch][f"val_{db}_{k}"] = v
                
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

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {"epoch": checkpoint['epoch'], "val_metric": checkpoint['val_metric']}

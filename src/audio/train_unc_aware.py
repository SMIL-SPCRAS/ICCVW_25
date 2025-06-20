import os
import sys
import logging
from datetime import datetime
from collections import Counter
from typing import Any


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import mlflow

sys.path.append('src')

from audio.utils.factories import create_dataloaders, create_scheduler, create_metrics
from audio.models.uncertainty_aware_models import *
from audio.trainer.trainer import Trainer
from audio.trainer.early_stopping import EarlyStopping
from audio.trainer.metric_manager import MetricManager
from audio.trainer.evaluator import Evaluator
from audio.utils.utils import load_config, define_seed
from audio.utils.loss import EmotionFocalLossStable
from audio.utils.mlflow_logger import MLflowLogger
from audio.utils.schedulers import GradientBasedUnfreezeScheduler

def compute_class_weights(
    dataloader: DataLoader,
    emotion_labels: list,
    logger: Any = None
) -> torch.Tensor:
    """
    Computes class weights for a multi-class classification task.
    """
    counts = Counter()
    dataset = dataloader.dataset
    num_classes = len(emotion_labels)

    if isinstance(dataset, ConcatDataset):
        for subds in dataset.datasets:
            if hasattr(subds, "df") and hasattr(subds, "emotion_labels"):
                df = subds.df
                labels = df[emotion_labels].values
                indices = labels.argmax(axis=1)
                counts.update(indices.tolist())
    elif hasattr(dataset, "df") and hasattr(dataset, "emotion_labels"):
        df = dataset.df
        labels = df[emotion_labels].values
        indices = labels.argmax(axis=1)
        counts.update(indices.tolist())
    else:
        for _, labels, _ in dataloader:
            for label in labels["emo"]:
                label_idx = label.argmax().item() if label.ndim >= 1 else label.item()
                counts[label_idx] += 1

    total = sum(counts.values())
    if logger:
        logger.info("📊 Class distribution:")
        for i in range(num_classes):
            count = counts.get(i, 0)
            logger.info(f"  Class {i}: {count} samples ({(count / total * 100):.2f}%)")

    weights = [total / counts.get(i, 1) for i in range(num_classes)]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    weights_tensor = weights_tensor / weights_tensor.sum() * num_classes
    return weights_tensor

def is_debugging():
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace is not None and gettrace()

def setup_logging(log_dir: str):
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ])
    return logging.getLogger("train")


def setup_directories(cfg: dict, run_name: str, debug: bool = False):
    log_dir = os.path.join(cfg["log_root"], run_name)
    plot_dir = os.path.join(log_dir, "plots")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not debug:
        # log base paths to MLflow for traceability
        mlflow.log_param("log_dir", log_dir)
        mlflow.log_param("checkpoint_dir", checkpoint_dir)
    
    return log_dir, plot_dir, checkpoint_dir


def main(cfg: dict, debug: bool = False):
    define_seed(42)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir, plot_dir, checkpoint_dir = setup_directories(cfg, run_name, debug)
    logger = setup_logging(log_dir)
    
    ml_logger = None
    if not debug:
        ml_logger = MLflowLogger(
            project_name=cfg["mlflow_project"],
            run_name=run_name,
            config=cfg,
            artifact_dir="src"
        )
    
    logger.info(f"🚀 Starting run: {run_name}")
    logger.info(f"📸 Logging to: {log_dir}")
    logger.info(f"💅 Model: {cfg['pretrained_model']}")
    logger.info(f"👠 Scheduler: {cfg['scheduler_type']}")

    dataloaders = create_dataloaders(cfg)

    model = WavLMEmotionClassifierV5(
        pretrained_model_name=cfg["pretrained_model"],
        num_emotions=len(cfg["emotion_labels"])
    ).to(torch.device(cfg["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = create_scheduler(cfg, optimizer)

    class_weights = compute_class_weights(dataloaders["train"], cfg["emotion_labels"], logger=logger)
    loss_fn = EmotionFocalLossStable(class_weights={"emo": class_weights})

    metrics = create_metrics(cfg)
    metric_manager = MetricManager(metrics)
    evaluator = Evaluator(
        metric_manager=metric_manager,
        label_names={"emo": cfg["emotion_labels"]},
        logger=logger,
        plot_dir=plot_dir,
        ml_logger=ml_logger
    )

    unfreeze_scheduler = GradientBasedUnfreezeScheduler(
        model=model,
        logger=logger,
        layers_to_track=[
            "encoder.layers.6", "encoder.layers.7", "encoder.layers.8",
            "encoder.layers.9", "encoder.layers.10", "encoder.layers.11",
            "projector", "layernorm"
        ],
        max_unfreeze_per_epoch=1,
        temperature_schedule={
            0: 1.5,
            10: 1.0,
            15: 0.7
        }
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=torch.device(cfg["device"]),
        metrics=metrics,
        logger=logger,
        log_dir=log_dir,
        plot_dir=plot_dir,
        checkpoint_dir=checkpoint_dir,
        final_activations={"emo": lambda x: x},
        ml_logger=ml_logger,
        unfreeze_scheduler=unfreeze_scheduler
    )

    early_stopper = EarlyStopping(
        patience=cfg["early_stopping_patience"],
        verbose=True,
        monitor="val_loss",
        delta=cfg["early_stopping_delta"]
    )

    for epoch in range(1, cfg["num_epochs"] + 1):
        trainer.train_epoch(dataloaders["train"], epoch)
        val_result = trainer.validate_epoch(dataloaders["dev"], epoch, evaluator)
        if "test" in dataloaders:
            test_result = trainer.validate_epoch(dataloaders["test"], epoch, evaluator, phase_name="test")
                    
        if early_stopper({"val_loss": val_result["loss"]}, trainer):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        else:
            logger.info(f"[Epoch {epoch}] Wait counter: {early_stopper.patience - early_stopper.counter} epochs left before early stop")

    logger.info("Training complete")
    trainer.save_metrics(os.path.join(log_dir, "metrics.csv"))

    if ml_logger:
        ml_logger.log_artifact(os.path.join(checkpoint_dir, "best_model.pth"))
        ml_logger.end()


if __name__ == "__main__":
    cfg = load_config("audio_config_without_others.yaml")
    debug = is_debugging()
    main(cfg, debug=debug)

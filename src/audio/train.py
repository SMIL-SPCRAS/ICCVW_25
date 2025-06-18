import os
import sys
import logging
from datetime import datetime
from collections import Counter

import torch

sys.path.append('src')

from audio.utils.factories import create_dataloaders, create_scheduler, create_metrics
from audio.models.models import WavLMEmotionClassifier
from audio.trainer.trainer import Trainer
from audio.trainer.early_stopping import EarlyStopping
from audio.trainer.metric_manager import MetricManager
from audio.trainer.evaluator import Evaluator
from audio.utils.utils import load_config, define_seed
from audio.utils.loss import SoftCrossEntropyLoss


def compute_class_weights(dataloader, num_classes):
    counts = Counter()
    for _, labels, _ in dataloader:
        for label in labels["emo"]:
            label_idx = label.argmax().item() if label.ndim >= 1 else label.item()
            counts[label_idx] += 1

    total = sum(counts.values())
    weights = [total / counts.get(i, 1) for i in range(num_classes)]
    weights = torch.tensor(weights, dtype=torch.float)
    weights = weights / weights.sum() * num_classes
    return weights


def setup_logging(log_dir):
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ])
    return logging.getLogger("train")


def setup_directories(cfg, run_name):
    log_dir = os.path.join(cfg["log_root"], run_name)
    plot_dir = os.path.join(log_dir, "plots")
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return log_dir, plot_dir, checkpoint_dir


def main(cfg: dict):
    define_seed(42)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir, plot_dir, checkpoint_dir = setup_directories(cfg, run_name)
    logger = setup_logging(log_dir)

    logger.info(f"Starting run: {run_name}")
    logger.info(f"Model: {cfg['pretrained_model']}, Scheduler: {cfg['scheduler_type']}, Logging to: {log_dir}")

    dataloaders = create_dataloaders(cfg)

    model = WavLMEmotionClassifier(
        pretrained_model_name=cfg["pretrained_model"],
        num_emotions=len(cfg["emotion_labels"])
    ).to(torch.device(cfg["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = create_scheduler(cfg, optimizer)

    class_weights = compute_class_weights(dataloaders["train"], len(cfg["emotion_labels"])).to(cfg["device"])
    loss_fn = SoftCrossEntropyLoss(class_weights={"emo": class_weights})

    metrics = create_metrics(cfg)
    metric_manager = MetricManager(metrics)
    evaluator = Evaluator(
        metric_manager=metric_manager,
        label_names={"emo": cfg["emotion_labels"]},
        logger=logger,
        plot_dir=plot_dir,
        use_wandb=cfg.get("use_wandb", False)
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
        checkpoint_dir=checkpoint_dir,
        final_activations={"emo": torch.nn.Softmax(dim=1)},
        use_wandb=cfg.get("use_wandb", False)
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

        if early_stopper({"val_loss": val_result["loss"]}, trainer):
            logger.info(f"Early stopping at epoch {epoch}")
            break
        else:
            logger.info(f"[Epoch {epoch}] Wait counter: {early_stopper.patience - early_stopper.counter} epochs left before early stop")

    logger.info("Training complete")
    trainer.save_metrics(os.path.join(log_dir, "metrics.csv"))

if __name__ == "__main__":
    cfg = load_config("audio_config.yaml")
    main(cfg)

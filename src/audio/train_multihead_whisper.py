import os
import sys
from datetime import datetime

import torch

sys.path.append('src')

from common.utils.factories import create_scheduler, create_metrics
from common.trainer.trainer import Trainer
from common.trainer.early_stopping import EarlyStopping
from common.trainer.metric_manager import MetricManager
from common.trainer.evaluator import Evaluator
from common.utils.utils import load_config, define_seed, setup_directories, \
    setup_logging, is_debugging, wait_for_it
from common.loss import MultiHeadFocalLoss
from common.mlflow_logger import MLflowLogger

from audio.data.utils import create_dataloaders, compute_class_weights
from audio.data.collate import speech_only_collate_fn
from audio.models.multihead_models import *


def main(cfg: dict[str, any], debug: bool = False) -> None:
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

    dataloaders = create_dataloaders(cfg, cfg["pretrained_model"], collate_fn=speech_only_collate_fn)
    # dataloaders = create_dataloaders(cfg, collate_fn=speech_only_collate_fn)

    model = MultiHeadWhisperEmotionClassifier(
        pretrained_model_name=cfg["pretrained_model"],
        num_emotions=len(cfg["emotion_labels"]),
        num_heads=3,
        max_position=200 # 4 seconds → 200 frames
    ).to(torch.device(cfg["device"]))

    # model = MultiHeadWavLMEmotionClassifier(
    #     pretrained_model_name=cfg["pretrained_model"],
    #     num_emotions=len(cfg["emotion_labels"]),
    #     num_heads=3,
    # ).to(torch.device(cfg["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = create_scheduler(cfg, optimizer)

    class_weights = compute_class_weights(dataloaders["train"], cfg["emotion_labels"], logger=logger)
    loss_fn = MultiHeadFocalLoss(class_weights=class_weights, gamma=2.0)

    metrics = create_metrics(cfg)
    metric_manager = MetricManager(metrics)
    evaluator = Evaluator(
        metric_manager=metric_manager,
        label_names={"emo": cfg["emotion_labels"]},
        logger=logger,
        plot_dir=plot_dir,
        ml_logger=ml_logger
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
        final_activations={"emo": torch.nn.Softmax(dim=-1)},
        ml_logger=ml_logger
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
                    
        if early_stopper(metrics={"val_loss": val_result["loss"]}, trainer=trainer, epoch=epoch):
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
    cfg = load_config("audio_config_multihead.yaml")
    debug = is_debugging()
    main(cfg, debug=debug)

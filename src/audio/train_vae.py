import os
import sys
from datetime import datetime

import torch

sys.path.append('src')

from audio.utils.factories import create_dataloaders, create_scheduler, create_metrics
from audio.models.vae_models import *
from audio.trainer.vae_trainer import Trainer
from audio.trainer.early_stopping import EarlyStopping
from audio.trainer.metric_manager import MetricManager
from audio.trainer.evaluator import Evaluator
from audio.utils.utils import load_config, define_seed, setup_directories, \
    setup_logging, compute_class_weights, is_debugging, wait_for_it
from audio.utils.loss import EmotionNLLLossStable
from audio.utils.mlflow_logger import MLflowLogger
from audio.data.collate import speech_only_collate_fn
from audio.utils.schedulers import UnfreezeScheduler


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

    dataloaders = create_dataloaders(cfg, collate_fn=speech_only_collate_fn)

    model = WavLMEmotionClassifierV5(
        pretrained_model_name=cfg["pretrained_model"],
        num_emotions=len(cfg["emotion_labels"])
    ).to(torch.device(cfg["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = create_scheduler(cfg, optimizer)

    class_weights = compute_class_weights(dataloaders["train"], cfg["emotion_labels"], logger=logger)
    loss_fn = EmotionNLLLossStable(class_weights={"emo": class_weights})

    metrics = create_metrics(cfg)
    metric_manager = MetricManager(metrics)
    evaluator = Evaluator(
        metric_manager=metric_manager,
        label_names={"emo": cfg["emotion_labels"]},
        logger=logger,
        plot_dir=plot_dir,
        ml_logger=ml_logger
    )

    unfreeze_scheduler = UnfreezeScheduler(
        model=model, 
        logger=logger,
        schedule={
            2: ['encoder.layers.6'],
            3: ['encoder.layers.7'],
            4: ['encoder.layers.8'],
            5: ['encoder.layers.9'],
            6: ['encoder.layers.10'],
            7: ['encoder.layers.11'],
            8: ['projector', 'layernorm']
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
    cfg = load_config("audio_config_vae_without_others.yaml")
    debug = is_debugging()
    main(cfg, debug=debug)

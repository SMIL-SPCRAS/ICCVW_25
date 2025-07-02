import os
import sys
import itertools
from datetime import datetime

import torch

sys.path.append('src')

from common.utils.factories import create_scheduler, create_metrics
from common.loss import SoftFocalLoss
from common.mlflow_logger import MLflowLogger
from common.trainer.trainer import Trainer
from common.trainer.early_stopping import EarlyStopping
from common.trainer.metric_manager import MetricManager
from common.trainer.evaluator import Evaluator
from common.utils.utils import load_config, define_seed, setup_directories, \
    setup_logging, is_debugging, wait_for_it
from common.telegram_notifier import TelegramNotifier

from fusion.models.multimodal_models import *
from fusion.data.utils import compute_class_weights, create_dataloaders
from fusion.data.collate import multimodal_collate_fn


def main(cfg: dict[str, any], model_cls: torch.nn.Module, debug: bool = False) -> None:
    define_seed(42)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir, plot_dir, checkpoint_dir = setup_directories(cfg, run_name, debug)
    logger = setup_logging(log_dir)

    telegram_notifier = TelegramNotifier(cfg)

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
    logger.info(f"💅 Modalities: {cfg['modalities']}")
    logger.info(f"💥 Model: {cfg['model_name']}")
    logger.info(f"👠 Scheduler: {cfg['scheduler_type']}")

    dataloaders = create_dataloaders(cfg, collate_fn=multimodal_collate_fn)

    model = model_cls(
        modality_dims=cfg["modalities"],
        num_emotions=len(cfg["emotion_labels"])
    ).to(torch.device(cfg["device"]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = create_scheduler(cfg, optimizer)

    class_weights = compute_class_weights(dataloaders["train"], logger=logger)
    loss_fn = SoftFocalLoss(class_weights={"emo": class_weights}, gamma=2.0, label_smoothing=0.1) #

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
        final_activations={"emo": torch.nn.Softmax(dim=1)},
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

    logger.info("🌋Training complete")
    telegram_notifier.send(f"🧪Experiment {run_name} completed")
    trainer.save_metrics(os.path.join(log_dir, "metrics.csv"))

    if ml_logger:
        ml_logger.log_artifact(os.path.join(checkpoint_dir, "best_model.pth"))
        ml_logger.end()


def run_experiments(cfg: dict[str, any], debug: bool = False) -> None:
    original_modalities = dict(cfg["modalities"])
    cfg["databases"].pop("C-EXPR-DB", None)
    all_modalities = list(cfg["modalities"].keys())
    all_combos = [combo for r in range(2, 7) for combo in itertools.combinations(all_modalities, r)]
    model_classes = {
        "EmotionFusionModelV3": EmotionFusionModelV3,
    }

    for combo in all_combos:        
        combo_dims = {mod: original_modalities[mod] for mod in combo}
        for model_name, model_cls in model_classes.items():
            cfg["modalities"] = combo_dims
            cfg["model_name"] = model_name

            try:
                main(cfg, model_cls, debug=debug)
            except Exception as e:
                print(f"❌ Experiment failed with error: {e}")
                

if __name__ == "__main__":
    # wait_for_it(4 * 60)
    cfg = load_config("multimodal_config.yaml")
    debug = is_debugging()
    run_experiments(cfg, debug)

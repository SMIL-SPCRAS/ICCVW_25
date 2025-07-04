import os
import re
import sys
import pickle
import shutil
from collections import defaultdict

import torch

sys.path.append('src')

from common.utils.utils import load_config, define_seed
from audio.models.models import WavLMEmotionClassifierV4
from audio.models.vae_models import WavLMEmotionClassifierV5
from audio.models.multihead_models import MultiHeadWhisperEmotionClassifier
from common.trainer.trainer import Trainer
from audio.data.utils import create_dataloaders


class DummyLogger:
    def info(self, *args: any, **kwargs: any)-> None:
        print(*args, **kwargs)

    def warning(self, *args: any, **kwargs: any)-> None: pass
    def error(self, *args: any, **kwargs: any)-> None: pass
    def debug(self, *args: any, **kwargs: any)-> None: pass


def main(cfg: dict[str, any], experiment_info: dict[str, any]) -> None:
    define_seed(42)

    logger = DummyLogger()
    logger.info(f"🧪 Running evaluation for saved model")
    
    dataloaders = create_dataloaders(cfg, 
                                     processor_name=cfg['pretrained_model'] if 'whisper' in cfg['pretrained_model'] else None)

    model = experiment_info["model"].to(torch.device(cfg["device"]))

    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        loss_fn=None,
        device=torch.device(cfg["device"]),
        metrics=[],
        logger=logger,
        log_dir='',
        plot_dir='',
        checkpoint_dir='',
        # final_activations={"emo": torch.nn.Softmax(dim=-1)},
        final_activations={"emo": lambda x: x},
        ml_logger=None
    )

    checkpoint_path = os.path.join(cfg["log_root"], experiment_info["experiment_name"], 
                                   "checkpoints", experiment_info["checkpoint_name"])
    state = trainer.load_checkpoint(checkpoint_path)
    logger.info(f"✅ Loaded checkpoint from epoch {state['epoch']} with val_metric {state['val_metric']}")

    out_path = "{0}_{1}_predictions".format(
        experiment_info["experiment_name"],
        re.search(r"(epoch_\d+|best)", experiment_info["checkpoint_name"]).group(1)
    )
    
    os.makedirs(out_path, exist_ok=True)
    for fname in os.listdir(os.path.join(cfg["log_root"], experiment_info["experiment_name"], "plots")):
        match = re.search(r"(epoch_\d+)", experiment_info["checkpoint_name"]).group(1)
        if match in fname:
            src_dir = os.path.join(cfg["log_root"], 
                                   experiment_info["experiment_name"], "plots", fname)
            shutil.copy(src_dir, os.path.join(out_path, fname))

    shutil.copy(os.path.join(cfg["log_root"], experiment_info["experiment_name"], "metrics.csv"), 
                os.path.join(out_path, "metrics.csv"))

    for name, loader in dataloaders.items():
        logger.info(f"🔍 Predicting for {name}...")
        result = trainer.predict(loader, return_features=True)

        predicts = result["predictions"]
        targets = result["targets"]
        metas = result["metas"]
        features = result["features"]

        results_by_db = defaultdict(list)
        for i in range(len(metas)):
            entry = {
                "predictions": {task: predicts[task][i].tolist() for task in predicts},
                "targets": {task: targets[task][i].tolist() for task in targets},
                "features": features[i] if len(features) > 0 else None,
                "metas": metas[i],
            }

            db_name = metas[i]["db"]
            results_by_db[db_name].append(entry)

        for db_name, entries in results_by_db.items():
            subset_path = os.path.join(out_path, f"{name}_{db_name}_audio_predictions.pkl")
            with open(subset_path, "wb") as f:
                pickle.dump(entries, f)
            
            logger.info(f"📁 Done: {name}_{db_name}")


if __name__ == "__main__":
    cfg = load_config("audio_config.yaml")

    # 8 classes, CMU, MELD
    # run_20250624_103834, epoch 8 + 
    
    # 8 classes
    # run_20250626_171326, epoch 5
    # run_20250626_140651, epoch 25 - Uncert Head

    # 7 classes
    # run_20250626_023120, epoch 27 - Uncert Head

    exps = [
        # {
        #     "log_root": "/media/maxim/WesternDigitalNew/9th_ABAW_8classes",
        #     "experiment_name": "run_20250624_103834",
        #     "checkpoint_name": "checkpoint_epoch_8.pt",
        #     "pretrained_model": "microsoft/wavlm-base-plus-sd",
        #     "model": WavLMEmotionClassifierV4(
        #         pretrained_model_name="microsoft/wavlm-base-plus-sd",
        #         num_emotions=len(cfg["emotion_labels"])
        #     )
        # },
        # {
        #     "log_root": "/media/maxim/WesternDigitalNew/9th_ABAW_8classes", # whisper
        #     "experiment_name": "run_20250626_171326",
        #     "checkpoint_name": "checkpoint_epoch_5.pt",
        #     "pretrained_model": "openai/whisper-small",
        #     "model": MultiHeadWhisperEmotionClassifier(
        #         pretrained_model_name="openai/whisper-small",
        #         num_emotions=len(cfg["emotion_labels"]),
        #         num_heads=3,
        #         max_position=200
        #     )
        # },
        {
            "log_root": "/media/maxim/WesternDigitalNew/9th_ABAW_8classes",
            "experiment_name": "run_20250626_140651",
            "checkpoint_name": "checkpoint_epoch_25.pt",
            "pretrained_model": "microsoft/wavlm-base-plus-sd",
            "model": WavLMEmotionClassifierV5(
                pretrained_model_name="microsoft/wavlm-base-plus-sd",
                num_emotions=len(cfg["emotion_labels"]),
            )
        },
        {
            "log_root": "/media/maxim/WesternDigitalNew/9th_ABAW_7classes",
            "experiment_name": "run_20250626_023120",
            "checkpoint_name": "checkpoint_epoch_27.pt",
            "pretrained_model": "microsoft/wavlm-base-plus-sd",
            "model": WavLMEmotionClassifierV5(
                pretrained_model_name="microsoft/wavlm-base-plus-sd",
                num_emotions=len(cfg["emotion_labels"]) - 1,
            )
        }
    ]
    
    for experiment_info in exps:
        cfg["log_root"] = experiment_info["log_root"]
        cfg["pretrained_model"] = experiment_info["pretrained_model"]

        main(cfg, experiment_info)

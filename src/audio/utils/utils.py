import os
import sys
import time
import random
import logging
from collections import Counter

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import mlflow

import seaborn as sns
import matplotlib.pyplot as plt


from audio.utils.mlflow_logger import MLflowLogger


def load_config(config_path: str) -> dict[str, any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def define_seed(seed: int = 12) -> None:
    """Fix seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def wait_for_it(time_left: int) -> None:
    """Wait time in minutes. before run the following statement"""
    t = time_left
    while t > 0:
        print("Time left: {0}".format(t))
        time.sleep(60) 
        t = t - 1


def compute_class_weights(
    dataloader: DataLoader,
    emotion_labels: list[str],
    logger: any = None) -> torch.Tensor:
    """Computes class weights for a multi-class classification task."""
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


def is_debugging() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    return gettrace is not None and gettrace()


def setup_logging(log_dir: str) -> any:
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(os.path.join(log_dir, "train.log")),
        logging.StreamHandler()
    ])
    return logging.getLogger("train")


def setup_directories(cfg: dict[any], run_name: str, debug: bool = False) -> tuple[str, str, str]:
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


def log_logvar(logger: any, ml_logger: MLflowLogger, plot_dir: str, 
               outputs: dict[str, torch.Tensor], 
               epoch: int, phase: str) -> None:
    """Logs average and per-class log-variance with emojis to indicate uncertainty levels."""
    if "logvar" not in outputs:
        return
    
    avg_logvar = outputs["logvar"].mean().item()
    ml_logger.log_metrics({f"{phase}_avg_logvar": avg_logvar}, step=epoch)
    logger.info(f"[Epoch {epoch}] 🧠 {phase.upper()} avg_logvar = {avg_logvar:.3f}")
    
    # Per-class logvar (mean across batch)
    per_class_logvar = outputs["logvar"].mean(dim=0)  # shape: (num_classes,)
    for i, lv in enumerate(per_class_logvar):
        ml_logger.log_metrics({f"{phase}_logvar_class_{i}": lv.item()}, step=epoch)

    # Emojis represent model's uncertainty level per emotion class, based on log-variance (logvar):
    # 😐 (logvar < 0)      → Confident prediction (variance < 1)
    # 😬 (0 ≤ logvar < 0.5) → Moderate uncertainty (variance ≈ 1 to 1.6)
    # 😱 (logvar ≥ 0.5)     → High uncertainty (variance > 1.6)
    class_log_parts = []
    for i, lv in enumerate(per_class_logvar):
        emoji = "😐" if lv < 0 else "😬" if lv < 0.5 else "😱"
        class_log_parts.append(f"cls_{i}: {lv.item():.3f}{emoji}")

    for i in range(0, len(class_log_parts), 4):
        line = " | ".join(class_log_parts[i:i+4])
        logger.info(f"[Epoch {epoch}] 📊 {phase.upper()} logvar → {line}")

    # Optional: also log histogram of mu values
    if "mu" in outputs:
        mu = outputs["mu"].detach().cpu().numpy()
        plt.figure(figsize=(8, 4))
        sns.histplot(mu.flatten(), bins=50, color='dodgerblue', kde=True, stat="density", edgecolor=None)
        plt.title(f"mu distribution at epoch {epoch} ({phase})")
        plt.xlabel("mu value")
        plt.ylabel("density")
        plt.tight_layout()

        save_path = os.path.join(plot_dir, f"mu_hist_{phase}_epoch_{epoch}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

        ml_logger.log_artifact(save_path, artifact_path=f"plots/{phase}/mu_hist_epoch_{epoch}")


def log_mu_statistics(mu: torch.Tensor, 
                      targets: torch.Tensor, 
                      epoch: int, phase: str, logger: any, ml_logger: MLflowLogger = None, 
                      plot_dir: str = None) -> None:
    """Logs mu distribution, top-1 predictions, and per-class mean mu."""
    mu = mu.detach().cpu()
    targets = targets.detach().cpu()
    pred_classes = mu.argmax(dim=-1)
    true_classes = targets.argmax(dim=-1)  # assumes soft labels

    # Count predictions
    pred_counts = torch.bincount(pred_classes, minlength=mu.shape[1]).numpy()
    pred_distribution = pred_counts / pred_counts.sum()

    # Log distribution
    for i, prob in enumerate(pred_distribution):
        logger.info(f"[Epoch {epoch}] 🔢 {phase.upper()} mu-predicted cls_{i}: {prob:.2%}")
        if ml_logger:
            ml_logger.log_metrics({f"{phase}_mu_pred_ratio_class_{i}": prob}, step=epoch)

    # Plot distribution
    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(range(mu.shape[1])), y=pred_distribution, color='royalblue')
    plt.title(f"{phase.upper()} top-1 prediction distribution (Epoch {epoch})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Ratio")
    plt.ylim(0, 1)

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"mu_pred_dist_{phase}_epoch_{epoch}.png")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        if ml_logger:
            ml_logger.log_artifact(save_path, artifact_path=f"plots/{phase}/mu_debug_epoch_{epoch}")
            ml_logger.log_artifact(save_path, artifact_path=f"plots/{phase}/mu_hist_epoch_{epoch}")




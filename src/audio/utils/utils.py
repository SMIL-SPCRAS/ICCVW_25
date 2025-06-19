import os
import sys
import time
import random
import logging
from typing import Dict, Any

import yaml
import torch
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def define_seed(seed: int = 12) -> None:
    """Fix seed for reproducibility

    Args:
        seed (int, optional): seed value. Defaults to 12.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
def wait_for_it(time_left: int) -> None:
    """Wait time in minutes. before run the following statement

    Args:
        time_left (int): time in minutes
    """
    t = time_left
    while t > 0:
        print("Time left: {0}".format(t))
        time.sleep(60) 
        t = t - 1


def log_logvar(logger, ml_logger, outputs, epoch, phase):
    """
    Logs average and per-class log-variance with emojis to indicate uncertainty levels.
    """
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

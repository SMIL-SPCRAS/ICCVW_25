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
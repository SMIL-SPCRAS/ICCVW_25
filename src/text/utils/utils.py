import yaml
import numpy as np
import torch
import random

def load_config(config_path: str) -> dict[str, any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def to_single_label(labels):
        return [np.argmax(label) for label in labels]  # converts one-hot/multi-hot into class index

# function to fixate seed
def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(dataloader, num_classes, device):
    all_indices = []
    for batch in dataloader:
        lbl = batch['labels']
        idx = lbl.cpu().numpy()
        all_indices.append(idx)
    all_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=int)

    if all_indices.size > 0:
        class_counts = np.bincount(all_indices, minlength=num_classes)
        total = len(all_indices)
        weights = torch.tensor((total / (class_counts + 1e-6)), dtype=torch.float32).to(device)
        return weights
    else:
        return None

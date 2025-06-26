import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset, default_collate

from audio.data.dataloaders import AudioEmotionDataset
from audio.utils.metrics import MacroF1, UAR


def create_dataloaders(cfg: dict[str, any], processor_name: str = None, 
                       collate_fn: callable = default_collate) -> dict[str, DataLoader]:
    """Create train/dev/test dataloaders from config."""
    datasets = defaultdict(list)

    for db_name, subsets in cfg["databases"].items():
        for subset, metadata_filename in subsets.items():
            csv_path = os.path.join(cfg["output_audio_dir"], db_name, metadata_filename)
            dataset = AudioEmotionDataset(
                csv_path=csv_path,
                audio_dir=os.path.join(cfg["output_audio_dir"], db_name, subset),
                db=db_name,
                emotion_labels=cfg["emotion_labels"],
                sample_rate=cfg["sample_rate"],
                max_length=cfg["max_length"],
                processor_name=processor_name
            )

            datasets[subset].append(dataset)

    dataloaders = {
       subset: DataLoader(
            ConcatDataset(ds_list),
            batch_size=cfg["batch_size"],
            shuffle=(subset == "train"),
            num_workers=cfg["num_workers"],
            collate_fn=collate_fn
        )
        for subset, ds_list in datasets.items()
    }
    
    return dataloaders


def create_scheduler(
        cfg: dict[str, any],
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
    """Factory method to create a learning rate scheduler."""
    scheduler_type = cfg["scheduler_type"]
    scheduler_params = cfg["scheduler_params"]

    schedulers = {
        "ReduceLROnPlateau": lambda opt, params: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, **params),
        "CosineAnnealingLR": lambda opt, params: torch.optim.lr_scheduler.CosineAnnealingLR(opt, **params),
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type](optimizer, scheduler_params)


def create_metrics(cfg: dict[str, any], task: str = "emo") -> list[any]:
    """Create a list of metrics based on config."""
    metric_names = cfg.get("metrics", ["UAR", "MacroF1"])
    available = {
        "UAR": UAR,
        "MacroF1": MacroF1,
    }
    
    return [available[name](task=task) for name in metric_names if name in available]



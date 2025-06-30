from collections import defaultdict

import torch
from torch.utils.data import DataLoader, ConcatDataset, default_collate
from fusion.data.dataloaders import MultimodalEmotionDataset


def create_dataloaders(cfg: dict[str, any],
                       collate_fn: callable = default_collate) -> dict[str, DataLoader]:
    """Create train/dev/test dataloaders from config."""
    datasets = defaultdict(list)

    for db_name, subsets in cfg["databases"].items():
        for idx, subset in enumerate(subsets):
            dataset = MultimodalEmotionDataset(
                features_dir=cfg["features_dir"],
                db=db_name,
                subset=subset,
                modalities=list(cfg["modalities"].keys())
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


def compute_class_weights(
    dataloader: torch.utils.data.DataLoader,
    logger: any = None
) -> torch.Tensor:
    """Computes class weights from dataset.class_counts without needing num_classes."""
    dataset = dataloader.dataset

    if isinstance(dataset, ConcatDataset):
        class_counts = None
        for subds in dataset.datasets:
            if hasattr(subds, "class_counts"):
                if class_counts is None:
                    class_counts = subds.class_counts.clone()
                else:
                    class_counts += subds.class_counts
            else:
                raise AttributeError("Sub-dataset missing 'class_counts'")
    elif hasattr(dataset, "class_counts"):
        class_counts = dataset.class_counts
    else:
        raise AttributeError("Dataset missing 'class_counts' attribute")

    total = class_counts.sum().item()
    num_classes = len(class_counts)

    if logger:
        logger.info("📊 Class distribution (from class_counts):")
        for i in range(num_classes):
            count = int(class_counts[i].item())
            logger.info(f"  Class {i}: {count} samples ({(count / total * 100):.2f}%)")

    weights = total / (class_counts + 1e-6)
    weights = weights / weights.mean()  # normalize to mean=1

    return weights

import os
from collections import defaultdict, Counter

import torch
from torch.utils.data import DataLoader, ConcatDataset, default_collate

from audio.data.dataloaders import AudioEmotionDataset


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

import torch
from torch.utils.data import default_collate


def speech_only_collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]]) \
    -> list[tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, any]]]:
    """
    Collate function that filters out samples without speech (based on metadata).
    """
    filtered_batch = [b for b in batch if b[2].get("has_speech", True)]
    return default_collate(filtered_batch if filtered_batch else batch)
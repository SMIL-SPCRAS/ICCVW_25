from torch.utils.data import default_collate
from typing import Callable, Any, List, Tuple, Dict


def speech_only_collate_fn(batch: List[Tuple[Any, Dict[str, Any], Dict[str, Any]]]) -> Any:
    """
    Collate function that filters out samples without speech (based on metadata).
    """
    filtered_batch = [b for b in batch if b[2].get("has_speech", True)]
    return default_collate(filtered_batch if filtered_batch else batch)
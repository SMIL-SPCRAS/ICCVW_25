import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset
from collections import defaultdict
from transformers import AutoTokenizer, CLIPTokenizer

from utils.clean_whisper import clean
from utils.extract_labels import extract_labels
from utils.utils import to_single_label
from data.dataloaders import TextDataset


def load_text_dataset(path, ds_label=None):
    df = pd.read_csv(path)
    df = df[df["text"].notna()].dropna().reset_index(drop=True)
    texts = [clean(str(t)) for t in df["text"].tolist()]
    labels = extract_labels(df)
    ds_labels = [ds_label] * len(texts) if ds_label else None
    return texts, labels, ds_labels


def create_text_dataloaders(cfg: dict, tokenizer=None, collate_fn=None):
    """
    Create DataLoaders from YAML-style config.

    Args:
        cfg (dict): Configuration dictionary.
        tokenizer: Optional tokenizer; otherwise initialized from model name.
        collate_fn: Optional custom collation function.

    Returns:
        dict[str, DataLoader]: Dataloaders for 'train' and 'eval'.
    """
    model_name = cfg["model"]["name"]
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 0)

    if tokenizer is None:
        if "clip" in model_name.lower():
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 77 if "clip" in model_name.lower() else (1024 if "jina" in model_name.lower() else None)

    datasets = defaultdict(list)

    for subset_name in ["train", "eval"]:
        subset_info = cfg["datasets"].get(subset_name, [])
        all_texts, all_labels, all_ds_labels = [], [], []

        for entry in subset_info:
            texts, labels, ds_labels = load_text_dataset(entry["csv"], ds_label=entry.get("ds_label"))
            all_texts.extend(texts)
            all_labels.extend(labels)
            if ds_labels:
                all_ds_labels.extend(ds_labels)

        # Filter out all-zero labels
        filtered = [
            (t, l, d if all_ds_labels else None)
            for t, l, d in zip(all_texts, all_labels, all_ds_labels or [None] * len(all_labels))
            if not all(v == 0 for v in l)
        ]

        if not filtered:
            continue

        texts, labels, ds_labels = zip(*filtered)
        labels = to_single_label(labels)

        tokenized = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=max_length)

        dataset = TextDataset(
            tokenized_texts=tokenized,
            labels=labels,
            ds_labels=list(ds_labels) if ds_labels[0] is not None else None
        )
        datasets[subset_name].append(dataset)

    dataloaders = {
        subset: DataLoader(
            ConcatDataset(ds_list),
            batch_size=batch_size,
            shuffle=(subset == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        for subset, ds_list in datasets.items()
    }

    return dataloaders
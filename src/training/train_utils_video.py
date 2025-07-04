# coding: utf-8
# train_utils.py

import torch
import logging
import random
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from typing import Type
import os

from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

from utils.losses import WeightedCrossEntropyLoss
from utils.measures import uar, war, mf1, wf1
from models.models import (VideoFormer,VideoMamba
)
from utils.schedulers import SmartScheduler
from data_loading.dataset_multimodal import DatasetVideo
from sklearn.utils.class_weight import compute_class_weight

def custom_collate_fn(batch):
    """Combines list of samples into single batch, dropping None (invalid) samples."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    videos = [b["video"] for b in batch]
    video_tensor = pad_sequence(videos, batch_first=True)

    return {
        "video": video_tensor,
        "label": label_tensor,
    }

def get_class_weights_from_loader(train_loader, num_classes):
    """
    Calculates class weights from train_loader, handles missing classes.
    If any class is missing from the sample, it gets weight 0.0.

    :param train_loader: DataLoader with one-hot labels
    :param num_classes: Total number of classes
    :return: np.ndarray of weights with length num_classes
    """
    all_labels = []
    for batch in train_loader:
        if batch is None:
            continue
        all_labels.extend(batch["label"].argmax(dim=1).tolist())

    if not all_labels:
        raise ValueError("No labels in train_loader to calculate class weights.")

    present_classes = np.unique(all_labels)

    if len(present_classes) < num_classes:
        missing = set(range(num_classes)) - set(present_classes)
        logging.info(f"[!] Missing labels for classes: {sorted(missing)}")

    # Calculate weights only for present classes
    weights_partial = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=all_labels
    )

    # Build full weights vector
    full_weights = np.zeros(num_classes, dtype=np.float32)
    for cls, w in zip(present_classes, weights_partial):
        full_weights[cls] = w

    return full_weights

def make_dataset_and_loader(config, split: str, image_feature_extractor: Type = None, only_dataset: str = None):
    """
    Universal function: combines datasets or returns single one if only_dataset specified.
    When combining train datasets - uses WeightedRandomSampler for balancing.
    """
    datasets = []

    if not hasattr(config, "datasets") or not config.datasets:
        raise ValueError("⛔ Config missing [datasets] section.")

    for dataset_name, dataset_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        csv_path = dataset_cfg["csv_path"].format(base_dir=dataset_cfg["base_dir"], task=dataset_cfg["task"], split=split)
        video_dir = dataset_cfg["video_dir"].format(base_dir=dataset_cfg["base_dir"], task=dataset_cfg["task"], split=split)
        task = dataset_cfg["task"]

        logging.info(f"[{dataset_name.upper()}], Task={task}, Split={split}: CSV={csv_path}, Video_DIR={video_dir}")

        dataset = DatasetVideo(
            csv_path=csv_path, 
            video_dir=video_dir, 
            config=config, 
            split=split, 
            image_feature_extractor=image_feature_extractor,
            dataset_name=dataset_name,
            task=task)

        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ No suitable datasets found for split='{split}'.")

    if len(datasets) == 1:
        full_dataset = datasets[0]
        loader = DataLoader(
            full_dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
    else:
        # Multiple datasets - calculate weights
        lengths = [len(d) for d in datasets]
        total = sum(lengths)

        logging.info(f"[!] Combining {len(datasets)} datasets: {lengths} (total={total})")

        weights = []
        for d_len in lengths:
            w = 1.0 / d_len
            weights += [w] * d_len
            logging.info(f"  ➜ Samples from dataset with {d_len} examples get weight {w:.6f}")

        full_dataset = ConcatDataset(datasets)

        if split == "train":
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )
        else:
            loader = DataLoader(
                full_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                collate_fn=custom_collate_fn
            )

    return full_dataset, loader

def run_eval(model, loader, criterion, device="cuda"):
    """
    Evaluates model on loader. Returns (loss, uar, war, mf1, wf1).
    """
    model.eval()
    total_loss = 0.0
    total_preds = []
    total_targets = []
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            labels = batch["label"].to(device)
            target = labels.argmax(dim=1)
            video = batch["video"].to(device)

            logits = model(video)
            loss = criterion(logits, target)
            bs = video.shape[0]
            total_loss += loss.item() * bs
            total += bs

            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())

    avg_loss = total_loss / total

    uar_m = uar(total_targets, total_preds)
    war_m = war(total_targets, total_preds)
    mf1_m = mf1(total_targets, total_preds)
    wf1_m = wf1(total_targets, total_preds)

    return avg_loss, uar_m, war_m, mf1_m, wf1_m

def train_once(config, train_loader, dev_loaders, test_loaders, metrics_csv_path=None):
    """
    Training logic (train/dev/test).
    Returns best metric on dev and metrics dictionary.
    """

    logging.info("== Starting training (train/dev/test) ==")

    checkpoint_dir = None
    if config.save_best_model:
        checkpoint_dir = f"{metrics_csv_path[:-4]}_timestamp"
        os.makedirs(checkpoint_dir, exist_ok=True)

    csv_writer = None
    csv_file = None

    if config.path_to_df_ls:
        df_ls = pd.read_csv(config.path_to_df_ls)

    if metrics_csv_path:
        csv_file = open(metrics_csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["split", "epoch", "dataset", "loss", "uar", "war", "mf1", "wf1", "mean"])

    # Seed
    if config.random_seed > 0:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(config.random_seed)
        generator = torch.Generator()
        generator.manual_seed(config.random_seed)
        logging.info(f"== Setting random seed: {config.random_seed}")
    else:
        logging.info("== Random seed not set (0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Parameters
    num_classes = len(config.emotion_columns)
    weight_decay = config.weight_decay
    momentum = config.momentum
    lr = config.lr
    num_epochs = config.num_epochs
    max_patience = config.max_patience
    scheduler_type = config.scheduler_type

    dict_models = {
        "VideoFormer": VideoFormer,
        "VideoMamba": VideoMamba,
    }

    model_cls = dict_models[config.model_name]
    model_name = config.model_name.lower()

    if model_name == 'videoformer':
        model = model_cls(
            input_dim=config.image_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_transformer_heads=config.num_transformer_heads,
            dropout=config.dropout,
            positional_encoding=config.positional_encoding,
            out_features=config.out_features,
            seg_len=config.counter_need_frames,
            tr_layer_number=config.tr_layer_number,
            num_classes=num_classes
        ).to(device)

    elif model_name == 'videomamba':
        model = model_cls(
            input_dim=config.image_embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            mamba_d_state=config.mamba_d_state,
            mamba_ker_size=config.mamba_ker_size,
            mamba_layer_number=config.mamba_layer_number,
            out_features=config.out_features,
            seg_len=config.counter_need_frames,
            num_classes=num_classes,
            device=device,
        ).to(device)

    # Optimizer and loss
    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum
        )
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"⛔ Unknown optimizer: {config.optimizer}")

    logging.info(f"Using optimizer: {config.optimizer}, learning rate: {lr}")

    class_weights = get_class_weights_from_loader(train_loader, num_classes)
    criterion = WeightedCrossEntropyLoss(class_weights)

    logging.info("Class weights: " + ", ".join(f"{name}={weight:.4f}" for name, weight in zip(config.emotion_columns, class_weights)))

    # LR Scheduler
    steps_per_epoch = sum(1 for batch in train_loader if batch is not None)
    scheduler = SmartScheduler(
        scheduler_type=scheduler_type,
        optimizer=optimizer,
        config=config,
        steps_per_epoch=steps_per_epoch
    )

    # Early stopping on dev
    best_dev_mean = float("-inf")
    best_dev_metrics = {}
    patience_counter = 0

    for epoch in range(num_epochs):
        logging.info(f"\n=== Epoch {epoch} ===")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds = []
        total_targets = []

        for batch in tqdm(train_loader):
            if batch is None:
                continue
            
            labels = batch["label"].to(device)
            video = batch["video"].to(device)
            logits = model(video)
            target = labels.argmax(dim=1)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # For One cycle or Hugging Face schedulers
            scheduler.step(batch_level=True)

            bs = video.shape[0]
            total_loss += loss.item() * bs

            preds = logits.argmax(dim=1)
            total_preds.extend(preds.cpu().numpy().tolist())
            total_targets.extend(target.cpu().numpy().tolist())
            total_samples += bs

        train_loss = total_loss / total_samples
        uar_m = uar(total_targets, total_preds)
        war_m = war(total_targets, total_preds)
        mf1_m = mf1(total_targets, total_preds)
        wf1_m = wf1(total_targets, total_preds)
        mean_train = np.mean([uar_m, war_m, mf1_m, wf1_m])

        logging.info(
            f"[TRAIN] Loss={train_loss:.4f}, UAR={uar_m:.4f}, WAR={war_m:.4f}, "
            f"MF1={mf1_m:.4f}, WF1={wf1_m:.4f}, MEAN={mean_train:.4f}"
        )

        # --- DEV ---
        dev_means = []
        dev_metrics_by_dataset = []

        for name, loader in dev_loaders:
            d_loss, d_uar, d_war, d_mf1, d_wf1 = run_eval(
                model, loader, criterion, device
            )
            d_mean = np.mean([d_uar, d_war, d_mf1, d_wf1])
            dev_means.append(d_mean)

            if csv_writer:
                csv_writer.writerow(["dev", epoch, name, d_loss, d_uar, d_war, d_mf1, d_wf1, d_mean])

            logging.info(
                f"[DEV:{name}] Loss={d_loss:.4f}, UAR={d_uar:.4f}, WAR={d_war:.4f}, "
                f"MF1={d_mf1:.4f}, WF1={d_wf1:.4f}, MEAN={d_mean:.4f}"
            )

            dev_metrics_by_dataset.append({
                "name": name,
                "loss": d_loss,
                "uar": d_uar,
                "war": d_war,
                "mf1": d_mf1,
                "wf1": d_wf1,
                "mean": d_mean,
            })

        mean_dev = np.mean(dev_means)

        # --- TEST ---
        test_means = []
        test_metrics_by_dataset = []

        for name, loader in test_loaders:
            t_loss, t_uar, t_war, t_mf1, t_wf1 = run_eval(
                model, loader, criterion, device
            )
            t_mean = np.mean([t_uar, t_war, t_mf1, t_wf1])
            test_means.append(t_mean)
            logging.info(
                f"[TEST:{name}] Loss={t_loss:.4f}, UAR={t_uar:.4f}, WAR={t_war:.4f}, "
                f"MF1={t_mf1:.4f}, WF1={t_wf1:.4f}, MEAN={t_mean:.4f}"
            )

            test_metrics_by_dataset.append({
                "name": name,
                "loss": t_loss,
                "uar": t_uar,
                "war": t_war,
                "mf1": t_mf1,
                "wf1": t_wf1,
                "mean": t_mean,
            })

            if csv_writer:
                csv_writer.writerow(["test", epoch, name, t_loss, t_uar, t_war, t_mf1, t_wf1, t_mean])

        
        mean_test = np.mean(test_means)

        if config.opt_set == "test":
            scheduler.step(mean_test)
            mean_target = mean_test
        else:
            scheduler.step(mean_dev)
            mean_target = mean_dev

        if mean_target > best_dev_mean:
            best_dev_mean = mean_target
            patience_counter = 0

            if config.save_best_model:
                model_path = os.path.join(checkpoint_dir, f"best_model_dev.pt")
                torch.save(model.state_dict(), model_path)
                logging.info(f"💾 Model saved with best dev (epoch {epoch}): {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logging.info(f"Early stopping: {max_patience} epochs without improvement.")
                break

    logging.info("Training completed. All splits processed!")

    if csv_file:
        csv_file.close()
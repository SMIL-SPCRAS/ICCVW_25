# train.py
# coding: utf-8
import logging
import os
import shutil
import datetime
# import whisper
import toml
# os.environ["HF_HOME"] = "models"

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from training.train_utils_video import (
    make_dataset_and_loader,
    train_once
)
from data_loading.feature_extractor import PretrainedImageEmbeddingExtractor

def main():
    # Load config file
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = f"results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # Configure logging
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)

    # Load config
    base_config.show_config()

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    # File where greedy search will write its results
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix = os.path.join(epochlog_dir, "metrics_epochlog")

    audio_feature_extractor = None
    text_feature_extractor = None
    image_feature_extractor = PretrainedImageEmbeddingExtractor(base_config)

    # Initialize Whisper model once
    logging.info(f"Initializing Whisper: model={base_config.whisper_model}, device={base_config.whisper_device}")

    # Create datasets/loaders
    # Common train_loader
    if "affwild2" in base_config.datasets:
        _, train_loader = make_dataset_and_loader(base_config, "train", image_feature_extractor)

    # Separate dev/test loaders
    dev_loaders = []
    test_loaders = []

    for dataset_name in base_config.datasets:
        if dataset_name in ["affwild2", "afew"]:
            _, dev_loader = make_dataset_and_loader(base_config, "dev", image_feature_extractor, only_dataset=dataset_name)
            if os.path.exists(base_config.datasets[dataset_name]["csv_path"].format(base_dir=base_config.datasets[dataset_name]["base_dir"], task=base_config.datasets[dataset_name]["task"], split="test")):
                _, test_loader = make_dataset_and_loader(base_config, "test", image_feature_extractor, only_dataset=dataset_name)
            else:
                test_loader = dev_loader
            
            dev_loaders.append((dataset_name, dev_loader))
            test_loaders.append((dataset_name, test_loader))

        elif dataset_name in ["c_expr_db"]:
                _, test_loader = make_dataset_and_loader(base_config, "test", image_feature_extractor, only_dataset=dataset_name)
                test_loaders.append((dataset_name, test_loader))

    if base_config.prepare_only:
        logging.info("== prepare_only mode: data preparation only, no training ==")
        return

    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config       = base_config,
            train_loader      = train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            default_values    = default_values,
            csv_prefix        = csv_prefix
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config       = base_config,
            train_loader      = train_loader,
            dev_loader        = dev_loaders,
            test_loader       = test_loaders,
            train_fn          = train_once,
            overrides_file    = overrides_file,
            param_grid        = param_grid,
            csv_prefix        = csv_prefix
        )

    elif base_config.search_type == "none":
        logging.info("== Single training mode (no parameter search) ==")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

        train_once(
            config           = base_config,
            train_loader     = train_loader,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
            metrics_csv_path = csv_file_path
        )

    else:
        raise ValueError(f"⛔️ Invalid search_type in config: '{base_config.search_type}'. Use 'greedy', 'exhaustive' or 'none'.")


if __name__ == "__main__":
    main()
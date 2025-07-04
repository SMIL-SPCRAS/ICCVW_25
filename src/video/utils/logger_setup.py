# utils/logger_setup.py

import logging
from colorlog import ColoredFormatter

def setup_logger(level=logging.INFO, log_file=None):
    """
    Configures the root logger for colored console output and 
    (optionally) writing to a file.

    :param level: Logging level (e.g., logging.DEBUG)
    :param log_file: Path to log file (if not None, logs will be written to this file)
    """
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with colorlog
    console_handler = logging.StreamHandler()
    log_format = (
        "%(log_color)s%(asctime)s [%(levelname)s]%(reset)s %(blue)s%(message)s"
    )
    console_formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # If log_file is specified, add file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    return logger
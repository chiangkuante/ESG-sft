"""
Unified logging configuration for the ESG 10-K pipeline.
"""
import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "10k_1A")


def setup_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Create a logger with console (INFO) + file (DEBUG) handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        log_file: Log file name, placed under data/10k_1A/.

    Returns:
        Configured logger instance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    # Console handler — INFO level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # File handler — DEBUG level
    fh = logging.FileHandler(os.path.join(LOG_DIR, log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

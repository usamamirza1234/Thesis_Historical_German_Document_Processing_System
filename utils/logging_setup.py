import logging
import sys
from typing import Optional

from config.settings import LoggingConfig


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger('document_processor')
    logger.setLevel(getattr(logging, config.level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if config.file_path:
        file_handler = logging.FileHandler(config.file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline"""
    dpi: int = 300
    ocr_language: str = 'deu_frak'
    enable_white_space_removal: bool = True
    confidence_threshold: float = 0.7
    output_directory: str = "output_dir"
    enable_caching: bool = True
    max_workers: int = 4
    enable_debug: bool = False

    # OCR specific settings
    tesseract_config: str = ""
    image_scale_factor: float = 2.5
    white_threshold: int = 240
    min_content_area: int = 1000

    # Pattern matching settings
    fuzzy_match_threshold: float = 0.8
    date_confidence_bonus: float = 0.9


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
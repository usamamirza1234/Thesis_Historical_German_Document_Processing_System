from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline"""
    dpi: int = 300
    confidence_threshold: float = 0.7
    output_directory: str = "output_dir"
    enable_caching: bool = True
    max_workers: int = 4
    enable_debug: bool = True

    # OCR specific settings - THESE WERE MISSING!
    ocr_language: str = 'deu_frak'  # Primary OCR language
    tesseract_config: str = ""  # Tesseract configuration string

    # Image processing settings
    enable_white_space_removal: bool = True
    image_scale_factor: float = 2.5
    white_threshold: int = 240
    min_content_area: int = 1000

    # Pattern matching settings
    fuzzy_match_threshold: float = 0.8
    date_confidence_bonus: float = 0.9

    # Model configurations (for ML components)
    model_configs: Dict = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
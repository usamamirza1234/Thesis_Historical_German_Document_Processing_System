from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
# ===================================================================
# CONFIGURATION SYSTEM
# ===================================================================

@dataclass
class ProcessingConfig:
    """Complete configuration for the processing system"""

    # Basic settings
    dpi: int = 300
    confidence_threshold: float = 0.7
    output_directory: str = "output_smart_ocr"
    enable_debug: bool = False

    # OCR settings
    ocr_language: str = 'deu'
    tesseract_config: str = ""
    fallback_languages: List[str] = field(default_factory=lambda: ['deu', 'deu_frak'])

    # Smart processing settings
    enable_smart_analysis: bool = True
    enable_interactive_mode: bool = True
    auto_select_best_approach: bool = True
    max_approaches_to_try: int = 6
    early_exit_confidence: float = 0.8
    early_exit_min_length: int = 100

    # Approach preferences
    preferred_approaches: List[str] = field(default_factory=list)
    excluded_approaches: List[str] = field(default_factory=list)

    # Performance settings
    processing_timeout_seconds: int = 300
    enable_parallel_processing: bool = False
    max_workers: int = 4

    # Output settings
    save_intermediate_images: bool = False
    save_analysis_results: bool = True
    save_detailed_logs: bool = True

    def __post_init__(self):
        """Validate and setup configuration"""
        os.makedirs(self.output_directory, exist_ok=True)
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))

        if self.enable_debug:
            self.save_intermediate_images = True
            self.save_analysis_results = True

    @classmethod
    def for_fraktur_documents(cls, output_dir: str = "output_fraktur") -> 'ProcessingConfig':
        """Configuration optimized for Fraktur documents"""
        return cls(
            ocr_language='deu_frak',
            preferred_approaches=["fraktur_traditional", "fraktur_enhanced", "mixed_period"],
            confidence_threshold=0.6,
            output_directory=output_dir,
            enable_debug=True
        )

    @classmethod
    def for_modern_documents(cls, output_dir: str = "output_modern") -> 'ProcessingConfig':
        """Configuration optimized for modern documents"""
        return cls(
            ocr_language='deu',
            preferred_approaches=["modern_german", "official_document", "light"],
            confidence_threshold=0.75,
            output_directory=output_dir
        )

    @classmethod
    def for_quick_processing(cls, output_dir: str = "output_quick") -> 'ProcessingConfig':
        """Configuration for quick processing"""
        return cls(
            enable_interactive_mode=False,
            max_approaches_to_try=3,
            preferred_approaches=["modern_german", "light", "direct"],
            confidence_threshold=0.65,
            output_directory=output_dir,
            enable_debug=False
        )

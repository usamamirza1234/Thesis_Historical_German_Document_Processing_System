from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from enum import Enum

class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    RESEARCH = "research"  # Maximum accuracy, all approaches
    PRODUCTION = "production"  # Balanced speed/accuracy
    ARCHIVE = "archive"  # High-volume processing
    INTERACTIVE = "interactive"  # User-guided processing


class DocumentEra(Enum):
    """Historical document eras with different characteristics"""
    WEIMAR_REPUBLIC = "1920-1933"
    NAZI_PERIOD = "1933-1945"
    EARLY_BRD = "1945-1970"
    MODERN_BRD = "1970-1990"
    CONTEMPORARY = "1990-2025"


@dataclass
class ProcessingConfig:
    """Enhanced configuration for the intelligent processing system"""

    # Core processing settings
    dpi: int = 300
    confidence_threshold: float = 0.7
    output_directory: str = "output_smart_ocr"
    enable_debug: bool = False
    processing_mode: ProcessingMode = ProcessingMode.INTERACTIVE

    # OCR engine configuration
    ocr_language: str = 'deu'
    tesseract_config: str = ""
    fallback_languages: List[str] = field(default_factory=lambda: ['deu', 'deu_frak'])
    enable_transkribus: bool = False
    transkribus_model_id: Optional[str] = None

    # Smart processing settings
    enable_smart_analysis: bool = True
    enable_interactive_mode: bool = True
    auto_select_best_approach: bool = True
    max_approaches_to_try: int = 6
    early_exit_confidence: float = 0.8
    early_exit_min_length: int = 100

    # Approach preferences by document era
    era_preferences: Dict[DocumentEra, List[str]] = field(default_factory=dict)
    excluded_approaches: List[str] = field(default_factory=list)

    # Performance optimization
    processing_timeout_seconds: int = 300
    enable_parallel_processing: bool = False
    max_workers: int = 4
    memory_limit_mb: int = 2048

    # Enhanced output settings
    save_intermediate_images: bool = False
    save_analysis_results: bool = True
    save_detailed_logs: bool = True
    save_confidence_maps: bool = False
    generate_processing_report: bool = True

    # Multi-page processing
    enable_cross_page_analysis: bool = True
    metadata_consolidation_strategy: str = "confidence_weighted"
    max_pages_to_process: Optional[int] = None

    # Evaluation and research settings
    enable_ground_truth_comparison: bool = False
    ground_truth_directory: Optional[str] = None
    save_extraction_metrics: bool = False

    def __post_init__(self):
        """Validate and setup configuration"""
        os.makedirs(self.output_directory, exist_ok=True)
        self.confidence_threshold = max(0.0, min(1.0, self.confidence_threshold))

        # Set era preferences if not provided
        if not self.era_preferences:
            self.era_preferences = {
                DocumentEra.WEIMAR_REPUBLIC: ["fraktur_traditional", "fraktur_enhanced", "mixed_period"],
                DocumentEra.NAZI_PERIOD: ["fraktur_enhanced", "official_document", "mixed_period"],
                DocumentEra.EARLY_BRD: ["mixed_period", "official_document", "modern_german"],
                DocumentEra.MODERN_BRD: ["official_document", "modern_german", "standard"],
                DocumentEra.CONTEMPORARY: ["modern_german", "official_document", "light"]
            }

        if self.enable_debug:
            self.save_intermediate_images = True
            self.save_analysis_results = True
            self.save_detailed_logs = True

        # Adjust settings based on processing mode
        self._configure_for_mode()

    def _configure_for_mode(self):
        """Configure settings based on processing mode"""
        if self.processing_mode == ProcessingMode.RESEARCH:
            self.max_approaches_to_try = 8
            self.enable_debug = True
            self.save_confidence_maps = True
            self.enable_ground_truth_comparison = True

        elif self.processing_mode == ProcessingMode.PRODUCTION:
            self.max_approaches_to_try = 4
            self.early_exit_confidence = 0.75
            self.enable_parallel_processing = True
            self.save_intermediate_images = False

        elif self.processing_mode == ProcessingMode.ARCHIVE:
            self.max_approaches_to_try = 3
            self.early_exit_confidence = 0.7
            self.enable_parallel_processing = True
            self.max_workers = 8
            self.enable_interactive_mode = False

        elif self.processing_mode == ProcessingMode.INTERACTIVE:
            self.enable_interactive_mode = True
            self.save_intermediate_images = True

    @classmethod
    def for_historical_era(cls, era: DocumentEra, output_dir: str = None) -> 'ProcessingConfig':
        """Configuration optimized for specific historical eras"""
        if output_dir is None:
            output_dir = f"output_{era.value.replace('-', '_')}"

        config = cls(output_directory=output_dir)

        # Era-specific optimizations
        if era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
            config.ocr_language = 'deu_frak'
            config.confidence_threshold = 0.6
            config.max_approaches_to_try = 8

        elif era == DocumentEra.EARLY_BRD:
            config.confidence_threshold = 0.65
            config.max_approaches_to_try = 6

        else:  # Modern eras
            config.confidence_threshold = 0.75
            config.max_approaches_to_try = 4

        return config

    @classmethod
    def for_research_evaluation(cls, output_dir: str = "output_research") -> 'ProcessingConfig':
        """Configuration for research and evaluation"""
        return cls(
            processing_mode=ProcessingMode.RESEARCH,
            output_directory=output_dir,
            enable_debug=True,
            save_confidence_maps=True,
            enable_ground_truth_comparison=True,
            save_extraction_metrics=True,
            max_approaches_to_try=10,
            confidence_threshold=0.5  # Lower threshold to capture more data
        )

    @classmethod
    def for_archive_batch_processing(cls, output_dir: str = "output_batch") -> 'ProcessingConfig':
        """Configuration for high-volume archive processing"""
        return cls(
            processing_mode=ProcessingMode.ARCHIVE,
            output_directory=output_dir,
            enable_interactive_mode=False,
            enable_parallel_processing=True,
            max_workers=8,
            max_approaches_to_try=3,
            early_exit_confidence=0.7,
            save_intermediate_images=False,
            processing_timeout_seconds=120
        )

    def get_preferred_approaches_for_era(self, era: DocumentEra) -> List[str]:
        """Get preferred processing approaches for a specific era"""
        return self.era_preferences.get(era, ["modern_german", "standard", "light"])

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization"""
        return {
            'processing_mode': self.processing_mode.value,
            'dpi': self.dpi,
            'confidence_threshold': self.confidence_threshold,
            'ocr_language': self.ocr_language,
            'max_approaches_to_try': self.max_approaches_to_try,
            'early_exit_confidence': self.early_exit_confidence,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_workers': self.max_workers
        }
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import json


# ===================================================================
# CORE DATA MODELS
# ===================================================================

class DocumentType(Enum):
    FRAKTUR = "fraktur"
    MODERN_GERMAN = "modern_german"
    OFFICIAL_DOCUMENT = "official_document"
    MIXED_PERIOD = "mixed_period"
    HANDWRITTEN = "handwritten"
    ITALIC_HEAVY = "italic_heavy"
    NEWSPAPER = "newspaper"
    UNKNOWN = "unknown"


class ImageQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class ProcessingStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class Match:
    """Represents a pattern match in text"""
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    pattern_name: str
    context: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of a processing operation"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageAnalysis:
    """Comprehensive image analysis results"""
    document_type: DocumentType
    confidence: float
    image_quality: ImageQuality
    brightness: float
    contrast: float
    noise_level: float
    text_density: float
    has_inverted_text: bool
    estimated_text_size: str
    resolution: Tuple[int, int]
    color_mode: str
    recommended_approaches: List[str]
    reasoning: List[str]


@dataclass
class OCRAttempt:
    """Record of an OCR attempt"""
    approach_name: str
    success: bool
    text_length: int
    confidence: float
    processing_time: float
    error: Optional[str] = None
    text_preview: str = ""


@dataclass
class ExtractedMetadata:
    """Complete metadata extracted from document"""
    # Core metadata
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    profession: Optional[str] = None

    # Confidence and processing info
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    raw_text_preview: Optional[str] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis results
    image_analysis: Optional[ImageAnalysis] = None
    ocr_attempts: List[OCRAttempt] = field(default_factory=list)

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence score"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'profession': self.profession,
            'confidence_scores': self.confidence_scores,
            'overall_confidence': self.get_overall_confidence(),
            'raw_text_preview': self.raw_text_preview,
            'processing_metadata': self.processing_metadata,
            'ocr_attempts_count': len(self.ocr_attempts),
            'successful_approach': self.processing_metadata.get('successful_approach'),
            'analysis_document_type': self.image_analysis.document_type.value if self.image_analysis else None,
            'analysis_quality': self.image_analysis.image_quality.value if self.image_analysis else None
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)


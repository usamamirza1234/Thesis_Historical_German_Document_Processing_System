from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import json


class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ProcessingResult:
    """Result of a processing step"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0


@dataclass
class Match:
    """Represents a pattern match"""
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    pattern_name: str
    context: str = ""


@dataclass
class Prediction:
    """ML model prediction result"""
    value: str
    confidence: float
    model_name: str
    raw_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExtractedMetadata:
    """Enhanced metadata container with improved typing and validation"""
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    profession: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    raw_text_preview: Optional[str] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary format with serializable types"""
        return {
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'profession': self.profession,
            'confidence_scores': self.confidence_scores,
            'raw_text_preview': self.raw_text_preview,
            'processing_metadata': self.processing_metadata
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def get_overall_confidence(self) -> float:
        """Calculate overall confidence score"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)


@dataclass
class ValidationResult:
    """Result of metadata validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)
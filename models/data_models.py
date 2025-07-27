from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import json

class DocumentType(Enum):
    FRAKTUR = "fraktur"
    MODERN_GERMAN = "modern_german"
    OFFICIAL_DOCUMENT = "official_document"
    MIXED_PERIOD = "mixed_period"
    HANDWRITTEN = "handwritten"
    ITALIC_HEAVY = "italic_heavy"
    NEWSPAPER = "newspaper"
    HISTORICAL_MANUSCRIPT = "historical_manuscript"
    TYPEWRITER = "typewriter"
    UNKNOWN = "unknown"


class ImageQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    DAMAGED = "damaged"


class ProcessingStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class ExtractionMethod(Enum):
    PATTERN_BASED = "pattern_based"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"
    ERA_SPECIFIC = "era_specific"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class Match:
    """ pattern match with era context"""
    value: str
    confidence: float
    start_pos: int
    end_pos: int
    pattern_name: str
    context: Optional[str] = None
    era_relevance: float = 0.0
    extraction_method: ExtractionMethod = ExtractionMethod.PATTERN_BASED
    validation_score: float = 0.0


@dataclass
class ProcessingResult:
    """ processing operation result"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    era_context: Optional[str] = None


@dataclass
class ImageAnalysis:
    """Comprehensive image analysis with era prediction"""
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

    #  era-specific fields
    predicted_era: Optional[str] = None
    era_confidence: float = 0.0
    era_indicators: List[str] = field(default_factory=list)
    script_characteristics: Dict[str, float] = field(default_factory=dict)
    layout_complexity: float = 0.0
    preservation_quality: float = 0.0


@dataclass
class OCRAttempt:
    """ OCR attempt record with era context"""
    approach_name: str
    success: bool
    text_length: int
    confidence: float
    processing_time: float
    error: Optional[str] = None
    text_preview: str = ""

    #  fields
    era_context: Optional[str] = None
    ocr_engine: str = "tesseract"
    language_used: str = "deu"
    preprocessing_steps: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    character_accuracy: Optional[float] = None
    word_accuracy: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    # OCR metrics
    character_error_rate: float = 0.0
    word_error_rate: float = 0.0
    confidence_score: float = 0.0

    # Metadata extraction metrics
    field_precision: Dict[str, float] = field(default_factory=dict)
    field_recall: Dict[str, float] = field(default_factory=dict)
    field_f1_score: Dict[str, float] = field(default_factory=dict)

    # Processing metrics
    processing_time: float = 0.0
    memory_usage: float = 0.0
    approach_success_rate: float = 0.0

    # Era-specific metrics
    era_detection_accuracy: float = 0.0
    era_specific_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExtractedMetadata:
    """ metadata with era context and validation"""
    # Core metadata
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    profession: Optional[str] = None

    #  metadata fields
    document_era: Optional[str] = None
    language: str = "German"
    page_count: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)

    # Confidence and validation
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    extraction_methods: Dict[str, ExtractionMethod] = field(default_factory=dict)

    # Processing information
    raw_text_preview: Optional[str] = None
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

    #  analysis results
    image_analysis: Optional[ImageAnalysis] = None
    ocr_attempts: List[OCRAttempt] = field(default_factory=list)
    evaluation_metrics: Optional[EvaluationMetrics] = None

    # Match details for debugging
    match_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Cross-validation results
    cross_page_consistency: Dict[str, float] = field(default_factory=dict)
    era_validation_results: Dict[str, Any] = field(default_factory=dict)

    def get_overall_confidence(self) -> float:
        """Calculate  overall confidence score"""
        if not self.confidence_scores:
            return 0.0

        # Weight different fields by importance
        field_weights = {
            'date': 0.25,
            'title': 0.20,
            'publisher': 0.20,
            'document_type': 0.15,
            'author': 0.10,
            'profession': 0.10
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for field, confidence in self.confidence_scores.items():
            weight = field_weights.get(field, 0.05)
            weighted_sum += confidence * weight
            total_weight += weight

        return weighted_sum / max(total_weight, 0.01)

    def get_era_consistency_score(self) -> float:
        """Calculate how consistent the extracted metadata is with the detected era"""
        if not self.document_era:
            return 0.0

        consistency_score = 0.0
        checks = 0

        # Date-era consistency
        if self.date and self.document_era:
            year = self.date.year
            era_ranges = {
                "1920-1933": (1918, 1933),
                "1933-1945": (1933, 1945),
                "1945-1970": (1945, 1970),
                "1970-1990": (1970, 1990),
                "1990-2025": (1990, 2030)
            }

            expected_range = era_ranges.get(self.document_era)
            if expected_range and expected_range[0] <= year <= expected_range[1]:
                consistency_score += 1.0
            checks += 1

        # Publisher-era consistency
        if self.publisher and self.document_era:
            era_publishers = {
                "1920-1933": ["reichs", "preußische"],
                "1933-1945": ["reichs", "deutsche arbeitsfront"],
                "1945-1970": ["bundes", "bundesrepublik"],
                "1970-1990": ["bundes", "bundesrepublik"],
                "1990-2025": ["bundes", "europäische"]
            }

            expected_terms = era_publishers.get(self.document_era, [])
            publisher_lower = self.publisher.lower()
            if any(term in publisher_lower for term in expected_terms):
                consistency_score += 1.0
            checks += 1

        return consistency_score / max(checks, 1)

    def validate_metadata_integrity(self) -> Dict[str, List[str]]:
        """Validate metadata for logical consistency and completeness"""
        issues = {
            'errors': [],
            'warnings': [],
            'suggestions': []
        }

        # Date validation
        if self.date:
            current_year = datetime.now().year
            if self.date.year < 1900 or self.date.year > current_year:
                issues['errors'].append(f"Date year {self.date.year} seems unrealistic")
            elif self.year and self.year != self.date.year:
                issues['warnings'].append(f"Year field ({self.year}) doesn't match date year ({self.date.year})")

        # Title validation
        if self.title:
            if len(self.title) < 10:
                issues['warnings'].append("Title seems unusually short")
            elif len(self.title) > 200:
                issues['warnings'].append("Title seems unusually long")

        # Publisher validation
        if self.publisher:
            if not any(term in self.publisher.lower() for term in ['ministerium', 'amt', 'behörde', 'verwaltung']):
                issues['suggestions'].append("Publisher doesn't contain typical German authority terms")

        # Confidence validation
        low_confidence_fields = [field for field, conf in self.confidence_scores.items() if conf < 0.5]
        if low_confidence_fields:
            issues['warnings'].append(f"Low confidence in fields: {', '.join(low_confidence_fields)}")

        return issues

    def to_dict(self) -> Dict:
        """Convert to comprehensive dictionary format"""
        return {
            # Core metadata
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'profession': self.profession,

            #  metadata
            'document_era': self.document_era,
            'language': self.language,
            'page_count': self.page_count,
            'keywords': self.keywords,
            'subjects': self.subjects,

            # Confidence and validation
            'confidence_scores': self.confidence_scores,
            'overall_confidence': self.get_overall_confidence(),
            'era_consistency_score': self.get_era_consistency_score(),
            'validation_scores': self.validation_scores,
            'extraction_methods': {k: v.value for k, v in self.extraction_methods.items()},

            # Processing information
            'processing_metadata': self.processing_metadata,
            'ocr_attempts_count': len(self.ocr_attempts),
            'successful_approach': self.processing_metadata.get('successful_approach'),

            # Analysis results
            'analysis_document_type': self.image_analysis.document_type.value if self.image_analysis else None,
            'analysis_quality': self.image_analysis.image_quality.value if self.image_analysis else None,
            'predicted_era': self.image_analysis.predicted_era if self.image_analysis else None,
            'era_confidence': self.image_analysis.era_confidence if self.image_analysis else None,

            # Quality metrics
            'character_error_rate': self.evaluation_metrics.character_error_rate if self.evaluation_metrics else None,
            'word_error_rate': self.evaluation_metrics.word_error_rate if self.evaluation_metrics else None,

            # Cross-validation
            'cross_page_consistency': self.cross_page_consistency,
            'era_validation_results': self.era_validation_results,

            # Metadata integrity
            'integrity_check': self.validate_metadata_integrity()
        }

    def to_json(self) -> str:
        """Convert to comprehensive JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)

    def to_research_format(self) -> Dict:
        """Convert to format suitable for research analysis"""
        return {
            'document_id': self.processing_metadata.get('file_path', 'unknown'),
            'extraction_results': {
                field: {
                    'value': getattr(self, field),
                    'confidence': self.confidence_scores.get(field, 0.0),
                    'method': self.extraction_methods.get(field, ExtractionMethod.PATTERN_BASED).value,
                    'validation_score': self.validation_scores.get(field, 0.0)
                }
                for field in ['title', 'date', 'publisher', 'author', 'document_type', 'profession']
                if getattr(self, field) is not None
            },
            'processing_analysis': {
                'era_detected': self.document_era,
                'era_confidence': self.image_analysis.era_confidence if self.image_analysis else 0.0,
                'overall_confidence': self.get_overall_confidence(),
                'era_consistency': self.get_era_consistency_score(),
                'ocr_attempts': len(self.ocr_attempts),
                'successful_attempts': len([a for a in self.ocr_attempts if a.success]),
                'processing_time': self.processing_metadata.get('processing_time_seconds', 0.0)
            },
            'quality_assessment': {
                'image_quality': self.image_analysis.image_quality.value if self.image_analysis else 'unknown',
                'text_length': len(self.raw_text_preview) if self.raw_text_preview else 0,
                'confidence_distribution': self.confidence_scores,
                'integrity_issues': self.validate_metadata_integrity()
            }
        }

    def export_for_training(self) -> Dict:
        """Export in format suitable for machine learning training"""
        return {
            'input_features': {
                'text_preview': self.raw_text_preview[:500] if self.raw_text_preview else "",
                'image_quality': self.image_analysis.image_quality.value if self.image_analysis else 'unknown',
                'document_type': self.image_analysis.document_type.value if self.image_analysis else 'unknown',
                'brightness': self.image_analysis.brightness if self.image_analysis else 0.0,
                'contrast': self.image_analysis.contrast if self.image_analysis else 0.0,
                'text_density': self.image_analysis.text_density if self.image_analysis else 0.0,
                'resolution': self.image_analysis.resolution if self.image_analysis else (0, 0),
                'era_indicators': self.image_analysis.era_indicators if self.image_analysis else []
            },
            'target_labels': {
                'title': self.title,
                'date': self.date.isoformat() if self.date else None,
                'publisher': self.publisher,
                'author': self.author,
                'document_type': self.document_type,
                'profession': self.profession,
                'era': self.document_era
            },
            'quality_indicators': {
                'confidence_scores': self.confidence_scores,
                'extraction_success': self.get_overall_confidence() > 0.7,
                'era_consistency': self.get_era_consistency_score() > 0.8,
                'validation_passed': len(self.validate_metadata_integrity()['errors']) == 0
            }
        }


@dataclass
class BatchProcessingResults:
    """Results from batch processing operations"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    processing_time: float
    results: List[ExtractedMetadata]

    #  batch metrics
    era_distribution: Dict[str, int] = field(default_factory=dict)
    approach_effectiveness: Dict[str, float] = field(default_factory=dict)
    quality_distribution: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0

    def get_success_rate(self) -> float:
        """Calculate overall success rate"""
        return self.successful_documents / max(self.total_documents, 1)

    def get_era_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get detailed statistics by era"""
        era_stats = {}

        for era in self.era_distribution:
            era_docs = [r for r in self.results if r.document_era == era]
            if era_docs:
                era_stats[era] = {
                    'count': len(era_docs),
                    'success_rate': len([r for r in era_docs if r.get_overall_confidence() > 0.7]) / len(era_docs),
                    'average_confidence': sum(r.get_overall_confidence() for r in era_docs) / len(era_docs),
                    'era_consistency': sum(r.get_era_consistency_score() for r in era_docs) / len(era_docs)
                }

        return era_stats

    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        return {
            'processing_summary': {
                'total_documents': self.total_documents,
                'successful_documents': self.successful_documents,
                'success_rate': self.get_success_rate(),
                'total_processing_time': self.processing_time,
                'average_time_per_document': self.processing_time / max(self.total_documents, 1)
            },
            'quality_metrics': {
                'average_confidence': self.average_confidence,
                'quality_distribution': self.quality_distribution,
                'high_confidence_docs': len([r for r in self.results if r.get_overall_confidence() > 0.8])
            },
            'era_analysis': {
                'era_distribution': self.era_distribution,
                'era_statistics': self.get_era_statistics()
            },
            'approach_performance': self.approach_effectiveness,
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on batch results"""
        recommendations = []

        success_rate = self.get_success_rate()
        if success_rate < 0.6:
            recommendations.append("Consider using higher quality scans or different preprocessing approaches")

        if self.average_confidence < 0.7:
            recommendations.append(
                "Review confidence thresholds and consider manual validation for low-confidence results")

        # Era-specific recommendations
        era_stats = self.get_era_statistics()
        for era, stats in era_stats.items():
            if stats['success_rate'] < 0.5:
                recommendations.append(f"Era {era} shows low success rate - consider era-specific optimization")

        return recommendations


# ===================================================================
# COMPATIBILITY ALIASES FOR EXISTING CODE
# ===================================================================

# Maintain backward compatibility
Match = Match
ImageAnalysis = ImageAnalysis
OCRAttempt = OCRAttempt
ExtractedMetadata = ExtractedMetadata
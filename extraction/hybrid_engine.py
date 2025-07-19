from typing import Dict, Tuple, Optional, Any
import logging
from extraction.pattern_matcher import PatternBasedExtractor
from extraction.ml_extractor import MLBasedExtractor, ModelManager
from models.data_models import ExtractedMetadata
from datetime import datetime

logger = logging.getLogger(__name__)


class HybridExtractionEngine:
    """Combines pattern-based and ML-based extraction approaches"""

    def __init__(self, pattern_extractor: PatternBasedExtractor,
                 ml_extractor: MLBasedExtractor,
                 confidence_threshold: float = 0.7):
        self.pattern_extractor = pattern_extractor
        self.ml_extractor = ml_extractor
        self.confidence_threshold = confidence_threshold

    def extract_metadata(self, text: str) -> ExtractedMetadata:
        """Extract metadata using hybrid approach"""
        logger.debug("Starting hybrid metadata extraction")

        # import pdb; pdb.set_trace()

        # Get pattern-based results
        pattern_results = self.pattern_extractor.extract_metadata_fields(text)

        # Get ML-based results
        ml_results = self.ml_extractor.extract_metadata_fields(text)

        # Combine results using confidence-based selection
        combined_results = self._combine_results(pattern_results, ml_results)

        # Create metadata object
        metadata = self._create_metadata_object(combined_results, text)

        logger.debug("Hybrid extraction completed")
        return metadata

    def _combine_results(self, pattern_results: Dict, ml_results: Dict) -> Dict[str, Tuple[Any, float, str]]:
        """Combine pattern and ML results using confidence-based selection"""
        combined = {}

        for field in pattern_results.keys():
            pattern_value, pattern_conf = pattern_results.get(field, (None, 0.0))
            ml_value, ml_conf = ml_results.get(field, (None, 0.0))

            # Choose based on confidence and availability
            if pattern_conf > self.confidence_threshold and pattern_conf >= ml_conf:
                combined[field] = (pattern_value, pattern_conf, "pattern")
            elif ml_conf > self.confidence_threshold and ml_conf > pattern_conf:
                combined[field] = (ml_value, ml_conf, "ml")
            elif pattern_value is not None:
                combined[field] = (pattern_value, pattern_conf, "pattern")
            elif ml_value is not None:
                combined[field] = (ml_value, ml_conf, "ml")
            else:
                combined[field] = (None, 0.0, "none")

        return combined

    def _create_metadata_object(self, combined_results: Dict, text: str) -> ExtractedMetadata:
        """Create ExtractedMetadata object from combined results"""
        metadata = ExtractedMetadata()
        confidence_scores = {}
        processing_metadata = {}

        for field, (value, confidence, method) in combined_results.items():
            if field == 'date' and isinstance(value, datetime):
                metadata.date = value
                metadata.year = value.year
                confidence_scores['date'] = confidence
                processing_metadata['date_method'] = method
            elif field == 'title':
                metadata.title = value
                confidence_scores['title'] = confidence
                processing_metadata['title_method'] = method
            elif field == 'publisher':
                metadata.publisher = value
                confidence_scores['publisher'] = confidence
                processing_metadata['publisher_method'] = method
            elif field == 'document_type':
                metadata.document_type = value
                confidence_scores['document_type'] = confidence
                processing_metadata['document_type_method'] = method
            elif field == 'profession':
                metadata.profession = value
                confidence_scores['profession'] = confidence
                processing_metadata['profession_method'] = method

        # Set confidence scores and processing metadata
        metadata.confidence_scores = confidence_scores
        metadata.processing_metadata = processing_metadata

        # Set text preview
        metadata.raw_text_preview = text[:500] + "..." if len(text) > 500 else text

        return metadata

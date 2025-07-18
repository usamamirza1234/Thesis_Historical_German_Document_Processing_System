from datetime import datetime
from models.data_models import ExtractedMetadata, ValidationResult


class MetadataValidator:
    """Validates extracted metadata for quality and consistency"""

    def __init__(self, min_confidence: float = 0.3):
        self.min_confidence = min_confidence

    def validate_extracted_data(self, metadata: ExtractedMetadata) -> ValidationResult:
        """Comprehensive validation of extracted metadata"""
        result = ValidationResult(is_valid=True)

        # Date validation
        self._validate_date(metadata, result)

        # Confidence validation
        self._validate_confidence_scores(metadata, result)

        # Content validation
        self._validate_content(metadata, result)

        return result

    def _validate_date(self, metadata: ExtractedMetadata, result: ValidationResult):
        """Validate date fields"""
        if metadata.date:
            if not self._is_reasonable_date(metadata.date):
                result.add_warning(f"Date {metadata.date} seems unreasonable for historical document")

            if metadata.year and metadata.year != metadata.date.year:
                result.add_error("Year field inconsistent with date field")

    def _validate_confidence_scores(self, metadata: ExtractedMetadata, result: ValidationResult):
        """Validate confidence scores"""
        for field, score in metadata.confidence_scores.items():
            if score < 0 or score > 1:
                result.add_error(f"Invalid confidence score for {field}: {score}")
            elif score < self.min_confidence:
                result.add_warning(f"Low confidence score for {field}: {score}")

    def _validate_content(self, metadata: ExtractedMetadata, result: ValidationResult):
        """Validate content fields"""
        if metadata.title and len(metadata.title) < 5:
            result.add_warning("Title seems too short")

        if metadata.publisher and len(metadata.publisher) < 3:
            result.add_warning("Publisher name seems too short")

    def _is_reasonable_date(self, date: datetime) -> bool:
        """Check if date is reasonable for historical documents"""
        return 1800 <= date.year <= 2030

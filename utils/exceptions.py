class DocumentProcessingError(Exception):
    """Base exception for document processing"""
    pass

class OCRError(DocumentProcessingError):
    """OCR-specific errors"""
    pass

class ValidationError(DocumentProcessingError):
    """Data validation errors"""
    pass

class PatternMatchingError(DocumentProcessingError):
    """Pattern matching errors"""
    pass

class ConfigurationError(DocumentProcessingError):
    """Configuration-related errors"""
    pass

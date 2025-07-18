from abc import ABC, abstractmethod
import pytesseract
import cv2
import os
from typing import Optional, Dict, Any
import logging
from models.data_models import ProcessingResult
from utils.exceptions import OCRError

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """Abstract base class for OCR engines"""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image"""
        pass

    @abstractmethod
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the OCR engine"""
        pass


class TesseractEngine(OCREngine):
    """Tesseract OCR engine implementation"""

    def __init__(self, language: str = 'deu_frak', config: str = ""):
        self.language = language
        self.config = config
        self._validate_tesseract()

    def _validate_tesseract(self) -> None:
        """Validate Tesseract installation"""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise OCRError(f"Tesseract not properly installed: {e}")

    def extract_text(self, image_path: str) -> str:
        """Extract text using Tesseract with Fraktur model"""
        if not os.path.exists(image_path):
            raise OCRError(f"Image file not found: {image_path}")

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise OCRError(f"Unable to load image: {image_path}")

            # Convert to RGB if needed (Tesseract expects RGB)
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Extract text
            text = pytesseract.image_to_string(
                image_rgb,
                lang=self.language,
                config=self.config
            )

            logger.debug(f"Extracted {len(text)} characters from {image_path}")
            return text

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {e}")
            raise OCRError(f"OCR extraction failed: {e}")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get Tesseract engine information"""
        try:
            version = pytesseract.get_tesseract_version()
            languages = pytesseract.get_languages()

            return {
                'engine': 'Tesseract',
                'version': str(version),
                'current_language': self.language,
                'available_languages': languages,
                'config': self.config
            }
        except Exception as e:
            return {
                'engine': 'Tesseract',
                'error': str(e)
            }


class CachedOCREngine(OCREngine):
    """OCR engine with caching capabilities"""

    def __init__(self, base_engine: OCREngine, enable_cache: bool = True):
        self.base_engine = base_engine
        self.enable_cache = enable_cache
        self.cache: Dict[str, str] = {}

    def _get_cache_key(self, image_path: str) -> str:
        """Generate cache key for image"""
        from utils.file_handlers import FileHandler
        return FileHandler.get_file_hash(image_path)

    def extract_text(self, image_path: str) -> str:
        """Extract text with caching"""
        if not self.enable_cache:
            return self.base_engine.extract_text(image_path)

        cache_key = self._get_cache_key(image_path)

        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Cache hit for {image_path}")
            return self.cache[cache_key]

        # Extract text and cache result
        text = self.base_engine.extract_text(image_path)
        self.cache[cache_key] = text

        logger.debug(f"Cached result for {image_path}")
        return text

    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine info including cache stats"""
        base_info = self.base_engine.get_engine_info()
        base_info.update({
            'cached': True,
            'cache_enabled': self.enable_cache,
            'cache_size': len(self.cache)
        })
        return base_info

    def clear_cache(self) -> None:
        """Clear the cache"""
        self.cache.clear()
        logger.info("OCR cache cleared")
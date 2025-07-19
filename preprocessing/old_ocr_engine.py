# preprocessing/ocr_engine.py
from abc import ABC, abstractmethod
import pytesseract
import cv2
import os
from typing import Optional, Dict, Any, List
import logging
from PIL import Image
import numpy as np
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
    """Tesseract OCR engine implementation with enhanced debugging"""

    def __init__(self, language: str = 'deu_frak', config: str = "", debug: bool = False):
        self.language = language
        self.config = config
        self.debug = debug
        self._validate_tesseract()

        # Fallback languages to try if primary fails
        self.fallback_languages = ['deu', 'eng', 'deu+eng']

        # Different OCR configurations to try
        self.ocr_configs = [
            "",  # Default
            "--psm 6",  # Uniform block of text
            "--psm 4",  # Single column of text
            "--psm 3",  # Fully automatic page segmentation
            "--psm 1",  # Automatic page segmentation with OSD
            "--oem 3 --psm 6",  # Use neural net LSTM engine
            "--oem 1 --psm 6",  # Use neural net LSTM engine only
        ]

    def _validate_tesseract(self) -> None:
        """Validate Tesseract installation"""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")

            # Check available languages
            available_langs = pytesseract.get_languages(config='')
            logger.info(f"Available languages: {available_langs}")

            if self.language not in available_langs:
                logger.warning(f"Language '{self.language}' not available. Available: {available_langs}")

        except Exception as e:
            raise OCRError(f"Tesseract not properly installed: {e}")

    def _preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Additional preprocessing specifically for better OCR"""
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply additional preprocessing for better OCR
            # 1. Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)

            # 2. Sharpen the image
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)

            # 3. Ensure proper contrast
            # Check if image is mostly dark text on light background or vice versa
            mean_intensity = np.mean(sharpened)
            if mean_intensity < 127:
                # Dark image, invert it
                sharpened = cv2.bitwise_not(sharpened)

            # 4. Final thresholding
            _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return thresh

        except Exception as e:
            logger.warning(f"OCR preprocessing failed: {e}")
            return image

    def _save_debug_image(self, image: np.ndarray, image_path: str, suffix: str) -> str:
        """Save debug image for inspection"""
        if not self.debug:
            return ""

        try:
            debug_dir = os.path.join(os.path.dirname(image_path), "debug_ocr")
            os.makedirs(debug_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            debug_path = os.path.join(debug_dir, f"{base_name}_{suffix}.png")

            cv2.imwrite(debug_path, image)
            return debug_path

        except Exception as e:
            logger.warning(f"Failed to save debug image: {e}")
            return ""

    def _analyze_image_properties(self, image_path: str) -> Dict[str, Any]:
        """Analyze image properties for debugging"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            properties = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "mean_intensity": float(np.mean(gray)),
                "std_intensity": float(np.std(gray)),
                "min_intensity": int(np.min(gray)),
                "max_intensity": int(np.max(gray)),
                "file_size_kb": os.path.getsize(image_path) / 1024,
                "is_mostly_dark": np.mean(gray) < 127
            }

            # Check for text-like regions
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            properties["num_contours"] = len(contours)
            properties["has_content"] = len(contours) > 10

            return properties

        except Exception as e:
            return {"error": str(e)}

    def extract_text(self, image_path: str) -> str:
        """Extract text using enhanced italic text support"""
        if not os.path.exists(image_path):
            raise OCRError(f"Image file not found: {image_path}")

        try:
            # First try the enhanced italic handler
            text = ItalicTextHandler.extract_text_with_italic_support(image_path, debug=self.debug)

            if text and len(text.strip()) > 20:
                # Good result from italic handler
                if self.debug:
                    logger.info(f"Italic handler succeeded with {len(text)} characters")
                return text
            else:
                # Fall back to original method if italic handler didn't work well
                if self.debug:
                    logger.info("Falling back to original OCR method")
                return self._original_extract_text(image_path)

        except Exception as e:
            logger.error(f"Enhanced OCR failed, using fallback: {e}")
            return self._original_extract_text(image_path)

    def _original_extract_text(self, image_path: str) -> str:
        """Your original extract_text method as fallback"""
        # Copy your entire original extract_text method here
        # This is your existing code from lines 85-170 in ocr_engine.py

        image = cv2.imread(image_path)
        if image is None:
            raise OCRError(f"Unable to load image: {image_path}")

        if self.debug:
            logger.info(f"Starting OCR extraction for: {image_path}")
            props = self._analyze_image_properties(image_path)
            logger.info(f"Image properties: {props}")

        # Try different preprocessing approaches
        preprocessing_results = []

        # Approach 1: Direct OCR on original image
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result1 = self._try_ocr_with_configs(original_rgb, "original")
        preprocessing_results.append(("original", result1))

        # Approach 2: Additional preprocessing
        preprocessed = self._preprocess_image_for_ocr(image)
        if self.debug:
            self._save_debug_image(preprocessed, image_path, "02_preprocessed")

        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)
        result2 = self._try_ocr_with_configs(preprocessed_rgb, "preprocessed")
        preprocessing_results.append(("preprocessed", result2))

        # Approach 3: Try with PIL Image
        try:
            from PIL import Image
            pil_image = Image.open(image_path)
            result3 = self._try_ocr_with_configs(pil_image, "pil")
            preprocessing_results.append(("pil", result3))
        except Exception as e:
            logger.warning(f"PIL approach failed: {e}")

        # Find the best result
        best_result = self._select_best_result(preprocessing_results)

        if self.debug:
            logger.info(f"OCR Results comparison:")
            for approach, result in preprocessing_results:
                logger.info(f"  {approach}: {len(result['text'])} chars, "
                            f"confidence: {result.get('confidence', 'unknown')}")
            logger.info(f"Selected: {best_result['approach']} with {len(best_result['text'])} characters")

        return best_result['text']

    def _try_ocr_with_configs(self, image, approach_name: str) -> Dict[str, Any]:
        """Try OCR with different configurations and languages"""
        best_result = {"text": "", "confidence": 0, "config": "", "language": ""}

        # Try primary language first
        languages_to_try = [self.language] + self.fallback_languages

        for lang in languages_to_try:
            for config in self.ocr_configs:
                try:
                    # Extract text
                    text = pytesseract.image_to_string(image, lang=lang, config=config)

                    # Get confidence if possible
                    try:
                        data = pytesseract.image_to_data(image, lang=lang, config=config,
                                                         output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    except:
                        avg_confidence = len(text)  # Use text length as proxy for confidence

                    # Update best result if this is better
                    if len(text.strip()) > len(best_result["text"].strip()) or avg_confidence > best_result[
                        "confidence"]:
                        best_result = {
                            "text": text,
                            "confidence": avg_confidence,
                            "config": config,
                            "language": lang,
                            "approach": approach_name
                        }

                    # Early exit if we got good results
                    if len(text.strip()) > 50 and avg_confidence > 50:
                        break

                except Exception as e:
                    if self.debug:
                        logger.debug(f"OCR attempt failed (lang={lang}, config={config}): {e}")
                    continue

            # If we got decent results, don't try other languages
            if len(best_result["text"].strip()) > 20:
                break

        return best_result

    def _select_best_result(self, results: List[tuple]) -> Dict[str, Any]:
        """Select the best OCR result from multiple approaches"""
        if not results:
            return {"text": "", "confidence": 0, "approach": "none"}

        # Score results based on text length and confidence
        scored_results = []
        for approach, result in results:
            text_length = len(result["text"].strip())
            confidence = result.get("confidence", 0)

            # Scoring formula: prioritize longer text with decent confidence
            score = text_length * 0.7 + confidence * 0.3

            scored_results.append((score, approach, result))

        # Return the highest scoring result
        scored_results.sort(key=lambda x: x[0], reverse=True)
        best_score, best_approach, best_result = scored_results[0]

        best_result["approach"] = best_approach
        best_result["score"] = best_score

        return best_result

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
                'config': self.config,
                'debug_mode': self.debug,
                'fallback_languages': self.fallback_languages
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


class ItalicTextHandler:
    """Handles italic text recognition better than standard approaches"""

    @staticmethod
    def extract_text_with_italic_support(image_path: str, debug: bool = True):
        """Enhanced text extraction with better italic text support"""
        import cv2
        import numpy as np
        import pytesseract
        import re

        if debug:
            print(f"Processing with italic support: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Unable to load image: {image_path}")

        # Try different approaches for italic text
        approaches = [
            # (name, preprocessor_function, language, config)
            ("german_modern", ItalicTextHandler._prep_modern, "deu", "--oem 3 --psm 6"),
            ("english_italic", ItalicTextHandler._prep_italic, "eng", "--oem 1 --psm 7"),
            ("mixed_lang", ItalicTextHandler._prep_mixed, "deu+eng", "--oem 3 --psm 6"),
            ("single_line", ItalicTextHandler._prep_line, "deu", "--psm 7"),
            ("word_level", ItalicTextHandler._prep_word, "eng", "--psm 8"),
        ]

        best_result = {"text": "", "score": 0, "approach": "none"}

        for name, prep_func, lang, config in approaches:
            try:
                # Preprocess image
                processed = prep_func(image)

                # Extract text
                text = pytesseract.image_to_string(processed, lang=lang, config=config)

                # Score result
                score = ItalicTextHandler._score_text(text)

                if debug:
                    print(f"{name}: {len(text)} chars, score: {score:.2f}")
                    first_line = text.split('\n')[0][:80] if text else ""
                    print(f"  Sample: {first_line}")

                if score > best_result["score"]:
                    best_result = {"text": text, "score": score, "approach": name}

                # Early exit for good results
                if score > 0.7:
                    break

            except Exception as e:
                if debug:
                    print(f"{name} failed: {e}")
                continue

        if debug:
            print(f"Best: {best_result['approach']} (score: {best_result['score']:.2f})")

        return best_result["text"]

    @staticmethod
    def _prep_modern(image):
        """Modern German text preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return cv2.convertScaleAbs(scaled, alpha=1.2, beta=10)

    @staticmethod
    def _prep_italic(image):
        """Italic text preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        scaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(scaled, -1, kernel)
        return cv2.convertScaleAbs(sharpened, alpha=1.3, beta=15)

    @staticmethod
    def _prep_mixed(image):
        """Mixed language preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(scaled)
        return cv2.convertScaleAbs(denoised, alpha=1.1, beta=5)

    @staticmethod
    def _prep_line(image):
        """Single line preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _prep_word(image):
        """Word level preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _score_text(text):
        """Score OCR result"""
        if not text or not text.strip():
            return 0.0

        score = 0.0
        text_lower = text.lower()

        # Length score
        score += min(len(text.strip()) / 100.0, 1.0) * 0.2

        # German words
        german_words = ['staatlich', 'anerkannt', 'erlaß', 'bundesminister', 'wirtschaft', 'berufsbild']
        matches = sum(1 for word in german_words if word in text_lower)
        score += min(matches / 4.0, 1.0) * 0.3

        # Specific patterns from your document
        import re
        patterns = [r'282094', r'10\.\s*5\.\s*1954', r'ii\s*a\s*4', r'vom']
        pattern_matches = sum(1 for p in patterns if re.search(p, text_lower))
        score += min(pattern_matches / 3.0, 1.0) * 0.5

        return min(score, 1.0)
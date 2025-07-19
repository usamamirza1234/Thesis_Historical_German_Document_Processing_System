import os
import time
import cv2
import re
from typing import Optional, List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path

from config.settings import ProcessingConfig, LoggingConfig
from preprocessing.image_processor import ImageProcessor
from preprocessing.ocr_engine import TesseractEngine, CachedOCREngine
from extraction.pattern_matcher import PatternBasedExtractor, PatternRegistry
from extraction.ml_extractor import MLBasedExtractor, ModelManager
from extraction.hybrid_engine import HybridExtractionEngine
from models.data_models import ExtractedMetadata, ProcessingResult
from utils.file_handlers import FileHandler
from utils.validation import MetadataValidator
from utils.logging_setup import setup_logging
from utils.exceptions import DocumentProcessingError, OCRError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Main processor for historical German legal documents.

    Combines OCR, image preprocessing, and metadata extraction using
    both pattern-based and machine learning approaches.

    Example:
        >>> config = ProcessingConfig()
        >>> processor = DocumentProcessor(config)
        >>> metadata = processor.process_document("document.pdf")
        >>> print(metadata.date, metadata.publisher)
    """

    def __init__(self, config: ProcessingConfig,
                 logging_config: Optional[LoggingConfig] = None):
        self.config = config

        # Setup logging
        if logging_config:
            self.logger = setup_logging(logging_config)
        else:
            self.logger = logging.getLogger(__name__)

        # Initialize components
        self._initialize_components()

        # Initialize validator
        self.validator = MetadataValidator(min_confidence=config.confidence_threshold)

        self.logger.info("DocumentProcessor initialized successfully")

    def _initialize_components(self):
        """Initialize all processing components"""
        try:
            # Image processor
            self.image_processor = ImageProcessor(self.config)

            # OCR engine
            base_ocr = TesseractEngine(
                language=self.config.ocr_language,
                config=self.config.tesseract_config
            )
            self.ocr_engine = CachedOCREngine(
                base_ocr,
                enable_cache=self.config.enable_caching
            )

            # Pattern-based extractor
            self.pattern_registry = PatternRegistry()
            self.pattern_extractor = PatternBasedExtractor(self.pattern_registry)

            # ML-based extractor
            model_configs = getattr(self.config, 'model_configs', {})
            self.model_manager = ModelManager(model_configs)
            self.ml_extractor = MLBasedExtractor(self.model_manager)

            # Hybrid extraction engine
            self.hybrid_engine = HybridExtractionEngine(
                self.pattern_extractor,
                self.ml_extractor,
                self.config.confidence_threshold
            )

            self.logger.info("All components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise DocumentProcessingError(f"Initialization failed: {e}")

    def process_document(self, file_path: str,
                         start_page: int = 1,
                         end_page: Optional[int] = None) -> ExtractedMetadata:
        """
        Extract metadata from a single document.

        Args:
            file_path: Path to PDF or image file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (None for all pages)

        Returns:
            ExtractedMetadata with confidence scores

        Raises:
            DocumentProcessingError: If processing fails
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing document: {file_path}")

            # Validate input
            if not FileHandler.validate_file_path(file_path):
                raise DocumentProcessingError(f"File not found or inaccessible: {file_path}")

            if not FileHandler.is_supported_file(file_path):
                raise DocumentProcessingError(f"Unsupported file format: {file_path}")

            # Extract text from document
            text = self._extract_text_from_document(file_path, start_page, end_page)

            if not text.strip():
                raise DocumentProcessingError("No text could be extracted from document")

            self.logger.info(f"Extracted Text: {text}")

            # Extract metadata using hybrid approach
            metadata = self.hybrid_engine.extract_metadata(text)

            # Validate results
            validation_result = self.validator.validate_extracted_data(metadata)
            if validation_result.warnings:
                self.logger.warning(f"Validation warnings: {validation_result.warnings}")

            # Add processing information
            processing_time = time.time() - start_time
            metadata.processing_metadata.update({
                'processing_time_seconds': processing_time,
                'file_path': file_path,
                'pages_processed': f"{start_page}-{end_page or 'end'}",
                'text_length': len(text),
                'validation_warnings': validation_result.warnings
            })

            self.logger.info(f"Successfully processed {file_path} in {processing_time:.2f}s")
            return metadata

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {file_path}: {e}")

            # Return partial metadata with error information
            error_metadata = ExtractedMetadata()
            error_metadata.processing_metadata = {
                'error': str(e),
                'processing_time_seconds': processing_time,
                'file_path': file_path
            }
            return error_metadata

    def _extract_text_from_document(self, file_path: str,
                                    start_page: int,
                                    end_page: Optional[int]) -> str:
        """Extract text from PDF or image file"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path, start_page, end_page)
        else:
            return self._extract_text_from_image(file_path)

    def _extract_text_from_pdf(self, pdf_path: str,
                               start_page: int,
                               end_page: Optional[int]) -> str:
        """Extract text from PDF using OCR"""
        try:
            # Convert PDF pages to images
            pages = convert_from_path(
                pdf_path,
                first_page=start_page,
                last_page=end_page,
                dpi=self.config.dpi
            )

            if not pages:
                raise OCRError("No pages could be converted from PDF")

            all_text = []

            for i, page in enumerate(pages):
                page_num = start_page + i
                self.logger.debug(f"Processing page {page_num}")

                # Save page as temporary image
                temp_image_path = os.path.join(
                    self.config.output_directory,
                    f"temp_page_{page_num}.png"
                )
                FileHandler.ensure_directory(os.path.dirname(temp_image_path))
                page.save(temp_image_path)

                try:
                    # Extract text from page
                    page_text = self._extract_text_from_image(temp_image_path)
                    if page_text.strip():
                        all_text.append(f"\n--- Page {page_num} ---\n{page_text}")

                    # Clean up temporary file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                except Exception as e:
                    self.logger.warning(f"Failed to process page {page_num}: {e}")
                    continue

            return "\n".join(all_text)

        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {e}")
            raise OCRError(f"PDF processing failed: {e}")

    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using multiple preprocessing approaches"""
        try:
            self.logger.info(f"Starting image text extraction: {image_path}")

            # Create multiple preprocessing approaches to try
            approaches = self._create_preprocessing_approaches()

            best_result = {"text": "", "length": 0, "approach": "none", "confidence": 0}

            for approach_name, pipeline in approaches:
                try:
                    self.logger.debug(f"Trying approach: {approach_name}")

                    # Process image with this approach
                    if pipeline is None:
                        # Direct OCR approach
                        text = self.ocr_engine.extract_text(image_path)
                        processed_image_path = image_path
                    else:
                        # Process with pipeline
                        original_image = cv2.imread(image_path)
                        if original_image is None:
                            self.logger.warning(f"Could not load image: {image_path}")
                            continue

                        processing_result = pipeline.process(original_image, os.path.basename(image_path))

                        if not processing_result.success:
                            self.logger.warning(f"Preprocessing failed for {approach_name}: {processing_result.error}")
                            continue

                        # Save processed image temporarily
                        processed_image_path = os.path.join(
                            self.config.output_directory,
                            f"processed_{approach_name}_{os.path.basename(image_path)}"
                        )

                        from utils.file_handlers import ImageHandler
                        ImageHandler.save_image(processing_result.data, processed_image_path)

                        # Extract text using OCR
                        text = self.ocr_engine.extract_text(processed_image_path)

                    text_length = len(text.strip())

                    # Calculate a simple confidence score based on text characteristics
                    confidence = self._calculate_text_confidence(text)

                    self.logger.info(f"Approach '{approach_name}': {text_length} chars, confidence: {confidence:.2f}")

                    # Update best result if this is better
                    if text_length > best_result["length"] or (
                            text_length == best_result["length"] and confidence > best_result["confidence"]):
                        best_result = {
                            "text": text,
                            "length": text_length,
                            "approach": approach_name,
                            "confidence": confidence
                        }

                    # Clean up temporary file if not in debug mode
                    if processed_image_path != image_path and not self.config.enable_debug and os.path.exists(
                            processed_image_path):
                        os.remove(processed_image_path)
                    elif self.config.enable_debug and processed_image_path != image_path:
                        self.logger.debug(f"Saved debug image: {processed_image_path}")

                    # Early exit if we got really good results
                    if text_length > 200 and confidence > 0.7:
                        self.logger.info(f"Early exit with good results from {approach_name}")
                        break

                except Exception as e:
                    self.logger.warning(f"Approach '{approach_name}' failed: {e}")
                    continue

            if best_result["length"] > 0:
                self.logger.info(f"Best result from '{best_result['approach']}': {best_result['length']} characters")
                return best_result["text"]
            else:
                self.logger.warning("All preprocessing approaches failed to extract meaningful text")
                return ""

        except Exception as e:
            self.logger.error(f"Image text extraction failed: {e}")
            raise OCRError(f"Image processing failed: {e}")

    def _create_preprocessing_approaches(self):
        """Create multiple preprocessing pipelines to try different approaches"""
        from preprocessing.processing_steps import (
            WhiteSpaceRemovalStep, InvertImageStep, RescaleImageStep,
            GrayscaleStep, BinarizeImageStep, NoiseRemovalStep,
            BorderRemovalStep, AddBordersStep
        )
        from preprocessing.image_processor import ProcessingPipeline

        approaches = []

        # Approach 1: Direct OCR (no preprocessing)
        approaches.append(("direct", None))

        # Approach 2: Minimal processing (like your working example)
        # This worked in your debug: invert + rescale
        minimal_steps = [
            InvertImageStep(),
            RescaleImageStep(scale_factor=2.5)
        ]
        approaches.append(("minimal", ProcessingPipeline(minimal_steps, save_intermediate=self.config.enable_debug)))

        # Approach 3: Light processing
        light_steps = [
            RescaleImageStep(scale_factor=2.0),
            GrayscaleStep(),
            AddBordersStep(border_size=50)
        ]
        approaches.append(("light", ProcessingPipeline(light_steps, save_intermediate=self.config.enable_debug)))

        # Approach 4: Your current default pipeline
        default_steps = []
        if self.config.enable_white_space_removal:
            default_steps.append(WhiteSpaceRemovalStep())
        default_steps.extend([
            InvertImageStep(),
            RescaleImageStep(scale_factor=2.5),
            GrayscaleStep(),
            NoiseRemovalStep(),
            BorderRemovalStep(),
            AddBordersStep()
        ])
        approaches.append(("default", ProcessingPipeline(default_steps, save_intermediate=self.config.enable_debug)))

        # Approach 5: Enhanced processing with binarization
        enhanced_steps = [
            InvertImageStep(),
            RescaleImageStep(scale_factor=3.0),
            GrayscaleStep(),
            BinarizeImageStep(threshold=127),
            NoiseRemovalStep(),
            AddBordersStep()
        ]
        approaches.append(("enhanced", ProcessingPipeline(enhanced_steps, save_intermediate=self.config.enable_debug)))

        # Approach 6: No white space removal (preserve all content)
        no_white_removal_steps = [
            InvertImageStep(),
            RescaleImageStep(scale_factor=2.5),
            GrayscaleStep(),
            NoiseRemovalStep(),
            AddBordersStep()
        ]
        approaches.append(("no_white_removal",
                           ProcessingPipeline(no_white_removal_steps, save_intermediate=self.config.enable_debug)))

        return approaches

    def _calculate_text_confidence(self, text: str) -> float:
        """Calculate confidence score for extracted text"""
        if not text or not text.strip():
            return 0.0

        confidence = 0.0

        # Length bonus (more text usually means better extraction)
        text_length = len(text.strip())
        length_score = min(text_length / 500.0, 1.0)  # Normalize to 0-1
        confidence += length_score * 0.3

        # German words bonus
        german_words = [
            'berufs', 'eignungsanforderungen', 'für', 'den', 'eintritt', 'lehrberuf',
            'deutschen', 'ausschuss', 'technisches', 'schulwesen', 'berlin',
            'arbeitsfront', 'reichsgruppe', 'industrie', 'verlag', 'stand', 'vom'
        ]

        text_lower = text.lower()
        german_matches = sum(1 for word in german_words if word in text_lower)
        german_score = min(german_matches / 5.0, 1.0)  # Normalize
        confidence += german_score * 0.4

        # Structure bonus (proper formatting)
        structure_indicators = [
            '\n',  # Line breaks
            '(',  # Parentheses
            ')',
            '.',  # Periods
            ',',  # Commas
        ]

        structure_count = sum(text.count(indicator) for indicator in structure_indicators)
        structure_score = min(structure_count / 20.0, 1.0)  # Normalize
        confidence += structure_score * 0.2

        # Date pattern bonus
        date_patterns = [
            r'\d{1,2}\.\s*[A-Za-z]+\s*\d{4}',  # German date format
            r'Stand\s+vom',  # "Stand vom" pattern
            r'\d{4}',  # Year
        ]

        date_matches = sum(1 for pattern in date_patterns if re.search(pattern, text))
        date_score = min(date_matches / 2.0, 1.0)
        confidence += date_score * 0.1

        return min(confidence, 1.0)

    def process_documents_batch(self, file_paths: List[str],
                                max_workers: Optional[int] = None) -> List[ExtractedMetadata]:
        """
        Process multiple documents in parallel.

        Args:
            file_paths: List of document paths to process
            max_workers: Maximum number of worker threads

        Returns:
            List of ExtractedMetadata objects
        """
        max_workers = max_workers or self.config.max_workers

        self.logger.info(f"Processing {len(file_paths)} documents with {max_workers} workers")

        results = []

        if max_workers == 1:
            # Sequential processing
            for file_path in file_paths:
                result = self.process_document(file_path)
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_document, file_path)
                    for file_path in file_paths
                ]

                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {e}")
                        # Add error metadata
                        error_metadata = ExtractedMetadata()
                        error_metadata.processing_metadata = {'error': str(e)}
                        results.append(error_metadata)

        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics and component information"""
        stats = {
            'config': {
                'dpi': self.config.dpi,
                'ocr_language': self.config.ocr_language,
                'confidence_threshold': self.config.confidence_threshold,
                'enable_caching': self.config.enable_caching
            },
            'ocr_engine': self.ocr_engine.get_engine_info(),
            'ml_models': self.model_manager.get_model_info(),
            'patterns': self.pattern_registry.get_pattern_info()
        }

        return stats

    def clear_cache(self) -> None:
        """Clear all caches"""
        if hasattr(self.ocr_engine, 'clear_cache'):
            self.ocr_engine.clear_cache()
        self.logger.info("Caches cleared")
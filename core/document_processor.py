import os
import time
from pathlib import Path

from typing import Optional, List, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path

from config.settings import ProcessingConfig
from extraction.pattern_matcher import MetadataExtractor

from models.data_models import ExtractedMetadata, ProcessingResult, ImageAnalysis
from ocr.ocr_engine import SmartOCREngine


# ===================================================================
# MAIN DOCUMENT PROCESSOR
# ===================================================================

class SmartGermanDocumentProcessor:
    """Main processor with intelligent OCR approach selection"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ocr_engine = SmartOCREngine(config)
        self.metadata_extractor = MetadataExtractor()

        # Setup logging
        self._setup_logging()

        self.logger.info("Smart German Document Processor initialized")

    def _setup_logging(self):
        """Setup logging system"""
        self.logger = logging.getLogger('smart_german_ocr')
        self.logger.setLevel(logging.INFO if not self.config.enable_debug else logging.DEBUG)

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler if enabled
            if self.config.save_detailed_logs:
                log_file = os.path.join(self.config.output_directory, "processing.log")
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(console_formatter)
                self.logger.addHandler(file_handler)

    def process_document(self, file_path: str, start_page: int = 1,
                         end_page: Optional[int] = None,
                         user_approach: Optional[str] = None) -> ExtractedMetadata:
        """Process a document with intelligent OCR"""

        start_time = time.time()
        self.logger.info(f"Processing document: {file_path}")

        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Extract text
            text, processing_info = self._extract_text_from_document(
                file_path, start_page, end_page, user_approach
            )

            if not text.strip():
                raise ValueError("No text could be extracted from document")

            self.logger.info(f"Successfully extracted {len(text)} characters")

            # Extract metadata
            metadata = self.metadata_extractor.extract_metadata(text)

            # Add processing information
            processing_time = time.time() - start_time
            metadata.processing_metadata.update({
                'processing_time_seconds': processing_time,
                'file_path': file_path,
                'pages_processed': f"{start_page}-{end_page or 'end'}",
                'text_length': len(text),
                **processing_info
            })

            self.logger.info(f"Processing completed in {processing_time:.2f}s")
            return metadata

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Processing failed: {e}")

            error_metadata = ExtractedMetadata()
            error_metadata.processing_metadata = {
                'error': str(e),
                'processing_time_seconds': processing_time,
                'file_path': file_path,
                'success': False
            }
            return error_metadata

    def _extract_text_from_document(self, file_path: str, start_page: int,
                                    end_page: Optional[int],
                                    user_approach: Optional[str]) -> Tuple[str, Dict]:
        """Extract text from document"""

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path, start_page, end_page, user_approach)
        else:
            return self._extract_from_image(file_path, user_approach)

    def _extract_from_pdf(self, pdf_path: str, start_page: int,
                          end_page: Optional[int],
                          user_approach: Optional[str]) -> Tuple[str, Dict]:
        """Extract text from PDF"""

        try:
            # Convert PDF to images
            pages = convert_from_path(
                pdf_path,
                first_page=start_page,
                last_page=end_page,
                dpi=self.config.dpi
            )

            if not pages:
                raise ValueError("No pages could be converted from PDF")

            all_text = []
            processing_info = {
                'total_pages': len(pages),
                'successful_pages': 0,
                'page_results': []
            }

            for i, page in enumerate(pages):
                page_num = start_page + i
                self.logger.info(f"Processing page {page_num}")

                # Save page as image
                temp_image_path = os.path.join(
                    self.config.output_directory,
                    f"temp_page_{page_num}.png"
                )
                os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                page.save(temp_image_path)

                try:
                    # Extract text from page
                    page_text, page_info = self._extract_from_image(
                        temp_image_path, user_approach
                    )

                    if page_text.strip():
                        all_text.append(f"\n--- Page {page_num} ---\n{page_text}")
                        processing_info['successful_pages'] += 1

                    page_info['page_number'] = page_num
                    processing_info['page_results'].append(page_info)

                    # Clean up unless debug mode
                    if not self.config.enable_debug:
                        os.remove(temp_image_path)

                except Exception as e:
                    self.logger.warning(f"Failed to process page {page_num}: {e}")
                    processing_info['page_results'].append({
                        'page_number': page_num,
                        'error': str(e),
                        'success': False
                    })

            return "\n".join(all_text), processing_info

        except Exception as e:
            raise ValueError(f"PDF processing failed: {e}")

    def _extract_from_image(self, image_path: str,
                            user_approach: Optional[str]) -> Tuple[str, Dict]:
        """Extract text from image using smart OCR"""

        # Get user choice if interactive mode is enabled
        if self.config.enable_interactive_mode and not user_approach:
            # First analyze the image
            analysis = self.ocr_engine.analyzer.analyze_image(image_path)
            user_choice = show_analysis_and_get_choice(analysis)
        else:
            user_choice = user_approach

        # Extract text using smart engine
        text, attempts, analysis = self.ocr_engine.extract_text_smart(
            image_path, user_choice
        )

        # Prepare processing info
        processing_info = {
            'analysis': {
                'document_type': analysis.document_type.value,
                'confidence': analysis.confidence,
                'image_quality': analysis.image_quality.value,
                'brightness': analysis.brightness,
                'contrast': analysis.contrast,
                'has_inverted_text': analysis.has_inverted_text,
                'reasoning': analysis.reasoning
            },
            'ocr_attempts': len(attempts),
            'successful_attempts': sum(1 for a in attempts if a.success),
            'user_choice': user_choice,
            'successful_approach': None,
            'final_confidence': 0.0
        }

        # Find best attempt
        successful_attempts = [a for a in attempts if a.success]
        if successful_attempts:
            best_attempt = max(successful_attempts, key=lambda a: a.confidence)
            processing_info['successful_approach'] = best_attempt.approach_name
            processing_info['final_confidence'] = best_attempt.confidence

        return text, processing_info

    def process_batch(self, file_paths: List[str], **kwargs) -> List[ExtractedMetadata]:
        """Process multiple documents"""
        results = []

        self.logger.info(f"Starting batch processing of {len(file_paths)} documents")

        if self.config.enable_parallel_processing and len(file_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [
                    executor.submit(self.process_document, file_path, **kwargs)
                    for file_path in file_paths
                ]

                for future in futures:
                    try:
                        result = future.result(timeout=self.config.processing_timeout_seconds)
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Batch processing error: {e}")
                        error_metadata = ExtractedMetadata()
                        error_metadata.processing_metadata = {'error': str(e)}
                        results.append(error_metadata)
        else:
            # Sequential processing
            for file_path in file_paths:
                try:
                    result = self.process_document(file_path, **kwargs)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    error_metadata = ExtractedMetadata()
                    error_metadata.processing_metadata = {'error': str(e)}
                    results.append(error_metadata)

        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results



# ===================================================================
# USER INTERACTION SYSTEM
# ===================================================================

def show_analysis_and_get_choice(analysis: ImageAnalysis, max_display: int = 5) -> Optional[str]:
    """Show analysis results and get user choice"""

    print(f"\n🔍 Smart Image Analysis Results:")
    print("=" * 50)
    print(f"📄 Document Type: {analysis.document_type.value.replace('_', ' ').title()}")
    print(f"⭐ Detection Confidence: {analysis.confidence:.2f}")
    print(f"🖼️  Image Quality: {analysis.image_quality.value.title()}")
    print(f"💡 Brightness: {analysis.brightness:.0f}/255")
    print(f"🌗 Contrast: {analysis.contrast:.1f}")
    print(f"📏 Resolution: {analysis.resolution[0]}x{analysis.resolution[1]}")
    print(f"📊 Text Density: {analysis.text_density:.2f}")
    print(f"📏 Estimated Text Size: {analysis.estimated_text_size}")

    if analysis.has_inverted_text:
        print("⚫ Inverted text detected (white text on dark background)")

    print(f"\n🧠 AI Analysis Reasoning:")
    for i, reason in enumerate(analysis.reasoning, 1):
        print(f"   {i}. {reason}")

    print(f"\n🚀 Recommended OCR Approaches (in priority order):")
    displayed_approaches = analysis.recommended_approaches[:max_display]

    for i, approach in enumerate(displayed_approaches, 1):
        print(f"   {i}. {approach.replace('_', ' ').title()}")

    print(f"\n⚡ Your Options:")
    print(f"   0. Use AI recommendation (auto-select best approach)")
    print(f"   1-{len(displayed_approaches)}. Choose specific approach")
    print(f"   a. Try all recommended approaches")
    print(f"   s. Skip analysis and use standard processing")

    while True:
        try:
            choice = input(f"\n👤 Your choice (0-{len(displayed_approaches)}, a, s): ").strip().lower()

            if choice == '0' or choice == '':
                return None  # Use AI recommendation
            elif choice == 'a':
                return "all_recommended"
            elif choice == 's':
                return "skip_analysis"
            elif choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(displayed_approaches):
                    return displayed_approaches[index]
                else:
                    print(f"❌ Please enter a number between 1 and {len(displayed_approaches)}")
            else:
                print("❌ Please enter a valid option (0, 1-5, a, or s)")

        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")


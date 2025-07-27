import json
import os
import time
from datetime import datetime
from pathlib import Path

from typing import Optional, List, Dict, Any, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path

from config.settings import ProcessingConfig, DocumentEra
from extraction.pattern_matcher import MetadataExtractor

from models.data_models import ExtractedMetadata, ProcessingResult, ImageAnalysis
from ocr.ocr_engine import SmartOCREngine
# ===================================================================
# ENHANCED DOCUMENT PROCESSOR WITH ERA DETECTION
# ===================================================================

class SmartGermanDocumentProcessor:
    """Enhanced processor with historical era detection and adaptive processing"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ocr_engine = SmartOCREngine(config)
        self.metadata_extractor = MetadataExtractor()

        # Setup logging
        self._setup_logging()

        # Initialize processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'successful_extractions': 0,
            'era_detections': {},
            'approach_success_rates': {},
            'average_processing_time': 0.0
        }

        self.logger.info("Intelligent German Document Processor initialized")
        self.logger.info(f"Processing mode: {config.processing_mode.value}")

    def _setup_logging(self):
        """Enhanced logging setup with structured output"""
        self.logger = logging.getLogger('intelligent_german_ocr')
        self.logger.setLevel(logging.INFO if not self.config.enable_debug else logging.DEBUG)

        if not self.logger.handlers:
            # Console handler with enhanced formatting
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # File handler for detailed logs
            if self.config.save_detailed_logs:
                log_file = os.path.join(self.config.output_directory, "intelligent_processing.log")
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s - %(extra_data)s',
                    defaults={'extra_data': ''}
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)

    def process_document(self, file_path: str, start_page: int = 1,
                         end_page: Optional[int] = None,
                         user_approach: Optional[str] = None,
                         expected_era: Optional[DocumentEra] = None) -> ExtractedMetadata:
        """Process document with intelligent era detection and adaptive processing"""

        start_time = time.time()
        self.logger.info(f"Processing document: {file_path}")

        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Extract text with era-aware processing
            text, processing_info = self._extract_text_with_era_detection(
                file_path, start_page, end_page, user_approach, expected_era
            )

            if not text.strip():
                raise ValueError("No text could be extracted from document")

            self.logger.info(f"Successfully extracted {len(text)} characters")

            # Extract metadata with enhanced analysis
            metadata = self.metadata_extractor.extract_metadata(text)

            # Detect document era from extracted content
            detected_era = self._detect_document_era(metadata, text)

            # Add comprehensive processing information
            processing_time = time.time() - start_time
            metadata.processing_metadata.update({
                'processing_time_seconds': processing_time,
                'file_path': file_path,
                'pages_processed': f"{start_page}-{end_page or 'end'}",
                'text_length': len(text),
                'detected_era': detected_era.value if detected_era else None,
                'expected_era': expected_era.value if expected_era else None,
                'era_match': detected_era == expected_era if expected_era else None,
                'processing_mode': self.config.processing_mode.value,
                **processing_info
            })

            # Update statistics
            self._update_processing_stats(metadata, processing_time, detected_era)

            # Generate processing report if enabled
            if self.config.generate_processing_report:
                self._generate_processing_report(metadata, file_path)

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

    def _extract_text_with_era_detection(self, file_path: str, start_page: int,
                                         end_page: Optional[int],
                                         user_approach: Optional[str],
                                         expected_era: Optional[DocumentEra]) -> Tuple[str, Dict]:
        """Extract text with intelligent era detection and approach selection"""

        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self._extract_from_pdf_with_era_detection(
                file_path, start_page, end_page, user_approach, expected_era
            )
        else:
            return self._extract_from_image_with_era_detection(
                file_path, user_approach, expected_era
            )

    def _extract_from_pdf_with_era_detection(self, pdf_path: str, start_page: int,
                                             end_page: Optional[int],
                                             user_approach: Optional[str],
                                             expected_era: Optional[DocumentEra]) -> Tuple[str, Dict]:
        """Extract text from PDF with era-aware processing"""

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
                'page_results': [],
                'era_indicators': [],
                'approach_adaptations': []
            }

            # Initial era detection from first page
            if not expected_era and len(pages) > 0:
                first_page_path = os.path.join(
                    self.config.output_directory,
                    f"temp_first_page.png"
                )
                pages[0].save(first_page_path)

                # Quick analysis for era detection
                analysis = self.ocr_engine.analyzer.analyze_image(first_page_path)
                expected_era = self._predict_era_from_analysis(analysis)

                if not self.config.enable_debug:
                    os.remove(first_page_path)

            for i, page in enumerate(pages):
                page_num = start_page + i
                self.logger.info(
                    f"Processing page {page_num} (era: {expected_era.value if expected_era else 'unknown'})")

                # Save page as image
                temp_image_path = os.path.join(
                    self.config.output_directory,
                    f"temp_page_{page_num}.png"
                )
                os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
                page.save(temp_image_path)

                try:
                    # Extract text with era-specific approach
                    page_text, page_info = self._extract_from_image_with_era_detection(
                        temp_image_path, user_approach, expected_era
                    )

                    if page_text.strip():
                        all_text.append(f"\n--- Page {page_num} ---\n{page_text}")
                        processing_info['successful_pages'] += 1

                    page_info['page_number'] = page_num
                    processing_info['page_results'].append(page_info)

                    # Collect era indicators for cross-validation
                    if 'era_indicators' in page_info:
                        processing_info['era_indicators'].extend(page_info['era_indicators'])

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

            # Consolidate cross-page metadata if enabled
            if self.config.enable_cross_page_analysis:
                consolidated_text = self._consolidate_cross_page_metadata(all_text, processing_info)
                return consolidated_text, processing_info
            else:
                return "\n".join(all_text), processing_info

        except Exception as e:
            raise ValueError(f"PDF processing failed: {e}")

    def _extract_from_image_with_era_detection(self, image_path: str,
                                               user_approach: Optional[str],
                                               expected_era: Optional[DocumentEra]) -> Tuple[str, Dict]:
        """Extract text from image with era-aware approach selection"""

        # Get user choice or era-based selection
        if self.config.enable_interactive_mode and not user_approach:
            # Analyze image and show interactive selection
            analysis = self.ocr_engine.analyzer.analyze_image(image_path)
            user_choice = show_enhanced_analysis_and_get_choice(analysis, expected_era)
        else:
            user_choice = user_approach

        # Select approaches based on era
        if expected_era and not user_choice:
            preferred_approaches = self.config.get_preferred_approaches_for_era(expected_era)
            self.logger.info(f"Using era-specific approaches for {expected_era.value}: {preferred_approaches}")
        else:
            preferred_approaches = None

        # Extract text using enhanced smart engine
        text, attempts, analysis = self.ocr_engine.extract_text_smart(
            image_path, user_choice, preferred_approaches, expected_era
        )

        # Enhanced processing info
        processing_info = {
            'analysis': {
                'document_type': analysis.document_type.value,
                'confidence': analysis.confidence,
                'image_quality': analysis.image_quality.value,
                'brightness': analysis.brightness,
                'contrast': analysis.contrast,
                'has_inverted_text': analysis.has_inverted_text,
                'reasoning': analysis.reasoning,
                'predicted_era': expected_era.value if expected_era else None
            },
            'ocr_attempts': len(attempts),
            'successful_attempts': sum(1 for a in attempts if a.success),
            'user_choice': user_choice,
            'era_based_selection': bool(expected_era and not user_choice),
            'successful_approach': None,
            'final_confidence': 0.0,
            'era_indicators': self._extract_era_indicators(text, analysis)
        }

        # Find best attempt
        successful_attempts = [a for a in attempts if a.success]
        if successful_attempts:
            best_attempt = max(successful_attempts, key=lambda a: a.confidence)
            processing_info['successful_approach'] = best_attempt.approach_name
            processing_info['final_confidence'] = best_attempt.confidence

        return text, processing_info

    def _detect_document_era(self, metadata: ExtractedMetadata, text: str) -> Optional[DocumentEra]:
        """Detect document era from extracted metadata and text content"""

        # Primary: Use extracted date
        if metadata.date:
            year = metadata.date.year
            if 1920 <= year <= 1933:
                return DocumentEra.WEIMAR_REPUBLIC
            elif 1933 <= year <= 1945:
                return DocumentEra.NAZI_PERIOD
            elif 1945 <= year <= 1970:
                return DocumentEra.EARLY_BRD
            elif 1970 <= year <= 1990:
                return DocumentEra.MODERN_BRD
            elif 1990 <= year <= 2025:
                return DocumentEra.CONTEMPORARY

        # Secondary: Use textual indicators
        text_lower = text.lower()

        # Nazi period indicators
        if any(term in text_lower for term in ['reichsministerium', 'deutsche arbeitsfront', 'führer']):
            return DocumentEra.NAZI_PERIOD

        # Weimar period indicators
        if any(term in text_lower for term in ['reichsarbeitsverwaltung', 'preußische', 'weimar']):
            return DocumentEra.WEIMAR_REPUBLIC

        # BRD indicators
        if any(term in text_lower for term in ['bundesministerium', 'bundesrepublik', 'grundgesetz']):
            if 'ddr' in text_lower or 'deutsche demokratische republik' in text_lower:
                return DocumentEra.MODERN_BRD  # Reunification era
            return DocumentEra.EARLY_BRD

        # Contemporary indicators
        if any(term in text_lower for term in ['europäische union', 'euro', 'internet', 'digital']):
            return DocumentEra.CONTEMPORARY

        return None

    def _predict_era_from_analysis(self, analysis: ImageAnalysis) -> Optional[DocumentEra]:
        """Predict era from image analysis characteristics"""

        # Fraktur script suggests earlier periods
        if analysis.document_type.value == 'fraktur':
            return DocumentEra.WEIMAR_REPUBLIC

        # Poor quality + official document suggests historical
        if (analysis.image_quality.value == 'poor' and
                analysis.document_type.value == 'official_document'):
            return DocumentEra.EARLY_BRD

        # High quality suggests contemporary
        if analysis.image_quality.value == 'excellent':
            return DocumentEra.CONTEMPORARY

        return None

    def _extract_era_indicators(self, text: str, analysis: ImageAnalysis) -> List[str]:
        """Extract indicators that help identify document era"""
        indicators = []

        # Script type indicators
        if analysis.document_type.value == 'fraktur':
            indicators.append('fraktur_script')

        # Language indicators
        text_lower = text.lower()
        era_terms = {
            'reich_terms': ['reichsministerium', 'reichsarbeit', 'deutsche arbeitsfront'],
            'weimar_terms': ['preußische', 'weimar', 'republik'],
            'brd_terms': ['bundesministerium', 'bundesrepublik', 'grundgesetz'],
            'contemporary_terms': ['europäische union', 'internet', 'digital', 'euro']
        }

        for category, terms in era_terms.items():
            if any(term in text_lower for term in terms):
                indicators.append(category)

        return indicators

    def _consolidate_cross_page_metadata(self, all_text: List[str], processing_info: Dict) -> str:
        """Consolidate metadata scattered across multiple pages"""

        if self.config.metadata_consolidation_strategy == "confidence_weighted":
            # Use confidence scores to weight information from different pages
            consolidated = self._confidence_weighted_consolidation(all_text, processing_info)
        else:
            # Simple concatenation
            consolidated = "\n".join(all_text)

        return consolidated

    def _confidence_weighted_consolidation(self, all_text: List[str], processing_info: Dict) -> str:
        """Consolidate text using confidence-weighted approach"""

        # For now, return simple concatenation
        # TODO: Implement sophisticated confidence weighting
        return "\n".join(all_text)

    def _update_processing_stats(self, metadata: ExtractedMetadata,
                                 processing_time: float,
                                 detected_era: Optional[DocumentEra]):
        """Update processing statistics for analysis"""

        self.processing_stats['documents_processed'] += 1

        if metadata.get_overall_confidence() > self.config.confidence_threshold:
            self.processing_stats['successful_extractions'] += 1

        if detected_era:
            era_key = detected_era.value
            if era_key not in self.processing_stats['era_detections']:
                self.processing_stats['era_detections'][era_key] = 0
            self.processing_stats['era_detections'][era_key] += 1

        # Update average processing time
        total_docs = self.processing_stats['documents_processed']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
                (current_avg * (total_docs - 1) + processing_time) / total_docs
        )

    def _generate_processing_report(self, metadata: ExtractedMetadata, file_path: str):
        """Generate detailed processing report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'metadata_extracted': metadata.to_dict(),
            'processing_config': self.config.to_dict(),
            'success': metadata.get_overall_confidence() > self.config.confidence_threshold
        }

        report_file = os.path.join(
            self.config.output_directory,
            f"report_{Path(file_path).stem}_{int(time.time())}.json"
        )

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        return {
            **self.processing_stats,
            'success_rate': (
                    self.processing_stats['successful_extractions'] /
                    max(1, self.processing_stats['documents_processed'])
            ),
            'config_summary': self.config.to_dict()
        }

    def process_batch_with_era_detection(self, file_paths: List[str],
                                         expected_eras: Optional[List[DocumentEra]] = None,
                                         **kwargs) -> List[ExtractedMetadata]:
        """Process multiple documents with era detection"""

        if expected_eras and len(expected_eras) != len(file_paths):
            raise ValueError("If provided, expected_eras must match file_paths length")

        results = []
        self.logger.info(f"Starting enhanced batch processing of {len(file_paths)} documents")

        for i, file_path in enumerate(file_paths):
            expected_era = expected_eras[i] if expected_eras else None

            try:
                result = self.process_document(
                    file_path,
                    expected_era=expected_era,
                    **kwargs
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                error_metadata = ExtractedMetadata()
                error_metadata.processing_metadata = {
                    'error': str(e),
                    'file_path': file_path
                }
                results.append(error_metadata)

        # Generate batch summary report
        if self.config.generate_processing_report:
            self._generate_batch_summary_report(results, file_paths)

        self.logger.info(f"Enhanced batch processing completed: {len(results)} results")
        return results

    def _generate_batch_summary_report(self, results: List[ExtractedMetadata], file_paths: List[str]):
        """Generate summary report for batch processing"""

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(results),
            'successful_documents': sum(
                1 for r in results if r.get_overall_confidence() > self.config.confidence_threshold),
            'processing_statistics': self.get_processing_statistics(),
            'era_distribution': {},
            'approach_effectiveness': {}
        }

        # Analyze era distribution
        for result in results:
            era = result.processing_metadata.get('detected_era')
            if era:
                if era not in summary['era_distribution']:
                    summary['era_distribution'][era] = 0
                summary['era_distribution'][era] += 1

        summary_file = os.path.join(
            self.config.output_directory,
            f"batch_summary_{int(time.time())}.json"
        )

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

def show_enhanced_analysis_and_get_choice(analysis: ImageAnalysis,
                                          expected_era: Optional[DocumentEra] = None,
                                          max_display: int = 5) -> Optional[str]:
    """Enhanced analysis display with era information"""

    print(f"\n🔍 Enhanced Smart Image Analysis Results:")
    print("=" * 60)
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

    if expected_era:
        print(f"🕰️  Expected Era: {expected_era.value}")

    print(f"\n🧠 AI Analysis Reasoning:")
    for i, reason in enumerate(analysis.reasoning, 1):
        print(f"   {i}. {reason}")

    print(f"\n🚀 Recommended OCR Approaches (in priority order):")
    displayed_approaches = analysis.recommended_approaches[:max_display]

    for i, approach in enumerate(displayed_approaches, 1):
        print(f"   {i}. {approach.replace('_', ' ').title()}")

    if expected_era:
        print(f"\n📚 Era-Specific Recommendations:")
        era_approaches = {
            DocumentEra.WEIMAR_REPUBLIC: ["Fraktur Traditional", "Fraktur Enhanced", "Mixed Period"],
            DocumentEra.NAZI_PERIOD: ["Fraktur Enhanced", "Official Document", "Mixed Period"],
            DocumentEra.EARLY_BRD: ["Mixed Period", "Official Document", "Modern German"],
            DocumentEra.MODERN_BRD: ["Official Document", "Modern German", "Standard"],
            DocumentEra.CONTEMPORARY: ["Modern German", "Official Document", "Light"]
        }

        era_recs = era_approaches.get(expected_era, [])
        for i, approach in enumerate(era_recs[:3], 1):
            print(f"   E{i}. {approach} (era-optimized)")

    print(f"\n⚡ Your Options:")
    print(f"   0. Use AI recommendation (auto-select best approach)")
    print(f"   1-{len(displayed_approaches)}. Choose specific approach")
    if expected_era:
        print(f"   e. Use era-optimized approaches automatically")
    print(f"   a. Try all recommended approaches")
    print(f"   s. Skip analysis and use standard processing")

    while True:
        try:
            choice = input(
                f"\n👤 Your choice (0-{len(displayed_approaches)}, {'e, ' if expected_era else ''}a, s): ").strip().lower()

            if choice == '0' or choice == '':
                return None  # Use AI recommendation
            elif choice == 'a':
                return "all_recommended"
            elif choice == 's':
                return "skip_analysis"
            elif choice == 'e' and expected_era:
                return "era_optimized"
            elif choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(displayed_approaches):
                    return displayed_approaches[index]
                else:
                    print(f"❌ Please enter a number between 1 and {len(displayed_approaches)}")
            else:
                valid_options = f"0, 1-{len(displayed_approaches)}, {'e, ' if expected_era else ''}a, or s"
                print(f"❌ Please enter a valid option ({valid_options})")

        except KeyboardInterrupt:
            print("\n👋 Cancelled by user")
            return None
        except Exception as e:
            print(f"❌ Error: {e}")
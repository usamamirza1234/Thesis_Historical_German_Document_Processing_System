# ===================================================================
# SMART OCR ENGINE
# ===================================================================
import logging
import os
import re
from typing import Optional, Tuple, List
import time
import os
import re
import cv2
import time
import json
import logging
import numpy as np
import pytesseract
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
from PIL import Image

import cv2
from pytesseract import pytesseract

from config.settings import ProcessingConfig
from models.data_models import OCRAttempt, ImageAnalysis, ImageQuality, ProcessingResult
from preprocessing.processing_pipeline import PipelineFactory
from preprocessing.smart_ocr_analyzer import SmartImageAnalyzer
import logging
import os
import re
from typing import Optional, Tuple, List
import time
import cv2
import numpy as np
import pytesseract
from datetime import datetime

from config.settings import ProcessingConfig, DocumentEra
from models.data_models import OCRAttempt, ImageAnalysis, ImageQuality, ProcessingResult, DocumentType
from preprocessing.processing_pipeline import PipelineFactory

class SmartOCREngine:
    """Intelligent OCR engine with approach selection"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.pipelines = PipelineFactory.create_all_pipelines()
        self.analyzer = SmartImageAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.performance_stats = {
            'total_attempts': 0,
            'successful_attempts': 0,
            'era_specific_successes': {},
            'approach_success_rates': {},
            'average_confidence': 0.0
        }

    def extract_text_smart(self, image_path: str,
                           user_choice: Optional[str] = None,
                           preferred_approaches: Optional[List[str]] = None,
                           expected_era: Optional[DocumentEra] = None) -> Tuple[str, List[OCRAttempt], ImageAnalysis]:
        """Enhanced text extraction with era-aware approach selection"""

        # Step 1: Enhanced image analysis
        analysis = self.analyzer.analyze_image_with_era_context(image_path, expected_era)
        self.logger.info(f"Enhanced analysis: {analysis.document_type.value}, quality: {analysis.image_quality.value}")

        if expected_era:
            self.logger.info(f"Era context: {expected_era.value}")

        # Step 2: Determine approaches with era awareness
        approaches = self._determine_optimal_approaches(
            analysis, user_choice, preferred_approaches, expected_era
        )

        # Step 3: Execute OCR attempts with adaptive strategy
        attempts = []
        best_result = {"text": "", "confidence": 0.0, "approach": "none"}

        for i, approach_name in enumerate(approaches):
            if approach_name in self.config.excluded_approaches:
                continue

            # Log attempt
            self.logger.info(f"Trying approach {i + 1}/{len(approaches)}: {approach_name}")

            attempt = self._try_approach_enhanced(image_path, approach_name, analysis, expected_era)
            attempts.append(attempt)

            # Update performance stats
            self._update_performance_stats(attempt, approach_name, expected_era)

            if attempt.success and attempt.confidence > best_result["confidence"]:
                best_result = {
                    "text": attempt.text_preview,
                    "confidence": attempt.confidence,
                    "approach": approach_name
                }

            # Enhanced early exit logic
            if self._should_exit_early(attempt, analysis, expected_era, i, len(approaches)):
                self.logger.info(f"Early exit with {approach_name} (confidence: {attempt.confidence:.3f})")
                break

        # Log final result
        success_rate = len([a for a in attempts if a.success]) / max(len(attempts), 1)
        self.logger.info(f"OCR completed: {len(attempts)} attempts, {success_rate:.1%} success rate")

        return best_result["text"], attempts, analysis

    def _determine_optimal_approaches(self, analysis: ImageAnalysis,
                                      user_choice: Optional[str],
                                      preferred_approaches: Optional[List[str]],
                                      expected_era: Optional[DocumentEra]) -> List[str]:
        """Determine optimal OCR approaches based on analysis and era"""

        # Handle user override
        if user_choice:
            if user_choice == "skip_analysis":
                return ["standard", "light", "direct"]
            elif user_choice == "all_recommended":
                return analysis.recommended_approaches
            elif user_choice == "era_optimized" and expected_era:
                return self.config.get_preferred_approaches_for_era(expected_era)
            elif user_choice in self.pipelines:
                return [user_choice]

        # Use preferred approaches if provided
        if preferred_approaches:
            return preferred_approaches[:self.config.max_approaches_to_try]

        # Era-specific approach selection
        if expected_era:
            era_approaches = self._get_era_optimized_approaches(expected_era, analysis)
            return era_approaches[:self.config.max_approaches_to_try]

        # Fallback to analysis-based recommendations
        return analysis.recommended_approaches[:self.config.max_approaches_to_try]

    def _get_era_optimized_approaches(self, era: DocumentEra, analysis: ImageAnalysis) -> List[str]:
        """Get era-optimized approach list"""

        base_approaches = self.config.get_preferred_approaches_for_era(era)

        # Adjust based on image analysis
        adjusted_approaches = []

        # For poor quality images, prioritize enhancement
        if analysis.image_quality in [ImageQuality.POOR, ImageQuality.FAIR]:
            if era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
                adjusted_approaches.extend(["fraktur_enhanced", "enhanced"])
            else:
                adjusted_approaches.extend(["enhanced", "standard"])

        # Add era-specific approaches
        adjusted_approaches.extend(base_approaches)

        # Add general fallbacks
        fallbacks = ["standard", "light", "direct"]
        for fallback in fallbacks:
            if fallback not in adjusted_approaches:
                adjusted_approaches.append(fallback)

        # Remove duplicates while preserving order
        seen = set()
        result = []
        for approach in adjusted_approaches:
            if approach not in seen and approach in self.pipelines:
                seen.add(approach)
                result.append(approach)

        return result

    def _try_approach_enhanced(self, image_path: str, approach_name: str,
                               analysis: ImageAnalysis, expected_era: Optional[DocumentEra]) -> OCRAttempt:
        """Enhanced OCR attempt with era-specific optimizations"""
        start_time = time.time()

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            # Apply pipeline processing
            if approach_name == "direct":
                processed_image = image
                pipeline_result = ProcessingResult(success=True, data=image)
            else:
                pipeline = self.pipelines.get(approach_name)
                if not pipeline:
                    raise ValueError(f"Unknown approach: {approach_name}")

                # Enhanced pipeline processing with era context
                pipeline_result = pipeline.process_with_era_context(
                    image,
                    expected_era=expected_era,
                    save_intermediate=self.config.save_intermediate_images,
                    output_dir=self.config.output_directory
                )

                if not pipeline_result.success:
                    raise Exception(pipeline_result.error)

                processed_image = pipeline_result.data

            # Save processed image for OCR
            processed_path = self._save_processed_image(processed_image, approach_name, image_path)

            # Perform enhanced OCR
            text = self._perform_enhanced_ocr(processed_path, approach_name, expected_era)

            # Clean up temporary file
            if not self.config.save_intermediate_images and os.path.exists(processed_path):
                os.remove(processed_path)

            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(text, analysis, expected_era, approach_name)
            processing_time = time.time() - start_time

            return OCRAttempt(
                approach_name=approach_name,
                success=True,
                text_length=len(text.strip()),
                confidence=confidence,
                processing_time=processing_time,
                text_preview=text,
                metadata={
                    'era_context': expected_era.value if expected_era else None,
                    'pipeline_success': pipeline_result.success,
                    'image_quality': analysis.image_quality.value,
                    'estimated_accuracy': self._estimate_text_accuracy(text)
                }
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.warning(f"Enhanced approach {approach_name} failed: {e}")

            return OCRAttempt(
                approach_name=approach_name,
                success=False,
                text_length=0,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e),
                metadata={
                    'era_context': expected_era.value if expected_era else None,
                    'failure_stage': 'processing' if 'pipeline' in str(e) else 'ocr'
                }
            )

    def _save_processed_image(self, processed_image: np.ndarray, approach_name: str, original_path: str) -> str:
        """Save processed image with proper naming"""
        if self.config.save_intermediate_images:
            processed_path = os.path.join(
                self.config.output_directory,
                f"processed_{approach_name}_{os.path.basename(original_path)}"
            )
        else:
            processed_path = os.path.join(
                self.config.output_directory,
                f"temp_{approach_name}_{int(time.time())}.png"
            )

        cv2.imwrite(processed_path, processed_image)
        return processed_path

    def _perform_enhanced_ocr(self, image_path: str, approach_name: str,
                              expected_era: Optional[DocumentEra]) -> str:
        """Enhanced OCR with era-specific configurations"""
        try:
            # Select OCR language based on era
            ocr_language = self._select_ocr_language(expected_era)

            # Era-specific Tesseract configurations
            tesseract_config = self._get_era_specific_config(expected_era, approach_name)

            # Primary OCR attempt
            text = pytesseract.image_to_string(
                image_path,
                lang=ocr_language,
                config=tesseract_config
            )

            # Era-specific post-processing
            text = self._post_process_text_for_era(text, expected_era)

            # Fallback OCR if result is poor
            if len(text.strip()) < 20:
                text = self._try_fallback_ocr(image_path, expected_era)

            return text

        except Exception as e:
            self.logger.error(f"Enhanced OCR failed: {e}")
            return ""

    def _select_ocr_language(self, expected_era: Optional[DocumentEra]) -> str:
        """Select optimal OCR language based on era"""
        if not expected_era:
            return self.config.ocr_language

        # Era-specific language selection
        if expected_era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
            return 'deu_frak'  # Fraktur script
        elif expected_era == DocumentEra.EARLY_BRD:
            return 'deu'  # Transitional period, prefer modern German
        else:
            return 'deu'  # Modern German

    def _get_era_specific_config(self, expected_era: Optional[DocumentEra], approach_name: str) -> str:
        """Get era-specific Tesseract configuration"""
        base_config = self.config.tesseract_config

        if not expected_era:
            return base_config

        # Era-specific PSM (Page Segmentation Mode) selection
        if expected_era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
            # Historical documents often have different layouts
            if approach_name in ['fraktur_traditional', 'fraktur_enhanced']:
                return base_config + " --psm 6"  # Uniform block of text
            else:
                return base_config + " --psm 4"  # Single column of text
        else:
            # Modern documents
            return base_config + " --psm 6"

    def _post_process_text_for_era(self, text: str, expected_era: Optional[DocumentEra]) -> str:
        """Apply era-specific post-processing to OCR text"""
        if not text or not expected_era:
            return text

        # Historical text corrections
        if expected_era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
            # Common Fraktur OCR errors
            text = re.sub(r'\bs\b', 'ſ', text)  # Long s correction
            text = re.sub(r'(?<=[a-z])s(?=[a-z])', 'ſ', text)  # Internal long s
            text = text.replace('ſs', 'ß')  # eszett correction

        # General German corrections
        text = re.sub(r'\b([Aa])e\b', r'\1ä', text)  # ae -> ä
        text = re.sub(r'\b([Oo])e\b', r'\1ö', text)  # oe -> ö
        text = re.sub(r'\b([Uu])e\b', r'\1ü', text)  # ue -> ü

        return text

    def _try_fallback_ocr(self, image_path: str, expected_era: Optional[DocumentEra]) -> str:
        """Try fallback OCR configurations"""
        fallback_text = ""

        # Try fallback languages
        for fallback_lang in self.config.fallback_languages:
            if fallback_lang != self._select_ocr_language(expected_era):
                try:
                    current_text = pytesseract.image_to_string(
                        image_path,
                        lang=fallback_lang,
                        config=self.config.tesseract_config
                    )
                    if len(current_text.strip()) > len(fallback_text.strip()):
                        fallback_text = current_text
                except Exception as e:
                    self.logger.debug(f"Fallback OCR with {fallback_lang} failed: {e}")

        return fallback_text

    def _calculate_enhanced_confidence(self, text: str, analysis: ImageAnalysis,
                                       expected_era: Optional[DocumentEra], approach_name: str) -> float:
        """Enhanced confidence calculation with era context"""
        if not text or not text.strip():
            return 0.0

        confidence = 0.0
        text_lower = text.lower()

        # Base length score
        length_score = min(len(text.strip()) / 500.0, 1.0)
        confidence += length_score * 0.2

        # Era-specific vocabulary scoring
        if expected_era:
            era_score = self._calculate_era_vocabulary_score(text_lower, expected_era)
            confidence += era_score * 0.3
        else:
            # Generic German words
            german_words = ['der', 'die', 'das', 'und', 'ein', 'eine', 'zu', 'von', 'mit']
            german_matches = sum(1 for word in german_words if word in text_lower)
            german_score = min(german_matches / 8.0, 1.0)
            confidence += german_score * 0.25

        # Document structure indicators
        structure_indicators = ['\n', '(', ')', '.', ',', ':', ';']
        structure_count = sum(text.count(indicator) for indicator in structure_indicators)
        structure_score = min(structure_count / 20.0, 1.0)
        confidence += structure_score * 0.15

        # Era-specific pattern bonuses
        if expected_era:
            pattern_score = self._calculate_era_pattern_score(text, expected_era)
            confidence += pattern_score * 0.2

        # Approach-specific adjustments
        approach_bonus = self._get_approach_confidence_bonus(approach_name, analysis, expected_era)
        confidence += approach_bonus

        # Image quality adjustments
        quality_multiplier = {
            ImageQuality.EXCELLENT: 1.1,
            ImageQuality.GOOD: 1.0,
            ImageQuality.FAIR: 0.9,
            ImageQuality.POOR: 0.8
        }.get(analysis.image_quality, 1.0)

        confidence *= quality_multiplier

        # Character quality assessment
        special_char_ratio = len(re.findall(r'[^\w\s\.\,\:\;\(\)\-äöüÄÖÜß]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            confidence *= 0.7

        return min(confidence, 1.0)

    def _calculate_era_vocabulary_score(self, text_lower: str, era: DocumentEra) -> float:
        """Calculate vocabulary score based on era-specific terms"""
        era_vocabularies = {
            DocumentEra.WEIMAR_REPUBLIC: [
                'republik', 'preußen', 'reichsarbeit', 'weimar', 'verfassung'
            ],
            DocumentEra.NAZI_PERIOD: [
                'reich', 'führer', 'deutsche', 'arbeitsfront', 'volk', 'rasse'
            ],
            DocumentEra.EARLY_BRD: [
                'bundesrepublik', 'grundgesetz', 'wiederaufbau', 'demokratie', 'bundesministerium'
            ],
            DocumentEra.MODERN_BRD: [
                'europäische', 'gemeinschaft', 'bundesrepublik', 'vereinigung', 'eu'
            ],
            DocumentEra.CONTEMPORARY: [
                'digital', 'internet', 'euro', 'europäische', 'union', 'globalisierung'
            ]
        }

        vocabulary = era_vocabularies.get(era, [])
        matches = sum(1 for term in vocabulary if term in text_lower)
        return min(matches / max(len(vocabulary), 1), 1.0)

    def _calculate_era_pattern_score(self, text: str, era: DocumentEra) -> float:
        """Calculate score based on era-specific patterns"""
        score = 0.0

        # Date patterns for era validation
        date_pattern = r'\b(19|20)\d{2}\b'
        dates = re.findall(date_pattern, text)

        if dates:
            era_ranges = {
                DocumentEra.WEIMAR_REPUBLIC: (1918, 1933),
                DocumentEra.NAZI_PERIOD: (1933, 1945),
                DocumentEra.EARLY_BRD: (1945, 1970),
                DocumentEra.MODERN_BRD: (1970, 1990),
                DocumentEra.CONTEMPORARY: (1990, 2030)
            }

            expected_range = era_ranges.get(era)
            if expected_range:
                for date_str in dates:
                    try:
                        year = int(date_str)
                        if expected_range[0] <= year <= expected_range[1]:
                            score += 0.2
                    except ValueError:
                        continue

        return min(score, 1.0)

    def _get_approach_confidence_bonus(self, approach_name: str, analysis: ImageAnalysis,
                                       expected_era: Optional[DocumentEra]) -> float:
        """Get confidence bonus based on approach appropriateness"""
        bonus = 0.0

        # Era-approach matching
        if expected_era:
            era_optimal_approaches = {
                DocumentEra.WEIMAR_REPUBLIC: ['fraktur_traditional', 'fraktur_enhanced'],
                DocumentEra.NAZI_PERIOD: ['fraktur_enhanced', 'official_document'],
                DocumentEra.EARLY_BRD: ['mixed_period', 'official_document'],
                DocumentEra.MODERN_BRD: ['official_document', 'modern_german'],
                DocumentEra.CONTEMPORARY: ['modern_german', 'light']
            }

            if approach_name in era_optimal_approaches.get(expected_era, []):
                bonus += 0.1

        # Quality-approach matching
        if analysis.image_quality == ImageQuality.POOR and approach_name == 'enhanced':
            bonus += 0.05
        elif analysis.image_quality == ImageQuality.EXCELLENT and approach_name == 'direct':
            bonus += 0.05

        return bonus

    def _should_exit_early(self, attempt: OCRAttempt, analysis: ImageAnalysis,
                           expected_era: Optional[DocumentEra], attempt_index: int, total_attempts: int) -> bool:
        """Enhanced early exit logic"""

        # Standard early exit conditions
        if (attempt.confidence > self.config.early_exit_confidence and
                attempt.text_length > self.config.early_exit_min_length):
            return True

        # Era-specific early exit adjustments
        if expected_era and attempt.success:
            # Lower threshold for historical documents (they're often harder to process)
            if expected_era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
                if attempt.confidence > 0.7 and attempt.text_length > 80:
                    return True

            # Higher threshold for contemporary documents (should be easier)
            elif expected_era == DocumentEra.CONTEMPORARY:
                if attempt.confidence > 0.85 and attempt.text_length > 120:
                    return True

        # Don't exit early if we haven't tried era-specific approaches yet
        if expected_era and attempt_index < 2:
            return False

        return False

    def _estimate_text_accuracy(self, text: str) -> float:
        """Estimate the accuracy of extracted text"""
        if not text:
            return 0.0

        # Simple heuristics for text quality
        total_chars = len(text)
        if total_chars == 0:
            return 0.0

        # Count problematic characters
        problematic_chars = len(re.findall(r'[^\w\s\.\,\:\;\(\)\-äöüÄÖÜßàáâãèéêëìíîïòóôõùúûü]', text))

        # Count reasonable words (length > 1, contains letters)
        words = re.findall(r'\b[a-zA-ZäöüÄÖÜß]{2,}\b', text)
        word_ratio = len(words) / max(total_chars / 5, 1)  # Approximate words per chars

        # Calculate accuracy estimate
        char_quality = 1.0 - (problematic_chars / total_chars)
        word_quality = min(word_ratio, 1.0)

        return (char_quality * 0.7 + word_quality * 0.3)

    def _update_performance_stats(self, attempt: OCRAttempt, approach_name: str,
                                  expected_era: Optional[DocumentEra]):
        """Update performance statistics"""
        self.performance_stats['total_attempts'] += 1

        if attempt.success:
            self.performance_stats['successful_attempts'] += 1

            # Era-specific success tracking
            if expected_era:
                era_key = expected_era.value
                if era_key not in self.performance_stats['era_specific_successes']:
                    self.performance_stats['era_specific_successes'][era_key] = 0
                self.performance_stats['era_specific_successes'][era_key] += 1

            # Approach success tracking
            if approach_name not in self.performance_stats['approach_success_rates']:
                self.performance_stats['approach_success_rates'][approach_name] = {'total': 0, 'success': 0}
            self.performance_stats['approach_success_rates'][approach_name]['total'] += 1
            self.performance_stats['approach_success_rates'][approach_name]['success'] += 1

            # Update average confidence
            total_successful = self.performance_stats['successful_attempts']
            current_avg = self.performance_stats['average_confidence']
            self.performance_stats['average_confidence'] = (
                    (current_avg * (total_successful - 1) + attempt.confidence) / total_successful
            )
        else:
            # Track failed attempts too
            if approach_name not in self.performance_stats['approach_success_rates']:
                self.performance_stats['approach_success_rates'][approach_name] = {'total': 0, 'success': 0}
            self.performance_stats['approach_success_rates'][approach_name]['total'] += 1

    def get_performance_statistics(self) -> dict:
        """Get comprehensive performance statistics"""
        total_attempts = self.performance_stats['total_attempts']
        if total_attempts == 0:
            return self.performance_stats

        # Calculate success rates
        overall_success_rate = self.performance_stats['successful_attempts'] / total_attempts

        # Calculate approach success rates
        approach_rates = {}
        for approach, stats in self.performance_stats['approach_success_rates'].items():
            if stats['total'] > 0:
                approach_rates[approach] = stats['success'] / stats['total']

        return {
            **self.performance_stats,
            'overall_success_rate': overall_success_rate,
            'approach_success_rates_calculated': approach_rates
        }
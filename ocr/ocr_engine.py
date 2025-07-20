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


class SmartOCREngine:
    """Intelligent OCR engine with approach selection"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.pipelines = PipelineFactory.create_all_pipelines()
        self.analyzer = SmartImageAnalyzer()
        self.logger = logging.getLogger(__name__)

    def extract_text_smart(self, image_path: str, user_choice: Optional[str] = None) -> Tuple[
        str, List[OCRAttempt], ImageAnalysis]:
        """Extract text using smart approach selection"""

        # Step 1: Analyze image
        analysis = self.analyzer.analyze_image(image_path)
        self.logger.info(f"Image analysis: {analysis.document_type.value}, quality: {analysis.image_quality.value}")

        # Step 2: Determine approaches to try
        if user_choice == "skip_analysis":
            approaches = ["standard", "light", "direct"]
        elif user_choice == "all_recommended":
            approaches = analysis.recommended_approaches
        elif user_choice and user_choice in self.pipelines:
            approaches = [user_choice]
        elif self.config.preferred_approaches:
            approaches = self.config.preferred_approaches
        else:
            approaches = analysis.recommended_approaches[:self.config.max_approaches_to_try]

        # Step 3: Try approaches
        attempts = []
        best_result = {"text": "", "confidence": 0.0, "approach": "none"}

        for approach_name in approaches:
            if approach_name in self.config.excluded_approaches:
                continue

            attempt = self._try_approach(image_path, approach_name, analysis)
            attempts.append(attempt)

            if attempt.success and attempt.confidence > best_result["confidence"]:
                best_result = {
                    "text": attempt.text_preview,  # This will be the full text
                    "confidence": attempt.confidence,
                    "approach": approach_name
                }

            # Early exit check
            if (attempt.confidence > self.config.early_exit_confidence and
                    attempt.text_length > self.config.early_exit_min_length):
                self.logger.info(f"Early exit with excellent result from {approach_name}")
                break

        return best_result["text"], attempts, analysis

    def _try_approach(self, image_path: str, approach_name: str, analysis: ImageAnalysis) -> OCRAttempt:
        """Try a specific OCR approach"""
        start_time = time.time()

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")

            # Apply pipeline
            if approach_name == "direct":
                processed_image = image
                pipeline_result = ProcessingResult(success=True, data=image)
            else:
                pipeline = self.pipelines.get(approach_name)
                if not pipeline:
                    raise ValueError(f"Unknown approach: {approach_name}")

                pipeline_result = pipeline.process(
                    image,
                    save_intermediate=self.config.save_intermediate_images,
                    output_dir=self.config.output_directory
                )

                if not pipeline_result.success:
                    raise Exception(pipeline_result.error)

                processed_image = pipeline_result.data

            # Save processed image if needed
            if self.config.save_intermediate_images:
                processed_path = os.path.join(
                    self.config.output_directory,
                    f"processed_{approach_name}_{os.path.basename(image_path)}"
                )
                cv2.imwrite(processed_path, processed_image)
            else:
                # Save to temporary file for OCR
                processed_path = os.path.join(
                    self.config.output_directory,
                    f"temp_{approach_name}.png"
                )
                cv2.imwrite(processed_path, processed_image)

            # Perform OCR
            text = self._perform_ocr(processed_path)

            # Clean up temporary file
            if not self.config.save_intermediate_images and os.path.exists(processed_path):
                os.remove(processed_path)

            # Calculate confidence
            confidence = self._calculate_confidence(text, analysis)
            processing_time = time.time() - start_time

            return OCRAttempt(
                approach_name=approach_name,
                success=True,
                text_length=len(text.strip()),
                confidence=confidence,
                processing_time=processing_time,
                text_preview=text  # Store full text here
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.warning(f"Approach {approach_name} failed: {e}")

            return OCRAttempt(
                approach_name=approach_name,
                success=False,
                text_length=0,
                confidence=0.0,
                processing_time=processing_time,
                error=str(e)
            )

    def _perform_ocr(self, image_path: str) -> str:
        """Perform OCR on processed image"""
        try:
            # Try primary language
            text = pytesseract.image_to_string(
                image_path,
                lang=self.config.ocr_language,
                config=self.config.tesseract_config
            )

            # If result is poor, try fallback languages
            if len(text.strip()) < 20:
                for fallback_lang in self.config.fallback_languages:
                    if fallback_lang != self.config.ocr_language:
                        fallback_text = pytesseract.image_to_string(
                            image_path,
                            lang=fallback_lang,
                            config=self.config.tesseract_config
                        )
                        if len(fallback_text.strip()) > len(text.strip()):
                            text = fallback_text

            return text

        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return ""

    def _calculate_confidence(self, text: str, analysis: ImageAnalysis) -> float:
        """Calculate confidence score for OCR result"""
        if not text or not text.strip():
            return 0.0

        confidence = 0.0
        text_lower = text.lower()

        # Length bonus
        length_score = min(len(text.strip()) / 500.0, 1.0)
        confidence += length_score * 0.25

        # German words bonus
        german_words = [
            'der', 'die', 'das', 'und', 'ein', 'eine', 'zu', 'von', 'mit',
            'staatslich', 'anerkannt', 'bundesminister', 'vom', 'für'
        ]
        german_matches = sum(1 for word in german_words if word in text_lower)
        german_score = min(german_matches / 8.0, 1.0)
        confidence += german_score * 0.35

        # Structure bonus
        structure_indicators = ['\n', '(', ')', '.', ',', ':', ';']
        structure_count = sum(text.count(indicator) for indicator in structure_indicators)
        structure_score = min(structure_count / 20.0, 1.0)
        confidence += structure_score * 0.15

        # Date pattern bonus
        date_patterns = [
            r'\d{1,2}\.\s*[A-Za-z]+\s*\d{4}',
            r'Stand\s+vom',
            r'vom\s+\d{1,2}\.\s*\d{1,2}\.\s*\d{4}'
        ]
        date_matches = sum(1 for pattern in date_patterns if re.search(pattern, text))
        date_score = min(date_matches / 3.0, 1.0)
        confidence += date_score * 0.15

        # Analysis-based adjustments
        if analysis.image_quality == ImageQuality.POOR:
            confidence *= 0.8
        elif analysis.image_quality == ImageQuality.EXCELLENT:
            confidence *= 1.1

        # Character quality assessment
        special_char_ratio = len(re.findall(r'[^\w\s\.\,\:\;\(\)\-]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            confidence *= 0.7

        return min(confidence, 1.0)


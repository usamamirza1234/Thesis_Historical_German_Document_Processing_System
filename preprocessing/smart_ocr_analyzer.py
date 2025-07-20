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

from models.data_models import ImageAnalysis, ImageQuality, DocumentType


# ===================================================================
# SMART IMAGE ANALYZER
# ===================================================================

class SmartImageAnalyzer:
    """Intelligent analyzer for document images"""

    def __init__(self):
        self.german_words = [
            'der', 'die', 'das', 'und', 'ein', 'eine', 'zu', 'den', 'von', 'mit',
            'ist', 'auf', 'für', 'als', 'sich', 'sie', 'nicht', 'werden', 'haben',
            'staatslich', 'anerkannt', 'bundesminister', 'reichsminister', 'vom'
        ]

        self.fraktur_indicators = ['ſ', 'ß', 'ff', 'fi', 'fl']
        self.official_terms = ['bundesminister', 'reichsminister', 'ministerium', 'verordnung']

    def analyze_image(self, image_path: str) -> ImageAnalysis:
        """Perform comprehensive image analysis"""

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        height, width = gray.shape

        # Basic metrics
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        has_inverted = brightness < 127

        # Quality assessment
        quality = self._assess_quality(contrast, width * height)

        # Text characteristics
        text_density = self._estimate_text_density(gray)
        text_size = self._estimate_text_size(gray)
        noise_level = self._estimate_noise(gray)

        # Document type detection
        doc_type, type_confidence = self._detect_document_type(image_path, gray)

        # Generate recommendations
        recommendations, reasoning = self._generate_recommendations(
            doc_type, quality, has_inverted, brightness, contrast, text_size, noise_level
        )

        return ImageAnalysis(
            document_type=doc_type,
            confidence=type_confidence,
            image_quality=quality,
            brightness=brightness,
            contrast=contrast,
            noise_level=noise_level,
            text_density=text_density,
            has_inverted_text=has_inverted,
            estimated_text_size=text_size,
            resolution=(width, height),
            color_mode="color" if len(image.shape) == 3 else "grayscale",
            recommended_approaches=recommendations,
            reasoning=reasoning
        )

    def _assess_quality(self, contrast: float, resolution: int) -> ImageQuality:
        """Assess overall image quality"""
        score = 0

        if contrast > 60:
            score += 3
        elif contrast > 40:
            score += 2
        elif contrast > 20:
            score += 1

        if resolution > 1000000:
            score += 2
        elif resolution > 500000:
            score += 1

        if score >= 4:
            return ImageQuality.EXCELLENT
        elif score >= 3:
            return ImageQuality.GOOD
        elif score >= 2:
            return ImageQuality.FAIR
        else:
            return ImageQuality.POOR

    def _estimate_text_density(self, gray_image: np.ndarray) -> float:
        """Estimate text density in image"""
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        total_area = gray_image.shape[0] * gray_image.shape[1]
        text_area = sum(cv2.contourArea(contour) for contour in contours)

        return min(text_area / total_area, 1.0)

    def _estimate_text_size(self, gray_image: np.ndarray) -> str:
        """Estimate text size"""
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return "medium"

        heights = [cv2.boundingRect(contour)[3] for contour in contours if cv2.contourArea(contour) > 50]

        if not heights:
            return "medium"

        avg_height = np.mean(heights)

        if avg_height < 20:
            return "small"
        elif avg_height > 40:
            return "large"
        else:
            return "medium"

    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level"""
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 1.0)

    def _detect_document_type(self, image_path: str, gray_image: np.ndarray) -> Tuple[DocumentType, float]:
        """Detect document type from image"""

        # Quick OCR sample
        sample_text = self._quick_ocr_sample(gray_image)

        scores = {doc_type: 0.0 for doc_type in DocumentType}

        if sample_text:
            text_lower = sample_text.lower()

            # Fraktur indicators
            fraktur_matches = sum(1 for indicator in self.fraktur_indicators if indicator in sample_text)
            scores[DocumentType.FRAKTUR] += fraktur_matches * 0.3

            # Official document terms
            official_matches = sum(1 for term in self.official_terms if term in text_lower)
            scores[DocumentType.OFFICIAL_DOCUMENT] += official_matches * 0.4

            # German words
            german_matches = sum(1 for word in self.german_words if word in text_lower)
            if german_matches > 3:
                scores[DocumentType.MODERN_GERMAN] += 0.4

            # Mixed period detection
            if fraktur_matches > 0 and german_matches > 2:
                scores[DocumentType.MIXED_PERIOD] += 0.3

        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]

        if confidence < 0.3:
            return DocumentType.MODERN_GERMAN, 0.5

        return best_type, min(confidence, 1.0)

    def _quick_ocr_sample(self, gray_image: np.ndarray) -> str:
        """Quick OCR sample for type detection"""
        try:
            # Take center sample
            h, w = gray_image.shape
            sample = gray_image[h // 4:3 * h // 4, w // 4:3 * w // 4]

            # Try quick OCR
            text = pytesseract.image_to_string(sample, lang='deu', config='--psm 6')
            return text.strip()
        except:
            return ""

    def _generate_recommendations(self, doc_type: DocumentType, quality: ImageQuality,
                                  has_inverted: bool, brightness: float, contrast: float,
                                  text_size: str, noise_level: float) -> Tuple[List[str], List[str]]:
        """Generate approach recommendations"""

        recommendations = []
        reasoning = []

        # Base recommendations by document type
        if doc_type == DocumentType.FRAKTUR:
            recommendations.extend(["fraktur_traditional", "fraktur_enhanced"])
            reasoning.append("Detected Fraktur script - using specialized preprocessing")

        elif doc_type == DocumentType.OFFICIAL_DOCUMENT:
            recommendations.extend(["official_document", "german_italic"])
            reasoning.append("Official document detected - using government document optimization")

        elif doc_type == DocumentType.MIXED_PERIOD:
            recommendations.extend(["mixed_period", "standard"])
            reasoning.append("Mixed text styles detected - using balanced approach")

        else:  # Modern German or unknown
            recommendations.extend(["modern_german", "standard"])
            reasoning.append("Modern German text - using standard preprocessing")

        # Quality adjustments
        if quality == ImageQuality.POOR:
            recommendations.insert(0, "enhanced")
            reasoning.append("Poor image quality - prioritizing noise reduction")

        elif quality == ImageQuality.EXCELLENT:
            recommendations.insert(0, "direct")
            reasoning.append("Excellent image quality - trying direct OCR first")

        # Specific adjustments
        if has_inverted:
            if "minimal" not in recommendations:
                recommendations.insert(1, "minimal")
            reasoning.append("Inverted image detected - using inversion preprocessing")

        if contrast < 30:
            recommendations.insert(0, "enhanced")
            reasoning.append("Low contrast - using contrast enhancement")

        if noise_level > 0.6:
            if "enhanced" not in recommendations:
                recommendations.insert(0, "enhanced")
            reasoning.append("High noise level - using noise reduction")

        # Always include fallbacks
        fallbacks = ["light", "standard", "direct"]
        for fallback in fallbacks:
            if fallback not in recommendations:
                recommendations.append(fallback)

        return recommendations[:6], reasoning


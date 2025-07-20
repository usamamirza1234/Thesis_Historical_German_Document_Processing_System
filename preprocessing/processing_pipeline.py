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

from models.data_models import ProcessingResult
from preprocessing.preprocessor import ProcessingStep, DirectOCRStep, ScaleStep, InvertStep, GrayscaleStep, \
    AddBordersStep, EnhanceContrastStep, DenoiseStep, AutoThresholdStep, BinarizeStep


# ===================================================================
# PROCESSING PIPELINES
# ===================================================================

class ProcessingPipeline:
    """Pipeline that applies multiple processing steps"""

    def __init__(self, name: str, steps: List[ProcessingStep]):
        self.name = name
        self.steps = steps

    def process(self, image: np.ndarray, save_intermediate: bool = False, output_dir: str = "") -> ProcessingResult:
        """Apply all steps in the pipeline"""
        start_time = time.time()

        try:
            current_image = image.copy()

            for i, step in enumerate(self.steps):
                current_image = step.apply(current_image)

                if save_intermediate and output_dir:
                    step_path = os.path.join(output_dir, f"{self.name}_step_{i + 1}_{step.name}.png")
                    cv2.imwrite(step_path, current_image)

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=current_image,
                processing_time=processing_time,
                metadata={'pipeline': self.name, 'steps': len(self.steps)}
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={'pipeline': self.name, 'failed_at': str(e)}
            )


class PipelineFactory:
    """Factory for creating processing pipelines"""

    @staticmethod
    def create_all_pipelines() -> Dict[str, ProcessingPipeline]:
        """Create all available processing pipelines"""

        pipelines = {}

        # Direct OCR
        pipelines["direct"] = ProcessingPipeline("direct", [DirectOCRStep()])

        # Minimal processing
        pipelines["minimal"] = ProcessingPipeline("minimal", [
            InvertStep(),
            ScaleStep(2.5)
        ])

        # Light processing
        pipelines["light"] = ProcessingPipeline("light", [
            ScaleStep(2.0),
            GrayscaleStep(),
            AddBordersStep(50)
        ])

        # Standard processing
        pipelines["standard"] = ProcessingPipeline("standard", [
            GrayscaleStep(),
            ScaleStep(2.5),
            EnhanceContrastStep(1.2, 10),
            AddBordersStep(30)
        ])

        # Enhanced processing
        pipelines["enhanced"] = ProcessingPipeline("enhanced", [
            GrayscaleStep(),
            ScaleStep(3.0),
            EnhanceContrastStep(1.3, 15),
            DenoiseStep(10),
            AutoThresholdStep(),
            AddBordersStep(40)
        ])

        # Fraktur traditional
        pipelines["fraktur_traditional"] = ProcessingPipeline("fraktur_traditional", [
            GrayscaleStep(),
            InvertStep(),
            ScaleStep(3.0),
            EnhanceContrastStep(1.3, 20),
            AddBordersStep(50)
        ])

        # Fraktur enhanced
        pipelines["fraktur_enhanced"] = ProcessingPipeline("fraktur_enhanced", [
            GrayscaleStep(),
            InvertStep(),
            ScaleStep(2.5),
            DenoiseStep(10),
            EnhanceContrastStep(1.2, 15),
            AddBordersStep(40)
        ])

        # Modern German
        pipelines["modern_german"] = ProcessingPipeline("modern_german", [
            GrayscaleStep(),
            ScaleStep(2.0),
            EnhanceContrastStep(1.2, 10),
            AddBordersStep(30)
        ])

        # German italic
        pipelines["german_italic"] = ProcessingPipeline("german_italic", [
            GrayscaleStep(),
            ScaleStep(2.5),
            EnhanceContrastStep(1.3, 15),
            AddBordersStep(40)
        ])

        # Official document
        pipelines["official_document"] = ProcessingPipeline("official_document", [
            GrayscaleStep(),
            ScaleStep(2.0),
            EnhanceContrastStep(1.1, 5),
            AddBordersStep(25)
        ])

        # Mixed period
        pipelines["mixed_period"] = ProcessingPipeline("mixed_period", [
            GrayscaleStep(),
            ScaleStep(2.2),
            EnhanceContrastStep(1.15, 8),
            AutoThresholdStep(),
            AddBordersStep(35)
        ])

        # Newspaper
        pipelines["newspaper"] = ProcessingPipeline("newspaper", [
            GrayscaleStep(),
            ScaleStep(2.0),
            EnhanceContrastStep(1.1, 5),
            BinarizeStep(127),
            AddBordersStep(30)
        ])

        # High quality
        pipelines["high_quality"] = ProcessingPipeline("high_quality", [
            GrayscaleStep(),
            ScaleStep(1.5),
            EnhanceContrastStep(1.1, 5),
            AddBordersStep(20)
        ])

        return pipelines


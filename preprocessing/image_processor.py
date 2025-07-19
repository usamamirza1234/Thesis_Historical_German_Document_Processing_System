from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import os
from preprocessing.processing_steps import ProcessingStep
from models.data_models import ProcessingResult
from utils.file_handlers import ImageHandler
import time
import logging

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Pipeline for applying multiple processing steps to images"""

    def __init__(self, steps: List[ProcessingStep], save_intermediate: bool = False,
                 output_dir: str = "temp"):
        self.steps = steps
        self.save_intermediate = save_intermediate
        self.output_dir = output_dir

        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)

    def process(self, image: np.ndarray, image_name: str = "image") -> ProcessingResult:
        """Apply all processing steps to image"""
        start_time = time.time()

        try:
            current_image = image.copy()

            for i, step in enumerate(self.steps):
                logger.debug(f"Applying step {i + 1}/{len(self.steps)}: {step}")

                try:
                    current_image = step.apply(current_image)

                    # Save intermediate result if requested
                    if self.save_intermediate:
                        step_filename = f"{image_name}_step_{i + 1:02d}_{step.name}.jpg"
                        step_path = os.path.join(self.output_dir, step_filename)
                        ImageHandler.save_image(current_image, step_path)

                except Exception as e:
                    logger.warning(f"Step {step.name} failed: {e}")
                    # Continue with previous image
                    continue

            processing_time = time.time() - start_time

            return ProcessingResult(
                success=True,
                data=current_image,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Pipeline processing failed: {e}")

            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )

    def add_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the pipeline"""
        self.steps.append(step)

    def remove_step(self, step_name: str) -> bool:
        """Remove a processing step by name"""
        original_length = len(self.steps)
        self.steps = [step for step in self.steps if step.name != step_name]
        return len(self.steps) < original_length

    def get_step_names(self) -> List[str]:
        """Get names of all processing steps"""
        return [step.name for step in self.steps]


class ImageProcessor:
    """High-level interface for image processing operations"""

    def __init__(self, config: 'ProcessingConfig'):
        self.config = config
        self.default_pipeline = self._create_default_pipeline()

    def _create_default_pipeline(self) -> ProcessingPipeline:
        """Create default processing pipeline"""
        from preprocessing.processing_steps import (
            WhiteSpaceRemovalStep, InvertImageStep, RescaleImageStep,
            GrayscaleStep, NoiseRemovalStep, BorderRemovalStep, AddBordersStep
        )

        steps = []

        # Add white space removal if enabled
        if self.config.enable_white_space_removal:
            steps.append(WhiteSpaceRemovalStep(
                white_threshold=self.config.white_threshold,
                min_content_area=self.config.min_content_area
            ))

        # Standard processing steps
        steps.extend([
            InvertImageStep(),
            RescaleImageStep(scale_factor=self.config.image_scale_factor),
            GrayscaleStep(),
            NoiseRemovalStep(),
            BorderRemovalStep(),
            AddBordersStep()
        ])

        return ProcessingPipeline(
            steps=steps,
            save_intermediate=self.config.enable_debug,
            output_dir=self.config.output_directory
        )

    def _is_image_in_output_directory(self, image_path: str) -> bool:
        """Check if the image is already in the output directory"""
        try:
            # Get the absolute paths to compare properly
            image_abs_path = os.path.abspath(image_path)
            output_abs_path = os.path.abspath(self.config.output_directory)

            # Check if the image path is within or is the output directory
            return image_abs_path.startswith(output_abs_path)
        except Exception as e:
            logger.debug(f"Error checking image path: {e}")
            return False

    def _should_skip_processing(self, image_path: str) -> bool:
        """Determine if processing should be skipped for this image"""
        # Check if image is in output directory
        if self._is_image_in_output_directory(image_path):
            return True

        # Check if it's a temporary or processed file based on filename patterns
        filename = os.path.basename(image_path).lower()
        skip_patterns = [
            'processed_',
            'temp_page_',
            '_step_',
            'debug_',
            'intermediate_',
            'pipeline_'
        ]

        for pattern in skip_patterns:
            if pattern in filename:
                return True

        return False

    def process_image(self, image_path: str,
                      custom_pipeline: Optional[ProcessingPipeline] = None) -> ProcessingResult:
        """Process image using default or custom pipeline"""
        try:
            # Check if we should skip processing this image
            if self._should_skip_processing(image_path):
                logger.warning(
                    f"Skipping processing - image appears to be already processed or in output directory: {image_path}")

                # Return the image as-is without processing
                try:
                    image = ImageHandler.load_image(image_path)
                    return ProcessingResult(
                        success=True,
                        data=image,
                        processing_time=0.0,
                        warnings=[f"Skipped processing - image in output directory"]
                    )
                except Exception as e:
                    return ProcessingResult(
                        success=False,
                        error=f"Failed to load already processed image: {e}",
                        processing_time=0.0
                    )

            # Load image
            image = ImageHandler.load_image(image_path)

            # Use custom pipeline or default
            pipeline = custom_pipeline or self.default_pipeline

            # Get image name for intermediate files
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Process image
            result = pipeline.process(image, image_name)

            if result.success:
                logger.info(f"Successfully processed {image_path} in {result.processing_time:.2f}s")
            else:
                logger.error(f"Failed to process {image_path}: {result.error}")

            return result

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return ProcessingResult(success=False, error=str(e))

    def force_process_image(self, image_path: str,
                            custom_pipeline: Optional[ProcessingPipeline] = None) -> ProcessingResult:
        """Force process image even if it's in output directory (bypass skip check)"""
        try:
            logger.info(f"Force processing image: {image_path}")

            # Load image
            image = ImageHandler.load_image(image_path)

            # Use custom pipeline or default
            pipeline = custom_pipeline or self.default_pipeline

            # Get image name for intermediate files
            image_name = os.path.splitext(os.path.basename(image_path))[0]

            # Process image
            result = pipeline.process(image, image_name)

            if result.success:
                logger.info(f"Successfully force processed {image_path} in {result.processing_time:.2f}s")
            else:
                logger.error(f"Failed to force process {image_path}: {result.error}")

            return result

        except Exception as e:
            logger.error(f"Error force processing image {image_path}: {e}")
            return ProcessingResult(success=False, error=str(e))

    def get_processing_info(self, image_path: str) -> Dict[str, Any]:
        """Get information about whether an image would be processed"""
        info = {
            'image_path': image_path,
            'exists': os.path.exists(image_path),
            'in_output_directory': self._is_image_in_output_directory(image_path),
            'would_skip': self._should_skip_processing(image_path),
            'output_directory': self.config.output_directory,
            'filename': os.path.basename(image_path)
        }

        if info['would_skip']:
            info['skip_reason'] = []
            if info['in_output_directory']:
                info['skip_reason'].append('in_output_directory')

            filename = os.path.basename(image_path).lower()
            skip_patterns = ['processed_', 'temp_page_', '_step_', 'debug_', 'intermediate_', 'pipeline_']
            for pattern in skip_patterns:
                if pattern in filename:
                    info['skip_reason'].append(f'filename_contains_{pattern.strip("_")}')

        return info

    def clean_output_directory(self, patterns: Optional[List[str]] = None) -> int:
        """Clean up files in output directory matching certain patterns"""
        if patterns is None:
            patterns = ['processed_*', 'temp_page_*', '*_step_*', 'debug_*']

        cleaned_count = 0

        try:
            import glob

            for pattern in patterns:
                full_pattern = os.path.join(self.config.output_directory, pattern)
                files_to_remove = glob.glob(full_pattern)

                for file_path in files_to_remove:
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                        logger.debug(f"Removed: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} files from output directory")

        except Exception as e:
            logger.error(f"Error cleaning output directory: {e}")

        return cleaned_count
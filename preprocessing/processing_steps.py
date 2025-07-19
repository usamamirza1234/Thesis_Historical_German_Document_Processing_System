from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProcessingStep(ABC):
    """Abstract base class for image processing steps"""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply processing step to image"""
        pass

    def __str__(self):
        return f"{self.name}({self.params})"


class InvertImageStep(ProcessingStep):
    """Invert image colors (black becomes white, white becomes black)"""

    def __init__(self, **kwargs):
        super().__init__("InvertImage", **kwargs)

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)


class RescaleImageStep(ProcessingStep):
    """Rescale image by given factor"""

    def __init__(self, scale_factor: float = 2.5, **kwargs):
        super().__init__("RescaleImage", scale_factor=scale_factor, **kwargs)
        self.scale_factor = scale_factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


class GrayscaleStep(ProcessingStep):
    """Convert image to grayscale"""

    def __init__(self, **kwargs):
        super().__init__("Grayscale", **kwargs)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


class BinarizeImageStep(ProcessingStep):
    """Convert image to binary (black and white)"""

    def __init__(self, threshold: int = 105, **kwargs):
        super().__init__("BinarizeImage", threshold=threshold, **kwargs)
        self.threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        return binary


class NoiseRemovalStep(ProcessingStep):
    """Remove noise from image using morphological operations"""

    def __init__(self, kernel_size: int = 1, **kwargs):
        super().__init__("NoiseRemoval", kernel_size=kernel_size, **kwargs)
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image


class WhiteSpaceRemovalStep(ProcessingStep):
    """Remove large white areas from document"""

    def __init__(self, white_threshold: int = 240, min_content_area: int = 1000, **kwargs):
        super().__init__("WhiteSpaceRemoval", white_threshold=white_threshold,
                         min_content_area=min_content_area, **kwargs)
        self.white_threshold = white_threshold
        self.min_content_area = min_content_area

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create binary mask for non-white areas
        _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Filter by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_content_area]

        if not valid_contours:
            return image

        # Find bounding box
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop image
        return image[y:y + h, x:x + w]


class BorderRemovalStep(ProcessingStep):
    """Remove borders from image"""

    def __init__(self, **kwargs):
        super().__init__("BorderRemoval", **kwargs)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return image[y:y + h, x:x + w]


class AddBordersStep(ProcessingStep):
    """Add white borders around image"""

    def __init__(self, border_size: int = 150, **kwargs):
        super().__init__("AddBorders", border_size=border_size, **kwargs)
        self.border_size = border_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        color = [255, 255, 255]  # White border
        return cv2.copyMakeBorder(image, self.border_size, self.border_size,
                                  self.border_size, self.border_size,
                                  cv2.BORDER_CONSTANT, value=color)


class FrakturPreprocessingStep(ProcessingStep):
    """Traditional Fraktur preprocessing (1920s-1940s)"""

    def __init__(self, scale_factor: float = 3.0, alpha: float = 1.3, beta: int = 20, **kwargs):
        super().__init__("FrakturPreprocessing", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Invert (Fraktur is typically black text on white)
        inverted = cv2.bitwise_not(gray)

        # Scale up significantly for Fraktur recognition
        scaled = cv2.resize(inverted, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Enhance contrast for old documents
        enhanced = cv2.convertScaleAbs(scaled, alpha=self.alpha, beta=self.beta)

        return enhanced


class FrakturEnhancedStep(ProcessingStep):
    """Enhanced Fraktur preprocessing with noise reduction"""

    def __init__(self, scale_factor: float = 2.5, denoise_h: int = 10,
                 alpha: float = 1.2, beta: int = 15, **kwargs):
        super().__init__("FrakturEnhanced", scale_factor=scale_factor,
                         denoise_h=denoise_h, alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.denoise_h = denoise_h
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        inverted = cv2.bitwise_not(gray)

        # Scale up
        scaled = cv2.resize(inverted, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Denoise specifically for old documents
        denoised = cv2.fastNlMeansDenoising(scaled, h=self.denoise_h)

        # Enhance contrast
        enhanced = cv2.convertScaleAbs(denoised, alpha=self.alpha, beta=self.beta)

        return enhanced


class ModernGermanStep(ProcessingStep):
    """Modern German text preprocessing (1950s-present)"""

    def __init__(self, scale_factor: float = 2.0, alpha: float = 1.2, beta: int = 10, **kwargs):
        super().__init__("ModernGerman", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Moderate scaling for modern text
        scaled = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Enhance contrast
        enhanced = cv2.convertScaleAbs(scaled, alpha=self.alpha, beta=self.beta)

        return enhanced


class GermanItalicStep(ProcessingStep):
    """German italic text preprocessing (good for official document headers)"""

    def __init__(self, scale_factor: float = 2.5, alpha: float = 1.3, beta: int = 15, **kwargs):
        super().__init__("GermanItalic", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Scale up for better italic recognition
        scaled = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Sharpen to enhance italic character edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(scaled, -1, kernel)

        # Enhance contrast specifically for italic
        enhanced = cv2.convertScaleAbs(sharpened, alpha=self.alpha, beta=self.beta)

        return enhanced


class MixedPeriodStep(ProcessingStep):
    """Mixed period preprocessing (documents with both old and new text styles)"""

    def __init__(self, scale_factor: float = 2.2, alpha: float = 1.15, beta: int = 8, **kwargs):
        super().__init__("MixedPeriod", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Balanced scaling
        scaled = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Gentle enhancement
        enhanced = cv2.convertScaleAbs(scaled, alpha=self.alpha, beta=self.beta)

        return enhanced


class OfficialDocumentStep(ProcessingStep):
    """Official document preprocessing (like BRD, government documents)"""

    def __init__(self, scale_factor: float = 2.0, alpha: float = 1.1, beta: int = 5, **kwargs):
        super().__init__("OfficialDocument", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Standard scaling for official documents
        scaled = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Moderate enhancement to preserve document structure
        enhanced = cv2.convertScaleAbs(scaled, alpha=self.alpha, beta=self.beta)

        return enhanced


class GermanHeaderStep(ProcessingStep):
    """Header/title preprocessing (often italic in German documents)"""

    def __init__(self, scale_factor: float = 2.3, **kwargs):
        super().__init__("GermanHeader", scale_factor=scale_factor, **kwargs)
        self.scale_factor = scale_factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Scale for header text
        scaled = cv2.resize(gray, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Light sharpening for headers
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(scaled, -1, kernel)

        return sharpened


class GermanOCRPreprocessingStep(ProcessingStep):
    """General German OCR preprocessing with automatic inversion detection"""

    def __init__(self, scale_factor: float = 2.0, alpha: float = 1.2, beta: int = 10, **kwargs):
        super().__init__("GermanOCRPreprocessing", scale_factor=scale_factor,
                         alpha=alpha, beta=beta, **kwargs)
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)

        # Sharpen the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)

        # Ensure proper contrast - auto-detect if inversion needed
        mean_intensity = np.mean(sharpened)
        if mean_intensity < 127:
            # Dark image, invert it
            sharpened = cv2.bitwise_not(sharpened)

        # Scale up for better recognition
        scaled = cv2.resize(sharpened, None, fx=self.scale_factor, fy=self.scale_factor,
                            interpolation=cv2.INTER_CUBIC)

        # Final enhancement
        enhanced = cv2.convertScaleAbs(scaled, alpha=self.alpha, beta=self.beta)

        # Final thresholding
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh


# GERMAN PIPELINE FACTORY

class GermanProcessingPipelineFactory:
    """Factory to create German-specific processing pipelines"""

    @staticmethod
    def create_fraktur_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create pipeline optimized for Fraktur text (1920s-1940s)"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            FrakturPreprocessingStep(),
            AddBordersStep(border_size=50)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)

    @staticmethod
    def create_modern_german_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create pipeline optimized for modern German text (1940s-present)"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            ModernGermanStep(),
            AddBordersStep(border_size=30)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)

    @staticmethod
    def create_german_italic_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create pipeline optimized for German italic text"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            GermanItalicStep(),
            AddBordersStep(border_size=40)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)

    @staticmethod
    def create_official_document_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create pipeline optimized for official German documents"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            OfficialDocumentStep(),
            GermanHeaderStep(),  # Good for headers/italic text
            AddBordersStep(border_size=25)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)

    @staticmethod
    def create_mixed_period_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create pipeline for documents with mixed text styles"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            MixedPeriodStep(),
            GermanOCRPreprocessingStep(),
            AddBordersStep(border_size=35)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)

    @staticmethod
    def create_comprehensive_german_pipeline(save_intermediate: bool = False, output_dir: str = "temp"):
        """Create comprehensive pipeline that handles multiple German text types"""
        from preprocessing.image_processor import ProcessingPipeline

        steps = [
            GrayscaleStep(),
            GermanOCRPreprocessingStep(),  # General German preprocessing
            AddBordersStep(border_size=30)
        ]
        return ProcessingPipeline(steps, save_intermediate, output_dir)


# USAGE EXAMPLE FOR YOUR DOCUMENT PROCESSOR

def get_german_preprocessing_approaches():
    """Get German-specific preprocessing approaches for your document processor"""

    approaches = []

    # Approach 1: Fraktur (for 1920s-1940s documents)
    approaches.append((
        "fraktur_traditional",
        GermanProcessingPipelineFactory.create_fraktur_pipeline()
    ))

    # Approach 2: Modern German (for 1940s-present)
    approaches.append((
        "modern_german",
        GermanProcessingPipelineFactory.create_modern_german_pipeline()
    ))

    # Approach 3: German Italic (for headers, official notices)
    approaches.append((
        "german_italic",
        GermanProcessingPipelineFactory.create_german_italic_pipeline()
    ))

    # Approach 4: Official Documents (like your BRD document)
    approaches.append((
        "official_document",
        GermanProcessingPipelineFactory.create_official_document_pipeline()
    ))

    # Approach 5: Mixed period (handles both old and new styles)
    approaches.append((
        "mixed_period",
        GermanProcessingPipelineFactory.create_mixed_period_pipeline()
    ))

    return approaches


# Integration with your existing _create_preprocessing_approaches method:
def _create_german_preprocessing_approaches(config):
    """Replace your _create_preprocessing_approaches method with this"""

    approaches = []

    # Direct OCR (no preprocessing)
    approaches.append(("direct", None))

    # German-specific approaches
    german_approaches = get_german_preprocessing_approaches()

    for name, pipeline in german_approaches:
        approaches.append((name, pipeline))

    return approaches
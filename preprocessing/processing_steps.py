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


import cv2
import numpy as np
from abc import ABC, abstractmethod


# ===================================================================
# IMAGE PROCESSING STEPS
# ===================================================================

class ProcessingStep(ABC):
    """Abstract base class for image processing steps"""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply processing step to image"""
        pass

    def __str__(self):
        return f"{self.name}({self.params})"


class DirectOCRStep(ProcessingStep):
    """No preprocessing - direct OCR"""

    def __init__(self):
        super().__init__("Direct")

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image


class GrayscaleStep(ProcessingStep):
    """Convert to grayscale"""

    def __init__(self):
        super().__init__("Grayscale")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


class InvertStep(ProcessingStep):
    """Invert image colors"""

    def __init__(self):
        super().__init__("Invert")

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.bitwise_not(image)


class ScaleStep(ProcessingStep):
    """Scale image by factor"""

    def __init__(self, factor: float = 2.5):
        super().__init__("Scale", factor=factor)
        self.factor = factor

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        new_size = (int(width * self.factor), int(height * self.factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)


class EnhanceContrastStep(ProcessingStep):
    """Enhance contrast"""

    def __init__(self, alpha: float = 1.2, beta: int = 10):
        super().__init__("EnhanceContrast", alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)


class DenoiseStep(ProcessingStep):
    """Remove noise from image"""

    def __init__(self, h: int = 10):
        super().__init__("Denoise", h=h)
        self.h = h

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.fastNlMeansDenoising(image, h=self.h)


class AddBordersStep(ProcessingStep):
    """Add white borders"""

    def __init__(self, size: int = 50):
        super().__init__("AddBorders", size=size)
        self.size = size

    def apply(self, image: np.ndarray) -> np.ndarray:
        color = [255, 255, 255]
        return cv2.copyMakeBorder(image, self.size, self.size, self.size, self.size,
                                  cv2.BORDER_CONSTANT, value=color)


class BinarizeStep(ProcessingStep):
    """Convert to binary (black and white)"""

    def __init__(self, threshold: int = 127):
        super().__init__("Binarize", threshold=threshold)
        self.threshold = threshold

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
        return binary


class AutoThresholdStep(ProcessingStep):
    """Automatic thresholding using Otsu's method"""

    def __init__(self):
        super().__init__("AutoThreshold")

    def apply(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

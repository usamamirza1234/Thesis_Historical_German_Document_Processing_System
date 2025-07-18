import os
from pathlib import Path
from typing import List, Optional
import hashlib
from PIL import Image
import cv2
import numpy as np


class FileHandler:
    """Handles file operations and validation"""

    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate if file exists and is accessible"""
        return os.path.exists(file_path) and os.path.isfile(file_path)

    @staticmethod
    def get_file_hash(file_path: str) -> str:
        """Generate hash for file caching"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    @staticmethod
    def ensure_directory(directory: str) -> None:
        """Ensure directory exists"""
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions"""
        return ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']

    @staticmethod
    def is_supported_file(file_path: str) -> bool:
        """Check if file extension is supported"""
        ext = Path(file_path).suffix.lower()
        return ext in FileHandler.get_supported_extensions()


class ImageHandler:
    """Handles image operations"""

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image with error handling"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        return image

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> None:
        """Save image with error handling"""
        FileHandler.ensure_directory(os.path.dirname(output_path))
        success = cv2.imwrite(output_path, image)
        if not success:
            raise ValueError(f"Failed to save image: {output_path}")
"""
Optimized Metadata Extraction System for Historical German Documents
Consolidated and optimized version with improved performance
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pytesseract
from PIL import Image
import os

import cv2
import os
import re
import pickle
import numpy as np
import pytesseract
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder




@dataclass
class ExtractedMetadata:
    """Structured metadata container with improved typing"""
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    profession: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    raw_text_preview: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'profession': self.profession,
            'confidence_scores': self.confidence_scores,
            'raw_text_preview': self.raw_text_preview
        }

class ImagePreprocessor:
    """
    ImagePreprocessor - Simple Guide

    This class helps prepare images for better text recognition (OCR).
    It applies various filters and adjustments to make text clearer and easier to read by computer programs.

    Core Functions:
    1. invert_image() - Flip Colors: Makes black text white, and white background black
    2. rescale_image(scale_factor) - Resize Image: Makes image bigger or smaller
    3. grayscale() - Remove Colors: Converts colorful image to black, white, and gray only
    4. binarize_image(threshold) - Make Pure Black & White: Converts gray pixels to pure black or white
    5. noise_removal() - Clean Up Spots: Removes small dots, specks, and unwanted marks
    6. thin_font()/thick_font() - Adjust Text Thickness: Make letters thinner or thicker
    7. deskew() - Straighten Tilted Text: Rotates image to make text lines horizontal
    8. remove_borders() - Crop to Content: Cuts away empty white space around text
    9. add_borders() - Add White Space: Adds padding around the image
    10. remove_white_areas() - NEW: Remove large white/blank areas from document
    """

    def __init__(self, image_path):
        """Initialize the preprocessor with an image path."""
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"The file '{image_path}' does not exist or cannot be read.")

    def display(self, im_path):
        """Display image with actual size using matplotlib."""
        dpi = 80
        im_data = plt.imread(im_path)

        height, width = im_data.shape[:2]

        # What size does the figure need to be in inches to fit the image?
        figsize = width / float(dpi), height / float(dpi)

        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])

        # Hide spines, ticks, etc.
        ax.axis('off')

        # Display the image.
        ax.imshow(im_data, cmap='gray')

        plt.show()

    def remove_white_areas(self, image=None, white_threshold=240, min_content_area=1000):
        """
        Remove large white/blank areas from document

        What it does: Detects and removes large white areas while preserving text content
        Variables:
        - white_threshold: Pixel intensity above which is considered "white" (0-255, default 240)
        - min_content_area: Minimum area in pixels to consider as content (default 1000)

        Effect: Focuses on actual document content, removes margins and blank spaces
        Returns: Cropped image with white areas removed
        """
        if image is None:
            image = self.original_image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create binary mask where non-white areas are marked
        # Values below white_threshold are considered content (non-white)
        _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours of non-white areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # If no contours found, return original image
            return image

        # Filter contours by area to remove noise
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_content_area]

        if not valid_contours:
            # If no valid contours, return original image
            return image

        # Find bounding box that encompasses all content areas
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add small padding to ensure we don't cut off text
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        # Crop the image to the content area
        if len(image.shape) == 3:
            cropped = image[y:y + h, x:x + w]
        else:
            cropped = image[y:y + h, x:x + w]

        return cropped

    def smart_crop_content(self, image=None, margin_threshold=0.95):
        """
        Alternative method: Smart cropping based on content density

        What it does: Analyzes rows and columns to find content boundaries
        Variables:
        - margin_threshold: Percentage of white pixels to consider as margin (default 0.95)

        Effect: More precise content detection, good for documents with varying layouts
        """
        if image is None:
            image = self.original_image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Calculate content density for each row and column
        height, width = gray.shape

        # For each row, calculate percentage of white pixels
        row_whiteness = []
        for i in range(height):
            white_pixels = np.sum(gray[i, :] > 240)
            whiteness_ratio = white_pixels / width
            row_whiteness.append(whiteness_ratio)

        # For each column, calculate percentage of white pixels
        col_whiteness = []
        for j in range(width):
            white_pixels = np.sum(gray[:, j] > 240)
            whiteness_ratio = white_pixels / height
            col_whiteness.append(whiteness_ratio)

        # Find content boundaries
        # Top boundary
        top = 0
        for i in range(height):
            if row_whiteness[i] < margin_threshold:
                top = i
                break

        # Bottom boundary
        bottom = height - 1
        for i in range(height - 1, -1, -1):
            if row_whiteness[i] < margin_threshold:
                bottom = i
                break

        # Left boundary
        left = 0
        for j in range(width):
            if col_whiteness[j] < margin_threshold:
                left = j
                break

        # Right boundary
        right = width - 1
        for j in range(width - 1, -1, -1):
            if col_whiteness[j] < margin_threshold:
                right = j
                break

        # Add small padding
        padding = 10
        top = max(0, top - padding)
        bottom = min(height - 1, bottom + padding)
        left = max(0, left - padding)
        right = min(width - 1, right + padding)

        # Crop the image
        if len(image.shape) == 3:
            cropped = image[top:bottom, left:right]
        else:
            cropped = image[top:bottom, left:right]

        return cropped

    def extract_fraktur_text(self, image_path):
        """Extract text from an image using Tesseract with the Fraktur language model."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file '{image_path}' does not exist.")

        # Load image
        image = cv2.imread(image_path)

        # Preprocess image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Optional: add more preprocessing like thresholding, denoising, etc.

        # Extract text using Fraktur model
        text = pytesseract.image_to_string(gray, lang='deu_frak')
        return text

    def invert_image(self, image=None):
        """
        1. Inverted Images - Flip Colors
        What it does: Makes black text white, and white background black
        When to use: Sometimes inverted images work better for text recognition
        Variables: None to change
        Effect: Complete color reversal
        """
        if image is None:
            image = self.original_image
        inverted_image = cv2.bitwise_not(image)
        return inverted_image

    def rescale_image(self, image=None, scale_factor=1.0):
        """
        2. Rescaling - Resize Image
        What it does: Makes image bigger or smaller
        Variables:
        - scale_factor = 1.0 → Same size
        - scale_factor = 2.0 → Double size
        - scale_factor = 0.5 → Half size
        Effect on text: Bigger images often give better OCR results
        """
        if image is None:
            image = self.original_image

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dimensions = (width, height)

        rescaled = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
        return rescaled

    def grayscale(self, image=None):
        """
        Convert image to grayscale - Remove Colors
        What it does: Converts colorful image to black, white, and gray only
        Variables: None to change
        Effect: Often improves text recognition, removes color distractions
        """
        if image is None:
            image = self.original_image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize_image(self, image=None, threshold=105):
        """3. Binarization - Convert to binary (black and white) image."""
        if image is None:
            image = self.original_image

        if len(image.shape) == 3:
            gray_image = self.grayscale(image)
        else:
            gray_image = image

        thresh, im_bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return im_bw

    def noise_removal(self, image):
        """
        4. Noise Removal - Clean Up Spots
        What it does: Removes small dots, specks, and unwanted marks
        Variables: Kernel sizes are hardcoded (1x1 pixels)
        Effect: Cleaner image but may remove small text details
        """
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def thin_font(self, image):
        """
        5. Dilation and Erosion - Make font thinner
        What it does: Makes letters thinner
        Variables: Kernel size (2x2 pixels) controls how much change
        Effect: Better for thick, bold text
        """
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def thick_font(self, image):
        """
        5. Dilation and Erosion - Make font thicker
        What it does: Makes letters thicker
        Variables: Kernel size (2x2 pixels) controls how much change
        Effect: Better for very thin, light text
        """
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def get_skew_angle(self, image):
        """6. Rotation / Deskewing - Get skew angle of image."""
        # Prep image, copy, convert to gray scale, blur, and threshold
        new_image = image.copy()
        gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into meaningful lines/paragraphs.
        # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
        # But use smaller kernel on Y axis to separate between different blocks of text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)

        # Find all contours
        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find largest contour and surround in min area box
        largest_contour = contours[0]
        print(len(contours))
        min_area_rect = cv2.minAreaRect(largest_contour)
        cv2.imwrite("../temp/boxes.jpg", new_image)
        # Determine the angle. Convert it to the value that was originally used to obtain skewed image
        angle = min_area_rect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotate_image(self, image, angle):
        """Rotate the image around its center."""
        new_image = image.copy()
        (h, w) = new_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return new_image

    def deskew(self, image):
        """
        6. Rotation / Deskewing - Straighten Tilted Text
        What it does: Rotates image to make text lines horizontal
        Variables: Automatically detects angle
        Effect: Improves OCR accuracy for crooked scanned documents
        """
        angle = self.get_skew_angle(image)
        return self.rotate_image(image, -1.0 * angle)

    def remove_borders(self, image):
        """
        7. Removing Borders - Crop to Content
        What it does: Cuts away empty white space around the text
        Variables: None to change
        Effect: Focuses OCR on actual content, may improve accuracy
        """
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_sorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cnts_sorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        return crop

    def add_borders(self, image, border_size=150, color=[255, 255, 255]):
        """
        8. Missing Borders - Add White Space
        What it does: Adds padding around the image
        Variables:
        - border_size = 150 → 150 pixels of padding (default)
        - color = [255,255,255] → White padding (default)
        Effect: Sometimes OCR works better with white space around text
        """
        top, bottom, left, right = [border_size] * 4
        image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image_with_border

    def add_transparency(self, image, alpha_value=255):
        """
        9. Transparency / Alpha Channel - Add transparency
        What it does: Add alpha channel to image for transparency effects
        Variables: alpha_value (0=fully transparent, 255=fully opaque)
        Effect: Useful for overlaying images or creating transparent backgrounds
        """
        if len(image.shape) == 3:
            # Convert BGR to BGRA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        else:
            # Convert grayscale to BGRA
            rgba_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)

        # Set alpha channel
        rgba_image[:, :, 3] = alpha_value
        return rgba_image

    def remove_transparency(self, image, background_color=[255, 255, 255]):
        """Remove alpha channel and replace with background color."""
        if len(image.shape) == 4:
            # Create a background image
            background = np.full(image.shape[:3], background_color, dtype=image.dtype)

            # Normalize alpha channel to 0-1 range
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0

            # Blend the image with background
            result = image[:, :, :3].astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
            return result.astype(np.uint8)
        else:
            return image

    def preprocess_pipeline(self, output_dir="temp/", save_intermediate=True, remove_white_spaces=True):
        """
        Complete Processing Pipeline - Do Everything
        Runs these steps in order:
        1. Remove white areas (if enabled)
        2. Invert colors
        3. Convert to grayscale
        4. Make black & white (binarize)
        5. Remove noise
        6. Remove borders
        7. Add new borders

        Variables:
        - remove_white_spaces: Boolean to enable/disable white space removal

        Files saved: Each step saves an image file so you can see the changes
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Start with original image
        current_image = self.original_image

        # 0. Remove white areas (NEW STEP)
        if remove_white_spaces:
            current_image = self.remove_white_areas(current_image)
            if save_intermediate:
                cv2.imwrite(f"{output_dir}white_removed.jpg", current_image)

        # 1. Inverted Images
        inverted_image = self.invert_image(current_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}inverted.jpg", inverted_image)

        # Convert to grayscale
        gray_image = self.grayscale(inverted_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}gray.jpg", gray_image)

        # 3. Binarization
        bw_image = self.binarize_image(gray_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}bw_image.jpg", bw_image)

        # 4. Noise Removal
        no_noise = self.noise_removal(bw_image)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}no_noise.jpg", no_noise)

        # 7. Remove Borders
        no_borders = self.remove_borders(no_noise)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}no_borders.jpg", no_borders)

        # 8. Add Borders
        with_borders = self.add_borders(no_borders)
        if save_intermediate:
            cv2.imwrite(f"{output_dir}image_with_border.jpg", with_borders)

        return with_borders

class ExtractorOCRText:
    def __init__(self, debug=False, remove_white_spaces=True):
        """
        Initialize the OCR extractor

        Args:
            debug: Enable debug output
            remove_white_spaces: Boolean to control white space removal before processing
        """
        self.debug = debug
        self.remove_white_spaces = remove_white_spaces

    def extract_text_from_pdf(self, pdf_path, start_page=1, end_page=None):
        """
        Extract text from PDF with improved OCR
        """
        text = ""
        if self.debug:
            print("✅ ExtractorOCRText.extract_text_from_pdf: ")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if end_page is None or end_page > total_pages:
                    end_page = total_pages

                if self.debug:
                    print(f"📄 Processing pages {start_page} to {end_page} of {total_pages} ..... "
                          f"in {os.path.basename(pdf_path)}")

                for i in range(start_page - 1, end_page):
                    page = pdf.pages[i]
                    print(f"📖 Processing page {i + 1}...")

                    ocr_text = self.extract_with_improved_ocr(pdf_path, page_num=i + 1)

                    if ocr_text and ocr_text.strip():
                        print(f"   ✅ OCR extraction: {len(ocr_text)} characters")
                        text += f"\n--- Page {i + 1} (OCR) ---\n{ocr_text}"
                    else:
                        print(f"   ❌ No text found on page {i + 1}")

        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return ""

        return text

    def extract_with_improved_ocr(self, pdf_path, page_num=1):
        """
        Extract text using improved OCR with multiple attempts.
        Saves all intermediate files under 'output_dir/{pdf_name}/'.
        Now includes optional white space removal.
        """
        try:
            if self.debug:
                print("✅ ExtractorOCRText.extract_with_improved_ocr: ")
                print(f"      🖼️ Converting page {page_num} to high-quality image...")

            # Convert PDF page to image with higher DPI
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)

            if not pages:
                return ""

            # Get base PDF name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # Define structured output directory
            output_dir = os.path.join("output_dir", pdf_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save the original page image
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            original_image = pages[0]
            original_image.save(image_path)

            if self.debug:
                print(f"💾 Saved original image as {image_path}")

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            save_intermediate = True

            # Start processing chain
            current_image = preprocessor.original_image

            # 0. Remove white areas FIRST (if enabled)
            if self.remove_white_spaces:
                if self.debug:
                    print("      🔍 Removing white areas...")

                # Try both methods and choose the better one
                method1_result = preprocessor.remove_white_areas(current_image)
                method2_result = preprocessor.smart_crop_content(current_image)

                # Save both results for comparison
                if save_intermediate:
                    method1_path = os.path.join(output_dir, f"white_removed_method1_page_{page_num}.jpg")
                    method2_path = os.path.join(output_dir, f"white_removed_method2_page_{page_num}.jpg")
                    cv2.imwrite(method1_path, method1_result)
                    cv2.imwrite(method2_path, method2_result)

                # Choose the method that results in more content (larger area)
                if method1_result.shape[0] * method1_result.shape[1] > method2_result.shape[0] * method2_result.shape[1]:
                    current_image = method1_result
                    chosen_method = "contour-based"
                else:
                    current_image = method2_result
                    chosen_method = "density-based"

                if self.debug:
                    print(f"      📏 Used {chosen_method} white removal method")

                # Save the chosen result
                if save_intermediate:
                    white_removed_path = os.path.join(output_dir, f"white_removed_page_{page_num}.jpg")
                    cv2.imwrite(white_removed_path, current_image)

            # 1. Invert image for better OCR results
            inverted_image = preprocessor.invert_image(current_image)
            if save_intermediate:
                inverted_path = os.path.join(output_dir, f"inverted_page_{page_num}.jpg")
                cv2.imwrite(inverted_path, inverted_image)

            # 2. Rescale image for better resolution
            rescale_image = preprocessor.rescale_image(inverted_image, 2.5)
            if save_intermediate:
                rescale_image_path = os.path.join(output_dir, f"rescale_image_page_{page_num}.jpg")
                cv2.imwrite(rescale_image_path, rescale_image)

            # Optional: Add binarization step (commented out but available)
            # binarize_image = preprocessor.binarize_image(rescale_image, threshold=115)
            # if save_intermediate:
            #     binarize_image_path = os.path.join(output_dir, f"binarize_image_page_{page_num}.jpg")
            #     cv2.imwrite(binarize_image_path, binarize_image)

            # Optional: Add grayscale conversion (commented out but available)
            # grayscale_image = preprocessor.grayscale(rescale_image)
            # if save_intermediate:
            #     grayscale_image_path = os.path.join(output_dir, f"grayscale_image_page_{page_num}.jpg")
            #     cv2.imwrite(grayscale_image_path, grayscale_image)

            if self.debug:
                print("      📝 Extracting text with Fraktur OCR...")

            # Extract text using the final processed image
            return preprocessor.extract_fraktur_text(rescale_image_path)

        except Exception as e:
            print(f"      ❌ OCR extraction failed: {e}")
            return ""

    def extract_with_custom_preprocessing(self, pdf_path, page_num=1, preprocessing_steps=None, end_page= None):
        """
        Extract text with custom preprocessing steps

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process
            preprocessing_steps: List of preprocessing steps to apply
                Available steps: ['remove_white', 'invert', 'rescale', 'grayscale', 'binarize', 'denoise']
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['remove_white', 'invert', 'rescale']

        try:
            # Convert PDF page to image
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=end_page, dpi=300)
            if not pages:
                return ""

            # Setup paths
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join("output_dir", pdf_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save original image
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            original_image = pages[0]
            original_image.save(image_path)

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            current_image = preprocessor.original_image

            # Apply preprocessing steps in order
            for step in preprocessing_steps:
                if step == 'remove_white' and self.remove_white_spaces:
                    current_image = preprocessor.remove_white_areas(current_image)
                elif step == 'invert':
                    current_image = preprocessor.invert_image(current_image)
                elif step == 'rescale':
                    current_image = preprocessor.rescale_image(current_image, 2.5)
                elif step == 'grayscale':
                    current_image = preprocessor.grayscale(current_image)
                elif step == 'binarize':
                    current_image = preprocessor.binarize_image(current_image, threshold=115)
                elif step == 'denoise':
                    current_image = preprocessor.noise_removal(current_image)

                # Save intermediate result
                if self.debug:
                    step_path = os.path.join(output_dir, f"{step}_page_{page_num}.jpg")
                    cv2.imwrite(step_path, current_image)

            # Save final processed image
            final_path = os.path.join(output_dir, f"final_processed_page_{page_num}.jpg")
            cv2.imwrite(final_path, current_image)

            # Extract text from final processed image
            return preprocessor.extract_fraktur_text(final_path)

        except Exception as e:
            print(f"❌ Custom preprocessing failed: {e}")
            return ""

    def batch_process_with_different_methods(self, pdf_path, page_num=1):
        """
        Process the same page with different preprocessing combinations
        and return the best result based on text length
        """
        methods = [
            {
                'name': 'method_1_basic',
                'steps': ['remove_white', 'invert', 'rescale'],
                'remove_white': True
            },
            {
                'name': 'method_2_enhanced',
                'steps': ['remove_white', 'invert', 'rescale', 'binarize'],
                'remove_white': True
            },
            {
                'name': 'method_3_no_white_removal',
                'steps': ['invert', 'rescale'],
                'remove_white': False
            },
            {
                'name': 'method_4_grayscale',
                'steps': ['remove_white', 'grayscale', 'rescale'],
                'remove_white': True
            }
        ]

        results = {}

        for method in methods:
            try:
                # Temporarily set remove_white_spaces
                original_setting = self.remove_white_spaces
                self.remove_white_spaces = method['remove_white']

                if self.debug:
                    print(f"🔄 Testing {method['name']}...")

                text = self.extract_with_custom_preprocessing(
                    pdf_path,
                    page_num,
                    method['steps']
                )

                results[method['name']] = {
                    'text': text,
                    'length': len(text.strip()) if text else 0,
                    'steps': method['steps']
                }

                # Restore original setting
                self.remove_white_spaces = original_setting

            except Exception as e:
                print(f"❌ Method {method['name']} failed: {e}")
                results[method['name']] = {
                    'text': '',
                    'length': 0,
                    'steps': method['steps']
                }

        # Find the method with the best result (most text extracted)
        best_method = max(results.keys(), key=lambda k: results[k]['length'])

        if self.debug:
            print(f"📊 Results summary:")
            for method_name, result in results.items():
                print(f"   {method_name}: {result['length']} characters")
            print(f"🏆 Best method: {best_method}")

        return results[best_method]['text'], results





class PDFExtractorWithOCR:
    def __init__(self, debug=False):
        self.debug = debug

        if self.debug:
            print("✅ PDFExtractorWithOCR.init PDF Extractor with OCR initialized")

        pytesseract.get_tesseract_version()

        if self.debug:
            print("✅ Tesseract OCR is available")




    #Step 1,
    def extract_text_from_pdf(self, pdf_path, start_page=1, end_page=None, type=None, preprocessing_steps=None):
        """
        Extract text from a PDF between specified pages (inclusive).
        Falls back to OCR if direct extraction fails.

        Pages are 1-indexed (first page is 1).
        """
        text = ""

        try:
            if self.debug:
                print("✅ PDFExtractorWithOCR.extract_text_from_pdf: ")
            extractor = ExtractorOCRText(self.debug)


            if type is None:
                text = extractor.extract_text_from_pdf(pdf_path, start_page=start_page, end_page=end_page)
            elif type == "custom_preprocessing":
                text = extractor.extract_with_custom_preprocessing(pdf_path, page_num=start_page, end_page=end_page, preprocessing_steps=preprocessing_steps)
            # elif type == "batch_process":
            #     text, all_results = extractor.batch_process_with_different_methods(pdf_path, page_num=start_page, end_page=end_page)




        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return ""

        return text



    #Step 2,
    def extract_metadata(self, text):
        """Extract metadata from text"""
        if not text.strip():
            return {"error": "No text to process"}

        if self.debug:
            print("✅ PDFExtractorWithOCR.extract_metadata: ")

        metadata = {}
        metadata_extractors = ExtractedMetadata(self.debug)
        return metadata_extractors.extract_date(text)


    def get_high_confidence_date(data):
        date_section = data.get('date', {})
        all_dates = date_section.get('all_dates', [])

        for date_entry in all_dates:
            if date_entry.get('confidence', '').lower() == 'high':
                return date_entry  # Return the first high confidence date found

        # If none found, fallback to the main date dict if its confidence is high
        if date_section.get('confidence', '').lower() == 'high':
            return date_section

        return None  # or {} if you prefer




class OptimizedMetadataExtractor:
    """Consolidated metadata extractor with all functionality"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.image_processor = PDFExtractorWithOCR(debug=debug)
        self._init_patterns()

        # ML components
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}

    def _init_patterns(self):
        """Initialize all regex patterns in one place"""

        # German month mapping with OCR error corrections
        self.german_months = {
            'januar': 1, 'jan': 1, 'zamuar': 1, 'zanuar': 1, 'jamuar': 1, 'jänner': 1,
            'februar': 2, 'feb': 2,
            'märz': 3, 'mär': 3, 'maerz': 3,
            'april': 4, 'apr': 4, 'aprıl': 4,
            'mai': 5,
            'juni': 6, 'jun': 6,
            'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11,
            'dezember': 12, 'dez': 12, 'deeember': 12, 'dezernber': 12
        }

        # Consolidated title patterns
        self.title_patterns = [
            r'(?:Berufs-?\s*)?([A-ZÄÖÜ][a-zäöüß]*anforderungen)\s*(?:\n.*?)?\s*(?:für|fü r)\s+([^.\n]{10,60})',
            r'(Berufs-?[Ee]ignungsanforderungen)\s*(?:\n.*?)?\s*(?:für.*?)\s*([A-ZÄÖÜ][a-zäöüß]{8,25})',
            r'(?:Verordnung|Anordnung|Gesetz|Bestimmungen|Richtlinien)\s+(?:über|für|zur|betreffend)\s+([^.\n]{20,100})',
            r'^([A-ZÄÖÜ][^.\n]{15,80})$'
        ]

        # Consolidated publisher patterns
        self.publisher_patterns = [
            r'bearbeitet\s+vom\s*\n?\s*([^\n.]{10,80})',
            r'(Deutscher\s+Ausschuss?\s+für\s+[^.\n]{5,40})\s*(?:\([^)]+\))?\s*[Ee]\.?[Vv]\.?',
            r'(Reichsministerium\s+für\s+[^.\n]+)',
            r'(Bundesministerium\s+für\s+[^.\n]+)',
            r'(Preußisches?\s+Ministerium\s+[^.\n]+)',
            r'(Deutsche\s+Arbeitsfront)',
            r'(Reichsgruppe\s+[A-ZÄÖÜ][a-zäöüß]+)',
            r'(Arbeitsgemeinschaft\s+[^.\n]{10,50})'
        ]

        # Document type patterns
        self.doc_type_patterns = {
            'Eignungsanforderungen': [
                r'[Ee]ignungsanforderungen', r'Berufs-?[Ee]ignungsanforderungen',
                r'Anforderungen.*Eintritt.*Lehrberuf'
            ],
            'Prüfungsordnung': [
                r'\bPrüfungsordnung\b', r'\bPrüfungsanforderungen\b',
                r'Ordnung.*Prüfung', r'Bestimmungen.*Prüfung'
            ],
            'Lehrplan': [
                r'\bLehrplan\b', r'\bLehrpläne\b', r'Plan.*Unterricht', r'Unterrichtsplan'
            ],
            'Ausbildungsordnung': [
                r'\bAusbildungsordnung\b', r'Ordnung.*Ausbildung', r'Bestimmungen.*Ausbildung'
            ],
            'Verordnung': [r'\bVerordnung\b', r'\bAnordnung\b'],
            'Gesetz': [r'\bGesetz\b', r'\bBerufsbildungsgesetz\b', r'\bHandwerksordnung\b']
        }

        # Profession patterns
        self.profession_patterns = [
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}fasser)',  # Steinfasser, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}macher)',  # Uhrmacher, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}schmidt)', # Goldschmidt, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}bauer)',   # Instrumentenbauer, etc.
            r'(Kaufmann|Mechaniker|Elektriker|Bäcker|Schneider|Tischler)'
        ]

        # Date publishing indicators (enhanced)
        self.publishing_indicators = [
            # Very strong indicators
            r'(?:Stand\s+vom|Stcmd\s+vom)',  # "Stand vom" or OCR error "Stcmd vom"
            r'(?:\(.*?Stand\s+vom.*?\))',     # "(Stand vom ...)" in parentheses
            r'(?:\(.*?Stcmd\s+vom.*?\))',     # "(Stcmd vom ...)" in parentheses

            # Strong indicators
            r'(?:Berlin,?\s+den)', r'(?:München,?\s+den)', r'(?:Hamburg,?\s+den)',
            r'(?:Datum\s*:)', r'(?:Ausgegeben\s+am)', r'(?:Verkündet\s+am)',

            # Medium indicators
            r'(?:Erlaß.*?vom)', r'(?:Erlass.*?vom)',  # "Erlaß ... vom"
            r'(?:mit\s+Wirkung\s+vom)',              # "mit Wirkung vom"
            r'(?:in\s+Kraft.*?vom)',                 # "in Kraft ... vom"
            r'(?:gültig\s+ab)',                      # "gültig ab"
        ]

        # Date pattern
        self.date_pattern = re.compile(r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})', re.IGNORECASE)

    def extract_text_from_pdf(self, pdf_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
        """Extract text from PDF with optimized OCR"""
        if self.debug:
            print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")

        try:
            text = self.image_processor.extract_text_from_pdf(pdf_path="../pdfs/brd/brd/brd_damenmaentelnaeherin_1954_bb.pdf",
                                                       start_page=1, end_page=1)
            return text
        except Exception as e:
            if self.debug:
                print(f"❌ Error processing PDF: {e}")
            return ""

    def extract_dates(self, text: str) -> Optional[datetime]:
        """
        Extracts the most probable publishing date from a block of text using a scoring system.

        This method scans the text for date-like patterns, validates them, and assigns scores
        based on context clues to determine which one is most likely to represent a publication date.

        Scoring heuristics include:
        - Presence of known publishing-related phrases (e.g., "Ausgegeben am", "Berlin, den ...")
        - Proximity to the beginning of the document (publishing dates often appear early)
        - Enclosure in parentheses (common in metadata and footnotes)

        Args:
        text (str): The input text to search for date patterns.

        Returns:
        Optional[datetime]: The most probable publishing date found in the text, or None if no valid date is found.
        """
        if self.debug:
            print("🗓️ Extracting dates...")

        # PRE-PROCESSING: Fix common OCR errors
        # Handle OCR error where "l." should be "1."
        text = re.sub(r'\bl\.\s*([a-zA-ZäöüÄÖÜß]+)', r'1. \1', text)

        if self.debug:
            print("   ✅ Applied OCR corrections")

        found_dates = []

        # Enhanced date pattern to catch more variations
        enhanced_date_patterns = [
            # Standard format: "22. September 1938"
            r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format with "vom": "vom 1. März 1938"
            r'vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format with "den": "den 27. April 1939"
            r'den\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format in parentheses: "(Stand vom 22. September 1938)"
            r'\(.*?(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4}).*?\)',
        ]

        for pattern in enhanced_date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    day_str, month_str, year_str = match.groups()
                    month_normalized = month_str.lower().strip()

                    if month_normalized in self.german_months:
                        day = int(day_str)
                        year = int(year_str)
                        month_num = self.german_months[month_normalized]
                        date_obj = datetime(year, month_num, day)

                        # Get context around the match (±150 characters)
                        context_start = max(0, match.start() - 150)
                        context_end = min(len(text), match.end() + 150)
                        context = text[context_start:context_end].lower()

                        # Initialize score
                        score = 0
                        match_text = match.group(0).lower()

                        if self.debug:
                            print(f"   Found date: {match.group(0)} -> {date_obj}")

                        # VERY HIGH PRIORITY: Document signature dates
                        very_high_priority_indicators = [
                            r'berlin,?\s+den',  # "Berlin, den" - official document signature
                            r'münchen,?\s+den',  # "München, den" - official document signature
                        ]

                        for indicator in very_high_priority_indicators:
                            if re.search(indicator, context):
                                score += 30  # Highest priority - document signature
                                if self.debug:
                                    print(f"     VERY HIGH PRIORITY: {indicator} -> +30")

                        # HIGH PRIORITY: Publishing date indicators
                        high_priority_indicators = [
                            r'stand\s+vom',  # "Stand vom"
                            r'stcmd\s+vom',  # OCR error "Stcmd vom"
                            r'\(.*?stand.*?vom',  # "(Stand vom ...)"
                            r'\(.*?stcmd.*?vom',  # "(Stcmd vom ...)"
                        ]

                        for indicator in high_priority_indicators:
                            if re.search(indicator, context):
                                score += 20  # Very high priority
                                if self.debug:
                                    print(f"     HIGH PRIORITY: {indicator} -> +20")

                        # MEDIUM PRIORITY: Official date patterns
                        medium_priority_indicators = [
                            r'ausgegeben\s+am',  # "Ausgegeben am"
                            r'verkündet\s+am',  # "Verkündet am"
                            r'wirkung\s+vom',  # "mit Wirkung vom"
                        ]

                        for indicator in medium_priority_indicators:
                            if re.search(indicator, context):
                                score += 12
                                if self.debug:
                                    print(f"     MEDIUM PRIORITY: {indicator} -> +12")

                        # LOW PRIORITY: Content-related dates (often not publishing dates)
                        low_priority_indicators = [
                            r'erlasz.*?vom',  # "Erlaß ... vom" - refers to decree dates, not publishing
                            r'erlass.*?vom',  # "Erlass ... vom" - refers to decree dates, not publishing
                        ]

                        for indicator in low_priority_indicators:
                            if re.search(indicator, context):
                                score += 5  # Lower priority - these are usually content dates
                                if self.debug:
                                    print(f"     LOW PRIORITY: {indicator} -> +5")

                        # PATTERN-SPECIFIC BONUSES
                        if 'vom' in match_text:
                            score += 8  # "vom" indicates publishing date
                            if self.debug:
                                print(f"     'vom' in match -> +8")

                        if 'den' in match_text:
                            score += 6  # "den" indicates official date
                            if self.debug:
                                print(f"     'den' in match -> +6")

                        # Parentheses bonus (official dates often in parentheses)
                        if '(' in context and ')' in context:
                            score += 10
                            if self.debug:
                                print(f"     Parentheses context -> +10")

                        # Position scoring (earlier = more likely publishing date)
                        relative_pos = match.start() / len(text)
                        if relative_pos < 0.2:  # First 20%
                            score += 8
                            if self.debug:
                                print(f"     Early position (top 20%) -> +8")
                        elif relative_pos < 0.4:  # First 40%
                            score += 4
                            if self.debug:
                                print(f"     Early position (top 40%) -> +4")

                        # Year reasonableness bonus
                        if 1920 <= year <= 1950:
                            score += 3
                        elif 1950 <= year <= 2025:
                            score += 2

                        # NEGATIVE INDICATORS (reduce score for content dates)
                        negative_indicators = [
                            r'seit\s+dem',  # "seit dem"
                            r'ab\s+dem',  # "ab dem"
                            r'erfolgte',  # "erfolgte"
                            r'geboren.*am',  # "geboren am"
                            r'verstorben.*am',  # "verstorben am"
                        ]

                        for neg_indicator in negative_indicators:
                            if re.search(neg_indicator, context):
                                score -= 5
                                if self.debug:
                                    print(f"     NEGATIVE: {neg_indicator} -> -5")

                        final_score = max(score, 0)  # Don't go negative
                        found_dates.append((date_obj, final_score, match.group(0)))

                        if self.debug:
                            print(f"     Final score: {final_score}")

                except (ValueError, KeyError) as e:
                    if self.debug:
                        print(f"     Error parsing date: {e}")
                    continue

        if not found_dates:
            if self.debug:
                print("   ❌ No valid dates found")
            return None

        # Sort by score (highest first)
        found_dates.sort(key=lambda x: x[1], reverse=True)

        if self.debug:
            print("   📊 All found dates ranked by score:")
            for i, (date_obj, score, original) in enumerate(found_dates):
                print(f"     {i + 1}. {original} -> {date_obj.strftime('%Y-%m-%d')} (score: {score})")

        # Return highest scoring date
        best_date = found_dates[0][0]
        if self.debug:
            print(f"   🏆 Selected date: {best_date.strftime('%Y-%m-%d')}")

        return best_date

    def extract_with_patterns(self, text: str, patterns: List[str], field_name: str) -> Tuple[Optional[str], float]:
        """Generic pattern-based extraction with confidence scoring"""
        candidates = []

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple matches (multiple groups)
                    if len(match) == 2:
                        result = f"{match[0]} {match[1]}"
                    else:
                        result = " ".join(match)
                else:
                    result = match

                # Clean and score
                clean_result = re.sub(r'\s+', ' ', result.strip())
                if 5 <= len(clean_result) <= 200:  # Reasonable length
                    confidence = 0.9 if len(clean_result) > 15 else 0.7
                    candidates.append((clean_result, confidence))

        if not candidates:
            return None, 0.0

        # Return best candidate
        best_match, best_score = max(candidates, key=lambda x: x[1])
        return best_match, best_score

    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type with profession enhancement"""
        best_type = None
        best_score = 0.0

        for doc_type, patterns in self.doc_type_patterns.items():
            score = 0.0
            match_count = 0

            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    match_count += matches
                    score += matches * 0.3

            if match_count > 0:
                # Bonus for early appearance
                early_text = text[:1000]
                if any(re.search(pattern, early_text, re.IGNORECASE) for pattern in patterns):
                    score *= 1.4

                if score > best_score:
                    best_score = score
                    best_type = doc_type

        # Try to extract profession and combine
        profession, prof_score = self.extract_with_patterns(text, self.profession_patterns, "profession")
        if profession and prof_score > 0.7 and best_type:
            best_type = f"{best_type} - {profession}"
            best_score = min(best_score + prof_score * 0.3, 1.0)

        return best_type, min(best_score, 1.0)

    def extract_all_metadata(self, text: str) -> ExtractedMetadata:
        """Extract all metadata in one optimized pass"""
        if self.debug:
            print("🔍 Extracting all metadata...")

        metadata = ExtractedMetadata()
        confidence_scores = {}

        # Extract date and year
        date_obj = self.extract_dates(text)
        if date_obj:
            metadata.date = date_obj
            metadata.year = date_obj.year
            confidence_scores['date'] = 0.9
            confidence_scores['year'] = 0.9

        # # Extract other fields
        # title, title_conf = self.extract_with_patterns(text, self.title_patterns, "title")
        # metadata.title = title
        # confidence_scores['title'] = title_conf
        #
        # publisher, pub_conf = self.extract_with_patterns(text, self.publisher_patterns, "publisher")
        # metadata.publisher = publisher
        # confidence_scores['publisher'] = pub_conf
        #
        # doc_type, type_conf = self.extract_document_type(text)
        # metadata.document_type = doc_type
        # confidence_scores['document_type'] = type_conf
        #
        # # Extract profession separately
        # profession, prof_conf = self.extract_with_patterns(text, self.profession_patterns, "profession")
        # metadata.profession = profession
        # if profession:
        #     confidence_scores['profession'] = prof_conf
        #
        # # Set confidence scores and preview
        # metadata.confidence_scores = confidence_scores
        # metadata.raw_text_preview = text[:500] + "..." if len(text) > 500 else text
        #
        # if self.debug:
        #     print("✅ Metadata extraction complete")
        #     for field, value in metadata.to_dict().items():
        #         if value and field != 'raw_text_preview':
        #             conf = confidence_scores.get(field, 0.0)
        #             print(f"   {field}: {value} (conf: {conf:.2f})")

        return metadata

    def train_ml_models(self, training_data: List[Dict]):
        """Train ML models for enhanced extraction"""
        if self.debug:
            print("🤖 Training ML models...")

        if len(training_data) < 5:
            if self.debug:
                print("⚠️ Not enough training data for ML models")
            return

        # Prepare training data
        texts = [item['text'] for item in training_data]

        # Train document type classifier
        doc_types = [item.get('document_type', '') for item in training_data]
        valid_types = [dt for dt in doc_types if dt]

        if len(valid_types) >= 3:
            self.vectorizers['doc_type'] = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X_tfidf = self.vectorizers['doc_type'].fit_transform(texts)

            self.label_encoders['doc_type'] = LabelEncoder()
            y_encoded = self.label_encoders['doc_type'].fit_transform(doc_types)

            self.models['doc_type'] = LogisticRegression(max_iter=1000)
            self.models['doc_type'].fit(X_tfidf, y_encoded)

            if self.debug:
                print("✅ Document type classifier trained")

    def predict_with_ml(self, text: str, field: str) -> Tuple[Optional[str], float]:
        """Predict using trained ML models"""
        if field not in self.models:
            return None, 0.0

        try:
            X_tfidf = self.vectorizers[field].transform([text])
            prediction = self.models[field].predict(X_tfidf)[0]
            probability = self.models[field].predict_proba(X_tfidf)[0].max()

            result = self.label_encoders[field].inverse_transform([prediction])[0]
            return result, probability
        except:
            return None, 0.0

    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoders': self.label_encoders
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        if self.debug:
            print(f"✅ Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.label_encoders = model_data['label_encoders']
            if self.debug:
                print(f"✅ Models loaded from {filepath}")
        except FileNotFoundError:
            if self.debug:
                print(f"⚠️ Model file not found: {filepath}")


class MetadataEvaluator:
    """Simplified evaluation framework"""

    @staticmethod
    def evaluate_extraction(predictions: List[ExtractedMetadata],
                          ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate extraction performance"""
        if not predictions or not ground_truth:
            return {}

        results = {}
        fields = ['title', 'year', 'publisher', 'document_type']

        for field in fields:
            pred_values = [getattr(pred, field, None) for pred in predictions]
            true_values = [gt.get(field) for gt in ground_truth]

            # Calculate accuracy
            matches = sum(1 for p, t in zip(pred_values, true_values)
                         if p and t and str(p).lower() == str(t).lower())
            total_with_truth = sum(1 for t in true_values if t)

            accuracy = matches / total_with_truth if total_with_truth > 0 else 0.0
            results[field] = accuracy

        results['overall'] = np.mean(list(results.values()))
        return results


# # Usage example and test function
# def quick_test():
#     """Quick test function"""
#     extractor = OptimizedMetadataExtractor(debug=True)
#
#     # Test text
#     test_text = """
#     Berufs-Eignungsanforderungen
#     für den Eintritt in den Lehrberuf
#     Schmucksteinfasser
#
#     bearbeitet vom
#     Deutschen Ausschuß für Technisches Schulwesen (Datsch) E.V.
#     Berlin NW7
#
#     (Stand vom 22. September 1938)
#
#     Verlag von B.G.Teubner in Leipzig und Berlin
#     """
#
#     # Extract metadata
#     metadata = extractor.extract_all_metadata(test_text)
#
#     print("\n" + "="*60)
#     print("EXTRACTION RESULTS:")
#     print("="*60)
#
#     result_dict = metadata.to_dict()
#     for key, value in result_dict.items():
#         if value and key != 'raw_text_preview':
#             print(f"{key.upper()}: {value}")
#
#
# # Test function for date extraction with your examples
# def test_date_extraction():
#     """Test date extraction with the problematic examples"""
#     extractor = OptimizedMetadataExtractor(debug=True)
#
#     # Test case 1: berufearchiv_6322 - should extract "1. März 1938" not "19. März 1938"
#     text_6322 = """
#     BERUFSAUSBILDUNC IN DER INDUSTRlE
#     Präfungsanfordetsungen
#     fiir den Lehrberuf
#     Teppichwebsper
#
#     Am 19. März 1938 als industrieller Lelirljeruf mit-klimmt
#     durch die Reichsgruppe Industrie mul die
#     Ärbeilsgemeinscltaft tlcr Industrie· uml Hamlelsliannncrn
#     in der Reichswiktscltuktslcammet-
#
#     Slimd vmn l. Miit-Z 1938
#     """
#
#     print("=" * 60)
#     print("TEST 1: berufearchiv_6322 (should be 1. März 1938)")
#     print("=" * 60)
#     result1 = extractor.extract_dates(text_6322)
#     print(f"Result: {result1}")
#     print()
#
#     # Test case 2: berufearchiv_5526 - should find a date
#     text_5526 = """
#     Industrie-
#     Facharbeiterausbildung
#     Berufsbildungsplan
#     für den Lehrberuf
#     Schokolademacher
#     bearbeitet vom
#     Deutschen Ausschuß für Technisches Schulwesen E. V. (Datsch)
#     Berlin NW 7
#     """
#
#     print("=" * 60)
#     print("TEST 2: berufearchiv_5526 (should find a date)")
#     print("=" * 60)
#     result2 = extractor.extract_dates(text_5526)
#     print(f"Result: {result2}")
#     print()
#
#     # Test case 3: berufearchiv_5542 - should work correctly
#     text_5542 = """
#     Fachliche Vorschriften zur Regelung
#     des Lehrlingswsfms im
#     Schornsteinfegerhandwerk
#
#     Der Neichswirtschaftsniinister hat sich mit den Fachlichen
#     Vorschriften zur Regelung des Lehrlingswesens im Schornstein-
#     fegerhandwerk mit dem Erlaß IIl sW 10 861J39 vom 22. April
#     1939 einverstanden erklärt. Sie treten mit Wirkung vom 1.Juli
#     1939 in Kraft.
#
#     Mit dem Erlasz dieser Vorschriften und dem im August 1936
#     erfolgten Crlasz der Fachlichen Vorschriften für die Meister-
#     priifung verfügt das Schornftcinfegerhandwerk«nunmehr über
#     eine einheitliche Grundlage.
#
#     Berlin, den 27. April 1939.
#     """
#
#     print("=" * 60)
#     print("TEST 3: berufearchiv_5542 (should be 27. April 1939)")
#     print("=" * 60)
#     result3 = extractor.extract_dates(text_5542)
#     print(f"Result: {result3}")
#
#
# if __name__ == "__main__":
#     # Run the original test
#     quick_test()
#
#     print("\n" + "="*80)
#     print("TESTING DATE EXTRACTION WITH PROBLEM CASES")
#     print("="*80)
#
#     # Run date extraction tests
#     test_date_extraction()
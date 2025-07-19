# debug_ocr_existing.py - Debug script using your existing code structure
import cv2
import numpy as np
import pytesseract
import os
from pdf2image import convert_from_path


class ImagePreprocessor:
    """Your existing ImagePreprocessor class with debug methods"""

    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"The file '{image_path}' does not exist or cannot be read.")

    def extract_fraktur_text(self, image_path):
        """Your existing extract_fraktur_text method"""
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
        """Your existing invert_image method"""
        if image is None:
            image = self.original_image
        inverted_image = cv2.bitwise_not(image)
        return inverted_image

    def rescale_image(self, image=None, scale_factor=1.0):
        """Your existing rescale_image method"""
        if image is None:
            image = self.original_image

        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        dimensions = (width, height)

        rescaled = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
        return rescaled

    def grayscale(self, image=None):
        """Your existing grayscale method"""
        if image is None:
            image = self.original_image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize_image(self, image=None, threshold=105):
        """Your existing binarize_image method"""
        if image is None:
            image = self.original_image

        if len(image.shape) == 3:
            gray_image = self.grayscale(image)
        else:
            gray_image = image

        thresh, im_bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return im_bw

    def noise_removal(self, image):
        """Your existing noise_removal method"""
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def remove_white_areas(self, image=None, white_threshold=240, min_content_area=1000):
        """Your existing remove_white_areas method"""
        if image is None:
            image = self.original_image

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Create binary mask where non-white areas are marked
        _, binary = cv2.threshold(gray, white_threshold, 255, cv2.THRESH_BINARY_INV)

        # Find contours of non-white areas
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Filter contours by area to remove noise
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_content_area]

        if not valid_contours:
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


class ExtractorOCRText:
    """Your existing ExtractorOCRText class with debug methods"""

    def __init__(self, debug=False, remove_white_spaces=True):
        self.debug = debug
        self.remove_white_spaces = remove_white_spaces

    def extract_with_improved_ocr_debug(self, pdf_path, page_num=1):
        """Debug version of your extract_with_improved_ocr method"""
        try:
            print("🔍 STARTING DEBUG OCR EXTRACTION")
            print("=" * 60)

            if self.debug:
                print("✅ ExtractorOCRText.extract_with_improved_ocr: ")
                print(f"      🖼️ Converting page {page_num} to high-quality image...")

            # Convert PDF page to image with higher DPI
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)

            if not pages:
                print("❌ No pages converted from PDF")
                return ""

            # Get base PDF name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # Define structured output directory
            output_dir = os.path.join("debug_output", pdf_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save the original page image
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            original_image = pages[0]
            original_image.save(image_path)

            print(f"💾 Saved original image as {image_path}")

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            save_intermediate = True

            # Test each processing step and OCR
            results = []

            # Test 0: Original image OCR
            print("\n📋 TEST 0: Original image")
            try:
                original_text = self.test_ocr_on_image(image_path, "original")
                results.append(("0_original", original_text, image_path))
                print(f"   Result: {len(original_text)} characters")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Start processing chain
            current_image = preprocessor.original_image

            # Test 1: Remove white areas (if enabled)
            if self.remove_white_spaces:
                print("\n📋 TEST 1: After white space removal")
                try:
                    current_image = preprocessor.remove_white_areas(current_image)

                    if save_intermediate:
                        white_removed_path = os.path.join(output_dir, f"01_white_removed_page_{page_num}.jpg")
                        cv2.imwrite(white_removed_path, current_image)
                        print(f"   💾 Saved: {white_removed_path}")

                        white_text = self.test_ocr_on_image(white_removed_path, "white_removed")
                        results.append(("1_white_removed", white_text, white_removed_path))
                        print(f"   Result: {len(white_text)} characters")
                except Exception as e:
                    print(f"   ❌ Failed: {e}")

            # Test 2: Invert image
            print("\n📋 TEST 2: After image inversion")
            try:
                inverted_image = preprocessor.invert_image(current_image)
                if save_intermediate:
                    inverted_path = os.path.join(output_dir, f"02_inverted_page_{page_num}.jpg")
                    cv2.imwrite(inverted_path, inverted_image)
                    print(f"   💾 Saved: {inverted_path}")

                    inverted_text = self.test_ocr_on_image(inverted_path, "inverted")
                    results.append(("2_inverted", inverted_text, inverted_path))
                    print(f"   Result: {len(inverted_text)} characters")

                # Update current image
                current_image = inverted_image
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Test 3: Rescale image
            print("\n📋 TEST 3: After rescaling (2.5x)")
            try:
                rescale_image = preprocessor.rescale_image(current_image, 2.5)
                if save_intermediate:
                    rescale_image_path = os.path.join(output_dir, f"03_rescale_image_page_{page_num}.jpg")
                    cv2.imwrite(rescale_image_path, rescale_image)
                    print(f"   💾 Saved: {rescale_image_path}")

                    rescale_text = self.test_ocr_on_image(rescale_image_path, "rescaled")
                    results.append(("3_rescaled", rescale_text, rescale_image_path))
                    print(f"   Result: {len(rescale_text)} characters")

                # Update current image
                current_image = rescale_image
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Test 4: Grayscale conversion
            print("\n📋 TEST 4: After grayscale conversion")
            try:
                grayscale_image = preprocessor.grayscale(current_image)
                if save_intermediate:
                    grayscale_image_path = os.path.join(output_dir, f"04_grayscale_image_page_{page_num}.jpg")
                    cv2.imwrite(grayscale_image_path, grayscale_image)
                    print(f"   💾 Saved: {grayscale_image_path}")

                    grayscale_text = self.test_ocr_on_image(grayscale_image_path, "grayscale")
                    results.append(("4_grayscale", grayscale_text, grayscale_image_path))
                    print(f"   Result: {len(grayscale_text)} characters")

                # Update current image
                current_image = grayscale_image
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Test 5: Binarization
            print("\n📋 TEST 5: After binarization")
            try:
                binarize_image = preprocessor.binarize_image(current_image, threshold=115)
                if save_intermediate:
                    binarize_image_path = os.path.join(output_dir, f"05_binarize_image_page_{page_num}.jpg")
                    cv2.imwrite(binarize_image_path, binarize_image)
                    print(f"   💾 Saved: {binarize_image_path}")

                    binarize_text = self.test_ocr_on_image(binarize_image_path, "binarized")
                    results.append(("5_binarized", binarize_text, binarize_image_path))
                    print(f"   Result: {len(binarize_text)} characters")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Test 6: Noise removal
            print("\n📋 TEST 6: After noise removal")
            try:
                noise_removed = preprocessor.noise_removal(current_image)
                if save_intermediate:
                    noise_removed_path = os.path.join(output_dir, f"06_noise_removed_page_{page_num}.jpg")
                    cv2.imwrite(noise_removed_path, noise_removed)
                    print(f"   💾 Saved: {noise_removed_path}")

                    noise_text = self.test_ocr_on_image(noise_removed_path, "noise_removed")
                    results.append(("6_noise_removed", noise_text, noise_removed_path))
                    print(f"   Result: {len(noise_text)} characters")
            except Exception as e:
                print(f"   ❌ Failed: {e}")

            # Show final results summary
            print("\n" + "=" * 60)
            print("🏆 RESULTS SUMMARY")
            print("=" * 60)

            # Sort by text length
            results.sort(key=lambda x: len(x[1].strip()), reverse=True)

            for i, (step_name, text, path) in enumerate(results):
                text_preview = text.strip()[:100].replace('\n', ' ') if text.strip() else "(empty)"
                print(f"{i + 1:2d}. {step_name:15s}: {len(text):4d} chars | {text_preview}...")

            # Save detailed results
            results_file = os.path.join(output_dir, f"debug_results_page_{page_num}.txt")
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write(f"Debug Results for: {pdf_path} (Page {page_num})\n")
                f.write("=" * 60 + "\n\n")

                for step_name, text, path in results:
                    f.write(f"=== {step_name} ===\n")
                    f.write(f"Path: {path}\n")
                    f.write(f"Length: {len(text)} characters\n")
                    f.write(f"Text:\n{text}\n\n")

            print(f"\n💾 Detailed results saved to: {results_file}")

            # Return the best result
            if results:
                best_result = results[0][1]  # Text from best result
                print(f"🎯 Best result: {results[0][0]} with {len(best_result)} characters")
                return best_result
            else:
                print("❌ No successful OCR results")
                return ""

        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return ""

    def test_ocr_on_image(self, image_path, step_name):
        """Test OCR with multiple approaches on a single image"""
        approaches = [
            ("deu_frak", "deu_frak"),
            ("deu", "deu"),
            ("eng", "eng"),
            ("auto", "")  # Let Tesseract auto-detect
        ]

        best_text = ""
        best_length = 0

        for lang_name, lang_code in approaches:
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    continue

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Extract text
                if lang_code:
                    text = pytesseract.image_to_string(gray, lang=lang_code)
                else:
                    text = pytesseract.image_to_string(gray)

                if len(text.strip()) > best_length:
                    best_text = text
                    best_length = len(text.strip())

                print(f"      {lang_name:8s}: {len(text.strip()):4d} chars")

            except Exception as e:
                print(f"      {lang_name:8s}: FAILED ({e})")
                continue

        return best_text


def debug_specific_image(image_path):
    """Debug OCR on a specific processed image"""
    print(f"\n🔍 DEBUGGING SPECIFIC IMAGE: {image_path}")
    print("=" * 60)

    if not os.path.exists(image_path):
        print("❌ Image file does not exist!")
        return

    # Analyze image properties
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image!")
        return

    print(f"📊 Image properties:")
    print(f"   Shape: {image.shape}")
    print(f"   Data type: {image.dtype}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    print(f"   Mean intensity: {np.mean(gray):.1f}")
    print(f"   Std intensity: {np.std(gray):.1f}")
    print(f"   Min/Max: {np.min(gray)}/{np.max(gray)}")

    # Check if mostly dark or light
    if np.mean(gray) < 127:
        print("   🌑 Image is mostly DARK")
    else:
        print("   🌕 Image is mostly LIGHT")

    # Test OCR with different approaches
    print(f"\n🔤 Testing OCR approaches:")

    extractor = ExtractorOCRText(debug=True)
    result = extractor.test_ocr_on_image(image_path, "debug")

    print(f"\n✅ Best result: {len(result)} characters")
    if result.strip():
        preview = result.strip()[:200].replace('\n', ' ')
        print(f"Preview: {preview}...")
    else:
        print("No text extracted!")


def main():
    """Main debug function"""
    print("🔍 OCR DEBUG TOOL - Using Your Existing Code")
    print("=" * 60)


    image_path = '../output/berufearchiv_6384/temp_page_1_step_02_InvertImage.jpg'  # Your example
    print(f"Using default: {image_path}")

    debug_specific_image(image_path)


if __name__ == "__main__":
    main()
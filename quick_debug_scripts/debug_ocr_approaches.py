import cv2
import pytesseract
import os
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def debug_ocr_approaches(image_path):
    """Debug different OCR approaches step by step"""

    print(f"🔍 Debugging OCR for: {image_path}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return

    print(f"📏 Image shape: {image.shape}")

    # Save debug images and try different approaches
    debug_dir = "debug_ocr_output"
    os.makedirs(debug_dir, exist_ok=True)

    approaches = [
        # Basic approaches
        ("1_original", image, "deu", "--psm 6"),
        ("2_original_fraktur", image, "deu_frak", "--psm 6"),

        # Try different PSM modes
        ("3_psm_4", image, "deu", "--psm 4"),  # Single column
        ("4_psm_7", image, "deu", "--psm 7"),  # Single line
        ("5_psm_8", image, "deu", "--psm 8"),  # Single word
        ("6_psm_13", image, "deu", "--psm 13"),  # Raw line

        # Preprocessing approaches
        ("7_grayscale", None, "deu", "--psm 6"),
        ("8_threshold", None, "deu", "--psm 6"),
        ("9_scaled", None, "deu", "--psm 6"),
        ("10_inverted", None, "deu", "--psm 6"),
    ]

    results = []

    for name, img, lang, config in approaches:
        try:
            print(f"\n🔄 Trying: {name}")

            # Apply preprocessing for certain approaches
            if img is None:
                if "grayscale" in name:
                    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif "threshold" in name:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                elif "scaled" in name:
                    img = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                elif "inverted" in name:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    img = cv2.bitwise_not(gray)
                else:
                    img = image

            # Save debug image
            debug_path = os.path.join(debug_dir, f"{name}.png")
            cv2.imwrite(debug_path, img)

            # Extract text
            text = pytesseract.image_to_string(img, lang=lang, config=config)

            # Clean and evaluate
            clean_text = text.strip()
            char_count = len(clean_text)
            word_count = len(clean_text.split()) if clean_text else 0

            # Simple scoring
            score = 0
            if char_count > 10:
                score += 0.3
            if word_count > 3:
                score += 0.3
            if any(word in clean_text.lower() for word in ['der', 'die', 'das', 'und', 'von']):
                score += 0.4

            result = {
                'name': name,
                'lang': lang,
                'config': config,
                'char_count': char_count,
                'word_count': word_count,
                'score': score,
                'text': clean_text[:100] + "..." if len(clean_text) > 100 else clean_text
            }

            results.append(result)

            print(f"   📊 {char_count} chars, {word_count} words, score: {score:.2f}")
            print(f"   📝 Text preview: {result['text']}")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue

    # Show best results
    print(f"\n🏆 BEST RESULTS:")
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    for i, result in enumerate(sorted_results[:5]):
        print(f"\n{i + 1}. {result['name']} (score: {result['score']:.2f})")
        print(f"   Language: {result['lang']}, Config: {result['config']}")
        print(f"   {result['char_count']} chars, {result['word_count']} words")
        print(f"   Text: {result['text']}")

    return sorted_results


def quick_image_analysis(image_path):
    """Quick analysis of image properties"""
    print(f"\n🔬 QUICK IMAGE ANALYSIS:")

    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return

    # Basic properties
    height, width = image.shape[:2]
    print(f"📐 Dimensions: {width}x{height}")

    # Color analysis
    if len(image.shape) == 3:
        print("🎨 Color image")
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        print("⚫ Grayscale image")
        gray = image

    # Brightness analysis
    mean_brightness = np.mean(gray)
    print(f"💡 Mean brightness: {mean_brightness:.1f}/255")

    # Contrast analysis
    contrast = np.std(gray)
    print(f"🌗 Contrast (std dev): {contrast:.1f}")

    # Text/background ratio estimation
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    total_pixels = binary.size

    print(f"⚪ White pixels: {white_pixels / total_pixels * 100:.1f}%")
    print(f"⚫ Black pixels: {black_pixels / total_pixels * 100:.1f}%")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if mean_brightness < 100:
        print("   - Image is quite dark, try inverting")
    if mean_brightness > 200:
        print("   - Image is quite bright, might need thresholding")
    if contrast < 50:
        print("   - Low contrast, try histogram equalization")
    if white_pixels / total_pixels > 0.8:
        print("   - Mostly white background, good for OCR")
    elif black_pixels / total_pixels > 0.8:
        print("   - Mostly black, definitely try inverting")


# Usage example:
if __name__ == "__main__":
    # Replace with your actual image path
    image_path = "output/berufearchiv_5542/temp_page_1.png"

    if os.path.exists(image_path):
        quick_image_analysis(image_path)
        debug_ocr_approaches(image_path)
    else:
        print(f"❌ Image not found: {image_path}")
        print("Make sure to run your main script first to generate the temp image")
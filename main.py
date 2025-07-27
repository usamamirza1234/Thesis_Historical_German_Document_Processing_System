# ===================================================================
# COMPLETE SMART OCR SYSTEM FOR GERMAN HISTORICAL DOCUMENTS
# ===================================================================

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

from config.settings import ProcessingConfig
from core.document_processor import SmartGermanDocumentProcessor


def create_example_config() -> ProcessingConfig:
    """Create example configuration"""
    return ProcessingConfig(
        dpi=300,
        enable_debug=True,
        confidence_threshold=0.6,
        output_directory="output/smart_ocr_example",
        ocr_language='deu',
        enable_interactive_mode=True,
        save_intermediate_images=True,
        max_approaches_to_try=5
    )


def main_example():
    """Main example function"""
    print("🚀 Smart German Document OCR System")
    print("=" * 50)

    # Create configuration
    config = create_example_config()

    # Initialize processor
    processor = SmartGermanDocumentProcessor(config)

    # Example document path
    file_path = "pdfs/30s/berufearchiv_5496.pdf"

    try:
        print(f"🔄 Processing: {file_path}")

        # Process document
        metadata = processor.process_document(file_path, start_page=1, end_page=2)

        print(f"\n✅ Processing Results:")
        print(f"📅 Date: {metadata.date}")
        print(f"🏢 Publisher: {metadata.publisher}")
        print(f"📄 Document Type: {metadata.document_type}")
        print(f"⭐ Overall Confidence: {metadata.get_overall_confidence():.2f}")

        # Show processing details
        if metadata.processing_metadata.get('successful_approach'):
            print(f"🚀 Best OCR Approach: {metadata.processing_metadata['successful_approach']}")
            print(f"🎯 Final Confidence: {metadata.processing_metadata.get('final_confidence', 0):.2f}")

        if metadata.processing_metadata.get('analysis'):
            analysis_info = metadata.processing_metadata['analysis']
            print(f"📊 Document Type Detected: {analysis_info['document_type']}")
            print(f"🖼️  Image Quality: {analysis_info['image_quality']}")

        # Show text preview
        if metadata.raw_text_preview:
            print(f"\n📝 Text Preview (first 300 characters):")
            preview = metadata.raw_text_preview[:300].replace('\n', ' ')
            print(f"   {preview}...")

        # Save results
        output_file = os.path.join(config.output_directory, "results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(metadata.to_json())

        print(f"\n💾 Results saved to: {output_file}")

        return metadata

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return None


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run example
    main_example()
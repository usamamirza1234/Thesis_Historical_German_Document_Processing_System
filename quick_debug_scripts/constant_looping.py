import re
import logging
from config.settings import ProcessingConfig
from core.document_processor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # This overrides any existing configuration
)
# Create configuration
config = ProcessingConfig(
    dpi=300,
    enable_debug=False,
    confidence_threshold=0.6,  # Lower threshold for German
    output_directory="output/berufearchiv_5542",
    ocr_language='deu_frak',  # Make sure this is set
    tesseract_config="",
    default_ocr_approaches= ["light", ]
)

# Initialize processor
processor = DocumentProcessor(config)

try:
    file_path = "../pdfs/30s/30s/berufearchiv_5542.pdf"
    metadata = processor.process_document(file_path=file_path,
                                          start_page=1,
                                          end_page=1)
    print(f"Extracted metadata:")
    print(f"Date: {metadata.date}")
    print(f"Publisher: {metadata.publisher}")
    print(f"Document Type: {metadata.document_type}")
    print(f"Overall Confidence: {metadata.get_overall_confidence():.2f}")

    # Save results
    with open("results.json", "w", encoding="utf-8") as f:
        f.write(metadata.to_json())

    raw_text = metadata.raw_text_preview or ""

    print(f"\n📝 Text Preview:")
    print(raw_text)



except Exception as e:
    print(f"Processing failed: {e}")
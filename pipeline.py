import argparse
import json
import sys
from pathlib import Path
from typing import List
from utils.file_handlers import FileHandler
from core.document_processor import DocumentProcessor
from config.settings import ProcessingConfig
# Create configuration
config = ProcessingConfig(
    dpi=300,
    enable_debug=True,
    confidence_threshold=0.7,
    output_directory="output/berufearchiv_6384"
)

# Initialize processor
processor = DocumentProcessor(config)

# Example: Process a single document
try:
    file_path = "pdfs/30s/30s/berufearchiv_6384.pdf"
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

except Exception as e:
    print(f"Processing failed: {e}")

# # Example: Batch processing
# document_paths = [
#     "doc1.pdf",
#     "doc2.pdf",
#     "doc3.jpg"
# ]
#
# try:
#     results = processor.process_documents_batch(document_paths)
#     print(f"Processed {len(results)} documents")
#
#     for i, result in enumerate(results):
#         print(f"Document {i + 1}: Confidence = {result.get_overall_confidence():.2f}")
#
# except Exception as e:
#     print(f"Batch processing failed: {e}")
#
# # Get processing statistics
# stats = processor.get_processing_statistics()
# print("Processing Statistics:")
# print(json.dumps(stats, indent=2))
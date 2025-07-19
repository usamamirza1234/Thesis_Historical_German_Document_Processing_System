# import argparse
# import json
# import sys
# from pathlib import Path
# from typing import List
# from utils.file_handlers import FileHandler
# from core.document_processor import DocumentProcessor
# from config.settings import ProcessingConfig
# # Create configuration
# # config = ProcessingConfig(
# #     dpi=300,
# #     enable_debug=True,
# #     confidence_threshold=0.7,
# #     output_directory="output/berufearchiv_6384"
# # )
# #
# # # Initialize processor
# # processor = DocumentProcessor(config)
# #
# # # Example: Process a single document
# # try:
# #     file_path = "pdfs/30s/30s/berufearchiv_6384.pdf"
# #     metadata = processor.process_document(file_path=file_path,
# #                                           start_page=1,
# #                                           end_page=1)
# #     print(f"Extracted metadata:")
# #     print(f"Date: {metadata.date}")
# #     print(f"Publisher: {metadata.publisher}")
# #     print(f"Document Type: {metadata.document_type}")
# #     print(f"Overall Confidence: {metadata.get_overall_confidence():.2f}")
# #
# #     # Save results
# #     with open("results.json", "w", encoding="utf-8") as f:
# #         f.write(metadata.to_json())
# #
# # except Exception as e:
# #     print(f"Processing failed: {e}")
#
# # # Example: Batch processing
# # document_paths = [
# #     "doc1.pdf",
# #     "doc2.pdf",
# #     "doc3.jpg"
# # ]
# #
# # try:
# #     results = processor.process_documents_batch(document_paths)
# #     print(f"Processed {len(results)} documents")
# #
# #     for i, result in enumerate(results):
# #         print(f"Document {i + 1}: Confidence = {result.get_overall_confidence():.2f}")
# #
# # except Exception as e:
# #     print(f"Batch processing failed: {e}")
# #
# # # Get processing statistics
# # stats = processor.get_processing_statistics()
# # print("Processing Statistics:")
# # print(json.dumps(stats, indent=2))
#
#
#
# import logging
# import os
# import re
#
# # logging.basicConfig(
# #     level=logging.DEBUG,
# #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# #     force=True
# # )
#
# # Import required modules
# from config.settings import ProcessingConfig
# from core.document_processor import DocumentProcessor
#
# # Create German-optimized configuration
# # config = ProcessingConfig(
# #     dpi=300,
# #     enable_debug=True,
# #     confidence_threshold=0.6,  # Lower threshold for German
# #     output_directory="output/brd_damenmaentelnaeherin_1954_bb",
# #     ocr_language='deu_frak',  # Make sure this is set
# #     tesseract_config=""
# # )
#
# # # Initialize processor
# # try:
# #     processor = DocumentProcessor(config)
# #     print("✅ DocumentProcessor initialized successfully")
# # except Exception as e:
# #     print(f"❌ Processor initialization failed: {e}")
# #     exit(1)
# #
# # # Process document
# # try:
# #     file_path = "pdfs/brd/brd/brd_damenmaentelnaeherin_1954_bb.pdf"
# #
# #     if not os.path.exists(file_path):
# #         print(f"❌ File not found: {file_path}")
# #         exit(1)
# #
# #     print(f"📄 Processing: {file_path}")
# #
# #     metadata = processor.process_document(
# #         file_path=file_path,
# #         start_page=1,
# #         end_page=1
# #     )
# #
# #     print(f"\n🎉 EXTRACTION RESULTS:")
# #     print(f"Date: {metadata.date}")
# #     print(f"Publisher: {metadata.publisher}")
# #     print(f"Document Type: {metadata.document_type}")
# #     print(f"Title: {metadata.title}")
# #     print(f"Overall Confidence: {metadata.get_overall_confidence():.2f}")
# #
# #     # Check for your specific italic text patterns
# #     raw_text = metadata.raw_text_preview or ""
# #     print(f"\n📝 Text Preview:")
# #     print(raw_text[:400] + "..." if len(raw_text) > 400 else raw_text)
# #
# #     # Pattern detection for your specific document
# #     print(f"\n🔍 PATTERN DETECTION:")
# #     if re.search(r'282094', raw_text):
# #         print("✅ Found document number: 282094")
# #     else:
# #         print("❌ Document number 282094 not found")
# #
# #     if re.search(r'10\.\s*5\.\s*1954', raw_text):
# #         print("✅ Found date: 10.5.1954")
# #     else:
# #         print("❌ Date 10.5.1954 not found")
# #
# #     if re.search(r'staatlich.*anerkannt', raw_text.lower()):
# #         print("✅ Found: 'Staatlich anerkannt'")
# #     else:
# #         print("❌ 'Staatlich anerkannt' not found")
# #
# #     # Save results
# #     with open("german_results.json", "w", encoding="utf-8") as f:
# #         f.write(metadata.to_json())
# #     print(f"\n💾 Results saved to german_results.json")
# #
# # except Exception as e:
# #     print(f"❌ Processing failed: {e}")
# #     import traceback
# #     traceback.print_exc()
#
#
#
# # Create configuration
# config = ProcessingConfig(
#     dpi=300,
#     enable_debug=False,
#     confidence_threshold=0.6,  # Lower threshold for German
#     output_directory="output/brd_damenmaentelnaeherin_1954_bb",
#     ocr_language='deu_frak',  # Make sure this is set
#     tesseract_config=""
# )
#
# # Initialize processor
# processor = DocumentProcessor(config)
#
# try:
#     file_path = "pdfs/brd/brd/brd_damenmaentelnaeherin_1954_bb.pdf"
#     metadata = (
#         processor.process_document(
#             file_path=file_path,
#             start_page=1,
#             end_page=1
#         )
#     )
#     print(f"Extracted metadata:")
#     print(f"Date: {metadata.date}")
#     print(f"Publisher: {metadata.publisher}")
#     print(f"Document Type: {metadata.document_type}")
#     print(f"Overall Confidence: {metadata.get_overall_confidence():.2f}")
#
#     # Save results
#     with open("results.json", "w", encoding="utf-8") as f:
#         f.write(metadata.to_json())
#
#     raw_text = metadata.raw_text_preview or ""
#
#     print(f"\n📝 Text Preview:")
#     print(raw_text)
#     if re.search(r'vom\s+(\d{1,2})\.(\d{1,2})\.\s+(\d{4})', raw_text):
#         print("✅ Found date: 10.5.1954")
#     else:
#         print("❌ Date 10.5.1954 not found")
#
# except Exception as e:
#     print(f"Processing failed: {e}")
from config.settings import ProcessingConfig
from core.document_processor import DocumentProcessor

# Create configuration
config = ProcessingConfig(
    dpi=300,
    enable_debug=True,
    confidence_threshold=0.6,
    output_directory="output/30s_werkgehilfin_1937_pruefungsanforderungen"
)

# Initialize processor
processor = DocumentProcessor(config)

# Example: Process a single document
try:
    file_path = "pdfs/30s/30s/30s_werkgehilfin_1937_pruefungsanforderungen.pdf"
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
import argparse
import json
import sys
from pathlib import Path
from typing import List


def main():
    """Main entry point for the document processing system"""
    parser = argparse.ArgumentParser(
        description="Extract metadata from historical German legal documents"
    )
    parser.add_argument(
        'input_path',
        help="Path to document file or directory"
    )
    parser.add_argument(
        '--output', '-o',
        help="Output file for results (JSON format)",
        default=None
    )
    parser.add_argument(
        '--config',
        help="Path to configuration file",
        default=None
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode with intermediate file saving"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help="Number of worker processes for batch processing"
    )
    parser.add_argument(
        '--start-page',
        type=int,
        default=1,
        help="First page to process (for PDFs)"
    )
    parser.add_argument(
        '--end-page',
        type=int,
        default=None,
        help="Last page to process (for PDFs)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = ProcessingConfig()
        if args.config:
            # Load from file (implementation would depend on config format)
            pass

        # Override config with command line arguments
        config.enable_debug = args.debug
        config.max_workers = args.workers

        # Setup logging
        logging_config = LoggingConfig(
            level="DEBUG" if args.debug else "INFO"
        )

        # Initialize processor
        processor = DocumentProcessor(config, logging_config)

        # Determine input files
        input_path = Path(args.input_path)

        if input_path.is_file():
            file_paths = [str(input_path)]
        elif input_path.is_dir():
            # Find all supported files in directory
            file_paths = []
            for ext in FileHandler.get_supported_extensions():
                file_paths.extend(input_path.glob(f"*{ext}"))
            file_paths = [str(p) for p in file_paths]
        else:
            print(f"Error: Input path not found: {args.input_path}")
            sys.exit(1)

        if not file_paths:
            print("No supported files found to process")
            sys.exit(1)

        print(f"Processing {len(file_paths)} file(s)...")

        # Process documents
        if len(file_paths) == 1:
            results = [processor.process_document(
                file_paths[0],
                args.start_page,
                args.end_page
            )]
        else:
            results = processor.process_documents_batch(file_paths)

        # Prepare output
        output_data = {
            'processing_statistics': processor.get_processing_statistics(),
            'results': [
                {
                    'file_path': file_paths[i],
                    'metadata': result.to_dict()
                }
                for i, result in enumerate(results)
            ]
        }

        # Save or print results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(output_data, indent=2, ensure_ascii=False))

        # Print summary
        successful = sum(1 for r in results if r.get_overall_confidence() > 0)
        print(f"\nProcessing complete: {successful}/{len(results)} files processed successfully")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

# # Example usage and integration
# if __name__ == "__main__":
#     # Example of how to use the system programmatically
#
#     # Create configuration
#     config = ProcessingConfig(
#         dpi=300,
#         enable_debug=True,
#         confidence_threshold=0.7,
#         output_directory="output"
#     )
#
#     # Initialize processor
#     processor = DocumentProcessor(config)
#
#     # Example: Process a single document
#     try:
#         metadata = processor.process_document("example_document.pdf")
#         print(f"Extracted metadata:")
#         print(f"Date: {metadata.date}")
#         print(f"Publisher: {metadata.publisher}")
#         print(f"Document Type: {metadata.document_type}")
#         print(f"Overall Confidence: {metadata.get_overall_confidence():.2f}")
#
#         # Save results
#         with open("results.json", "w", encoding="utf-8") as f:
#             f.write(metadata.to_json())
#
#     except Exception as e:
#         print(f"Processing failed: {e}")
#
#     # Example: Batch processing
#     document_paths = [
#         "doc1.pdf",
#         "doc2.pdf",
#         "doc3.jpg"
#     ]
#
#     try:
#         results = processor.process_documents_batch(document_paths)
#         print(f"Processed {len(results)} documents")
#
#         for i, result in enumerate(results):
#             print(f"Document {i + 1}: Confidence = {result.get_overall_confidence():.2f}")
#
#     except Exception as e:
#         print(f"Batch processing failed: {e}")
#
#     # Get processing statistics
#     stats = processor.get_processing_statistics()
#     print("Processing Statistics:")
#     print(json.dumps(stats, indent=2))
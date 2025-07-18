"""
# Historical German Document Processing System

## Overview
Automated metadata extraction system for historical German legal documents (1920-2025) 
using hybrid pattern-matching and machine learning approaches.

## Features
- Advanced OCR with German Fraktur support
- Configurable image preprocessing pipeline
- Pattern-based metadata extraction
- Machine learning integration
- Hybrid extraction engine
- Batch processing capabilities
- Comprehensive validation and error handling

## Installation
```bash
pip install -r requirements.txt
# Install Tesseract OCR with German language support
# Download German Fraktur model for Tesseract
```

## Quick Start
```python
from config.settings import ProcessingConfig
from core.document_processor import DocumentProcessor

# Configure
config = ProcessingConfig(dpi=300, enable_debug=True)
processor = DocumentProcessor(config)

# Process document
metadata = processor.process_document("document.pdf")
print(f"Date: {metadata.date}, Publisher: {metadata.publisher}")
```

## Command Line Usage
```bash
# Single document
python main.py document.pdf --output results.json

# Batch processing
python main.py documents/ --workers 4 --debug

# Specific pages
python main.py document.pdf --start-page 1 --end-page 3
```

## Configuration
See `config/settings.py` for all available configuration options.

## Testing
```bash
python -m pytest tests/
```

## Architecture
- **Core**: Main document processing pipeline
- **Preprocessing**: Image enhancement and OCR
- **Extraction**: Pattern matching and ML-based extraction
- **Models**: Data structures and ML model management
- **Utils**: File handling, validation, logging

## Performance
- Processes typical document in 2-5 seconds
- Batch processing with configurable parallelization
- Caching for improved performance on repeated documents
- Memory-efficient processing pipeline
"""
- 

conda env update -f environment.yml --prune
or
conda env remove -n Thesis_Historical_German_Document_Processing_System
conda env create -f environment.yml
conda activate Thesis_Historical_German_Document_Processing_System


# requirements.txt content (as comment for reference)
"""
opencv-python>=4.8.0
pytesseract>=0.3.10
pdf2image>=1.16.3
Pillow>=10.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
regex>=2023.6.3
"""
import unittest
from datetime import datetime
from extraction.pattern_matcher import PatternRegistry, GermanDateParser, PatternBasedExtractor


class TestPatternMatcher(unittest.TestCase):
    """Test pattern matching functionality"""

    def setUp(self):
        self.pattern_registry = PatternRegistry()
        self.date_parser = GermanDateParser()
        self.extractor = PatternBasedExtractor(self.pattern_registry)

    def test_date_extraction_standard_format(self):
        """Test standard German date format extraction"""
        text = "Das Dokument wurde am 22. September 1938 veröffentlicht."

        date_result, confidence = self.extractor.extract_date(text)

        self.assertIsNotNone(date_result)
        self.assertEqual(date_result.year, 1938)
        self.assertEqual(date_result.month, 9)
        self.assertEqual(date_result.day, 22)
        self.assertGreater(confidence, 0.5)

    def test_date_extraction_with_stand_vom(self):
        """Test high-priority 'Stand vom' date pattern"""
        text = "Berufs-Eignungsanforderungen (Stand vom 1. März 1938)"

        date_result, confidence = self.extractor.extract_date(text)

        self.assertIsNotNone(date_result)
        self.assertEqual(date_result.year, 1938)
        self.assertEqual(date_result.month, 3)
        self.assertEqual(date_result.day, 1)
        self.assertGreater(confidence, 0.9)  # High confidence for Stand vom

    def test_publisher_extraction(self):
        """Test publisher pattern extraction"""
        text = """
        bearbeitet vom
        Deutschen Ausschuß für Technisches Schulwesen (Datsch) E.V.
        Berlin NW7
        """

        results = self.extractor.extract_metadata_fields(text)
        publisher, confidence = results['publisher']

        self.assertIsNotNone(publisher)
        self.assertIn("Deutschen Ausschuß", publisher)
        self.assertGreater(confidence, 0.7)

    def test_document_type_extraction(self):
        """Test document type pattern extraction"""
        text = "Berufs-Eignungsanforderungen für den Eintritt in den Lehrberuf"

        results = self.extractor.extract_metadata_fields(text)
        doc_type, confidence = results['document_type']

        self.assertIsNotNone(doc_type)
        self.assertGreater(confidence, 0.8)

    def test_ocr_error_handling(self):
        """Test handling of common OCR errors"""
        # Test 'l.' -> '1.' correction
        text = "Stand vom l. März 1938"  # OCR error: l instead of 1

        date_result, confidence = self.extractor.extract_date(text)

        self.assertIsNotNone(date_result)
        self.assertEqual(date_result.day, 1)  # Should correct l. to 1.


class TestDocumentProcessor(unittest.TestCase):
    """Test main document processor"""

    def setUp(self):
        from config.settings import ProcessingConfig
        self.config = ProcessingConfig(enable_debug=False)

    def test_processor_initialization(self):
        """Test processor initializes without errors"""
        from core.document_processor import DocumentProcessor

        # This should not raise any exceptions
        processor = DocumentProcessor(self.config)

        self.assertIsNotNone(processor.image_processor)
        self.assertIsNotNone(processor.ocr_engine)
        self.assertIsNotNone(processor.hybrid_engine)

    def test_file_validation(self):
        """Test file validation functionality"""
        from utils.file_handlers import FileHandler

        # Test supported extensions
        self.assertTrue(FileHandler.is_supported_file("test.pdf"))
        self.assertTrue(FileHandler.is_supported_file("test.jpg"))
        self.assertFalse(FileHandler.is_supported_file("test.txt"))


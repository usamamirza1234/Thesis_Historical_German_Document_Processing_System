# ===================================================================
# PATTERN MATCHING SYSTEM
# ===================================================================
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import re
from models.data_models import Match, ExtractedMetadata


@dataclass
class Pattern:
    """Pattern for metadata extraction"""
    name: str
    regex: str
    confidence: float
    description: str
    field_type: str
    parse_groups: Optional[Tuple[int, ...]] = None


class PatternRegistry:
    """Registry for extraction patterns"""

    def __init__(self):
        self.patterns: Dict[str, List[Pattern]] = {}
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize German document patterns"""

        # Date patterns
        self.patterns["date"] = [
            Pattern(
                name="standard_date",
                regex=r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.8,
                description="Standard German date: DD. Month YYYY",
                field_type="date",
                parse_groups=(1, 2, 3)
            ),
            Pattern(
                name="vom_date",
                regex=r'vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.9,
                description="Date with 'vom' prefix",
                field_type="date",
                parse_groups=(1, 2, 3)
            ),
            Pattern(
                name="numeric_date",
                regex=r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
                confidence=0.7,
                description="Numeric date: DD.MM.YYYY",
                field_type="date",
                parse_groups=(1, 2, 3)
            ),
            Pattern(
                name="stand_vom_date",
                regex=r'Stand\s+vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.95,
                description="Stand vom date",
                field_type="date",
                parse_groups=(1, 2, 3)
            )
        ]

        # Publisher patterns
        self.patterns["publisher"] = [
            Pattern(
                name="bundesministerium",
                regex=r'(Bundesministerium\s+für\s+[^.\n]+)',
                confidence=0.9,
                description="Federal Ministry",
                field_type="publisher"
            ),
            Pattern(
                name="reichsministerium",
                regex=r'(Reichsministerium\s+für\s+[^.\n]+)',
                confidence=0.9,
                description="Reich Ministry",
                field_type="publisher"
            ),
            Pattern(
                name="deutscher_ausschuss",
                regex=r'(Deutscher\s+Ausschuss\s+für\s+[^.\n]{5,40})',
                confidence=0.85,
                description="German Committee",
                field_type="publisher"
            )
        ]

        # Document type patterns — generic only
        self.patterns["document_type"] = [
            Pattern(
                name="doc_type_verordnung",
                regex=r'^\s*(Verordnung)\b',
                confidence=0.95,
                description="Document starts with 'Verordnung'",
                field_type="document_type"
            ),
            Pattern(
                name="doc_type_anordnung",
                regex=r'\bAnordnung\b',
                confidence=0.95,
                description="Anordnung as document type",
                field_type="document_type"
            ),
            Pattern(
                name="doc_type_pruefungsordnung",
                regex=r'\b(Prüfungsordnung|Prüfungsanforderungen)\b',
                confidence=0.95,
                description="Prüfungsordnung or related",
                field_type="document_type"
            ),
            Pattern(
                name="doc_type_satzung",
                regex=r'\bSatzung\b',
                confidence=0.9,
                description="Satzung as document type",
                field_type="document_type"
            )
        ]

        # Title patterns — full regulation titles
        self.patterns["title"] = [
            Pattern(
                name="title_verordnung_long",
                regex=r'Verordnung\s+über\s+die\s+Berufsausbildung\s+(?:zum|zur|in der|für|von)?\s?[^\n\(\)]+',
                confidence=0.92,
                description="Long regulation title for training professions",
                field_type="title"
            ),
            Pattern(
                name="title_named_parenthesis",
                regex=r'\([\w\s\-–]+Ausbildungsverordnung\s*[-–]\s*[^\)]+\)',
                confidence=0.95,
                description="Parenthetical named regulation title (e.g., – FKüAusbV)",
                field_type="title"
            )
        ]

        # Author patterns
        self.patterns["author"] = [
            Pattern(
                name="author_herausgeber",
                regex=r'(?i)(Herausgegeben\s+von\s+.+?)(?:\.|\n)',
                confidence=0.9,
                description="Author from 'Herausgegeben von'",
                field_type="author"
            ),
            Pattern(
                name="author_named",
                regex=r'(?i)(Autor(?:in)?(?:en)?:\s*[^\n\.]+)',
                confidence=0.85,
                description="Line naming author(s)",
                field_type="author"
            ),
            Pattern(
                name="author_ministry_named",
                regex=r'(Bundesministerium\s+für\s+[^\n\.]{3,40})',
                confidence=0.8,
                description="Ministry as author",
                field_type="author"
            )
        ]

    def extract_field(self, text: str, field: str) -> List[Match]:
        """Extract matches for a specific field"""
        matches = []
        patterns = self.patterns.get(field, [])

        for pattern in patterns:
            try:
                for match in re.finditer(pattern.regex, text, re.IGNORECASE | re.MULTILINE):
                    start_context = max(0, match.start() - 50)
                    end_context = min(len(text), match.end() + 50)
                    context = text[start_context:end_context]

                    match_obj = Match(
                        value=match.group(0),
                        confidence=pattern.confidence,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        pattern_name=pattern.name,
                        context=context
                    )
                    matches.append(match_obj)
            except re.error as e:
                logging.warning(f"Regex error in pattern {pattern.name}: {e}")

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches


class GermanDateParser:
    """Specialized German date parser"""

    def __init__(self):
        self.german_months = {
            'januar': 1, 'jan': 1, 'februar': 2, 'feb': 2, 'märz': 3, 'mär': 3,
            'april': 4, 'apr': 4, 'mai': 5, 'juni': 6, 'jun': 6, 'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11, 'dezember': 12, 'dez': 12
        }

    def parse_date_from_match(self, match: Match) -> Optional[datetime]:
        """Parse datetime from a match"""
        try:
            # Try numeric date first
            numeric_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', match.value)
            if numeric_match:
                day, month, year = map(int, numeric_match.groups())
                return datetime(year, month, day)

            # Try month names
            text_match = re.search(r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})', match.value)
            if text_match:
                day_str, month_str, year_str = text_match.groups()
                day = int(day_str)
                year = int(year_str)
                month_normalized = month_str.lower().strip()

                if month_normalized in self.german_months:
                    month = self.german_months[month_normalized]
                    return datetime(year, month, day)

            return None

        except (ValueError, AttributeError):
            return None


class MetadataExtractor:
    """Main metadata extraction engine"""

    def __init__(self):
        self.pattern_registry = PatternRegistry()
        self.date_parser = GermanDateParser()

    def extract_metadata(self, text: str) -> ExtractedMetadata:
        """Extract all metadata from text"""
        metadata = ExtractedMetadata()

        # Extract each field
        for field in ['title', 'publisher', 'document_type', 'profession', 'author']:
            matches = self.pattern_registry.extract_field(text, field)
            if matches:
                best_match = matches[0]
                setattr(metadata, field, best_match.value)
                metadata.confidence_scores[field] = best_match.confidence

        # Special handling for dates
        date_result, date_confidence = self._extract_date(text)
        if date_result:
            metadata.date = date_result
            metadata.year = date_result.year
            metadata.confidence_scores['date'] = date_confidence

        # Set text preview
        metadata.raw_text_preview = text[:1000] + "..." if len(text) > 1000 else text

        return metadata

    def _extract_date(self, text: str) -> Tuple[Optional[datetime], float]:
        """Extract date with confidence"""
        date_matches = self.pattern_registry.extract_field(text, "date")
        if not date_matches:
            return None, 0.0

        best_date = None
        best_confidence = 0.0

        for match in date_matches:
            parsed_date = self.date_parser.parse_date_from_match(match)
            if parsed_date:
                # Score based on context
                context_bonus = 0.0
                context_lower = match.context.lower()

                if 'stand vom' in context_lower:
                    context_bonus += 0.2
                elif 'vom' in context_lower:
                    context_bonus += 0.1

                total_confidence = match.confidence + context_bonus

                if total_confidence > best_confidence:
                    best_date = parsed_date
                    best_confidence = total_confidence

        return best_date, min(best_confidence, 1.0)

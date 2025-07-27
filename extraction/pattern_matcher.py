# ===================================================================
# PATTERN MATCHING SYSTEM
# ===================================================================
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import re

from config.settings import DocumentEra
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
    context_boost: float = 0.0  # Additional confidence for context matches


class PatternRegistry:
    """Registry for extraction patterns"""

    def __init__(self):
        self.patterns: Dict[str, List[Pattern]] = {}
        self._initialize_enhanced_patterns()

    def _initialize_enhanced_patterns(self):
        """Initialize comprehensive German document patterns with era specificity"""

        # Enhanced date patterns with era context
        self.patterns["date"] = [
            Pattern(
                name="standard_date",
                regex=r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.8,
                description="Standard German date: DD. Month YYYY",
                field_type="date",
                applicable_eras=list(DocumentEra),
                parse_groups=(1, 2, 3)
            ),
            Pattern(
                name="vom_date",
                regex=r'vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.9,
                description="Date with 'vom' prefix",
                field_type="date",
                applicable_eras=list(DocumentEra),
                parse_groups=(1, 2, 3),
                context_boost=0.1
            ),
            Pattern(
                name="stand_vom_date",
                regex=r'Stand\s+vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.95,
                description="Stand vom date (official documents)",
                field_type="date",
                applicable_eras=[DocumentEra.EARLY_BRD, DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY],
                parse_groups=(1, 2, 3),
                context_boost=0.2
            ),
            Pattern(
                name="reich_date",
                regex=r'(?:im\s+Jahre?\s+des\s+Führers\s+|Anno\s+Domini\s+)?(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence=0.85,
                description="Historical Reich-era date formats",
                field_type="date",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD],
                parse_groups=(1, 2, 3)
            ),
            Pattern(
                name="numeric_date",
                regex=r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
                confidence=0.7,
                description="Numeric date: DD.MM.YYYY",
                field_type="date",
                applicable_eras=list(DocumentEra),
                parse_groups=(1, 2, 3)
            )
        ]

        # Enhanced publisher patterns with historical context
        self.patterns["publisher"] = [
            # Federal Republic patterns
            Pattern(
                name="bundesministerium",
                regex=r'(Bundesministerium\s+für\s+[^.\n]{5,60})',
                confidence=0.9,
                description="Federal Ministry (BRD era)",
                field_type="publisher",
                applicable_eras=[DocumentEra.EARLY_BRD, DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY],
                context_boost=0.1
            ),
            # Reich-era patterns
            Pattern(
                name="reichsministerium",
                regex=r'(Reichsministerium\s+(?:für\s+|des\s+)?[^.\n]{5,60})',
                confidence=0.95,
                description="Reich Ministry (Nazi/Weimar era)",
                field_type="publisher",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD],
                context_boost=0.2
            ),
            Pattern(
                name="reichsarbeitsministerium",
                regex=r'(Reichsarbeitsministerium|Reichsarbeitsverwaltung)',
                confidence=0.98,
                description="Reich Labor Ministry",
                field_type="publisher",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD],
                context_boost=0.3
            ),
            # Weimar specific
            Pattern(
                name="preussisches_ministerium",
                regex=r'(Preußisches\s+Ministerium\s+für\s+[^.\n]{5,40})',
                confidence=0.92,
                description="Prussian Ministry (Weimar era)",
                field_type="publisher",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC],
                context_boost=0.25
            ),
            # Nazi specific
            Pattern(
                name="deutsche_arbeitsfront",
                regex=r'(Deutsche\s+Arbeitsfront|DAF)',
                confidence=0.95,
                description="German Labor Front (Nazi era)",
                field_type="publisher",
                applicable_eras=[DocumentEra.NAZI_PERIOD],
                context_boost=0.3
            ),
            # General patterns
            Pattern(
                name="deutscher_ausschuss",
                regex=r'(Deutscher\s+Ausschuss\s+für\s+[^.\n]{5,40})',
                confidence=0.85,
                description="German Committee",
                field_type="publisher",
                applicable_eras=list(DocumentEra)
            ),
            # Contemporary patterns
            Pattern(
                name="bundesagentur",
                regex=r'(Bundesagentur\s+für\s+Arbeit|BA)',
                confidence=0.9,
                description="Federal Employment Agency (contemporary)",
                field_type="publisher",
                applicable_eras=[DocumentEra.CONTEMPORARY],
                context_boost=0.15
            )
        ]

        # Enhanced document type patterns with era context
        self.patterns["document_type"] = [
            # General document types
            Pattern(
                name="verordnung",
                regex=r'^\s*(Verordnung)\b',
                confidence=0.95,
                description="Regulation/Ordinance",
                field_type="document_type",
                applicable_eras=list(DocumentEra)
            ),
            Pattern(
                name="anordnung",
                regex=r'\b(Anordnung)\b',
                confidence=0.9,
                description="Order/Directive",
                field_type="document_type",
                applicable_eras=list(DocumentEra)
            ),
            # Era-specific document types
            Pattern(
                name="reichsgesetz",
                regex=r'\b(Reichsgesetz|Gesetz\s+des\s+Deutschen\s+Reiches)\b',
                confidence=0.98,
                description="Reich Law (historical)",
                field_type="document_type",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD],
                context_boost=0.3
            ),
            Pattern(
                name="fuehrerverordnung",
                regex=r'\b(Führerverordnung|Verordnung\s+des\s+Führers)\b',
                confidence=0.98,
                description="Führer Ordinance (Nazi era)",
                field_type="document_type",
                applicable_eras=[DocumentEra.NAZI_PERIOD],
                context_boost=0.4
            ),
            # Professional training specific
            Pattern(
                name="ausbildungsordnung",
                regex=r'\b(Ausbildungsordnung|Ausbildungsverordnung)\b',
                confidence=0.92,
                description="Training Regulation",
                field_type="document_type",
                applicable_eras=[DocumentEra.EARLY_BRD, DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY],
                context_boost=0.2
            ),
            Pattern(
                name="pruefungsordnung",
                regex=r'\b(Prüfungsordnung|Prüfungsanforderungen)\b',
                confidence=0.9,
                description="Examination Regulation",
                field_type="document_type",
                applicable_eras=list(DocumentEra)
            ),
            Pattern(
                name="eignungsanforderungen",
                regex=r'\b(Eignungsanforderungen|Eignungsprüfung)\b',
                confidence=0.88,
                description="Aptitude Requirements",
                field_type="document_type",
                applicable_eras=list(DocumentEra)
            ),
            # Contemporary types
            Pattern(
                name="rahmenplan",
                regex=r'\b(Rahmenplan|Rahmenlehrplan)\b',
                confidence=0.85,
                description="Framework Curriculum",
                field_type="document_type",
                applicable_eras=[DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY],
                context_boost=0.1
            )
        ]

        # Enhanced title patterns for German legal documents
        self.patterns["title"] = [
            # Training regulation titles
            Pattern(
                name="training_regulation_title",
                regex=r'(Verordnung\s+über\s+die\s+Berufsausbildung\s+(?:zum|zur|in\s+der|für|von)?\s?[^\n\(\)]{10,80})',
                confidence=0.92,
                description="Professional training regulation title",
                field_type="title",
                applicable_eras=[DocumentEra.EARLY_BRD, DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY],
                context_boost=0.15
            ),
            # Requirements documents
            Pattern(
                name="requirements_title",
                regex=r'(Eignungsanforderungen\s+für\s+[^\n\(\)]{5,60})',
                confidence=0.9,
                description="Aptitude requirements title",
                field_type="title",
                applicable_eras=list(DocumentEra),
                context_boost=0.1
            ),
            # Historical titles
            Pattern(
                name="reich_regulation_title",
                regex=r'((?:Reichs)?[Vv]erordnung\s+(?:über|betreffend|zur)\s+[^\n\(\)]{10,80})',
                confidence=0.95,
                description="Reich regulation title",
                field_type="title",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD],
                context_boost=0.2
            ),
            # Parenthetical titles with abbreviations
            Pattern(
                name="abbreviated_title",
                regex=r'\(([^)]*(?:Ausbildungsverordnung|AusbV|Verordnung)[^)]*[-–]\s*[A-ZÄÖÜ][a-zA-ZäöüÄÖÜß]*)\)',
                confidence=0.88,
                description="Abbreviated regulation title in parentheses",
                field_type="title",
                applicable_eras=list(DocumentEra)
            )
        ]

        # Enhanced author patterns
        self.patterns["author"] = [
            Pattern(
                name="herausgeber",
                regex=r'(?i)(Herausgegeben\s+von\s+.+?)(?:\.|\n)',
                confidence=0.9,
                description="Published by (Herausgegeben von)",
                field_type="author",
                applicable_eras=list(DocumentEra)
            ),
            Pattern(
                name="verfasser",
                regex=r'(?i)(Verfasser|Bearbeiter):\s*([^\n\.]+)',
                confidence=0.85,
                description="Author/Editor designation",
                field_type="author",
                applicable_eras=list(DocumentEra)
            ),
            Pattern(
                name="ministry_as_author",
                regex=r'((?:Bundes|Reichs)ministerium\s+für\s+[^\n\.]{3,40})',
                confidence=0.8,
                description="Ministry as author",
                field_type="author",
                applicable_eras=list(DocumentEra)
            )
        ]

        # Profession-specific patterns
        self.patterns["profession"] = [
            # Traditional crafts
            Pattern(
                name="traditional_crafts",
                regex=r'\b(Schreiner|Tischler|Schlosser|Schneider|Bäcker|Metzger|Schmied)\b',
                confidence=0.85,
                description="Traditional craft professions",
                field_type="profession",
                applicable_eras=list(DocumentEra)
            ),
            # Industrial professions
            Pattern(
                name="industrial_professions",
                regex=r'\b(Elektriker|Mechaniker|Maschinenbauer|Chemiker|Techniker)\b',
                confidence=0.88,
                description="Industrial professions",
                field_type="profession",
                applicable_eras=[DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD, DocumentEra.EARLY_BRD,
                                 DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY]
            ),
            # Modern professions
            Pattern(
                name="modern_professions",
                regex=r'\b(Informatiker|Mediengestalter|Kaufmann|Bürokaufmann|Einzelhandelskaufmann)\b',
                confidence=0.9,
                description="Modern service professions",
                field_type="profession",
                applicable_eras=[DocumentEra.MODERN_BRD, DocumentEra.CONTEMPORARY]
            ),
            # Generic profession patterns
            Pattern(
                name="profession_suffix",
                regex=r'\b([A-ZÄÖÜ][a-zäöüß]{4,}(?:er|in|mann|frau))\b',
                confidence=0.7,
                description="Generic profession with common suffixes",
                field_type="profession",
                applicable_eras=list(DocumentEra)
            )
        ]

    def extract_field_with_era_context(self, text: str, field: str,
                                       era: Optional[DocumentEra] = None,
                                       context_window: int = 100) -> List[Match]:
        """Extract matches for a specific field with era context awareness"""
        matches = []
        patterns = self.patterns.get(field, [])

        for pattern in patterns:
            # Skip patterns not applicable to the current era
            if era and era not in pattern.applicable_eras:
                continue

            try:
                for match in re.finditer(pattern.regex, text, re.IGNORECASE | re.MULTILINE):
                    start_context = max(0, match.start() - context_window)
                    end_context = min(len(text), match.end() + context_window)
                    context = text[start_context:end_context]

                    # Calculate confidence with era and context boosts
                    base_confidence = pattern.confidence

                    # Era match bonus
                    era_bonus = 0.1 if era and era in pattern.applicable_eras else 0.0

                    # Context relevance bonus
                    context_bonus = self._calculate_context_bonus(context, pattern, era)

                    final_confidence = min(1.0, base_confidence + era_bonus + context_bonus + pattern.context_boost)

                    match_obj = Match(
                        value=match.group(0),
                        confidence=final_confidence,
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

    def _calculate_context_bonus(self, context: str, pattern: Pattern,
                                 era: Optional[DocumentEra]) -> float:
        """Calculate confidence bonus based on surrounding context"""
        context_lower = context.lower()
        bonus = 0.0

        # Era-specific context indicators
        if era:
            era_indicators = {
                DocumentEra.WEIMAR_REPUBLIC: ['republik', 'preußen', 'weimar'],
                DocumentEra.NAZI_PERIOD: ['reich', 'führer', 'deutsche arbeitsfront'],
                DocumentEra.EARLY_BRD: ['bundesrepublik', 'grundgesetz', 'wiederaufbau'],
                DocumentEra.MODERN_BRD: ['bundesrepublik', 'eu', 'europäisch'],
                DocumentEra.CONTEMPORARY: ['digital', 'internet', 'euro', 'union']
            }

            indicators = era_indicators.get(era, [])
            if any(indicator in context_lower for indicator in indicators):
                bonus += 0.05

        # Field-specific context bonuses
        if pattern.field_type == "date":
            if any(term in context_lower for term in ['stand', 'datum', 'erlassen', 'verkündet']):
                bonus += 0.05
        elif pattern.field_type == "publisher":
            if any(term in context_lower for term in ['herausgeber', 'verlegt', 'veröffentlicht']):
                bonus += 0.05
        elif pattern.field_type == "title":
            if context.count('\n') <= 2:  # Likely in header area
                bonus += 0.1

        return bonus

    def extract_field(self, text: str, field: str) -> List[Match]:
        """Backward compatibility method"""
        return self.extract_field_with_era_context(text, field)


class GermanDateParser:
    """Specialized German date parser"""

    def __init__(self):
        self.german_months = {
            'januar': 1, 'jan': 1, 'februar': 2, 'feb': 2, 'märz': 3, 'mär': 3,
            'april': 4, 'apr': 4, 'mai': 5, 'juni': 6, 'jun': 6, 'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11, 'dezember': 12, 'dez': 12
        }

        # Historical month variations
        self.historical_month_variations = {
            'jänner': 1,  # Austrian/historical variant
            'hornung': 2,  # Historical German for February
            'lenzing': 3,  # Historical German for March
            'ostermond': 4,  # Historical German for April
            'wonnemond': 5,  # Historical German for May
            'brachmond': 6,  # Historical German for June
            'heumond': 7,  # Historical German for July
            'erntemond': 8,  # Historical German for August
            'herbstmond': 9,  # Historical German for September
            'weinmond': 10,  # Historical German for October
            'nebelmond': 11,  # Historical German for November
            'julmond': 12  # Historical German for December
        }

    def parse_date_from_match(self, match: Match, era: Optional[DocumentEra] = None) -> Optional[datetime]:
        """Parse datetime from a match with era context"""
        try:
            # Try numeric date first
            numeric_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', match.value)
            if numeric_match:
                day, month, year = map(int, numeric_match.groups())
                return datetime(year, month, day)

            # Try standard month names
            text_match = re.search(r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})', match.value)
            if text_match:
                day_str, month_str, year_str = text_match.groups()
                day = int(day_str)
                year = int(year_str)
                month_normalized = month_str.lower().strip()

                # Try standard months first
                if month_normalized in self.german_months:
                    month = self.german_months[month_normalized]
                    return datetime(year, month, day)

                # Try historical variants for older documents
                if era in [DocumentEra.WEIMAR_REPUBLIC, DocumentEra.NAZI_PERIOD]:
                    if month_normalized in self.historical_month_variations:
                        month = self.historical_month_variations[month_normalized]
                        return datetime(year, month, day)

            return None

        except (ValueError, AttributeError) as e:
            logging.debug(f"Date parsing failed for '{match.value}': {e}")
            return None


class MetadataExtractor:
    """Main metadata extraction engine"""

    def __init__(self):
        self.pattern_registry = PatternRegistry()
        self.date_parser = GermanDateParser()

    def extract_metadata(self, text: str, era: Optional[DocumentEra] = None) -> ExtractedMetadata:
        """Extract all metadata from text with era context"""
        metadata = ExtractedMetadata()

        # Extract each field with era context
        fields_to_extract = ['title', 'publisher', 'document_type', 'profession', 'author']

        for field in fields_to_extract:
            matches = self.pattern_registry.extract_field_with_era_context(text, field, era)
            if matches:
                best_match = matches[0]
                setattr(metadata, field, best_match.value)
                metadata.confidence_scores[field] = best_match.confidence

                # Store additional match information
                if not hasattr(metadata, 'match_details'):
                    metadata.match_details = {}
                metadata.match_details[field] = {
                    'pattern_name': best_match.pattern_name,
                    'position': (best_match.start_pos, best_match.end_pos),
                    'context': best_match.context[:100] + "..." if len(best_match.context) > 100 else best_match.context
                }

        # Enhanced date extraction with era context
        date_result, date_confidence = self._extract_date_with_era(text, era)
        if date_result:
            metadata.date = date_result
            metadata.year = date_result.year
            metadata.confidence_scores['date'] = date_confidence

        # Store era information
        if era:
            metadata.processing_metadata['detected_era'] = era.value
            metadata.processing_metadata['era_specific_extraction'] = True

        # Set text preview
        metadata.raw_text_preview = text[:1000] + "..." if len(text) > 1000 else text

        # Calculate enhanced overall confidence
        metadata.processing_metadata['extraction_method'] = 'enhanced_pattern_matching'
        metadata.processing_metadata['era_aware'] = era is not None

        return metadata

    def _extract_date_with_era(self, text: str, era: Optional[DocumentEra]) -> Tuple[Optional[datetime], float]:
        """Extract date with era-specific parsing"""
        date_matches = self.pattern_registry.extract_field_with_era_context(text, "date", era)
        if not date_matches:
            return None, 0.0

        best_date = None
        best_confidence = 0.0

        for match in date_matches:
            parsed_date = self.date_parser.parse_date_from_match(match, era)
            if parsed_date:
                # Validate date against era expectations
                era_bonus = self._validate_date_against_era(parsed_date, era)
                total_confidence = match.confidence + era_bonus

                if total_confidence > best_confidence:
                    best_date = parsed_date
                    best_confidence = total_confidence

        return best_date, min(best_confidence, 1.0)

    def _validate_date_against_era(self, date: datetime, era: Optional[DocumentEra]) -> float:
        """Validate if extracted date makes sense for the expected era"""
        if not era:
            return 0.0

        year = date.year
        era_ranges = {
            DocumentEra.WEIMAR_REPUBLIC: (1918, 1933),
            DocumentEra.NAZI_PERIOD: (1933, 1945),
            DocumentEra.EARLY_BRD: (1945, 1970),
            DocumentEra.MODERN_BRD: (1970, 1990),
            DocumentEra.CONTEMPORARY: (1990, 2030)
        }

        expected_range = era_ranges.get(era)
        if expected_range and expected_range[0] <= year <= expected_range[1]:
            return 0.1  # Bonus for era-appropriate date
        elif expected_range and abs(year - expected_range[0]) <= 5:
            return 0.05  # Small bonus for close dates

        return 0.0  # No bonus or potential penalty for mismatched dates

    def extract_metadata_multi_approach(self, text: str,
                                        primary_era: Optional[DocumentEra] = None,
                                        fallback_eras: Optional[List[DocumentEra]] = None) -> ExtractedMetadata:
        """Extract metadata trying multiple era approaches"""

        # Primary extraction with specified era
        primary_result = self.extract_metadata(text, primary_era)

        # If confidence is high enough, return primary result
        if primary_result.get_overall_confidence() > 0.8:
            return primary_result

        # Try fallback eras if provided
        if fallback_eras:
            best_result = primary_result
            best_confidence = primary_result.get_overall_confidence()

            for fallback_era in fallback_eras:
                fallback_result = self.extract_metadata(text, fallback_era)
                fallback_confidence = fallback_result.get_overall_confidence()

                if fallback_confidence > best_confidence:
                    best_result = fallback_result
                    best_confidence = fallback_confidence

            # Mark as multi-approach extraction
            best_result.processing_metadata['multi_era_extraction'] = True
            best_result.processing_metadata['tried_eras'] = [era.value for era in [primary_era] + fallback_eras if era]

            return best_result

        return primary_result

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from models.data_models import Match
from utils.exceptions import PatternMatchingError

logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Represents a regex pattern for metadata extraction"""
    name: str
    regex: str
    confidence_weight: float
    description: str
    flags: int = re.IGNORECASE | re.MULTILINE

    def find_matches(self, text: str) -> List[Match]:
        """Find all matches of this pattern in text"""
        try:
            matches = []
            for match in re.finditer(self.regex, text, self.flags):
                # Get context around match
                start_context = max(0, match.start() - 100)
                end_context = min(len(text), match.end() + 100)
                context = text[start_context:end_context]

                match_obj = Match(
                    value=match.group(0),
                    confidence=self.confidence_weight,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    pattern_name=self.name,
                    context=context
                )
                matches.append(match_obj)

            return matches

        except re.error as e:
            logger.error(f"Regex error in pattern {self.name}: {e}")
            return []


class PatternRegistry:
    """Registry for managing extraction patterns"""

    def __init__(self):
        self.patterns: Dict[str, List[Pattern]] = {}
        self._initialize_default_patterns()

    def _initialize_default_patterns(self):
        """Initialize default patterns for German documents"""
        self._add_date_patterns()
        # self._add_publisher_patterns()
        # self._add_title_patterns()
        # self._add_document_type_patterns()
        # self._add_profession_patterns()

    def _add_date_patterns(self):
        """Add date extraction patterns"""
        date_patterns = [
            Pattern(  # Example match: "12. März 2023"
                name="standard_date",
                regex=r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence_weight=0.8,
                description="Standard German date format: DD. Month YYYY"
            ),
            Pattern( # Example match: "vom 5. Oktober 2022"
                name="vom_date",
                regex=r'vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence_weight=0.9,
                description="Date with 'vom' prefix"
            ),
            Pattern(  # Example match: "den 14. Juli 2021"
                name="den_date",
                regex=r'den\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence_weight=0.9,
                description="Date with 'den' prefix"
            ),
            Pattern(  # Example match: "(Ausgestellt am 3. Mai 2020)"
                name="parentheses_date",
                regex=r'\(.*?(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4}).*?\)',
                confidence_weight=0.85,
                description="Date in parentheses"
            ),
            Pattern(  # Example match: "Stand vom 7. Januar 2024" or "Stcmd vom 7. Januar 2024"
                name="stand_vom_date",
                regex=r'(?:Stand|Stcmd)\s+vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence_weight=0.95,
                description="High priority: Stand vom date"
            ),
            Pattern(  # Example match: "Berlin, den 30. November 2022"
                name="berlin_den_date",
                regex=r'Berlin,?\s+den\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
                confidence_weight=0.98,
                description="Highest priority: Official Berlin signature date"
            ),
            Pattern( # Example match: 19.9.1961
                name="numeric_date",
                regex=r'(\d{1,2})\.(\d{1,2})\.(\d{4})',
                confidence_weight=0.7,
                description="Fully numeric date format: DD.MM.YYYY"
            ),
            Pattern(  # Example match: "vom 10.9. 1954"
                name="vom_numeric_date_space",
                regex=r'vom\s+(\d{1,2})\.(\d{1,2})\.\s+(\d{4})',
                confidence_weight=0.9,
                description="vom with numeric date and space: vom DD.MM. YYYY"
            )
        ]

        for pattern in date_patterns:
            self.register_pattern("date", pattern)

    # def _add_publisher_patterns(self):
    #     """Add publisher extraction patterns"""
    #     publisher_patterns = [
    #         Pattern(
    #             name="bearbeitet_vom",
    #             regex=r'bearbeitet\s+vom\s*\n?\s*([^\n.]{10,80})',
    #             confidence_weight=0.9,
    #             description="'bearbeitet vom' publisher indicator"
    #         ),
    #         Pattern(
    #             name="deutscher_ausschuss",
    #             regex=r'(Deutscher\s+Ausschuss?\s+für\s+[^.\n]{5,40})\s*(?:\([^)]+\))?\s*[Ee]\.?[Vv]\.?',
    #             confidence_weight=0.85,
    #             description="German Committee organizations"
    #         ),
    #         Pattern(
    #             name="reichsministerium",
    #             regex=r'(Reichsministerium\s+für\s+[^.\n]+)',
    #             confidence_weight=0.9,
    #             description="Reich Ministry"
    #         ),
    #         Pattern(
    #             name="bundesministerium",
    #             regex=r'(Bundesministerium\s+für\s+[^.\n]+)',
    #             confidence_weight=0.9,
    #             description="Federal Ministry"
    #         ),
    #         Pattern(
    #             name="preussisches_ministerium",
    #             regex=r'(Preußisches?\s+Ministerium\s+[^.\n]+)',
    #             confidence_weight=0.85,
    #             description="Prussian Ministry"
    #         ),
    #         Pattern(
    #             name="deutsche_arbeitsfront",
    #             regex=r'(Deutsche\s+Arbeitsfront)',
    #             confidence_weight=0.8,
    #             description="German Labor Front"
    #         )
    #     ]
    #
    #     for pattern in publisher_patterns:
    #         self.register_pattern("publisher", pattern)
    #
    # def _add_title_patterns(self):
    #     """Add title extraction patterns"""
    #     title_patterns = [
    #         Pattern(
    #             name="berufs_eignungsanforderungen",
    #             regex=r'(?:Berufs-?\s*)?([A-ZÄÖÜ][a-zäöüß]*anforderungen)\s*(?:\n.*?)?\s*(?:für|fü r)\s+([^.\n]{10,60})',
    #             confidence_weight=0.9,
    #             description="Professional requirements format"
    #         ),
    #         Pattern(
    #             name="verordnung_title",
    #             regex=r'(?:Verordnung|Anordnung|Gesetz|Bestimmungen|Richtlinien)\s+(?:über|für|zur|betreffend)\s+([^.\n]{20,100})',
    #             confidence_weight=0.85,
    #             description="Regulation/law titles"
    #         ),
    #         Pattern(
    #             name="general_title",
    #             regex=r'^([A-ZÄÖÜ][^.\n]{15,80})',
    #             confidence_weight=0.6,
    #             description="General title pattern"
    #         )
    #     ]
    #
    #     for pattern in title_patterns:
    #         self.register_pattern("title", pattern)
    #
    # def _add_document_type_patterns(self):
    #     """Add document type patterns"""
    #     doc_type_patterns = [
    #         Pattern(
    #             name="eignungsanforderungen",
    #             regex=r'[Ee]ignungsanforderungen|Berufs-?[Ee]ignungsanforderungen',
    #             confidence_weight=0.9,
    #             description="Professional aptitude requirements"
    #         ),
    #         Pattern(
    #             name="pruefungsordnung",
    #             regex=r'\bPrüfungsordnung\b|\bPrüfungsanforderungen\b',
    #             confidence_weight=0.9,
    #             description="Examination regulations"
    #         ),
    #         Pattern(
    #             name="lehrplan",
    #             regex=r'\bLehrplan\b|\bLehrpläne\b',
    #             confidence_weight=0.85,
    #             description="Curriculum/teaching plan"
    #         ),
    #         Pattern(
    #             name="ausbildungsordnung",
    #             regex=r'\bAusbildungsordnung\b',
    #             confidence_weight=0.9,
    #             description="Training regulations"
    #         ),
    #         Pattern(
    #             name="verordnung",
    #             regex=r'\bVerordnung\b|\bAnordnung\b',
    #             confidence_weight=0.8,
    #             description="Regulation/decree"
    #         )
    #     ]
    #
    #     for pattern in doc_type_patterns:
    #         self.register_pattern("document_type", pattern)
    #
    # def _add_profession_patterns(self):
    #     """Add profession extraction patterns"""
    #     profession_patterns = [
    #         Pattern(
    #             name="suffix_fasser",
    #             regex=r'([A-ZÄÖÜ][a-zäöüß]{6,20}fasser)',
    #             confidence_weight=0.8,
    #             description="Professions ending in 'fasser'"
    #         ),
    #         Pattern(
    #             name="suffix_macher",
    #             regex=r'([A-ZÄÖÜ][a-zäöüß]{6,20}macher)',
    #             confidence_weight=0.8,
    #             description="Professions ending in 'macher'"
    #         ),
    #         Pattern(
    #             name="suffix_schmidt",
    #             regex=r'([A-ZÄÖÜ][a-zäöüß]{6,20}schmidt)',
    #             confidence_weight=0.8,
    #             description="Professions ending in 'schmidt'"
    #         ),
    #         Pattern(
    #             name="suffix_bauer",
    #             regex=r'([A-ZÄÖÜ][a-zäöüß]{6,20}bauer)',
    #             confidence_weight=0.8,
    #             description="Professions ending in 'bauer'"
    #         ),
    #         Pattern(
    #             name="common_professions",
    #             regex=r'(Kaufmann|Mechaniker|Elektriker|Bäcker|Schneider|Tischler)',
    #             confidence_weight=0.9,
    #             description="Common profession names"
    #         )
    #     ]
    #
    #     for pattern in profession_patterns:
    #         self.register_pattern("profession", pattern)

    def register_pattern(self, field: str, pattern: Pattern):
        """Register a pattern for a specific field"""
        if field not in self.patterns:
            self.patterns[field] = []
        self.patterns[field].append(pattern)
        logger.debug(f"Registered pattern '{pattern.name}' for field '{field}'")

    def extract_field(self, text: str, field: str) -> List[Match]:
        """Extract all matches for a specific field"""
        matches = []
        field_patterns = self.patterns.get(field, [])

        for pattern in field_patterns:
            try:
                pattern_matches = pattern.find_matches(text)
                matches.extend(pattern_matches)
            except Exception as e:
                logger.warning(f"Pattern {pattern.name} failed: {e}")

        # Sort by confidence score (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def get_pattern_info(self) -> Dict[str, List[Dict]]:
        """Get information about all registered patterns"""
        info = {}
        for field, patterns in self.patterns.items():
            info[field] = [
                {
                    'name': p.name,
                    'description': p.description,
                    'confidence_weight': p.confidence_weight
                }
                for p in patterns
            ]
        return info


class GermanDateParser:
    """Specialized parser for German dates in historical documents"""

    def __init__(self):
        self.german_months = {
            'januar': 1, 'jan': 1, 'zamuar': 1, 'zanuar': 1, 'jamuar': 1, 'jänner': 1,
            'februar': 2, 'feb': 2,
            'märz': 3, 'mär': 3, 'maerz': 3,
            'april': 4, 'apr': 4, 'aprıl': 4,
            'mai': 5,
            'juni': 6, 'jun': 6,
            'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11,
            'dezember': 12, 'dez': 12, 'deeember': 12, 'dezernber': 12
        }

    def parse_date_from_match(self, match: Match) -> Optional[datetime]:
        """Parse a datetime from a date match"""
        try:
            # Try numeric date with space first (DD.MM. YYYY)
            numeric_space_match = re.search(r'(\d{1,2})\.(\d{1,2})\.\s+(\d{4})', match.value)
            if numeric_space_match:
                day_str, month_str, year_str = numeric_space_match.groups()
                day = int(day_str)
                month = int(month_str)
                year = int(year_str)
                return datetime(year, month, day)

            # Try numeric date without space (DD.MM.YYYY)
            numeric_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', match.value)
            if numeric_match:
                day_str, month_str, year_str = numeric_match.groups()
                day = int(day_str)
                month = int(month_str)
                year = int(year_str)
                return datetime(year, month, day)

            # Then try month names (existing code)
            regex_match = re.search(r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})', match.value)
            if regex_match:
                day_str, month_str, year_str = regex_match.groups()
                day = int(day_str)
                year = int(year_str)
                month_normalized = month_str.lower().strip()

                if month_normalized in self.german_months:
                    month = self.german_months[month_normalized]
                    return datetime(year, month, day)

            return None

        except (ValueError, AttributeError) as e:
            logger.debug(f"Failed to parse date from match: {e}")
            return None

    def score_date_context(self, match: Match) -> float:
        """Score a date match based on context indicators"""
        context = match.context.lower()
        score = match.confidence

        # Very high priority indicators
        very_high_indicators = [
            r'berlin,?\s+den',
            r'münchen,?\s+den',
        ]

        for indicator in very_high_indicators:
            if re.search(indicator, context):
                score += 0.3

        # High priority indicators
        high_indicators = [
            r'stand\s+vom',
            r'stcmd\s+vom',
            r'\(.*?stand.*?vom',
            r'\(.*?stcmd.*?vom',
        ]

        for indicator in high_indicators:
            if re.search(indicator, context):
                score += 0.2

        # Medium priority indicators
        medium_indicators = [
            r'ausgegeben\s+am',
            r'verkündet\s+am',
            r'wirkung\s+vom',
        ]

        for indicator in medium_indicators:
            if re.search(indicator, context):
                score += 0.12

        # Pattern-specific bonuses
        if 'vom' in match.value.lower():
            score += 0.08

        if 'den' in match.value.lower():
            score += 0.06

        # Parentheses bonus
        if '(' in context and ')' in context:
            score += 0.10

        # Position scoring (earlier text gets bonus)
        relative_pos = match.start_pos / len(match.context) if match.context else 0.5
        if relative_pos < 0.2:
            score += 0.08
        elif relative_pos < 0.4:
            score += 0.04

        # Negative indicators
        negative_indicators = [
            r'seit\s+dem',
            r'ab\s+dem',
            r'erfolgte',
            r'geboren.*am',
            r'verstorben.*am',
        ]

        for neg_indicator in negative_indicators:
            if re.search(neg_indicator, context):
                score -= 0.05

        return max(score, 0.0)


class PatternBasedExtractor:
    """Main pattern-based extraction engine"""

    def __init__(self, pattern_registry: Optional[PatternRegistry] = None):
        self.pattern_registry = pattern_registry or PatternRegistry()
        self.date_parser = GermanDateParser()

    def extract_metadata_fields(self, text: str) -> Dict[str, Tuple[Optional[str], float]]:
        """Extract all metadata fields using patterns"""
        results = {}

        # Extract each field type
        for field in ['title', 'publisher', 'document_type', 'profession']:
            matches = self.pattern_registry.extract_field(text, field)
            if matches:
                best_match = matches[0]  # Highest confidence
                results[field] = (best_match.value, best_match.confidence)
            else:
                results[field] = (None, 0.0)

        # Special handling for dates
        date_result = self.extract_date(text)
        results['date'] = date_result

        return results

    def extract_date(self, text: str) -> Tuple[Optional[datetime], float]:
        """Extract the best date from text with enhanced scoring"""
        # Pre-process text to fix common OCR errors
        text = re.sub(r'\bl\.\s*([a-zA-ZäöüÄÖÜß]+)', r'1. \1', text)

        # Get all date matches
        date_matches = self.pattern_registry.extract_field(text, "date")

        if not date_matches:
            return None, 0.0

        # Parse and score dates
        scored_dates = []
        for match in date_matches:
            parsed_date = self.date_parser.parse_date_from_match(match)
            # import pdb; pdb.set_trace()
            if parsed_date:
                context_score = self.date_parser.score_date_context(match)
                scored_dates.append((parsed_date, context_score, match))

        if not scored_dates:
            return None, 0.0

        # Return highest scoring date
        scored_dates.sort(key=lambda x: x[1], reverse=True)
        best_date, best_score, best_match = scored_dates[0]

        logger.debug(f"Selected date: {best_date} with score {best_score:.2f} from pattern {best_match.pattern_name}")

        return best_date, best_score

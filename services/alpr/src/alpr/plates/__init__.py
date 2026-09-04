"""Country-aware plate grammars.

Phase 5. The region-specific value of the project lives here: detection is
trained region-agnostically because a plate detector transfers across
countries, but *reading* a plate does not — `DA-XY 123` and `MH12AB1234` obey
different rules, and applying the wrong one turns a correct read into a
rejected one.

    >>> from alpr.plates import parse
    >>> parse("MH 12 A8 1Z34").text
    'MH12AB1234'
"""

from __future__ import annotations

from alpr.data.schema import Region
from alpr.plates.base import PlateFormat, PlateMatch, normalize
from alpr.plates.correct import (
    DIGIT_TO_LETTERS,
    EDIT_PENALTY,
    LETTER_TO_DIGITS,
    MIN_CONFIDENCE,
    alternatives,
    correct_to_format,
    parse_plate,
)
from alpr.plates.germany import DISTRICT_PREFIXES, GermanyFormat
from alpr.plates.india import STATE_CODES, IndiaFormat
from alpr.plates.poland import VOIVODESHIPS, PolandFormat

INDIA = IndiaFormat()
GERMANY = GermanyFormat()
POLAND = PolandFormat()

FORMATS: list[PlateFormat] = [INDIA, GERMANY, POLAND]

FORMATS_BY_REGION: dict[Region, PlateFormat] = {
    Region.INDIA: INDIA,
    Region.GERMANY: GERMANY,
    Region.POLAND: POLAND,
}


def parse(
    raw: str,
    *,
    region: Region | None = None,
    max_edits: int = 2,
    min_confidence: float = MIN_CONFIDENCE,
) -> PlateMatch | None:
    """Parse a raw OCR string against every known grammar.

    Args:
        raw: OCR output, unnormalized.
        region: restrict to one country's grammar. Set it when the deployment
            site is known — it stops an Indian reading winning on a German road.
        max_edits: how many OCR confusions may be repaired.
        min_confidence: floor below which a reading is discarded. Noise can
            be corrected into something plausible, and a false plate in the
            log is worse than a missed one.

    Returns:
        The best match, or None when nothing fits.
    """
    return parse_plate(
        raw, FORMATS, region=region, max_edits=max_edits, min_confidence=min_confidence
    )


def format_display(match: PlateMatch) -> str:
    """Render a match the way its country writes it."""
    fmt = FORMATS_BY_REGION.get(match.region)
    return fmt.format_display(match) if fmt else match.text


__all__ = [
    "DIGIT_TO_LETTERS",
    "DISTRICT_PREFIXES",
    "EDIT_PENALTY",
    "FORMATS",
    "FORMATS_BY_REGION",
    "GERMANY",
    "INDIA",
    "POLAND",
    "VOIVODESHIPS",
    "LETTER_TO_DIGITS",
    "MIN_CONFIDENCE",
    "STATE_CODES",
    "GermanyFormat",
    "IndiaFormat",
    "PolandFormat",
    "PlateFormat",
    "PlateMatch",
    "alternatives",
    "correct_to_format",
    "format_display",
    "normalize",
    "parse",
    "parse_plate",
]

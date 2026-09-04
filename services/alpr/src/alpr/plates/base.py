"""Plate grammar interface and normalization.

A plate format is a grammar. That grammar is what turns a raw OCR string into
a decision, and it is worth more than better OCR: `MH12A81234` is unreadable
as a plate until you know position 8 must be a digit, at which point the `B`
that OCR saw is obviously an `8`.

Each format is responsible for one country. `alpr.plates.correct` uses them to
repair OCR confusions, and Phase 6 uses the confidence to pick a winner from
several frames of the same vehicle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from alpr.data.schema import Region

# Kept out of normalization: German district prefixes genuinely contain
# umlauts (LÖ Lörrach, GÖ Göttingen, WÜ Würzburg), so folding them to ASCII
# would corrupt real plates rather than clean them up.
_UMLAUTS = "ÄÖÜ"
_KEEP = re.compile(rf"[^A-Z0-9{_UMLAUTS}]")

# Indian plates often carry "IND" beside the state emblem on the embossed
# strip. OCR reads it as part of the string; it is not part of the number.
_IND_PREFIX = re.compile(r"^IND(?=[A-Z]{2}\d)")


# German plates are written district-first with a separator: "M-AB 123".
# Normalization has to drop that separator, but it is the only thing that
# distinguishes M-AB (München) from MA-B (Mannheim), so it is captured before
# being discarded.
_DISTRICT_HINT = re.compile(rf"^\s*([A-Z{_UMLAUTS}]{{1,3}})\s*[-–—.\s]\s*(?=[A-Z])", re.IGNORECASE)


def district_hint(raw: str) -> str | None:
    """Extract the district code delimited by a separator in a raw OCR string.

    Returns None when no separator is visible, which is common — plenty of
    reads lose it. The hint is advisory: it breaks ties between otherwise
    equally valid segmentations rather than forcing a reading.
    """
    if not raw:
        return None
    found = _DISTRICT_HINT.match(raw.upper())
    return found.group(1) if found else None


def normalize(text: str) -> str:
    """Strip a raw OCR string down to plate characters.

    Uppercases, removes separators (spaces, hyphens, dots) and drops the
    decorative "IND" strip. Does not attempt any character correction — that
    is `alpr.plates.correct`'s job and needs a grammar to do safely.
    """
    if not text:
        return ""
    cleaned = _KEEP.sub("", text.upper())
    return _IND_PREFIX.sub("", cleaned)


@dataclass(frozen=True)
class PlateMatch:
    """A plate string parsed against a grammar.

    Attributes:
        text: the normalized, possibly corrected plate string.
        region: which grammar matched.
        format_name: the specific rule within that grammar (a country can have
            several — India's older series and its newer BH series differ).
        edits: how many characters correction had to change. 0 means OCR read
            it cleanly; a high count means the match was a stretch.
        confidence: 0..1 grammar fit. Combines an exact-match base, a penalty
            per edit, and a bonus when the region code is a real one.
        components: the named parts of the match (state code, district, series
            number), useful for the Excel log and for debugging.
        original: the normalized input before correction.
    """

    text: str
    region: Region
    format_name: str
    edits: int = 0
    confidence: float = 1.0
    components: dict[str, str] = field(default_factory=dict)
    original: str = ""

    @property
    def corrected(self) -> bool:
        return self.edits > 0

    def __str__(self) -> str:
        return self.text


class PlateFormat(ABC):
    """A country's plate grammar."""

    region: Region
    name: str

    @abstractmethod
    def match(self, text: str, *, district_hint: str | None = None) -> PlateMatch | None:
        """Parse an already-normalized string, or return None if it does not fit.

        Args:
            text: normalized candidate.
            district_hint: the region code as delimited in the raw OCR string,
                when a separator was visible. `MAB123` is ambiguous between
                M-AB (München) and MA-B (Mannheim); if OCR actually read
                "M-AB", that hyphen settles it.
        """

    @abstractmethod
    def format_display(self, match: PlateMatch) -> str:
        """Render a match the way the country writes it, with separators."""

    def accepts_length(self, length: int) -> bool:
        """Cheap pre-filter used to skip correction search on hopeless inputs."""
        return 1 <= length <= 12

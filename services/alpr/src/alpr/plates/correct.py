"""Grammar-constrained correction of OCR confusions.

OCR confuses characters that look alike: `0/O`, `1/I`, `5/S`, `8/B`. Correcting
these blindly makes accuracy *worse* — rewriting every `0` to `O` breaks every
plate that legitimately contains a zero.

Correcting them **against a grammar** is safe, because the grammar says which
positions may hold a letter and which a digit. `MH12A81234` has a `B`-shaped
`8` where India's rule demands a digit, so exactly one reading survives. The
search is a bounded edit search rather than a positional rewrite, because the
formats have variable-length parts — Germany's district prefix is one to three
letters, so position alone does not determine the character class.

Cost is bounded: at most `max_edits` positions change, each with at most three
alternatives, over a string of at most twelve characters.
"""

from __future__ import annotations

from itertools import combinations, product

from alpr.data.schema import Region
from alpr.plates.base import PlateFormat, PlateMatch, district_hint, normalize

# A letter OCR may have meant, given the digit it reported.
DIGIT_TO_LETTERS: dict[str, str] = {
    "0": "ODQ",
    "1": "IL",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

# The reverse: a digit OCR may have meant, given the letter it reported.
LETTER_TO_DIGITS: dict[str, str] = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

# Each correction costs confidence. Two edits still beats no match at all, but
# should never outrank a clean read from another frame of the same vehicle.
EDIT_PENALTY = 0.18


def alternatives(char: str) -> str:
    """Characters OCR might have meant when it reported `char`."""
    return DIGIT_TO_LETTERS.get(char, "") or LETTER_TO_DIGITS.get(char, "")


# A reading that is *both* corrected and has an unrecognized region code is
# too weak to trust: 0.80 (unknown district) − 0.18 (one edit) = 0.62 is what
# "hello" scores after O→0 turns it into the plausible-looking HEL-L 0. Every
# genuine reading observed sits at 0.80 or above.
MIN_CONFIDENCE = 0.7


def _score(match: PlateMatch, edits: int) -> float:
    # Rounded because the value lands in a spreadsheet cell, and
    # 0.8200000000000001 is not something to show a user.
    return round(max(0.0, match.confidence - edits * EDIT_PENALTY), 3)


def correct_to_format(
    text: str,
    fmt: PlateFormat,
    *,
    max_edits: int = 2,
    district_hint: str | None = None,
) -> PlateMatch | None:
    """Find the closest reading of `text` that satisfies `fmt`.

    Args:
        text: an already-normalized candidate.
        fmt: the grammar to satisfy.
        max_edits: how many characters may be reinterpreted. Beyond two, a
            "match" says more about the search than about the plate.

    Returns:
        The best-scoring match, or None if no reading within `max_edits` fits.
        Fewer edits always wins; ties break on the grammar's own confidence,
        so a real state code beats an invented one.
    """
    if not text or not fmt.accepts_length(len(text)):
        return None

    exact = fmt.match(text, district_hint=district_hint)
    # Only a *confident* exact match ends the search. An exact reading with an
    # unrecognized region code can legitimately lose to a one-edit reading with
    # a real one: `DAXYI23` parses as-is into the unknown district DAX (0.80),
    # but repairing I→1 yields Darmstadt's DA-XY 123 (1.00 − 0.18 = 0.82).
    # Short-circuiting on any match at all would never consider that.
    if exact is not None and exact.confidence >= 1.0:
        return exact

    positions = [i for i, ch in enumerate(text) if alternatives(ch)]
    if not positions:
        return exact

    best: PlateMatch | None = exact
    best_score = exact.confidence if exact else -1.0

    for n_edits in range(1, max_edits + 1):
        for chosen in combinations(positions, n_edits):
            option_sets = [alternatives(text[i]) for i in chosen]
            for replacements in product(*option_sets):
                candidate = list(text)
                for index, replacement in zip(chosen, replacements, strict=True):
                    candidate[index] = replacement

                found = fmt.match("".join(candidate), district_hint=district_hint)
                if not found:
                    continue

                score = _score(found, n_edits)
                if score > best_score:
                    best_score = score
                    best = PlateMatch(
                        text=found.text,
                        region=found.region,
                        format_name=found.format_name,
                        edits=n_edits,
                        confidence=score,
                        components=found.components,
                        original=text,
                    )

        # Stop once this edit count has produced something better than what we
        # started with: a two-edit reading can never beat a one-edit reading of
        # the same string, since each edit costs a fixed penalty.
        if best is not None and best.edits > 0:
            break

    return best


def parse_plate(
    raw: str,
    formats: list[PlateFormat],
    *,
    region: Region | None = None,
    max_edits: int = 2,
    min_confidence: float = MIN_CONFIDENCE,
) -> PlateMatch | None:
    """Parse a raw OCR string against the given grammars.

    Args:
        raw: the OCR output, unnormalized.
        formats: grammars to try.
        region: restrict to one region's grammar. Worth setting when the
            deployment site is known — it removes the chance of an Indian
            reading winning on a German road.
        max_edits: passed through to `correct_to_format`.
        min_confidence: floor below which a match is discarded. Grammars are
            permissive enough that noise can be corrected into something
            plausible — "hello" becomes HEL-L 0 once O→0 is applied — and a
            false plate in the log is worse than a missed one.

    Returns:
        The highest-confidence match across the grammars, or None.
    """
    text = normalize(raw)
    if not text:
        return None

    # Derived from the raw string: normalization is about to drop the
    # separator that distinguishes M-AB from MA-B.
    hint = district_hint(raw)

    candidates = [f for f in formats if region is None or f.region is region]

    best: PlateMatch | None = None
    for fmt in candidates:
        found = correct_to_format(text, fmt, max_edits=max_edits, district_hint=hint)
        if found is None:
            continue
        if best is None or (found.confidence, -found.edits) > (best.confidence, -best.edits):
            best = found

    if best is not None and best.confidence < min_confidence:
        return None
    return best

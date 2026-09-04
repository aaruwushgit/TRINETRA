"""Polish plate grammar.

`PO 2427W`: a two-or-three letter territorial code, then four or five
characters of which **at most two may be letters**. The first letter of the
code is the voivodeship; a second letter marks an urban district, a second and
third a rural one.

Two rules make this grammar unusually strong for OCR correction, more so than
either India's or Germany's:

**Q is unused entirely.** It does not appear in Polish, so it appears on no
plate. A `Q` anywhere is a misread.

**B, D, I, O and Z are banned from the series** — the part after the code —
precisely *because they look like digits*. Poland removed the ambiguity at the
design stage. So a `B` in the series is an `8`, an `O` is a `0`, an `I` is a
`1`, and the correction is not a guess but a certainty. This is the cleanest
case in the project of a real-world format encoding exactly the constraint an
OCR pipeline needs.

**Individual (vanity) plates carry a single voivodeship letter** followed by
three to five characters, and they had to be modelled for a reason found by
measurement rather than foreseen. Every grammar-corrupted read in the
end-to-end run was a 1-letter plate that a *single* edit pushed into the
2-letter standard shape: `P74103` became `PT4103`, `W2515T` became `WZ515T`.
The grammar was inventing a standard plate out of a valid individual one.

Modelling them at slightly lower confidence fixes that: the individual reading
matches exactly, and an exact match outscores a corrected one. Longer vanity
strings (`Z1NORGE`, `DOROSSO`) exceed five characters and are still rejected —
they cannot be validated without a registry, and a false plate in the log is
worse than a missing one.
"""

from __future__ import annotations

import re

from alpr.data.schema import Region
from alpr.plates.base import PlateFormat, PlateMatch

# The 16 voivodeships. The first letter of every standard plate is one of
# these, which makes an unrecognised first letter a strong misread signal.
VOIVODESHIPS: dict[str, str] = {
    "B": "Podlaskie",
    "C": "Kujawsko-Pomorskie",
    "D": "Dolnośląskie",
    "E": "Łódzkie",
    "F": "Lubuskie",
    "G": "Pomorskie",
    "K": "Małopolskie",
    "L": "Lubelskie",
    "N": "Warmińsko-Mazurskie",
    "O": "Opolskie",
    "P": "Wielkopolskie",
    "R": "Podkarpackie",
    "S": "Śląskie",
    "T": "Świętokrzyskie",
    "W": "Mazowieckie",
    "Z": "Zachodniopomorskie",
}

# Banned from the series because they resemble digits. This is the rule that
# turns an ambiguous OCR read into a determined one.
SERIES_BANNED = "BDIOZQ"

# Q is absent from Polish and appears on no plate, anywhere.
_CODE = "[A-PR-Z]"
_SERIES_LETTER = "[ACEFGHJKLMNPRSTUVWXY]"
_SERIES_CHAR = f"(?:[0-9]|{_SERIES_LETTER})"

_PATTERN = re.compile(rf"^(?P<code>{_CODE}{{2,3}})(?P<series>{_SERIES_CHAR}{{4,5}})$")

# Individual plates: one voivodeship letter, then 3-5 free characters. Scored
# below the standard format because they are far rarer — so where a string
# fits both, the standard reading wins.
_INDIVIDUAL = re.compile(rf"^(?P<code>{_CODE})(?P<series>[A-PR-Z0-9]{{3,5}})$")
INDIVIDUAL_CONFIDENCE = 0.85

MAX_SERIES_LETTERS = 2


class PolandFormat(PlateFormat):
    region = Region.POLAND
    name = "PL"

    def match(self, text: str, *, district_hint: str | None = None) -> PlateMatch | None:
        del district_hint  # Polish codes are unambiguous: the split is by shape.

        found = _PATTERN.match(text)
        if not found:
            return self._match_individual(text)

        code = found.group("code")
        series = found.group("series")

        # "Up to two may be letters" is the rule that stops the series
        # swallowing a code, and it is what makes 5-character series mostly
        # digits.
        letters = sum(char.isalpha() for char in series)
        if letters > MAX_SERIES_LETTERS:
            return self._match_individual(text)

        known = code[0] in VOIVODESHIPS
        return PlateMatch(
            text=text,
            region=Region.POLAND,
            format_name="PL",
            # Harsher than Germany's unknown-district penalty: the voivodeship
            # set is closed and tiny, so an unrecognised first letter is far
            # more likely a misread than a plate this grammar has not seen.
            confidence=1.0 if known else 0.6,
            components={
                "code": code,
                "series": series,
                "voivodeship": VOIVODESHIPS.get(code[0], ""),
            },
            original=text,
        )

    def _match_individual(self, text: str) -> PlateMatch | None:
        found = _INDIVIDUAL.match(text)
        if not found or found.group("code") not in VOIVODESHIPS:
            return None
        # At least one digit. This is a false-positive guard rather than a
        # legal requirement: the format is loose enough that "ZZZZZZ" would
        # otherwise read as a Zachodniopomorskie vanity plate. It costs the
        # rare all-letter vanity plate, which is the cheaper mistake — those
        # are unverifiable without a registry anyway.
        if not any(char.isdigit() for char in found.group("series")):
            return None
        return PlateMatch(
            text=text,
            region=Region.POLAND,
            format_name="PL-individual",
            confidence=INDIVIDUAL_CONFIDENCE,
            components={
                "code": found.group("code"),
                "series": found.group("series"),
                "voivodeship": VOIVODESHIPS[found.group("code")],
            },
            original=text,
        )

    def accepts_length(self, length: int) -> bool:
        # Individual plates start at 1 + 3 = 4; standard runs to 3 + 5 = 8.
        return 4 <= length <= 8

    def format_display(self, match: PlateMatch) -> str:
        return f"{match.components['code']} {match.components['series']}"

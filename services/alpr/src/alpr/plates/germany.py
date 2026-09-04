"""German plate grammar.

`DA-XY 1234`: a one-to-three letter district prefix, one or two letters, and
one to four digits. Two suffixes carry meaning — `E` for electric vehicles and
`H` for historic ones — and both sit after the digits.

The hard constraint is a total of at most **eight** alphanumerics: a three
letter prefix leaves room for two letters and three digits, not four. That
rule is what makes the grammar useful for correction, because it rules out
otherwise plausible-looking readings.

Unlike India's closed list of state codes, Germany has roughly 700 district
prefixes and the set changes. `DISTRICT_PREFIXES` therefore holds the common
ones and is used only to *raise* confidence — never to reject a plate, which
would make every unlisted district unreadable.
"""

from __future__ import annotations

import re

from alpr.data.schema import Region
from alpr.plates.base import PlateFormat, PlateMatch

MAX_CHARS = 8

# A representative subset: state capitals, large cities and common districts.
# Extending it improves confidence scoring and correction tie-breaks; it never
# changes what is accepted.
DISTRICT_PREFIXES: dict[str, str] = {
    "A": "Augsburg",
    "AA": "Ostalbkreis",
    "AC": "Aachen",
    "B": "Berlin",
    "BA": "Bamberg",
    "BI": "Bielefeld",
    "BN": "Bonn",
    "BO": "Bochum",
    "BS": "Braunschweig",
    "C": "Chemnitz",
    "CB": "Cottbus",
    "CO": "Coburg",
    "D": "Düsseldorf",
    "DA": "Darmstadt",
    "DD": "Dresden",
    "DO": "Dortmund",
    "DU": "Duisburg",
    "E": "Essen",
    "EF": "Erfurt",
    "EN": "Ennepe-Ruhr-Kreis",
    "ER": "Erlangen",
    "F": "Frankfurt am Main",
    "FD": "Fulda",
    "FR": "Freiburg",
    "FÜ": "Fürth",
    "G": "Gera",
    "GI": "Gießen",
    "GÖ": "Göttingen",
    "H": "Hannover",
    "HB": "Bremen",
    "HD": "Heidelberg",
    "HH": "Hamburg",
    "HL": "Lübeck",
    "HN": "Heilbronn",
    "HRO": "Rostock",
    "J": "Jena",
    "K": "Köln",
    "KA": "Karlsruhe",
    "KI": "Kiel",
    "KL": "Kaiserslautern",
    "KO": "Koblenz",
    "KS": "Kassel",
    "L": "Leipzig",
    "LU": "Ludwigshafen",
    "LÖ": "Lörrach",
    "M": "München",
    "MA": "Mannheim",
    "MD": "Magdeburg",
    "MI": "Minden-Lübbecke",
    "MZ": "Mainz",
    "N": "Nürnberg",
    "OB": "Oberhausen",
    "OF": "Offenbach",
    "OL": "Oldenburg",
    "OS": "Osnabrück",
    "P": "Potsdam",
    "PB": "Paderborn",
    "R": "Regensburg",
    "RO": "Rosenheim",
    "S": "Stuttgart",
    "SB": "Saarbrücken",
    "SN": "Schwerin",
    "SU": "Rhein-Sieg-Kreis",
    "TR": "Trier",
    "TÜ": "Tübingen",
    "UL": "Ulm",
    "W": "Wuppertal",
    "WI": "Wiesbaden",
    "WÜ": "Würzburg",
    "Z": "Zwickau",
}

# Split off the digits first: they unambiguously separate the letter block
# from the optional E/H suffix. How the letter block divides into district +
# letters is decided afterwards, because that split is genuinely ambiguous.
_SHAPE = re.compile(r"^(?P<letters>[A-ZÄÖÜ]+)(?P<number>\d{1,4})(?P<suffix>[EH]?)$")

_SUFFIX_MEANING = {"E": "electric", "H": "historic"}

# A plate of three body characters (B-A 1) is legal but vanishingly rare,
# while three-character noise out of an OCR pass is common. Requiring four
# trades an almost-never-seen plate for a large drop in false positives —
# which matters because correction can otherwise manufacture a "valid" plate
# out of junk like ZZZ.
MIN_CHARS = 4


class GermanyFormat(PlateFormat):
    region = Region.GERMANY
    name = "DE"

    def match(self, text: str, *, district_hint: str | None = None) -> PlateMatch | None:
        shape = _SHAPE.match(text)
        if not shape:
            return None

        letters = shape.group("letters")
        number = shape.group("number")
        suffix = shape.group("suffix")

        # The eight-character cap is the rule that gives this grammar its
        # discriminating power; without it the pattern would accept strings
        # no real plate can have.
        body_length = len(letters) + len(number)
        if not (MIN_CHARS <= body_length <= MAX_CHARS):
            return None

        # `DAXY123` could split as DAX|Y, DA|XY or D|AXY. Greedy matching picks
        # DAX|Y and loses Darmstadt entirely; the prefix table is what resolves
        # the ambiguity. A known prefix always wins, and among equally known
        # candidates the longer one does — a two-letter district is a better
        # explanation than a one-letter district that merely also fits.
        splits = [
            (letters[:n], letters[n:])
            for n in (1, 2, 3)
            if 1 <= len(letters) - n <= 2  # the letter group is 1-2 characters
        ]
        if not splits:
            return None

        # A separator seen in the raw read outranks everything: "M-AB 123" is
        # München, not Mannheim, and no amount of prefix-table reasoning can
        # recover that once the hyphen is gone.
        district, letter_group = max(
            splits,
            key=lambda s: (
                s[0] == district_hint,
                s[0] in DISTRICT_PREFIXES,
                len(s[0]),
            ),
        )
        known = district in DISTRICT_PREFIXES

        components = {
            "district": district,
            "letters": letter_group,
            "number": number,
            "suffix": suffix,
            "district_name": DISTRICT_PREFIXES.get(district, ""),
        }
        if suffix:
            components["suffix_meaning"] = _SUFFIX_MEANING[suffix]

        return PlateMatch(
            text=text,
            region=Region.GERMANY,
            format_name="DE",
            # A softer penalty than India's: the prefix list here is knowingly
            # partial, so an unlisted district is ordinary rather than suspect.
            confidence=1.0 if known else 0.8,
            components=components,
            original=text,
        )

    def accepts_length(self, length: int) -> bool:
        return MIN_CHARS <= length <= MAX_CHARS + 1

    def format_display(self, match: PlateMatch) -> str:
        c = match.components
        return f"{c['district']}-{c['letters']} {c['number']}{c['suffix']}"

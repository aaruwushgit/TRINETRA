"""Indian plate grammar.

Two series are in use:

**Standard** — `MH 12 AB 1234`: a two-letter state code, a one-or-two digit
RTO district code, an optional one-to-three letter series, and a number of
**one to four** digits. Both the optional series and the short numbers are
real: early registrations in a district have no series letters, and plates
like `KL 54 H 369` and `TN 58 AM 1` are ordinary.

The number was originally written as exactly four digits. End-to-end
measurement against hand-labelled plates found that wrong — it rejected
`KL54H369`, `TN58AM1` and `KL7BZ99`, all genuine. Unit tests had not caught
it because they were written from the same wrong assumption as the code.

**BH (Bharat)** — `21 BH 1234 AB`: introduced for vehicles that move between
states. Two year digits, the literal `BH`, four digits, then one or two
letters. It reverses the usual letters-then-digits shape, which is exactly
why it needs its own rule rather than a looser regex: a pattern permissive
enough to cover both would accept a lot of OCR noise as valid.
"""

from __future__ import annotations

import re

from alpr.data.schema import Region
from alpr.plates.base import PlateFormat, PlateMatch

# Every current state and union territory code. Used to score confidence and
# to disambiguate correction candidates — not to reject: a genuinely new code
# should degrade the score, not make the plate unreadable.
STATE_CODES: dict[str, str] = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha (former)",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TG": "Telangana",
    "TR": "Tripura",
    "TS": "Telangana (former)",
    "UA": "Uttarakhand (former)",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

_STANDARD = re.compile(
    r"^(?P<state>[A-Z]{2})(?P<district>\d{1,2})(?P<series>[A-Z]{0,3})(?P<number>\d{1,4})$"
)
_BH = re.compile(r"^(?P<year>\d{2})BH(?P<number>\d{4})(?P<series>[A-Z]{1,2})$")


class IndiaFormat(PlateFormat):
    region = Region.INDIA
    name = "IN"

    def match(self, text: str, *, district_hint: str | None = None) -> PlateMatch | None:
        # The hint is a German-plate disambiguator; Indian plates have no
        # ambiguous split, since letters and digits alternate at fixed points.
        del district_hint
        bh = _BH.match(text)
        if bh:
            return PlateMatch(
                text=text,
                region=Region.INDIA,
                format_name="IN-BH",
                confidence=1.0,
                components=bh.groupdict(),
                original=text,
            )

        standard = _STANDARD.match(text)
        if not standard:
            return None

        parts = standard.groupdict()
        known = parts["state"] in STATE_CODES
        return PlateMatch(
            text=text,
            region=Region.INDIA,
            format_name="IN",
            # An unknown state code is a strong signal the read is wrong: the
            # list is closed in practice, so "XZ12AB1234" is almost certainly
            # a misread rather than a new state.
            confidence=1.0 if known else 0.55,
            components={**parts, "state_name": STATE_CODES.get(parts["state"], "")},
            original=text,
        )

    def accepts_length(self, length: int) -> bool:
        # Shortest is state + 1 district digit + 1 number digit = 4; BH plates
        # run to 10.
        return 4 <= length <= 11

    def format_display(self, match: PlateMatch) -> str:
        c = match.components
        if match.format_name == "IN-BH":
            return f"{c['year']} BH {c['number']} {c['series']}"
        parts = [c["state"], c["district"]]
        if c["series"]:
            parts.append(c["series"])
        parts.append(c["number"])
        return " ".join(parts)

"""Polish plate grammar.

Poland is the strongest case in the project for grammar-constrained
correction, because the format itself bans the letters OCR confuses.
"""

from __future__ import annotations

import pytest

from alpr.data.schema import Region
from alpr.plates import POLAND, VOIVODESHIPS, correct_to_format, format_display, parse


class TestStandardFormat:
    @pytest.mark.parametrize(
        "plate",
        [
            "PO2427W",  # 2-letter code, 5-char series, 1 letter
            "SBE81PF",  # 3-letter code, 4-char series, 2 letters
            "PCTC843",  # letter first in the series
            "WOS3020C",
            "ZGL17775",  # all digits
            "WA11003",
            "ZKOE187",
            "NO0994A",
            "PLE24JP",
        ],
    )
    def test_accepts_real_plates_from_the_dataset(self, plate):
        match = POLAND.match(plate)
        assert match is not None, plate
        assert match.region is Region.POLAND

    def test_series_may_hold_at_most_two_letters(self):
        # The rule that stops a series swallowing a code. Three letters is not
        # a standard plate — it may still fall through to the far looser
        # individual format, so the check is on which format matched.
        assert POLAND.match("SBE81PF").format_name == "PL"  # exactly two
        three = POLAND.match("POACEF")
        assert three is None or three.format_name != "PL"

    def test_rejects_q_anywhere(self):
        # Q is absent from Polish and appears on no plate.
        assert POLAND.match("QO2427W") is None

    @pytest.mark.parametrize("banned", ["B", "D", "I", "O", "Z"])
    def test_banned_letters_cannot_appear_in_the_series(self, banned):
        # Poland removed these at the design stage precisely because they look
        # like digits. That is what makes the correction a certainty.
        assert POLAND.match(f"PO2{banned}27W") is None

    def test_unknown_voivodeship_lowers_confidence(self):
        known = POLAND.match("PO2427W")
        unknown = POLAND.match("XY2427W")
        assert unknown is not None
        assert unknown.confidence < known.confidence

    def test_resolves_the_voivodeship(self):
        assert POLAND.match("PO2427W").components["voivodeship"] == "Wielkopolskie"

    def test_display_format(self):
        assert format_display(POLAND.match("PO2427W")) == "PO 2427W"

    def test_voivodeship_codes_are_single_letters(self):
        assert all(len(c) == 1 and c.isupper() for c in VOIVODESHIPS)
        assert len(VOIVODESHIPS) == 16


class TestIndividualPlates:
    @pytest.mark.parametrize("plate", ["P74103", "W2515T", "P1RRA"])
    def test_accepts_one_letter_vanity_plates(self, plate):
        match = POLAND.match(plate)
        assert match is not None, plate
        assert match.format_name == "PL-individual"

    def test_scored_below_the_standard_format(self):
        # Individual plates are far rarer, so where a string fits both, the
        # standard reading should win.
        assert POLAND.match("P74103").confidence < POLAND.match("PO2427W").confidence

    def test_rejects_an_unknown_voivodeship_letter(self):
        assert POLAND.match("X1234") is None

    def test_long_vanity_strings_are_still_rejected(self):
        # Over five series characters, and unvalidatable without a registry.
        # A false plate in the log is worse than a missing one.
        for plate in ("Z1NORGE", "DOROSSO"):
            assert POLAND.match(plate) is None, plate

    def test_gogra01_is_an_ordinary_standard_plate(self):
        # It looks like a vanity plate and is not: GOG is a Pomorskie district
        # code and RA01 a legal series. Worth pinning, because I mistook it.
        match = POLAND.match("GOGRA01")
        assert match.format_name == "PL"
        assert match.components["code"] == "GOG"


class TestCorrection:
    def test_a_banned_letter_in_the_series_is_a_determined_fix(self):
        # Not a guess: B cannot legally appear there, so it is an 8.
        match = correct_to_format("PO2B27W", POLAND)
        assert match is not None
        assert match.text == "PO2827W"
        assert match.edits == 1

    def test_repairs_an_o_that_should_be_zero(self):
        match = correct_to_format("PCTCB43", POLAND)
        assert match.text == "PCTC843"

    def test_individual_plates_are_not_corrupted_into_standard_ones(self):
        # The bug end-to-end measurement found: a single edit pushed a valid
        # 1-letter plate into the 2-letter shape — P74103 became PT4103 and
        # W2515T became WZ515T. Modelling individual plates makes the exact
        # reading win.
        for plate in ("P74103", "W2515T"):
            match = parse(plate)
            assert match is not None, plate
            assert match.text == plate
            assert match.edits == 0


class TestAgainstOtherGrammars:
    def test_a_polish_plate_is_not_read_as_german(self):
        match = parse("PO2427W")
        assert match.region is Region.POLAND

    def test_an_indian_plate_is_still_indian(self):
        assert parse("MH12AB1234").region is Region.INDIA

    def test_a_german_plate_is_still_german(self):
        assert parse("DA-XY 123").region is Region.GERMANY

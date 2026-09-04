"""Plate grammars, and grammar-constrained OCR correction."""

from __future__ import annotations

import pytest

from alpr.data.schema import Region
from alpr.plates import (
    DISTRICT_PREFIXES,
    GERMANY,
    INDIA,
    STATE_CODES,
    alternatives,
    correct_to_format,
    format_display,
    normalize,
    parse,
)


class TestNormalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("mh12ab1234", "MH12AB1234"),
            ("MH 12 AB 1234", "MH12AB1234"),
            ("MH-12-AB-1234", "MH12AB1234"),
            ("DA-XY 123", "DAXY123"),
            ("  DA XY 123  ", "DAXY123"),
            ("DA.XY.123", "DAXY123"),
            ("", ""),
        ],
    )
    def test_strips_separators_and_uppercases(self, raw, expected):
        assert normalize(raw) == expected

    def test_drops_the_decorative_ind_strip(self):
        # Indian plates carry "IND" beside the state emblem; OCR reads it as
        # part of the string but it is not part of the number.
        assert normalize("IND MH12AB1234") == "MH12AB1234"

    def test_keeps_ind_when_it_is_not_the_strip(self):
        # Must not eat characters that merely start with those letters.
        assert normalize("INDORE") == "INDORE"

    def test_preserves_umlauts(self):
        # German district prefixes genuinely contain them (LÖ, GÖ, WÜ);
        # folding to ASCII would corrupt real plates.
        assert normalize("LÖ-AB 123") == "LÖAB123"


class TestIndiaFormat:
    @pytest.mark.parametrize(
        "plate",
        ["MH12AB1234", "DL01C0001", "KA05MN9999", "TN10A1234", "UP32ABC1234"],
    )
    def test_accepts_valid_standard_plates(self, plate):
        match = INDIA.match(plate)
        assert match is not None, plate
        assert match.region is Region.INDIA

    def test_series_letters_are_optional(self):
        # Early registrations in a district have no series letters.
        match = INDIA.match("MH121234")
        assert match is not None
        assert match.components["series"] == ""

    def test_accepts_the_bh_series(self):
        match = INDIA.match("21BH1234AB")
        assert match is not None
        assert match.format_name == "IN-BH"
        assert match.components["year"] == "21"

    def test_unknown_state_code_lowers_confidence(self):
        # The list is closed in practice, so an unlisted code is far more
        # likely a misread than a new state.
        known = INDIA.match("MH12AB1234")
        unknown = INDIA.match("XZ12AB1234")
        assert unknown is not None
        assert unknown.confidence < known.confidence

    def test_resolves_the_state_name(self):
        assert INDIA.match("MH12AB1234").components["state_name"] == "Maharashtra"

    @pytest.mark.parametrize(
        "plate",
        [
            "M12AB1234",  # one-letter state code
            "MH12AB12345",  # five trailing digits
            "MHXXAB1234",  # district must be digits
            "1234567890",
            "",
        ],
    )
    def test_rejects_malformed(self, plate):
        assert INDIA.match(plate) is None

    def test_display_format(self):
        assert format_display(INDIA.match("MH12AB1234")) == "MH 12 AB 1234"
        assert format_display(INDIA.match("21BH1234AB")) == "21 BH 1234 AB"

    def test_state_codes_are_two_uppercase_letters(self):
        assert all(len(c) == 2 and c.isupper() for c in STATE_CODES)


class TestGermanyFormat:
    @pytest.mark.parametrize(
        "plate",
        ["DAXY123", "BAB1", "MA1234", "HHAB12", "KAB123", "FXY99"],
    )
    def test_accepts_valid_plates(self, plate):
        match = GERMANY.match(plate)
        assert match is not None, plate
        assert match.region is Region.GERMANY

    def test_accepts_electric_and_historic_suffixes(self):
        electric = GERMANY.match("DAXY123E")
        historic = GERMANY.match("DAXY123H")
        assert electric.components["suffix_meaning"] == "electric"
        assert historic.components["suffix_meaning"] == "historic"

    def test_enforces_the_eight_character_cap(self):
        # Three-letter district + two letters leaves room for three digits,
        # not four. This cap is what gives the grammar its discriminating
        # power during correction.
        assert GERMANY.match("HROAB123") is not None  # 8 chars
        assert GERMANY.match("HROAB1234") is None  # 9

    def test_unknown_district_is_accepted_with_lower_confidence(self):
        # ~700 prefixes exist and the set changes; rejecting unlisted ones
        # would make whole districts unreadable.
        unknown = GERMANY.match("XQZAB123")
        assert unknown is not None
        assert unknown.confidence < GERMANY.match("DAXY123").confidence

    def test_resolves_the_district_name(self):
        assert GERMANY.match("DAXY123").components["district_name"] == "Darmstadt"

    def test_accepts_umlaut_districts(self):
        assert GERMANY.match("LÖAB123") is not None

    @pytest.mark.parametrize(
        "plate",
        [
            "DAXY",  # no digits
            "DAXY12345",  # five digits
            "1234",  # no letters
            "ABC",  # no digits, and under the length floor
            "",
        ],
    )
    def test_rejects_malformed(self, plate):
        assert GERMANY.match(plate) is None

    def test_da123_is_a_dusseldorf_plate_not_a_darmstadt_one(self):
        # "DA123" is genuinely valid — as D-A 123 (Düsseldorf), not DA-? 123.
        # The letter group is mandatory, so DA cannot be the district here.
        match = GERMANY.match("DA123")
        assert match is not None
        assert match.components["district"] == "D"
        assert match.components["letters"] == "A"

    def test_prefers_a_known_prefix_over_a_greedy_split(self):
        # Greedy matching splits DAXY123 as DAX|Y and loses Darmstadt.
        match = GERMANY.match("DAXY123")
        assert match.components["district"] == "DA"
        assert match.components["letters"] == "XY"

    def test_rejects_bodies_below_the_length_floor(self):
        # B-A 1 is legal but vanishingly rare, while 3-character OCR noise is
        # common — and correction could otherwise manufacture a plate from it.
        assert GERMANY.match("BA1") is None

    def test_display_format(self):
        assert format_display(GERMANY.match("DAXY123")) == "DA-XY 123"

    def test_district_prefixes_are_plausible(self):
        assert all(1 <= len(p) <= 3 and p.isupper() for p in DISTRICT_PREFIXES)


class TestAlternatives:
    def test_digits_offer_letters_and_vice_versa(self):
        assert "O" in alternatives("0")
        assert "0" in alternatives("O")
        assert "B" in alternatives("8")
        assert "8" in alternatives("B")

    def test_unambiguous_characters_have_none(self):
        # Correction must not invent options for characters OCR rarely confuses.
        assert alternatives("X") == ""
        assert alternatives("9") == ""


class TestCorrection:
    def test_repairs_a_letter_where_a_digit_belongs(self):
        # A letter cannot sit where the district digits belong, so the I
        # after the state code is a 1.
        match = correct_to_format("MHI2AB1234", INDIA)
        assert match is not None
        assert match.text == "MH12AB1234"
        assert match.edits == 1

    def test_short_numbers_are_real_plates_not_errors(self):
        # Measured on hand-labelled data: KL54H369, TN58AM1 and KL7BZ99 are
        # ordinary plates. Demanding four trailing digits rejected all three.
        for plate in ("KL54H369", "TN58AM1", "KL7BZ99"):
            assert INDIA.match(plate) is not None, plate

    def test_looser_numbers_cost_some_correction_power(self):
        # The tradeoff, recorded rather than hidden: MH12ABB234 is now a valid
        # reading (series ABB, number 234), so the grammar no longer repairs
        # the B as an 8. End-to-end this was still net positive — +3 plates
        # recovered against ~2 points of precision — but it is a real loss.
        match = correct_to_format("MH12ABB234", INDIA)
        assert match is not None
        assert match.edits == 0
        assert match.components["series"] == "ABB"

    def test_repairs_a_digit_where_a_letter_belongs(self):
        # The state code must be letters, so 0 is O and 5 is S.
        match = correct_to_format("0R12AB1234", INDIA)
        assert match is not None
        assert match.text == "OR12AB1234"

    def test_repairs_a_german_plate(self):
        match = correct_to_format("DAXY1Z3", GERMANY)
        assert match is not None
        assert match.text == "DAXY123"
        assert match.edits == 1

    def test_a_known_district_can_beat_an_exact_unknown_one(self):
        # "DAXYI23" parses as-is into the unknown district DAX (0.80). But
        # repairing I→1 gives Darmstadt's DA-XY 123, worth 1.00 − 0.18 = 0.82.
        # The likelier reading wins even though it costs an edit.
        match = correct_to_format("DAXYI23", GERMANY)
        assert match is not None
        assert match.text == "DAXY123"
        assert match.components["district"] == "DA"
        assert match.edits == 1

    def test_separator_disambiguates_munich_from_mannheim(self):
        # "MAB123E" fits both M-AB (München) and MA-B (Mannheim), and both
        # prefixes are real, so the string alone cannot settle it. The hyphen
        # OCR actually read does — and normalization is about to discard it.
        munich = parse("M-AB 123E")
        assert munich.components["district"] == "M"
        assert munich.components["district_name"] == "München"

        mannheim = parse("MA-B 123E")
        assert mannheim.components["district"] == "MA"
        assert mannheim.components["district_name"] == "Mannheim"

    def test_without_a_separator_the_longer_known_prefix_wins(self):
        # No hint available; fall back to the tie-break. Still a real plate,
        # just the more specific reading.
        assert parse("MAB123E").components["district"] == "MA"

    def test_hint_is_advisory_not_binding(self):
        # A hint that cannot produce a valid split must not veto the match.
        match = parse("XYZ-AB 123")
        assert match is not None
        assert match.region is Region.GERMANY

    def test_a_confident_exact_match_is_never_second_guessed(self):
        match = correct_to_format("DAXY123", GERMANY)
        assert match.edits == 0
        assert match.confidence == 1.0

    def test_clean_plates_are_never_altered(self):
        # The critical property. Blind confusable substitution damages every
        # plate that legitimately contains 0, 1, 5 or 8; grammar-constrained
        # correction must leave a valid read exactly as it is.
        for plate in ("MH12AB1234", "DL01C0001", "MH05BS8850", "KA51OD1058"):
            match = correct_to_format(plate, INDIA)
            assert match is not None, plate
            assert match.text == plate
            assert match.edits == 0

    def test_prefers_fewer_edits(self):
        match = correct_to_format("MH12AB1Z34", INDIA)
        assert match is not None
        assert match.edits == 1
        assert match.text == "MH12AB1234"

    def test_confidence_falls_with_each_edit(self):
        clean = correct_to_format("MH12AB1234", INDIA)
        repaired = correct_to_format("MHI2AB1234", INDIA)
        assert repaired.confidence < clean.confidence

    def test_gives_up_beyond_max_edits(self):
        # A string needing many reinterpretations is noise, and "matching" it
        # says more about the search than about the plate.
        assert correct_to_format("OOOOOOOOOO", INDIA, max_edits=1) is None

    def test_returns_none_for_hopeless_input(self):
        assert correct_to_format("XXXX", INDIA) is None
        assert correct_to_format("", INDIA) is None

    def test_records_the_original_reading(self):
        match = correct_to_format("MHI2AB1234", INDIA)
        assert match.original == "MHI2AB1234"
        assert match.corrected is True


class TestParse:
    def test_parses_an_indian_plate_end_to_end(self):
        match = parse("mh 12 ab 1234")
        assert match is not None
        assert match.text == "MH12AB1234"
        assert match.region is Region.INDIA

    def test_parses_a_german_plate_end_to_end(self):
        match = parse("DA-XY 123")
        assert match is not None
        assert match.text == "DAXY123"
        assert match.region is Region.GERMANY

    def test_region_hint_restricts_the_grammar(self):
        # A deployment on a German road should never return an Indian reading.
        match = parse("DAXY123", region=Region.GERMANY)
        assert match.region is Region.GERMANY
        assert parse("MH12AB1234", region=Region.GERMANY) is None

    def test_returns_none_for_junk(self):
        for junk in ("", "!!!", "ZZZ", "hello world"):
            assert parse(junk) is None

    def test_rejects_noise_corrected_into_a_plausible_plate(self):
        # "hello" uppercases to HELLO, and O->0 turns it into HEL-L 0 — a
        # structurally valid German plate. It scores 0.62 (unknown district,
        # one edit), below the floor. A false plate in the log is worse than
        # a missed one.
        assert parse("hello") is None
        assert parse("hello", min_confidence=0.0) is not None

    def test_confidence_is_rounded_for_display(self):
        # The value goes into a spreadsheet cell; 0.8200000000000001 is not
        # something to show a user.
        match = parse("MH12A81234")
        assert match.confidence == 0.82

    def test_a_corrected_known_code_clears_the_floor(self):
        # The floor must not reject genuine repairs — only noise.
        assert parse("MH12A81234") is not None  # 0.82
        assert parse("DAXY1Z3") is not None  # 0.82

    def test_correction_and_normalization_compose(self):
        # Separators stripped, then the 8-shaped B and Z-shaped 2 repaired.
        match = parse("MH 12 AB 8Z34")
        assert match is not None
        assert match.text == "MH12AB8234"

    @pytest.mark.parametrize(
        ("raw", "expected", "region"),
        [
            ("MH12AB1234", "MH12AB1234", Region.INDIA),
            ("DL 01 CAB 1234", "DL01CAB1234", Region.INDIA),
            ("21 BH 1234 AB", "21BH1234AB", Region.INDIA),
            ("DA-XY 1234", "DAXY1234", Region.GERMANY),
            ("M-AB 123E", "MAB123E", Region.GERMANY),
            ("HH-AB 12H", "HHAB12H", Region.GERMANY),
        ],
    )
    def test_real_world_shapes(self, raw, expected, region):
        match = parse(raw)
        assert match is not None, raw
        assert match.text == expected
        assert match.region is region

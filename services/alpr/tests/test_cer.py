"""Character error rate and the ablation harness."""

from __future__ import annotations

import pytest

from alpr.cer import ablate, cer, compare, grammar_gain, levenshtein, score


class TestLevenshtein:
    def test_identical(self):
        assert levenshtein("MH12AB1234", "MH12AB1234") == 0

    def test_substitution(self):
        assert levenshtein("MH12AB1234", "MH12A81234") == 1

    def test_insertion_and_deletion(self):
        assert levenshtein("ABC", "ABBC") == 1
        assert levenshtein("ABBC", "ABC") == 1

    def test_empty_strings(self):
        assert levenshtein("", "") == 0
        assert levenshtein("", "ABC") == 3
        assert levenshtein("ABC", "") == 3

    def test_is_symmetric(self):
        assert levenshtein("KA05MN", "KA05NM") == levenshtein("KA05NM", "KA05MN")


class TestCer:
    def test_perfect_read(self):
        assert cer("MH12AB1234", "MH12AB1234") == 0.0

    def test_one_wrong_character_in_ten(self):
        assert cer("MH12AB1234", "MH12A81234") == pytest.approx(0.1)

    def test_empty_hypothesis_is_total_failure(self):
        assert cer("MH12AB1234", "") == 1.0

    def test_invented_plate_where_there_was_none(self):
        # Not a division by zero: the reader hallucinated a plate.
        assert cer("", "ABC") == 1.0
        assert cer("", "") == 0.0


class TestScore:
    def test_aggregates_over_the_corpus(self):
        # Corpus CER, not the mean of per-sample CERs: one badly-read short
        # plate must not dominate the score.
        report = score([("MH12AB1234", "MH12A81234"), ("DAXY123", "DAXY123")])
        assert report.total_reference_chars == 17
        assert report.total_distance == 1
        assert report.cer == pytest.approx(1 / 17)

    def test_exact_match_accuracy(self):
        # The metric a plate log actually needs: one wrong character makes a
        # wrong plate, not a 90%-right one.
        report = score([("ABC123", "ABC123"), ("ABC123", "ABC124")])
        assert report.accuracy == 0.5
        assert report.cer < 0.5

    def test_counts_empty_reads(self):
        assert score([("ABC123", ""), ("ABC123", "ABC123")]).empty_reads == 1

    def test_keeps_the_worst_samples(self):
        report = score(
            [("ABC123", "ABC123"), ("ABC123", "XYZ999"), ("ABC123", "ABC124")],
            keep_worst=2,
        )
        assert len(report.worst) == 2
        assert report.worst[0][1] == "XYZ999"

    def test_perfect_run_has_no_worst_list(self):
        assert score([("ABC123", "ABC123")]).worst == []

    def test_empty_input(self):
        report = score([])
        assert report.cer == 0.0
        assert report.accuracy == 0.0


class TestCompare:
    def test_orders_by_cer(self):
        good = score([("ABC123", "ABC123")], name="good")
        bad = score([("ABC123", "XYZ999")], name="bad")
        table = compare([bad, good])
        assert table.index("good") < table.index("bad")


class TestAblate:
    def test_scores_each_variant_over_the_same_samples(self):
        samples = [("a.png", "ABC123"), ("b.png", "DAXY123")]

        def read(path, variant):
            # "good" reads correctly; "bad" drops a character.
            truth = dict(samples)[path]
            return truth if variant == "good" else truth[:-1]

        reports = ablate(samples, read, {"good": "good", "bad": "bad"})
        by_name = {r.name: r for r in reports}
        assert by_name["good"].cer == 0.0
        assert by_name["bad"].cer > 0.0
        assert by_name["good"].samples == 2


class TestGrammarGain:
    def test_reports_an_improvement(self):
        raw = [("MH12AB1234", "MH12A81234")]
        corrected = [("MH12AB1234", "MH12AB1234")]
        text = grammar_gain(raw, corrected)
        assert "improved" in text
        assert "not earning its place" not in text

    def test_flags_a_grammar_that_hurts(self):
        # Entirely possible: a grammar that "corrects" a correct read has made
        # things worse, and the comparison must say so rather than flatter it.
        raw = [("MH12AB1234", "MH12AB1234")]
        corrected = [("MH12AB1234", "MH12AB1284")]
        text = grammar_gain(raw, corrected)
        assert "not earning its place" in text

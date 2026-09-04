"""Multi-frame voting."""

from __future__ import annotations

import pytest

from alpr.vote import Read, TrackVoter, vote


def reads(*items) -> list[Read]:
    """(text, confidence) pairs, or bare strings at confidence 1.0."""
    out = []
    for item in items:
        if isinstance(item, tuple):
            out.append(Read(item[0], item[1]))
        else:
            out.append(Read(item))
    return out


class TestRead:
    def test_rejects_impossible_confidence(self):
        with pytest.raises(ValueError):
            Read("ABC", 1.5)


class TestVote:
    def test_unanimous_reads(self):
        result = vote(reads("MH12AB1234", "MH12AB1234", "MH12AB1234"))
        assert result.text == "MH12AB1234"
        assert result.unanimous
        assert result.reads == 3

    def test_recovers_a_plate_no_frame_read_correctly(self):
        # The whole reason voting is per character. Every read is wrong, no
        # string has a majority, and each error appears in only one frame.
        result = vote(
            reads(
                "MH12A81234",  # B -> 8
                "MH12AB1Z34",  # 2 -> Z
                "NH12AB1234",  # M -> N
                "MH12AB1234",  # correct
            )
        )
        assert result.text == "MH12AB1234"
        assert not result.unanimous

    def test_majority_wins_per_position(self):
        result = vote(reads("ABC123", "ABC123", "ABC124"))
        assert result.text == "ABC123"

    def test_confidence_outweighs_count(self):
        # A crisp close-up should beat two blurred distant reads, not merely
        # be outnumbered by them.
        result = vote(reads(("ABC999", 0.2), ("ABC999", 0.2), ("ABC123", 0.95)))
        assert result.text == "ABC123"

    def test_length_chosen_by_confidence_weight(self):
        # Voting per position across mixed lengths would smear characters into
        # the wrong slots, so length is settled first.
        result = vote(reads(("ABC12", 0.3), ("ABC123", 0.9), ("ABC123", 0.9)))
        assert result.text == "ABC123"

    def test_reads_of_other_lengths_are_excluded(self):
        result = vote(reads("ABC123", "ABC123", "AB12"))
        assert result.reads == 2

    def test_agreement_reflects_disagreement(self):
        perfect = vote(reads("ABC123", "ABC123"))
        split = vote(reads("ABC123", "ABC124"))
        assert perfect.agreement == 1.0
        assert split.agreement < 1.0

    def test_confidence_combines_agreement_and_read_quality(self):
        # Unanimity among uncertain reads is weaker evidence than unanimity
        # among confident ones.
        confident = vote(reads(("ABC123", 0.9), ("ABC123", 0.9)))
        unsure = vote(reads(("ABC123", 0.3), ("ABC123", 0.3)))
        assert confident.confidence > unsure.confidence

    def test_identifies_the_weakest_position(self):
        # ABC123 vs ABC183 differ at index 4 (the "2"/"8"), not index 3.
        result = vote(reads("ABC123", "ABC123", "ABC183"))
        assert result.weakest_position() == 4
        assert result.per_character[4] < 1.0

    def test_unanimous_has_no_weakest_position(self):
        assert vote(reads("ABC123", "ABC123")).weakest_position() is None

    def test_reports_a_runner_up(self):
        result = vote(reads("ABC123", "ABC123", "ABC124"))
        assert result.runner_up == "ABC124"

    def test_no_runner_up_when_unanimous(self):
        assert vote(reads("ABC123", "ABC123")).runner_up is None

    def test_ties_break_deterministically(self):
        # A nondeterministic plate would be worse than a wrong one.
        first = vote(reads("ABC123", "ABD123"))
        second = vote(reads("ABD123", "ABC123"))
        assert first.text == second.text

    def test_too_few_reads_returns_none(self):
        assert vote(reads("ABC123")) is None
        assert vote([]) is None

    def test_min_reads_is_configurable(self):
        assert vote(reads("ABC123"), min_reads=1) is not None

    def test_empty_texts_are_ignored(self):
        result = vote([Read(""), Read("ABC123"), Read("ABC123")])
        assert result.text == "ABC123"
        assert result.reads == 2

    def test_per_character_agreement_is_reported(self):
        result = vote(reads("ABC123", "ABC124"))
        assert len(result.per_character) == 6
        assert result.per_character[0] == 1.0  # all agree
        assert result.per_character[5] < 1.0  # they differ here


class TestTrackVoter:
    def test_accumulates_per_track(self):
        voter = TrackVoter()
        voter.add(1, Read("MH12AB1234"))
        voter.add(1, Read("MH12AB1234"))
        voter.add(2, Read("DAXY123"))
        assert voter.tracked == 2
        assert voter.result(1).text == "MH12AB1234"

    def test_tracks_do_not_mix(self):
        voter = TrackVoter()
        for _ in range(3):
            voter.add(1, Read("AAA111"))
            voter.add(2, Read("BBB222"))
        assert voter.result(1).text == "AAA111"
        assert voter.result(2).text == "BBB222"

    def test_pop_emits_once(self):
        # One row per vehicle: a second pop must not re-emit the same track.
        voter = TrackVoter()
        voter.add(1, Read("ABC123"))
        voter.add(1, Read("ABC123"))
        assert voter.pop(1).text == "ABC123"
        assert voter.pop(1) is None

    def test_unknown_track_is_none(self):
        assert TrackVoter().result(99) is None

    def test_empty_reads_are_not_buffered(self):
        voter = TrackVoter()
        voter.add(1, Read(""))
        assert voter.tracked == 0

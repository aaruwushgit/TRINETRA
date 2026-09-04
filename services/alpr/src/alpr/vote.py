"""Multi-frame voting over a track's OCR reads.

This is the largest accuracy mechanism in the pipeline that costs no model
quality at all — it is pure aggregation of information the pipeline already
has.

**Voting per character, not per string.** String-level majority is the obvious
approach and it throws away most of the signal. Consider one vehicle read
across four frames:

    MH12AB1234    correct
    MH12A81234    B misread as 8
    MH12AB1Z34    2 misread as Z
    NH12AB1234    M misread as N

No string has a majority — every read differs from every other. String voting
picks one arbitrarily and is right by luck. Per-character voting is right at
every position, because each error appears in only one frame and is outvoted
three to one. **The result is a plate no single frame read correctly.**

Reads are weighted by confidence, so a crisp close-up outvotes a blurred
distant one rather than merely outnumbering it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

# A track needs at least this many reads before its vote is trustworthy. One
# read is not a vote; two cannot break a tie.
DEFAULT_MIN_READS = 2


@dataclass(frozen=True)
class Read:
    """One frame's reading of a plate."""

    text: str
    confidence: float = 1.0
    frame: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class VoteResult:
    """The consensus reading of a track."""

    text: str
    confidence: float
    reads: int
    agreement: float
    per_character: tuple[float, ...] = ()
    runner_up: str | None = None

    @property
    def unanimous(self) -> bool:
        return self.agreement >= 1.0

    def weakest_position(self) -> int | None:
        """Index of the least-agreed character, or None if unanimous.

        Useful for review: a single low-agreement position usually means one
        genuinely ambiguous glyph rather than a bad read overall.
        """
        if not self.per_character or self.unanimous:
            return None
        return min(range(len(self.per_character)), key=lambda i: self.per_character[i])


def vote(
    reads: Sequence[Read],
    *,
    min_reads: int = DEFAULT_MIN_READS,
) -> VoteResult | None:
    """Combine a track's reads into one consensus plate.

    Args:
        reads: every reading of this track. Empty texts are ignored.
        min_reads: below this, return None rather than a fragile answer.

    Returns:
        The voted plate, or None when there is not enough to vote on.
    """
    usable = [r for r in reads if r.text]
    if len(usable) < min_reads:
        return None

    # Length is decided first, by confidence weight rather than by count: a
    # clear 10-character read should outweigh several blurred 9-character
    # ones. Voting per position across mixed lengths would otherwise smear
    # characters into the wrong slots.
    weight_by_length: dict[int, float] = defaultdict(float)
    for read in usable:
        weight_by_length[len(read.text)] += read.confidence
    target_length = max(weight_by_length, key=lambda n: (weight_by_length[n], n))

    candidates = [r for r in usable if len(r.text) == target_length]

    per_position_votes: list[dict[str, float]] = [defaultdict(float) for _ in range(target_length)]
    for read in candidates:
        for index, char in enumerate(read.text):
            per_position_votes[index][char] += read.confidence

    chars: list[str] = []
    agreements: list[float] = []
    for votes in per_position_votes:
        total = sum(votes.values())
        # Ties break alphabetically so the same input always votes the same
        # way — a nondeterministic plate would be worse than a wrong one.
        char = max(sorted(votes), key=lambda c: votes[c])
        chars.append(char)
        agreements.append(votes[char] / total if total else 0.0)

    text = "".join(chars)
    agreement = sum(agreements) / len(agreements) if agreements else 0.0
    mean_confidence = sum(r.confidence for r in candidates) / len(candidates)

    # Confidence combines how well the reads agreed with how confident they
    # were: unanimous agreement between four uncertain reads is not the same
    # evidence as unanimous agreement between four confident ones.
    confidence = round(agreement * mean_confidence, 3)

    runner_up = _runner_up(candidates, text)

    return VoteResult(
        text=text,
        confidence=confidence,
        reads=len(candidates),
        agreement=round(agreement, 3),
        per_character=tuple(round(a, 3) for a in agreements),
        runner_up=runner_up,
    )


def _runner_up(candidates: Sequence[Read], winner: str) -> str | None:
    """The most-supported reading that is not the winner, if any.

    Worth surfacing: when the runner-up differs by one character, the vote was
    close and the row is worth a human glance.
    """
    totals: dict[str, float] = defaultdict(float)
    for read in candidates:
        totals[read.text] += read.confidence
    others = {text: weight for text, weight in totals.items() if text != winner}
    if not others:
        return None
    return max(sorted(others), key=lambda t: others[t])


@dataclass
class TrackVoter:
    """Accumulates reads per track and votes when the track ends.

    The buffer is what makes one row per vehicle possible: reads arrive frame
    by frame, and the answer only exists once the vehicle has gone.
    """

    min_reads: int = DEFAULT_MIN_READS
    _reads: dict[int, list[Read]] = field(default_factory=lambda: defaultdict(list))

    def add(self, track_id: int, read: Read) -> None:
        if read.text:
            self._reads[track_id].append(read)

    def reads_for(self, track_id: int) -> list[Read]:
        return list(self._reads.get(track_id, []))

    def result(self, track_id: int) -> VoteResult | None:
        return vote(self._reads.get(track_id, []), min_reads=self.min_reads)

    def pop(self, track_id: int) -> VoteResult | None:
        """Vote and discard the buffer — a track is emitted once."""
        result = self.result(track_id)
        self._reads.pop(track_id, None)
        return result

    @property
    def tracked(self) -> int:
        return len(self._reads)

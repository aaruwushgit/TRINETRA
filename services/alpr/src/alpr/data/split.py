"""Deterministic, grouped, region-stratified train/val/test splitting.

Three properties, each protecting against a specific way the numbers can lie:

**Grouped** — frames sampled from one video are near-duplicates. A random
per-image split puts frame 100 in train and frame 101 in val, so the model is
evaluated on pictures it has effectively memorized and val accuracy reads far
higher than reality. Everything sharing a `group_key` goes to one split.

**Stratified by region** — with India and Germany in one dataset, an unlucky
split can leave test almost entirely Indian, and the German number then rests
on a handful of images. Each region is split to the target ratios separately.

**Deterministic** — assignment comes from a hash of the group key, not from
list order or a global RNG. Re-running with the same seed reproduces the split
exactly, even if the records arrive in a different order or new images are
appended. That is what makes a training run repeatable.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from alpr.data.schema import DatasetError, ImageRecord, Region, Split

DEFAULT_RATIOS: dict[Split, float] = {
    Split.TRAIN: 0.70,
    Split.VAL: 0.15,
    Split.TEST: 0.15,
}


def _group_rank(group_key: str, seed: int) -> int:
    """Stable pseudo-random rank for a group.

    blake2b rather than `hash()`: Python salts string hashing per process, so
    `hash()` would produce a different split on every run.
    """
    digest = hashlib.blake2b(f"{seed}:{group_key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


@dataclass(frozen=True)
class SplitAssignment:
    """Which split each group — and therefore each image — belongs to."""

    by_group: Mapping[str, Split]
    seed: int
    ratios: Mapping[Split, float]

    def of(self, record: ImageRecord) -> Split:
        try:
            return self.by_group[record.group_key]
        except KeyError:
            raise DatasetError(
                f"{record.image_id!r} has group {record.group_key!r}, which was not in the "
                "records this split was computed from"
            ) from None

    def partition(self, records: Iterable[ImageRecord]) -> dict[Split, list[ImageRecord]]:
        out: dict[Split, list[ImageRecord]] = {s: [] for s in Split}
        for record in records:
            out[self.of(record)].append(record)
        return out

    def counts(self, records: Iterable[ImageRecord]) -> dict[Split, int]:
        return {split: len(items) for split, items in self.partition(records).items()}


def _normalize_ratios(ratios: Mapping[Split, float]) -> dict[Split, float]:
    if not ratios:
        raise DatasetError("ratios must not be empty")
    for split, value in ratios.items():
        if value < 0:
            raise DatasetError(f"negative ratio for {split.value}: {value}")
    total = sum(ratios.values())
    if total <= 0:
        raise DatasetError("ratios must sum to more than zero")
    return {split: value / total for split, value in ratios.items()}


def split_records(
    records: Sequence[ImageRecord],
    ratios: Mapping[Split, float] | None = None,
    seed: int = 0,
) -> SplitAssignment:
    """Assign every record to a split, grouped and stratified by region.

    Groups are placed largest-first, each going to whichever split is furthest
    below its target image count. Largest-first matters: placing a 400-frame
    group after the small ones can overshoot a split by more than the whole
    val set is meant to hold.

    Args:
        records: the dataset. May be empty.
        ratios: target proportions; normalized if they do not sum to 1.
        seed: changes the assignment; the same seed always reproduces it.

    Returns:
        A `SplitAssignment` covering every group present in `records`.
    """
    resolved = _normalize_ratios(dict(ratios) if ratios else DEFAULT_RATIOS)

    # Group by region first so each region is split to the target ratios
    # independently, then by group key within it.
    by_region: dict[Region, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        by_region[record.primary_region][record.group_key] += 1

    # One group can hold images of several regions (its primary region is per
    # image). Whichever region claims it first owns it, so a group is never
    # split across regions — and therefore never across splits.
    claimed: set[str] = set()
    assignment: dict[str, Split] = {}

    for region in sorted(by_region, key=lambda r: r.value):
        groups = by_region[region]
        pending = [(g, n) for g, n in groups.items() if g not in claimed]
        if not pending:
            continue
        claimed.update(g for g, _ in pending)

        total = sum(n for _, n in pending)
        targets = {split: share * total for split, share in resolved.items()}
        placed: dict[Split, int] = {split: 0 for split in resolved}

        # Largest first, hash as the deterministic tiebreak among equal sizes.
        pending.sort(key=lambda item: (-item[1], _group_rank(item[0], seed)))

        for group_key, size in pending:
            # Deficit, not ratio: an empty split with a large target should win
            # over one that is merely slightly under.
            best = max(
                resolved,
                key=lambda s: (
                    targets[s] - placed[s],
                    -_group_rank(f"{group_key}:{s.value}", seed),
                ),
            )
            assignment[group_key] = best
            placed[best] += size

    return SplitAssignment(by_group=assignment, seed=seed, ratios=resolved)


def verify_split(
    records: Sequence[ImageRecord],
    assignment: SplitAssignment,
    *,
    require_all_splits: bool = True,
    require_exact_coverage: bool = True,
) -> None:
    """Raise if the split is unsound for these records.

    Grouping itself needs no checking — `by_group` maps each group to exactly
    one split, so a group cannot span splits by construction. What *can* go
    wrong is the assignment not matching the dataset: reusing a split saved
    before more images were added silently drops or misplaces the new ones.
    That is what this checks, and it runs in CI rather than in a notebook.

    Args:
        records: the dataset the assignment should cover.
        assignment: the split to check.
        require_all_splits: require every non-zero ratio to receive an image.
            Disable for tiny fixtures.
        require_exact_coverage: require the assignment to describe exactly the
            groups present — no unassigned records, no stale leftovers.

    Raises:
        DatasetError: on an unassigned record, a stale assignment, or an empty
            split when one was required.
    """
    record_groups = {record.group_key for record in records}

    if require_exact_coverage:
        unassigned = sorted(record_groups - set(assignment.by_group))
        if unassigned:
            raise DatasetError(
                f"{len(unassigned)} group(s) have no split assignment "
                f"(e.g. {unassigned[:3]}) — the dataset grew after this split was computed; "
                "recompute it rather than training on a partial set"
            )
        stale = sorted(set(assignment.by_group) - record_groups)
        if stale:
            raise DatasetError(
                f"{len(stale)} assigned group(s) are absent from the dataset "
                f"(e.g. {stale[:3]}) — this split belongs to a different dataset version"
            )
    else:
        for record in records:
            assignment.of(record)  # raises on an unassigned group

    if require_all_splits and records:
        counts = assignment.counts(records)
        empty = [s.value for s, share in assignment.ratios.items() if share > 0 and counts[s] == 0]
        if empty:
            raise DatasetError(
                f"split(s) {', '.join(sorted(empty))} received no images — "
                f"{len(record_groups)} group(s) cannot fill {len(assignment.ratios)} splits"
            )

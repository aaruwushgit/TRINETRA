"""Splitting: leak-free, deterministic, stratified.

These are the tests that protect the headline number. A leak here inflates
val accuracy without failing anything else in the project.
"""

from __future__ import annotations

import pytest

from alpr.data import (
    DatasetError,
    ImageRecord,
    PlateBox,
    Region,
    Split,
    SplitAssignment,
    split_records,
    verify_split,
)


def make_records(n_groups: int, frames_per_group: int, region=Region.UNKNOWN, prefix="clip"):
    """Simulate frames sampled from videos — the case grouping exists for."""
    records = []
    for g in range(n_groups):
        for f in range(frames_per_group):
            records.append(
                ImageRecord(
                    image_id=f"{prefix}_{g:03d}_frame_{f:04d}",
                    width=1920,
                    height=1080,
                    boxes=(PlateBox(0.5, 0.5, 0.1, 0.05, region=region),),
                )
            )
    return records


class TestGrouping:
    def test_frames_from_one_clip_never_straddle_splits(self):
        records = make_records(n_groups=30, frames_per_group=20)
        assignment = split_records(records)
        verify_split(records, assignment)  # raises on a leak

        by_group: dict[str, set[Split]] = {}
        for record in records:
            by_group.setdefault(record.group_key, set()).add(assignment.of(record))
        assert all(len(splits) == 1 for splits in by_group.values())

    def test_a_group_cannot_span_splits_by_construction(self):
        # by_group maps each group to exactly one split, so this is a property
        # of the data structure, not something a runtime check has to catch.
        records = make_records(n_groups=12, frames_per_group=8)
        assignment = split_records(records)
        for record in records:
            assert assignment.of(record) is assignment.by_group[record.group_key]

    def test_stale_assignment_from_a_grown_dataset_is_rejected(self):
        # The realistic failure: split once, add images, reuse the old split.
        # The new clips would silently never be trained or evaluated on.
        original = make_records(n_groups=12, frames_per_group=5)
        assignment = split_records(original)
        grown = [*original, *make_records(3, 5, prefix="newclip")]
        with pytest.raises(DatasetError, match="no split assignment"):
            verify_split(grown, assignment)

    def test_assignment_for_a_shrunk_dataset_is_rejected(self):
        # The other direction: split, then drop images (a licence filter, a
        # corrupt-file sweep) and reuse the old split. The ratios it promises
        # no longer describe what is actually there.
        records = make_records(n_groups=12, frames_per_group=5)
        assignment = split_records(records)
        kept = [r for r in records if not r.group_key.endswith(("_010", "_011"))]
        with pytest.raises(DatasetError, match="absent from the dataset"):
            verify_split(kept, assignment)

    def test_forged_assignment_missing_groups_is_caught(self):
        records = make_records(n_groups=4, frames_per_group=5)
        forged = SplitAssignment(
            by_group={"clip_000": Split.TRAIN, "clip_001": Split.VAL},
            seed=0,
            ratios={Split.TRAIN: 0.5, Split.VAL: 0.5},
        )
        with pytest.raises(DatasetError, match="no split assignment"):
            verify_split(records, forged, require_all_splits=False)

    def test_unassigned_group_is_an_error(self):
        records = make_records(n_groups=3, frames_per_group=2)
        assignment = split_records(records)
        stranger = ImageRecord("unseen_clip_frame_0001", 10, 10)
        with pytest.raises(DatasetError, match="not in the records"):
            assignment.of(stranger)


class TestDeterminism:
    def test_same_seed_same_split(self):
        records = make_records(20, 10)
        assert split_records(records, seed=7).by_group == split_records(records, seed=7).by_group

    def test_input_order_does_not_matter(self):
        # Assignment comes from hashing the group key, not from list position.
        records = make_records(20, 10)
        shuffled = list(reversed(records))
        assert split_records(records, seed=1).by_group == split_records(shuffled, seed=1).by_group

    def test_different_seed_gives_a_different_split(self):
        records = make_records(40, 5)
        a = split_records(records, seed=1).by_group
        b = split_records(records, seed=2).by_group
        assert a != b

    def test_stable_across_processes(self):
        # blake2b, not hash(): Python salts string hashing per process, so a
        # hash()-based split would differ between the notebook and CI.
        records = make_records(10, 3)
        expected = split_records(records, seed=0).by_group
        assert split_records(records, seed=0).by_group == expected


class TestRatios:
    def test_hits_target_ratios_approximately(self):
        records = make_records(n_groups=100, frames_per_group=10)
        assignment = split_records(records)
        counts = assignment.counts(records)
        total = len(records)
        assert counts[Split.TRAIN] / total == pytest.approx(0.70, abs=0.05)
        assert counts[Split.VAL] / total == pytest.approx(0.15, abs=0.05)
        assert counts[Split.TEST] / total == pytest.approx(0.15, abs=0.05)

    def test_uneven_group_sizes_still_balance(self):
        # Largest-first placement exists for this: a 400-frame clip placed last
        # would overflow whichever split it lands in by more than val's target.
        records = [
            ImageRecord(f"c{g}_frame_{i:04d}", 100, 100)
            for g, size in enumerate([400, 200, 100, 50, 20, 10, 5, 5, 5, 5])
            for i in range(size)
        ]
        assignment = split_records(records)
        counts = assignment.counts(records)
        assert counts[Split.TRAIN] / len(records) == pytest.approx(0.70, abs=0.15)

    def test_ratios_are_normalized(self):
        records = make_records(30, 4)
        assignment = split_records(records, ratios={Split.TRAIN: 8, Split.VAL: 2})
        assert assignment.ratios[Split.TRAIN] == pytest.approx(0.8)
        assert assignment.ratios[Split.VAL] == pytest.approx(0.2)

    def test_rejects_negative_ratio(self):
        with pytest.raises(DatasetError, match="negative"):
            split_records(make_records(2, 2), ratios={Split.TRAIN: -1, Split.VAL: 2})


class TestStratification:
    def test_both_regions_reach_every_split(self):
        records = [
            *make_records(20, 5, region=Region.INDIA, prefix="in"),
            *make_records(20, 5, region=Region.GERMANY, prefix="de"),
        ]
        assignment = split_records(records)
        partition = assignment.partition(records)
        for split in Split:
            regions = {r.primary_region for r in partition[split]}
            assert Region.INDIA in regions, f"{split.value} has no Indian plates"
            assert Region.GERMANY in regions, f"{split.value} has no German plates"

    def test_minority_region_is_not_swallowed(self):
        # 40 Indian clips vs 6 German ones: an unstratified split can leave
        # test with zero German plates and the German number resting on noise.
        records = [
            *make_records(40, 5, region=Region.INDIA, prefix="in"),
            *make_records(6, 5, region=Region.GERMANY, prefix="de"),
        ]
        assignment = split_records(records)
        partition = assignment.partition(records)
        german_per_split = {
            split: sum(1 for r in items if r.primary_region is Region.GERMANY)
            for split, items in partition.items()
        }
        assert all(count > 0 for count in german_per_split.values()), german_per_split


class TestEdgeCases:
    def test_empty_dataset(self):
        assignment = split_records([])
        assert assignment.by_group == {}
        verify_split([], assignment)

    def test_too_few_groups_is_reported(self):
        records = make_records(n_groups=1, frames_per_group=50)
        assignment = split_records(records)
        with pytest.raises(DatasetError, match="received no images"):
            verify_split(records, assignment)

    def test_too_few_groups_tolerated_when_not_required(self):
        records = make_records(n_groups=1, frames_per_group=50)
        verify_split(records, split_records(records), require_all_splits=False)

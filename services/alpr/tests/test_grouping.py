"""Clip-level split integrity: the invariant, and adversarial pressure on it.

The other test modules check the pieces. This one checks the property those
pieces exist to deliver:

    two frames of one recognised clip can never land in different splits

That is a guarantee by construction rather than a statistic, and the proof is a
chain of four steps. Each step is asserted below by the test named after it, so
that if someone breaks a link the failure names which one.

    1. clip identity is a pure function of the filename, and the upstream
       Roboflow directory is not part of it        -> TestInvariant.test_step1_*
    2. ingest therefore gives every frame of a clip the same `group`
                                                   -> TestInvariant.test_step2_*
    3. `regroup_by_duplicates` only ever merges groups, never splits one
                                                   -> TestInvariant.test_step3_*
    4. `split_records` maps a `group_key` to exactly one split
                                                   -> TestInvariant.test_step4_*

Steps 1+2 put a clip in one group; step 3 says nothing downstream can pull it
apart; step 4 says one group means one split. Together: one clip, one split.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image, ImageEnhance

from alpr.build import group_duplicates
from alpr.data import (
    ImageRecord,
    Split,
    clip_id,
    from_roboflow_export,
    split_records,
    verify_split,
)
from alpr.dupes import DuplicatePair, DuplicateReport, regroup_by_duplicates

# Real filenames from this project's surviving identifiers (labels/index.json).
DAYRIDE = [f"dayride_type1_001-mp4-t-{n}" for n in (451, 491, 751, 1115, 1208)]
NIGHTRIDE = [f"nightride_type3_001-mp4-t-{n}" for n in (256, 540, 585)]
VIDEO3 = [f"video3_{n}" for n in (640, 1220, 2190)]
VIDEO11 = [f"video11_{n}" for n in (1870, 1900)]
STILLS = [f"pl_license_plate_{n}" for n in (205, 242, 327, 486, 188)]

_HEX = "0123456789abcdef"


def rf(stem: str, *, ext: str = "jpg", seed: int | None = None) -> str:
    """Attach a realistic Roboflow export suffix to a stem."""
    rng = random.Random(seed if seed is not None else stem)
    return f"{stem}_{ext}.rf." + "".join(rng.choice(_HEX) for _ in range(32))


def picture(path, key: int, *, bright: float = 1.0) -> None:
    """An image whose perceptual hash is determined by `key`."""
    rng = random.Random(key)
    small = Image.new("L", (9, 8))
    small.putdata([rng.choice((0, 255)) for _ in range(72)])
    out = small.convert("RGB").resize((162, 120), Image.Resampling.NEAREST)
    if bright != 1.0:
        out = ImageEnhance.Brightness(out).enhance(bright)
    out.save(path)


def export(tmp_path, layout, name="export"):
    """Build a Roboflow-shaped export.

    `layout` is `{upstream_dir: [(stem, picture_key), ...]}`.
    """
    location = tmp_path / name
    for split, entries in layout.items():
        images = location / split / "images"
        labels = location / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for stem, key in entries:
            picture(images / f"{stem}.jpg", key)
            (labels / f"{stem}.txt").write_text("0 0.5 0.6 0.2 0.08\n")
    return location


def ingest(location, *, prefix="eu-"):
    return from_roboflow_export(location, source="s", id_prefix=prefix).records


def placement(records, assignment):
    return {r.image_id: assignment.of(r) for r in records}


def clips_of(records):
    """`{clip identity: [image_id, ...]}` computed independently of the code
    under test, from the raw stem, so a bug in grouping cannot hide itself."""
    out: dict[str, list[str]] = {}
    for record in records:
        stem = record.image_id.split("-", 2)[-1]
        clip = clip_id(stem)
        if clip:
            out.setdefault(clip, []).append(record.image_id)
    return out


def assert_no_clip_leaks(records, assignment):
    """No recognised clip may occupy more than one split."""
    where = placement(records, assignment)
    for clip, ids in clips_of(records).items():
        splits = {where[i] for i in ids}
        assert len(splits) == 1, f"clip {clip!r} split across {sorted(s.value for s in splits)}"


class TestInvariant:
    """The four links in the chain, asserted individually."""

    def test_step1_clip_identity_ignores_the_upstream_directory(self):
        # clip_id sees only a stem. The directory is not an input, so it cannot
        # be an output.
        for stem in DAYRIDE:
            assert clip_id(rf(stem)) == "dayride_type1_001"
        for stem in VIDEO3:
            assert clip_id(rf(stem)) == "video3"

    def test_step2_ingest_gives_one_clip_one_group(self, tmp_path):
        location = export(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 1), (rf(DAYRIDE[1]), 2)],
                "valid": [(rf(DAYRIDE[2]), 3)],
                "test": [(rf(DAYRIDE[3]), 4), (rf(DAYRIDE[4]), 5)],
            },
        )
        records = ingest(location)
        assert len({r.group for r in records}) == 1
        assert records[0].group == "eu-dayride_type1_001"

    def test_step3_regrouping_only_ever_merges(self, tmp_path):
        """The lemma that makes step 2 durable.

        `regroup_by_duplicates` rewrites `group`. If it could give two records
        that previously shared a group different ones, step 2 would buy nothing.
        It cannot: the union-find closes over the existing `group_key` relation
        as well as the hash pairs, so a component is either rewritten whole or
        left alone.
        """
        records = [
            ImageRecord("a", 10, 10, group="clip"),
            ImageRecord("b", 10, 10, group="clip"),
            ImageRecord("c", 10, 10, group="other"),
        ]
        # `c` is a perceptual duplicate of `a` only. `b` shares a's group.
        report = DuplicateReport(
            images_hashed=3,
            pairs=[DuplicatePair("a", "c", 0, Split.TRAIN, Split.TEST)],
        )
        out = {r.image_id: r.group_key for r in regroup_by_duplicates(records, report)}
        assert out["a"] == out["b"], "a group that was together was pulled apart"
        assert out["a"] == out["c"], "the duplicate was not merged in"

    def test_step3_holds_for_every_pre_existing_group(self, tmp_path):
        # Property form of the lemma over a mixed dataset.
        location = export(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 1), (rf(DAYRIDE[1]), 2), (rf(STILLS[0]), 3)],
                "test": [(rf(DAYRIDE[2]), 1), (rf(STILLS[1]), 4)],
            },
        )
        records = ingest(location)
        before: dict[str, list[str]] = {}
        for r in records:
            before.setdefault(r.group_key, []).append(r.image_id)

        regrouped, _ = group_duplicates(records, location.parent)
        after = {r.image_id: r.group_key for r in regrouped}
        for key, ids in before.items():
            assert len({after[i] for i in ids}) == 1, f"group {key!r} was split"

    def test_step4_one_group_key_is_one_split(self, tmp_path):
        # split_records builds dict[group_key, Split]; the mapping is
        # single-valued by construction. Asserted over a real assignment.
        location = export(
            tmp_path,
            {
                "train": [(rf(s), i) for i, s in enumerate(DAYRIDE + STILLS)],
                "test": [(rf(s), 100 + i) for i, s in enumerate(NIGHTRIDE + VIDEO3)],
            },
        )
        records = ingest(location)
        assignment = split_records(records, seed=0)
        by_key: dict[str, set] = {}
        for record in records:
            by_key.setdefault(record.group_key, set()).add(assignment.of(record))
        assert all(len(v) == 1 for v in by_key.values())


class TestTransitivity:
    """Task 5: video identity + perceptual similarity + existing groups, closed."""

    def test_video_identity_chains_into_perceptual_similarity(self, tmp_path):
        # A shares clip identity with B; B is perceptually identical to C, whose
        # filename relates it to nothing. All three must end in one cluster.
        location = export(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 11), (rf(DAYRIDE[1]), 22)],  # A, B
                "test": [(rf(STILLS[0]), 22)],  # C: B's twin, unrelated name
            },
        )
        records = ingest(location)
        regrouped, report = group_duplicates(records, location.parent)
        keys = {r.image_id: r.group_key for r in regrouped}
        assert len(set(keys.values())) == 1, f"not one cluster: {keys}"
        assert report.pairs, "fixture did not actually produce a duplicate pair"

    def test_chain_survives_into_the_split(self, tmp_path):
        location = export(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 11), (rf(DAYRIDE[1]), 22)]
                + [(rf(s), 200 + i) for i, s in enumerate(STILLS)],
                "test": [(rf(NIGHTRIDE[0]), 22)],
            },
        )
        records = ingest(location)
        regrouped, _ = group_duplicates(records, location.parent)
        assignment = split_records(regrouped, seed=0)
        landed = {
            assignment.of(r) for r in regrouped if "dayride" in r.image_id or "night" in r.image_id
        }
        assert len(landed) == 1


# --- Task 7: adversarial scenarios -------------------------------------------

SEEDS = (0, 1, 7, 42, 1234)


class TestAdversarial:
    """Thirteen shapes the grouping has to survive.

    Every case asserts the same property — no recognised clip occupies more than
    one split — and the cases that can also over-merge assert that unrelated
    images stayed apart.
    """

    def _run(self, tmp_path, layout, *, seeds=SEEDS, ratios=None, dupes=True):
        location = export(tmp_path, layout)
        records = ingest(location)
        if dupes:
            records, _ = group_duplicates(records, location.parent)
        for seed in seeds:
            assignment = split_records(records, ratios=ratios, seed=seed)
            verify_split(records, assignment, require_all_splits=False)
            assert_no_clip_leaks(records, assignment)
        return records

    def _filler(self, n=12, start=500):
        return [(rf(f"pl_license_plate_{start + i}"), start + i) for i in range(n)]

    # 1
    def test_clip_entirely_in_train(self, tmp_path):
        self._run(tmp_path, {"train": [(rf(s), i) for i, s in enumerate(DAYRIDE)] + self._filler()})

    # 2
    def test_clip_entirely_in_test(self, tmp_path):
        self._run(tmp_path, {"test": [(rf(s), i) for i, s in enumerate(DAYRIDE)] + self._filler()})

    # 3
    def test_clip_split_across_train_and_test(self, tmp_path):
        self._run(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 1), (rf(DAYRIDE[1]), 2)] + self._filler(),
                "test": [(rf(DAYRIDE[2]), 3), (rf(DAYRIDE[3]), 4)],
            },
        )

    # 4
    def test_clip_split_across_validation_and_test(self, tmp_path):
        self._run(
            tmp_path,
            {
                "valid": [(rf(DAYRIDE[0]), 1), (rf(DAYRIDE[1]), 2)] + self._filler(),
                "test": [(rf(DAYRIDE[2]), 3)],
            },
        )

    # 5
    def test_multiple_clips_with_similar_names_do_not_merge(self, tmp_path):
        layout = {
            "train": [(rf("dayride_type1_001-mp4-t-10"), 1), (rf("dayride_type1_002-mp4-t-10"), 2)],
            "test": [
                (rf("dayride_type1_0010-mp4-t-10"), 3),
                (rf("video1_10"), 4),
                (rf("video11_10"), 5),
            ],
        }
        layout["train"] += self._filler()
        records = self._run(tmp_path, layout)
        groups = {r.image_id: r.group_key for r in records}
        clip_groups = {v for k, v in groups.items() if "license_plate" not in k}
        assert len(clip_groups) == 5, f"similar names merged: {clip_groups}"

    # 6
    def test_identical_stems_but_unrelated_files_do_not_merge(self, tmp_path):
        # Same stem in two upstream directories, visually different, not a
        # recognised frame name. Nothing establishes these are one photograph,
        # so ingest must keep them apart.
        records = self._run(
            tmp_path,
            {
                "train": [(rf("plate_001", seed=1), 71)] + self._filler(),
                "test": [(rf("plate_001", seed=2), 72)],
            },
        )
        same_stem = [r for r in records if "plate_001" in r.image_id]
        assert len(same_stem) == 2
        assert len({r.group_key for r in same_stem}) == 2, "unrelated stems were merged"

    # 7
    def test_roboflow_suffixes_in_every_flavour(self, tmp_path):
        records = self._run(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0], ext="jpg"), 1), (rf(DAYRIDE[1], ext="jpeg"), 2)]
                + self._filler(),
                "test": [(rf(DAYRIDE[2], ext="png"), 3)],
            },
        )
        dayride = {r.group_key for r in records if "dayride" in r.image_id}
        assert len(dayride) == 1, "extension flavour changed the clip identity"

    # 8
    def test_video_n_convention(self, tmp_path):
        records = self._run(
            tmp_path,
            {
                "train": [(rf(VIDEO3[0]), 1), (rf(VIDEO11[0]), 2)] + self._filler(),
                "test": [(rf(VIDEO3[1]), 3), (rf(VIDEO3[2]), 4), (rf(VIDEO11[1]), 5)],
            },
        )
        v3 = {r.group_key for r in records if "video3" in r.image_id}
        v11 = {r.group_key for r in records if "video11" in r.image_id}
        assert len(v3) == 1 and len(v11) == 1
        assert v3 != v11, "video3 and video11 collapsed together"

    # 9
    def test_perceptual_duplicates_without_shared_filenames(self, tmp_path):
        records = self._run(
            tmp_path,
            {
                "train": [(rf(STILLS[0]), 90)] + self._filler(),
                "test": [(rf(STILLS[1]), 90)],  # same picture, unrelated name
            },
        )
        twins = [r for r in records if any(s in r.image_id for s in STILLS[:2])]
        assert len({r.group_key for r in twins}) == 1, "perceptual twins were not merged"

    # 10
    def test_filename_and_perceptual_grouping_interact(self, tmp_path):
        # One frame of a clip is also a perceptual twin of an unrelated still.
        # The closure must pull the whole clip and the still into one group.
        records = self._run(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 61), (rf(DAYRIDE[1]), 62)] + self._filler(),
                "test": [(rf(STILLS[0]), 61)],
            },
        )
        involved = [r for r in records if "dayride" in r.image_id or STILLS[0] in r.image_id]
        assert len(involved) == 3
        assert len({r.group_key for r in involved}) == 1

    # 11
    def test_all_four_upstream_directories(self, tmp_path):
        self._run(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 1)] + self._filler(),
                "valid": [(rf(DAYRIDE[1]), 2)],
                "val": [(rf(DAYRIDE[2]), 3)],
                "test": [(rf(DAYRIDE[3]), 4)],
            },
        )

    # 12
    @pytest.mark.parametrize(
        "ratios",
        [
            {Split.TRAIN: 0.70, Split.VAL: 0.15, Split.TEST: 0.15},
            {Split.TRAIN: 0.50, Split.VAL: 0.25, Split.TEST: 0.25},
            {Split.TRAIN: 0.34, Split.VAL: 0.33, Split.TEST: 0.33},
            {Split.TRAIN: 0.90, Split.VAL: 0.05, Split.TEST: 0.05},
            {Split.TRAIN: 0.20, Split.VAL: 0.40, Split.TEST: 0.40},
        ],
    )
    def test_every_split_ratio(self, tmp_path, ratios):
        # The Stage-2 counterexample was a placement artifact of the 70/15/15
        # ratios. With clip identity fixed the property must hold for any of
        # them, including ones that make a clip a majority of its region.
        self._run(
            tmp_path,
            {
                "train": [(rf(s), i) for i, s in enumerate(DAYRIDE)] + self._filler(4),
                "test": [(rf(s), 50 + i) for i, s in enumerate(NIGHTRIDE)],
            },
            ratios=ratios,
        )

    # 13
    def test_many_seeds(self, tmp_path):
        self._run(
            tmp_path,
            {
                "train": [(rf(DAYRIDE[0]), 1), (rf(DAYRIDE[1]), 2)] + self._filler(),
                "test": [(rf(DAYRIDE[2]), 3), (rf(VIDEO3[0]), 4)],
                "valid": [(rf(VIDEO3[1]), 5)],
            },
            seeds=range(50),
        )

    def test_the_stage_two_counterexample_is_closed(self, tmp_path):
        """The witness that leaked 25/25 seeds before this change.

        A clip whose larger half exceeds 55% of its region: under the old
        grouping the two halves were separate groups, and deficit-greedy
        placement pushed the second out of train. One group now, so there is
        nothing to place apart.
        """
        big = [(rf(f"ride_007-mp4-t-{i}"), 900 + i) for i in range(60)]
        small = [(rf(f"ride_007-mp4-t-{i}"), 900 + i) for i in range(60, 70)]
        self._run(
            tmp_path,
            {"train": big, "test": small, "valid": self._filler(30)},
            seeds=range(25),
        )

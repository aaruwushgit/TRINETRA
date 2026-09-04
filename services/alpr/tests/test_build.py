"""One-call dataset construction.

Downloads are stubbed — these tests exercise the part that decides whether a
notebook can run in a fresh session, not Roboflow's client.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image

from alpr.build import (
    SOURCES,
    BuildError,
    attribution,
    build_dataset,
    dataset_ready,
    ensure_dataset,
    group_duplicates,
    ingest_sources,
)
from alpr.data import read_manifest, split_records
from alpr.data.schema import Region
from alpr.dupes import DUPLICATE_GROUP_PREFIX


class TestSources:
    def test_both_regions_are_represented(self):
        assert {s.region for s in SOURCES} == {Region.EUROPE, Region.INDIA}

    def test_every_source_records_its_licence(self):
        # The licence decides what may be redistributed; a source without one
        # cannot be published safely.
        assert all(s.licence and s.url for s in SOURCES)

    def test_current_sources_are_redistributable(self):
        assert all(s.redistributable for s in SOURCES)

    @pytest.mark.parametrize(
        ("licence", "allowed"),
        [
            ("CC BY 4.0", True),
            ("CC BY-SA 4.0", True),
            ("ODbL-1.0", True),
            ("CC BY-NC-ND 4.0", False),  # the case a naive split got wrong
            ("CC BY-ND 4.0", False),
            ("CC BY-NC 4.0", False),
            ("cc by-nc-nd 4.0", False),  # case-insensitive
        ],
    )
    def test_licence_gates_redistribution(self, licence, allowed):
        # Regression: splitting "CC BY-NC-ND 4.0" on hyphens yields "ND 4.0",
        # so a membership test for "ND" silently marked NoDerivatives data as
        # publishable — a licence violation, not a cosmetic bug.
        from alpr.build import RoboflowSource

        source = RoboflowSource(
            workspace="w",
            project="p",
            version=1,
            directory="d",
            region=Region.INDIA,
            source_tag="t",
            id_prefix="x-",
            licence=licence,
            url="http://example.com",
        )
        assert source.redistributable is allowed

    def test_attribution_lists_redistributable_sources(self):
        text = attribution()
        for source in SOURCES:
            assert source.url in text


def _distinct_image(path, key: int) -> None:
    """Write an image whose perceptual hash is unlike every other key's.

    These used to be `Image.new("RGB", (640, 480))` — the same black rectangle
    every time, so the whole fixture was 48 copies of one picture. That was
    invisible while nothing hashed the dataset. Now that `build_dataset`
    audits for near-duplicates it is not: 48 identical images correctly collapse
    into a single split group, one split takes everything, and `verify_split`
    rejects the build. The fixture was wrong, not the audit.

    A high-contrast 9x8 pattern upscaled by an exact integer factor survives
    dhash's own 9x8 thumbnailing, so distinct keys land ~32 bits apart against a
    threshold of 5 — comfortably distinct, and identical keys hash identically,
    which is what `_twin` relies on.
    """
    rng = random.Random(key)
    small = Image.new("L", (9, 8))
    small.putdata([rng.choice((0, 255)) for _ in range(72)])
    small.convert("RGB").resize((648, 480), Image.Resampling.NEAREST).save(path)


def _fake_download(raw_dir, n=4):
    """Mimic what Roboflow writes: one directory per source, per split."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(SOURCES):
        for split_index, split in enumerate(("train", "valid", "test")):
            images = raw_dir / source.directory / split / "images"
            labels = raw_dir / source.directory / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for i in range(n):
                stem = f"{split}_{i}"
                # Unique per (source, upstream split, index) so no two images in
                # the fixture are accidentally the same picture.
                _distinct_image(images / f"{stem}.jpg", 1000 * index + 100 * split_index + i)
                (labels / f"{stem}.txt").write_text("0 0.5 0.6 0.2 0.08\n")
    return raw_dir


def _twin(raw_dir, source, split, stem, of_key):
    """Add an image to an upstream split that is a copy of `of_key`'s picture."""
    images = raw_dir / source.directory / split / "images"
    labels = raw_dir / source.directory / split / "labels"
    _distinct_image(images / f"{stem}.jpg", of_key)
    (labels / f"{stem}.txt").write_text("0 0.5 0.6 0.2 0.08\n")


def _split_of_exported(out_dir):
    """Map image_id -> split by reading the exported tree, not by recomputing.

    The tree is the artifact training consumes, so asserting against it proves
    the split that was actually written rather than the split the same code
    would compute a second time.
    """
    placement = {}
    for split in ("train", "val", "test"):
        for label in (out_dir / "labels" / split).glob("*.txt"):
            placement[label.stem] = split
    return placement


class TestIngest:
    def test_ingests_every_source(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        records = ingest_sources(raw)
        assert len(records) == len(SOURCES) * 3 * 4
        assert {r.source for r in records} == {s.source_tag for s in SOURCES}

    def test_regions_are_tagged(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        regions = {b.region for r in ingest_sources(raw) for b in r.boxes}
        assert regions == {s.region for s in SOURCES}

    def test_missing_source_is_an_error(self, tmp_path):
        with pytest.raises(BuildError, match="missing"):
            ingest_sources(tmp_path / "empty")


class TestBuildDataset:
    def test_produces_a_usable_export(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        data_yaml, stats = build_dataset(raw, tmp_path / "yolo", tmp_path / "manifest.jsonl")
        assert data_yaml.exists()
        assert stats.images == len(SOURCES) * 3 * 8
        assert dataset_ready(data_yaml)

    def test_is_deterministic(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        _, first = build_dataset(raw, tmp_path / "a", tmp_path / "m1.jsonl", seed=0)
        _, second = build_dataset(raw, tmp_path / "b", tmp_path / "m2.jsonl", seed=0)
        assert first.by_split == second.by_split


class TestDuplicateRegrouping:
    """The audit is part of the build, not a tool nobody runs.

    Every test here builds a real dataset containing known near-duplicates and
    asserts against the *exported tree*, which is what training consumes.
    """

    def _build_with_twins(self, tmp_path, *, n=8, **kwargs):
        raw = _fake_download(tmp_path / "raw", n=n)
        source = SOURCES[0]
        # The leak this exists to stop: the same picture in the upstream train
        # and test directories, under names nothing relates to each other.
        _twin(raw, source, "train", "twin_left", of_key=777)
        _twin(raw, source, "test", "twin_right", of_key=777)
        out = tmp_path / "yolo"
        build_dataset(raw, out, tmp_path / "manifest.jsonl", **kwargs)
        return raw, out

    def test_near_duplicates_cannot_cross_splits(self, tmp_path):
        _, out = self._build_with_twins(tmp_path)
        placement = _split_of_exported(out)
        left = placement[f"{SOURCES[0].id_prefix}train-twin_left"]
        right = placement[f"{SOURCES[0].id_prefix}test-twin_right"]
        assert left == right, "a memorized image reached a held-out split"

    def test_without_the_check_they_do_cross(self, tmp_path):
        # Proves the test above is actually testing something: the same fixture
        # with the audit disabled puts the twins on opposite sides.
        _, out = self._build_with_twins(tmp_path, check_duplicates=False)
        placement = _split_of_exported(out)
        left = placement[f"{SOURCES[0].id_prefix}train-twin_left"]
        right = placement[f"{SOURCES[0].id_prefix}test-twin_right"]
        assert left != right

    def test_a_whole_duplicate_group_stays_intact(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        source = SOURCES[0]
        # Five copies of one picture, spread across all three upstream splits.
        for split, stem in (
            ("train", "c1"),
            ("train", "c2"),
            ("valid", "c3"),
            ("test", "c4"),
            ("test", "c5"),
        ):
            _twin(raw, source, split, stem, of_key=555)
        out = tmp_path / "yolo"
        build_dataset(raw, out, tmp_path / "manifest.jsonl")

        placement = _split_of_exported(out)
        cluster = {
            placement[f"{source.id_prefix}{split}-{stem}"]
            for split, stem in (
                ("train", "c1"),
                ("train", "c2"),
                ("valid", "c3"),
                ("test", "c4"),
                ("test", "c5"),
            )
        }
        assert len(cluster) == 1, f"cluster of 5 was split across {cluster}"

    def test_the_merged_group_is_persisted_in_the_manifest(self, tmp_path):
        # Manifest-as-source-of-truth: the groups the split was computed from
        # have to be *in* the manifest, or a downstream read_manifest() +
        # split_records() disagrees with the tree that was exported.
        raw = _fake_download(tmp_path / "raw", n=8)
        source = SOURCES[0]
        _twin(raw, source, "train", "twin_left", of_key=777)
        _twin(raw, source, "test", "twin_right", of_key=777)
        manifest = tmp_path / "manifest.jsonl"
        build_dataset(raw, tmp_path / "yolo", manifest)

        groups = {r.image_id: r.group for r in read_manifest(manifest)}
        left = groups[f"{source.id_prefix}train-twin_left"]
        right = groups[f"{source.id_prefix}test-twin_right"]
        assert left == right
        assert left.startswith(DUPLICATE_GROUP_PREFIX)

    def test_the_exported_tree_matches_the_manifest(self, tmp_path):
        # The invariant notebooks 03+ rely on: read the manifest, re-split with
        # the same seed, and you get the split that was exported.
        raw, out = self._build_with_twins(tmp_path)
        manifest = tmp_path / "manifest.jsonl"

        records = read_manifest(manifest)
        assignment = split_records(records, seed=0)
        recomputed = {r.image_id: assignment.of(r).value for r in records}
        assert recomputed == _split_of_exported(out)

    def test_ordinary_images_keep_their_own_group(self, tmp_path):
        # A still with no twin must still split per image, not get swept into
        # some merged group.
        raw, _ = self._build_with_twins(tmp_path)
        records = read_manifest(tmp_path / "manifest.jsonl")
        untwinned = [r for r in records if "twin" not in r.image_id]
        twinned = [r for r in records if "twin" in r.image_id]

        assert len(twinned) == 2, "fixture should have produced exactly one twin pair"
        assert len(untwinned) == len(SOURCES) * 3 * 8
        # An ordinary still is its own group: nothing in the filename says two
        # stills are related, so ingest will not claim they are.
        assert all(r.group == r.image_id for r in untwinned)
        assert all(r.group.startswith(DUPLICATE_GROUP_PREFIX) for r in twinned)

    def test_builds_stay_deterministic(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        source = SOURCES[0]
        _twin(raw, source, "train", "twin_left", of_key=777)
        _twin(raw, source, "test", "twin_right", of_key=777)

        first_manifest = tmp_path / "m1.jsonl"
        second_manifest = tmp_path / "m2.jsonl"
        build_dataset(raw, tmp_path / "a", first_manifest, seed=0)
        build_dataset(raw, tmp_path / "b", second_manifest, seed=0)

        # Byte-identical manifests: same groups, same order, same everything.
        assert first_manifest.read_bytes() == second_manifest.read_bytes()
        assert _split_of_exported(tmp_path / "a") == _split_of_exported(tmp_path / "b")

    def test_region_stratification_survives(self, tmp_path):
        # Each region must still be spread over the splits; regrouping must not
        # let one region collapse into a single split.
        raw, out = self._build_with_twins(tmp_path, n=16)
        records = read_manifest(tmp_path / "manifest.jsonl")
        placement = _split_of_exported(out)

        by_region: dict[Region, set[str]] = {}
        for record in records:
            by_region.setdefault(record.primary_region, set()).add(placement[record.image_id])

        assert set(by_region) == {s.region for s in SOURCES}
        for region, splits in by_region.items():
            assert splits == {"train", "val", "test"}, f"{region} only reached {splits}"

    def test_verify_split_still_runs_on_every_build(self, tmp_path, monkeypatch):
        # Adding a step before the split must not have moved the guard that
        # rejects an unsound one.
        seen = []

        def spy(records, assignment, **kwargs):
            seen.append((len(list(records)), assignment))

        monkeypatch.setattr("alpr.build.verify_split", spy)
        raw = _fake_download(tmp_path / "raw", n=8)
        build_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")

        assert len(seen) == 1, f"verify_split ran {len(seen)} time(s)"
        count, assignment = seen[0]
        # It must see the post-regrouping records, not the ingested ones.
        assert count == len(SOURCES) * 3 * 8
        assert assignment is not None

    def test_an_unsound_split_still_aborts_the_build(self, tmp_path):
        # The whole fixture is one picture, so every image collapses into one
        # group and two splits get nothing. That must fail loudly rather than
        # export a dataset with an empty test split.
        from alpr.data.schema import DatasetError

        raw = tmp_path / "raw"
        for source in SOURCES:
            for split in ("train", "valid", "test"):
                (raw / source.directory / split / "images").mkdir(parents=True)
                (raw / source.directory / split / "labels").mkdir(parents=True)
                for i in range(4):
                    _twin(raw, source, split, f"{split}_{i}", of_key=1)

        with pytest.raises(DatasetError, match="received no images"):
            build_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")

    def test_the_report_is_handed_to_the_caller(self, tmp_path):
        seen = []
        raw = _fake_download(tmp_path / "raw", n=8)
        source = SOURCES[0]
        _twin(raw, source, "train", "twin_left", of_key=777)
        _twin(raw, source, "test", "twin_right", of_key=777)
        build_dataset(
            raw,
            tmp_path / "yolo",
            tmp_path / "m.jsonl",
            on_duplicates=seen.append,
        )
        assert len(seen) == 1
        # The report describes the pre-regrouping split, so the contamination
        # it names is what the build went on to prevent.
        assert seen[0].contaminating
        assert "measure memorization" in seen[0].report()

    def test_a_clip_straddling_upstream_splits_ends_in_one_split(self, tmp_path):
        """The exact leak the README describes, end to end.

        Roboflow put consecutive frames of one clip in its own train and test
        directories. Filename grouping alone cannot rejoin them, because the
        image id carries the upstream split (`eu-train-…` vs `eu-test-…`) to
        stop stems colliding. Hashing relates the frames that look alike, and
        the closure over both relations then pulls in the rest of each clip.
        """
        raw = _fake_download(tmp_path / "raw", n=8)
        source = SOURCES[0]
        # Four frames of one clip: two in the upstream train dir, two in test.
        # Frames 1 and 3 look alike (one moment), 2 and 4 do not.
        for split, frame, key in (
            ("train", 1062, 900),
            ("train", 1063, 901),
            ("test", 1064, 900),
            ("test", 1065, 902),
        ):
            _twin(raw, source, split, f"dayride_type1_001-mp4-t-{frame}", of_key=key)

        out = tmp_path / "yolo"
        build_dataset(raw, out, tmp_path / "manifest.jsonl")
        placement = _split_of_exported(out)

        landed = {
            placement[f"{source.id_prefix}{split}-dayride_type1_001-mp4-t-{frame}"]
            for split, frame in (
                ("train", 1062),
                ("train", 1063),
                ("test", 1064),
                ("test", 1065),
            )
        }
        assert len(landed) == 1, f"one clip was split across {landed}"

    def test_threshold_zero_still_catches_exact_copies(self, tmp_path):
        _, out = self._build_with_twins(tmp_path, duplicate_threshold=0)
        placement = _split_of_exported(out)
        assert (
            placement[f"{SOURCES[0].id_prefix}train-twin_left"]
            == placement[f"{SOURCES[0].id_prefix}test-twin_right"]
        )


class TestGroupDuplicates:
    """The seam `build_dataset` uses, in isolation."""

    def test_grouping_ignores_the_provisional_seed(self, tmp_path):
        # The provisional assignment exists only so a pair can name the splits
        # it crosses. If the seed leaked into the grouping, a rebuild with a
        # different seed would merge different images.
        raw = _fake_download(tmp_path / "raw", n=8)
        _twin(raw, SOURCES[0], "train", "twin_left", of_key=777)
        _twin(raw, SOURCES[0], "test", "twin_right", of_key=777)
        records = ingest_sources(raw)

        zero, _ = group_duplicates(records, raw, seed=0)
        seven, _ = group_duplicates(records, raw, seed=7)
        assert [r.group for r in zero] == [r.group for r in seven]

    def test_a_clean_dataset_is_returned_unchanged(self, tmp_path):
        raw = _fake_download(tmp_path / "raw", n=8)
        records = ingest_sources(raw)
        regrouped, report = group_duplicates(records, raw)
        assert regrouped == records
        assert report.pairs == []


class TestDatasetReady:
    def test_false_when_absent(self, tmp_path):
        assert dataset_ready(tmp_path / "nope.yaml") is False

    def test_true_after_a_build(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        data_yaml, _ = build_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")
        assert dataset_ready(data_yaml) is True


class TestEnsureDataset:
    def test_builds_when_missing(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        data_yaml = ensure_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl")
        assert data_yaml.exists()

    def test_is_a_no_op_when_present(self, tmp_path, monkeypatch):
        # The point of the function: any notebook can call it first, and it
        # costs nothing when the data is already there.
        raw = _fake_download(tmp_path / "raw")
        out = tmp_path / "yolo"
        ensure_dataset(raw, out, tmp_path / "m.jsonl")

        def _explode(*args, **kwargs):
            raise AssertionError("should not rebuild an existing dataset")

        monkeypatch.setattr("alpr.build.build_dataset", _explode)
        assert ensure_dataset(raw, out, tmp_path / "m.jsonl").exists()

    def test_force_rebuilds(self, tmp_path):
        raw = _fake_download(tmp_path / "raw")
        out = tmp_path / "yolo"
        ensure_dataset(raw, out, tmp_path / "m.jsonl")
        (out / "data.yaml").unlink()
        assert ensure_dataset(raw, out, tmp_path / "m.jsonl", force=True).exists()

    def test_does_not_ask_for_a_key_when_sources_are_present(self, tmp_path, monkeypatch):
        # Re-running in a session that already downloaded should not demand a
        # credential it does not need.
        raw = _fake_download(tmp_path / "raw")

        def _explode(*args, **kwargs):
            raise AssertionError("should not request a credential")

        monkeypatch.setattr("alpr.env.get_credential", _explode)
        assert ensure_dataset(raw, tmp_path / "yolo", tmp_path / "m.jsonl").exists()

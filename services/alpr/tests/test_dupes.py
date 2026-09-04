"""Near-duplicate detection across splits."""

from __future__ import annotations

from PIL import Image, ImageEnhance

from alpr.data.schema import ImageRecord, PlateBox, Region, Split
from alpr.data.split import SplitAssignment
from alpr.dupes import (
    DEFAULT_THRESHOLD,
    DuplicatePair,
    DuplicateReport,
    clean_subset,
    dhash,
    duplicate_clusters,
    find_duplicates,
    hamming,
    regroup_by_duplicates,
)


def photo(path, seed=0, size=(200, 150)):
    """A structured image — flat colour would hash identically for everything."""
    import random

    rng = random.Random(seed)
    img = Image.new("RGB", size, (30, 30, 40))
    pixels = img.load()
    for _ in range(400):
        x, y = rng.randrange(size[0]), rng.randrange(size[1])
        for dx in range(rng.randrange(5, 25)):
            for dy in range(rng.randrange(5, 25)):
                if x + dx < size[0] and y + dy < size[1]:
                    pixels[x + dx, y + dy] = (rng.randrange(256), rng.randrange(256), 200)
    img.save(path)
    return path


class TestDhash:
    def test_identical_images_hash_identically(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=1)
        Image.open(a).save(b)
        assert dhash(a) == dhash(b)

    def test_different_images_hash_differently(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=1)
        photo(b, seed=99)
        assert hamming(dhash(a), dhash(b)) > 5

    def test_survives_rescaling(self, tmp_path):
        # The point of a perceptual hash: an augmented copy is still the same
        # picture, where a cryptographic hash would see a different file.
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=7, size=(400, 300))
        Image.open(a).resize((200, 150)).save(b)
        assert hamming(dhash(a), dhash(b)) <= 5

    def test_survives_mild_brightness_change(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        photo(a, seed=3)
        ImageEnhance.Brightness(Image.open(a)).enhance(1.15).save(b)
        assert hamming(dhash(a), dhash(b)) <= 5

    def test_hash_is_64_bits(self, tmp_path):
        assert dhash(photo(tmp_path / "a.png")).bit_length() <= 64


class TestHamming:
    def test_identical(self):
        assert hamming(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self):
        assert hamming(0b1111, 0b1010) == 2


def _setup(tmp_path, layout):
    """Build images and an assignment from {image_id: (seed, split)}.

    The assignment is keyed by `group_key`, not by `image_id`: for an ordinary
    still those are the same string, but a frame-named id strips its suffix, and
    keying by id would make `assignment.of()` raise on exactly the video-frame
    case the grouping rules exist for.
    """
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    records, by_group = [], {}
    for image_id, (seed, split) in layout.items():
        photo(images / f"{image_id}.png", seed=seed)
        record = ImageRecord(image_id=image_id, width=200, height=150, file_name=f"{image_id}.png")
        records.append(record)
        by_group[record.group_key] = split
    assignment = SplitAssignment(by_group=by_group, seed=0, ratios={Split.TRAIN: 1.0})
    return records, assignment, images


class TestFindDuplicates:
    def test_clean_dataset_has_no_pairs(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path,
            {"a": (1, Split.TRAIN), "b": (2, Split.VAL), "c": (3, Split.TEST)},
        )
        report = find_duplicates(records, assignment, images)
        assert report.pairs == []
        assert "no train/held-out contamination" in report.report()

    def test_detects_a_twin_across_splits(self, tmp_path):
        # The case that inflates a score: the same picture in train and test.
        records, assignment, images = _setup(
            tmp_path, {"train_img": (5, Split.TRAIN), "test_img": (5, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        assert len(report.contaminating) == 1
        assert report.contaminated_images(Split.TEST) == {"test_img"}
        assert "measure memorization" in report.report()

    def test_duplicates_within_one_split_are_not_contamination(self, tmp_path):
        # Wasteful, but it does not inflate a held-out score.
        records, assignment, images = _setup(
            tmp_path, {"a": (5, Split.TRAIN), "b": (5, Split.TRAIN)}
        )
        report = find_duplicates(records, assignment, images)
        assert len(report.pairs) == 1
        assert report.contaminating == []

    def test_val_test_overlap_is_flagged_but_not_contaminating(self, tmp_path):
        # Neither was trained on, so no memorization — still worth knowing.
        records, assignment, images = _setup(tmp_path, {"v": (9, Split.VAL), "t": (9, Split.TEST)})
        report = find_duplicates(records, assignment, images)
        assert len(report.cross_split) == 1
        assert report.contaminating == []

    def test_counts_hashed_images(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (1, Split.TRAIN), "b": (2, Split.TEST)}
        )
        assert find_duplicates(records, assignment, images).images_hashed == 2

    def test_missing_file_is_reported_not_fatal(self, tmp_path):
        records, assignment, images = _setup(tmp_path, {"a": (1, Split.TRAIN)})
        (images / "a.png").unlink()
        report = find_duplicates(records, assignment, images)
        assert report.unreadable == ["a"]
        assert report.images_hashed == 0

    def test_threshold_zero_finds_only_exact_matches(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (4, Split.TRAIN), "b": (4, Split.TEST)}
        )
        # Perturb b slightly so it is near- but not exactly-identical.
        ImageEnhance.Brightness(Image.open(images / "b.png")).enhance(1.1).save(images / "b.png")
        loose = find_duplicates(records, assignment, images, threshold=5)
        strict = find_duplicates(records, assignment, images, threshold=0)
        assert len(loose.pairs) >= len(strict.pairs)

    def test_report_includes_split_percentages(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"tr": (6, Split.TRAIN), "te": (6, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        text = report.report(split_totals={Split.TEST: 4})
        assert "25.0%" in text


class TestDuplicatePair:
    def test_within_split_is_not_cross_split(self):
        pair = DuplicatePair("a", "b", 0, Split.TRAIN, Split.TRAIN)
        assert pair.crosses_splits is False
        assert pair.contaminates_evaluation is False

    def test_train_to_test_contaminates(self):
        pair = DuplicatePair("a", "b", 0, Split.TRAIN, Split.TEST)
        assert pair.contaminates_evaluation is True

    def test_val_to_test_does_not_contaminate(self):
        pair = DuplicatePair("a", "b", 0, Split.VAL, Split.TEST)
        assert pair.crosses_splits is True
        assert pair.contaminates_evaluation is False


class TestGroupKeyFromFilename:
    """The regex fix, in isolation from hashing."""

    def test_roboflow_video_frames_share_a_group(self):
        # The exact pair that leaked: frames 1062 and 1063 of one clip landed
        # in train and test because this pattern did not match.
        a = ImageRecord("dayride_type1_001-mp4-t-1062", 10, 10)
        b = ImageRecord("dayride_type1_001-mp4-t-1063", 10, 10)
        assert a.group_key == b.group_key == "dayride_type1_001"

    def test_classic_frame_naming_still_groups(self):
        assert ImageRecord("clip_042_frame_0137", 10, 10).group_key == "clip_042"

    def test_unrelated_stills_are_not_collapsed(self):
        # A bare trailing-number rule would put most of the dataset in one
        # group and hand it to a single split.
        a = ImageRecord("pl_license_plate_205", 10, 10)
        b = ImageRecord("pl_license_plate_242", 10, 10)
        assert a.group_key != b.group_key


class TestDuplicateClusters:
    def test_pairs_become_clusters(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (5, Split.TRAIN), "b": (5, Split.TEST)}
        )
        clusters = duplicate_clusters(find_duplicates(records, assignment, images))
        assert clusters["a"] == clusters["b"]

    def test_chains_merge_transitively(self):
        # A~B and B~C must give one cluster even if A and C never paired:
        # splitting a chain leaks exactly as badly as splitting a pair.
        report = DuplicateReport(
            images_hashed=3,
            pairs=[
                DuplicatePair("a", "b", 1, Split.TRAIN, Split.TRAIN),
                DuplicatePair("b", "c", 1, Split.TRAIN, Split.TEST),
            ],
        )
        clusters = duplicate_clusters(report)
        assert clusters["a"] == clusters["b"] == clusters["c"]

    def test_unduplicated_images_are_absent(self):
        assert duplicate_clusters(DuplicateReport(images_hashed=1)) == {}


class TestRegroup:
    def test_duplicates_end_up_in_one_split(self, tmp_path):
        from alpr.data.split import split_records, verify_split

        records, assignment, images = _setup(
            tmp_path,
            {
                "twin_a": (5, Split.TRAIN),
                "twin_b": (5, Split.TEST),
                **{f"solo{i}": (100 + i, Split.TRAIN) for i in range(12)},
            },
        )
        report = find_duplicates(records, assignment, images)
        regrouped = regroup_by_duplicates(records, report)

        new_assignment = split_records(regrouped, seed=0)
        verify_split(regrouped, new_assignment, require_all_splits=False)

        by_id = {r.image_id: r for r in regrouped}
        assert new_assignment.of(by_id["twin_a"]) is new_assignment.of(by_id["twin_b"])

    def test_records_without_duplicates_are_untouched(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path, {"a": (1, Split.TRAIN), "b": (2, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        assert regroup_by_duplicates(records, report) == list(records)

    def test_the_merged_group_id_is_stable(self, tmp_path):
        # Lowest id wins, so a rebuild produces the same group key and
        # therefore the same split.
        records, assignment, images = _setup(
            tmp_path, {"zebra": (5, Split.TRAIN), "alpha": (5, Split.TEST)}
        )
        report = find_duplicates(records, assignment, images)
        groups = {r.image_id: r.group for r in regroup_by_duplicates(records, report)}
        assert groups["alpha"] == groups["zebra"] == "dup:alpha"


class TestRegroupPreservesFilenameGrouping:
    """Hashing must not *undo* the frame-suffix rule.

    `regroup_by_duplicates` writes `group`, and `group_key` returns `group`
    whenever it is set — so a `dup:` group replaces whatever the filename
    implied. Where hashing only partially fires across a clip, that would split
    the clip: exactly the leak both rules exist to prevent. The returned
    grouping is therefore the closure over both relations.
    """

    def test_a_clip_is_not_split_when_only_one_frame_hash_matches(self, tmp_path):
        from alpr.data.split import split_records, verify_split

        # frame 1 and frame 2 of one clip look nothing alike (different seeds),
        # so hashing relates neither to the other. An unrelated photograph is a
        # twin of frame 1 only.
        records, assignment, images = _setup(
            tmp_path,
            {
                "ride_001-mp4-t-1": (5, Split.TRAIN),
                "ride_001-mp4-t-2": (42, Split.TRAIN),
                "elsewhere": (5, Split.TEST),
                **{f"solo{i}": (100 + i, Split.TRAIN) for i in range(12)},
            },
        )
        assert (
            ImageRecord("ride_001-mp4-t-1", 10, 10).group_key
            == ImageRecord("ride_001-mp4-t-2", 10, 10).group_key
        ), "fixture assumption: both frames share a filename group"

        report = find_duplicates(records, assignment, images)
        regrouped = regroup_by_duplicates(records, report)
        by_id = {r.image_id: r for r in regrouped}

        # All three end up in one group: the two frames by filename, the twin
        # by hash, and the whole component transitively.
        assert (
            by_id["ride_001-mp4-t-1"].group_key
            == by_id["ride_001-mp4-t-2"].group_key
            == by_id["elsewhere"].group_key
        )

        new_assignment = split_records(regrouped, seed=0)
        verify_split(regrouped, new_assignment, require_all_splits=False)
        assert (
            new_assignment.of(by_id["ride_001-mp4-t-1"])
            is new_assignment.of(by_id["ride_001-mp4-t-2"])
            is new_assignment.of(by_id["elsewhere"])
        )

    def test_sharing_a_filename_group_alone_does_not_merge(self, tmp_path):
        # The second relation only pulls in records whose component already
        # contains a duplicate. Two frames of a clip with no twin anywhere keep
        # group=None and go on splitting by filename, as before.
        records, assignment, images = _setup(
            tmp_path,
            {"ride_002-mp4-t-1": (11, Split.TRAIN), "ride_002-mp4-t-2": (12, Split.TRAIN)},
        )
        report = find_duplicates(records, assignment, images)
        assert regroup_by_duplicates(records, report) == list(records)


def levels_image(path, levels):
    """Write an image whose 9x8 dHash thumbnail is exactly `levels`.

    Upscaled by an exact integer factor so dHash's own LANCZOS downsample
    recovers the block means, which makes the resulting hash predictable.
    """
    img = Image.new("L", (9, 8))
    img.putdata([v for row in levels for v in row])
    img.resize((9 * 40, 8 * 40), Image.Resampling.NEAREST).save(path)
    return path


def gradient(descents=()):
    """A high-contrast image that ramps left to right.

    Every adjacent comparison is `left > right` = False, so the hash is 0 bits
    — *not* because the picture is empty but because it is monotonic. Three real
    images in the dataset look like this (popcount 4, 5 and 6), which is exactly
    why hash popcount cannot be used to detect degeneracy. Each entry in
    `descents` forces one comparison true, setting one bit.
    """
    rows = [[min(255, c * 30) for c in range(9)] for _ in range(8)]
    for r, c in descents:
        rows[r][c], rows[r][c + 1] = 255, 0
    return rows


def flat_image(path, level=0):
    """A degenerate frame: every pixel identical, so the thumbnail has no range."""
    Image.new("L", (360, 320), level).save(path)
    return path


class TestDegenerateBridge:
    """A blank frame must not weld unrelated pictures into one component.

    Reproduces the real failure found in the 3,105-image rebuild: a pure black
    dashcam frame (`stddev 0.00`, hash 0/64) sat within 5 bits of two unrelated
    high-contrast images, and union-find merged a European clip, an Indian clip
    and 49 stills into one 514-image group.

    A and C below are *normal* pictures — thumbnail range 238 and 239 — that are
    10 bits apart from each other. Only the blank frame connects them.
    """

    def _bridge(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        levels_image(images / "eu_like.png", gradient([(0, 0), (0, 2), (0, 4), (0, 6), (1, 0)]))
        flat_image(images / "blank.png")
        levels_image(images / "in_like.png", gradient([(5, 0), (5, 2), (5, 4), (5, 6), (6, 0)]))
        records = [
            ImageRecord(image_id=n, width=360, height=320, file_name=f"{n}.png")
            for n in ("eu_like", "blank", "in_like")
        ]
        assignment = SplitAssignment(
            by_group={"eu_like": Split.TRAIN, "blank": Split.TRAIN, "in_like": Split.TEST},
            seed=0,
            ratios={Split.TRAIN: 1.0},
        )
        return records, assignment, images

    def test_the_fixture_really_is_a_bridge(self, tmp_path):
        # Guards the test itself: if these distances drift, the scenario below
        # stops testing anything.
        records, _, images = self._bridge(tmp_path)
        h = {r.image_id: dhash(images / r.file_name) for r in records}
        assert hamming(h["eu_like"], h["blank"]) <= DEFAULT_THRESHOLD
        assert hamming(h["blank"], h["in_like"]) <= DEFAULT_THRESHOLD
        assert hamming(h["eu_like"], h["in_like"]) > DEFAULT_THRESHOLD

    def test_unrelated_images_do_not_merge_through_a_blank_frame(self, tmp_path):
        records, assignment, images = self._bridge(tmp_path)
        report = find_duplicates(records, assignment, images)
        clusters = duplicate_clusters(report)
        assert clusters.get("eu_like") != clusters.get("in_like") or not clusters, (
            "a blank frame welded two unrelated images into one component"
        )

    def test_the_blank_frame_produces_no_approximate_edges(self, tmp_path):
        records, assignment, images = self._bridge(tmp_path)
        report = find_duplicates(records, assignment, images)
        approx = [p for p in report.pairs if p.distance > 0]
        assert not any("blank" in (p.left, p.right) for p in approx)

    def test_the_degenerate_image_is_reported(self, tmp_path):
        records, assignment, images = self._bridge(tmp_path)
        assert find_duplicates(records, assignment, images).degenerate == ["blank"]

    def test_it_still_counts_as_hashed(self, tmp_path):
        # Excluded from approximate matching, not dropped from the dataset.
        records, assignment, images = self._bridge(tmp_path)
        assert find_duplicates(records, assignment, images).images_hashed == 3


class TestDuplicatesSurviveTheDegenerateRule:
    """The rule must not cost genuine duplicate detection."""

    def _run(self, tmp_path, build):
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        names = build(images)
        records = [
            ImageRecord(image_id=n, width=360, height=320, file_name=f"{n}.png") for n in names
        ]
        assignment = SplitAssignment(
            by_group={n: Split.TRAIN for n in names}, seed=0, ratios={Split.TRAIN: 1.0}
        )
        return find_duplicates(records, assignment, images)

    def test_1_exact_identical_normal_images_are_duplicates(self, tmp_path):
        def build(d):
            photo(d / "a.png", seed=11)
            Image.open(d / "a.png").save(d / "b.png")
            return ["a", "b"]

        report = self._run(tmp_path, build)
        assert len(report.pairs) == 1 and report.pairs[0].distance == 0

    def test_2_near_identical_normal_images_are_duplicates(self, tmp_path):
        def build(d):
            photo(d / "a.png", seed=12)
            ImageEnhance.Brightness(Image.open(d / "a.png")).enhance(1.15).save(d / "b.png")
            return ["a", "b"]

        report = self._run(tmp_path, build)
        assert report.pairs, "a mild brightness shift stopped being a duplicate"

    def test_3_exact_identical_degenerate_images_still_group(self, tmp_path):
        # Two copies of the same blank frame genuinely are the same picture, and
        # exact equality is decidable without approximate matching.
        def build(d):
            flat_image(d / "a.png")
            flat_image(d / "b.png")
            return ["a", "b"]

        report = self._run(tmp_path, build)
        assert len(report.pairs) == 1
        assert report.pairs[0].distance == 0
        assert sorted(report.degenerate) == ["a", "b"]

    def test_4_many_identical_degenerate_images_form_one_cluster(self, tmp_path):
        def build(d):
            for i in range(4):
                flat_image(d / f"z{i}.png")
            return [f"z{i}" for i in range(4)]

        report = self._run(tmp_path, build)
        clusters = duplicate_clusters(report)
        assert len(set(clusters.values())) == 1
        assert len(clusters) == 4

    def test_5_blank_image_does_not_match_a_smooth_image(self, tmp_path):
        def build(d):
            flat_image(d / "blank.png")
            levels_image(d / "smooth.png", gradient())
            return ["blank", "smooth"]

        report = self._run(tmp_path, build)
        assert report.pairs == [], "a blank frame matched an unrelated smooth image"

    def test_6_low_information_image_does_not_match_an_unrelated_one(self, tmp_path):
        def build(d):
            flat_image(d / "blank.png", level=128)
            photo(d / "real.png", seed=21)
            return ["blank", "real"]

        assert self._run(tmp_path, build).pairs == []

    def test_7_unrelated_normal_images_are_not_duplicates(self, tmp_path):
        def build(d):
            photo(d / "a.png", seed=31)
            photo(d / "b.png", seed=97)
            return ["a", "b"]

        assert self._run(tmp_path, build).pairs == []

    def test_8_degenerate_between_two_unrelated_normals_forms_no_component(self, tmp_path):
        def build(d):
            levels_image(d / "left.png", gradient([(0, 0), (0, 2), (0, 4), (0, 6), (1, 0)]))
            flat_image(d / "mid.png")
            levels_image(d / "right.png", gradient([(5, 0), (5, 2), (5, 4), (5, 6), (6, 0)]))
            return ["left", "mid", "right"]

        report = self._run(tmp_path, build)
        clusters = duplicate_clusters(report)
        assert clusters.get("left") != clusters.get("right") or not clusters

    def test_a_normal_monotonic_image_keeps_its_duplicates(self, tmp_path):
        # The trap: this image's hash has 0 bits set, exactly like a blank
        # frame, but it is a real high-contrast picture. Three such images exist
        # in the real dataset. It must keep normal duplicate behaviour.
        def build(d):
            levels_image(d / "a.png", gradient())
            levels_image(d / "b.png", gradient())
            return ["a", "b"]

        report = self._run(tmp_path, build)
        assert report.degenerate == []
        assert len(report.pairs) == 1


class TestRegionAwareApproximateMatching:
    """Approximate matching is held to a tighter bound across regions.

    Same region keeps the threshold of 5. Different regions allow 2. Exact hash
    equality is untouched and still crosses regions, because two datasets built
    independently really can share a photograph.

    Distances are constructed, not sampled: `gradient()` ramps left to right so
    every comparison is False and the hash is 0, and each descent flips exactly
    one bit. An image with `n` descents therefore sits exactly `n` bits from a
    plain gradient — which is how each case below pins its distance.
    """

    def _pair(self, tmp_path, descents, left_region, right_region):
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        levels_image(images / "left.png", gradient())
        levels_image(images / "right.png", gradient(descents))
        records = [
            ImageRecord(
                image_id=name,
                width=360,
                height=320,
                file_name=f"{name}.png",
                boxes=(PlateBox(0.5, 0.5, 0.2, 0.1, region=region),),
            )
            for name, region in (("left", left_region), ("right", right_region))
        ]
        assignment = SplitAssignment(
            by_group={"left": Split.TRAIN, "right": Split.TEST},
            seed=0,
            ratios={Split.TRAIN: 1.0},
        )
        report = find_duplicates(records, assignment, images)
        distance = hamming(dhash(images / "left.png"), dhash(images / "right.png"))
        return distance, report

    # One descent per row, each flipping exactly one bit. Verified: prefixes of
    # length 1..5 sit exactly 1..5 bits from a plain gradient. Descents packed
    # into a single row interfere through the LANCZOS thumbnail and flip more
    # bits than they set, so each case asserts its measured distance.
    _D = [(row, 0) for row in range(5)]

    def test_1_same_region_exact_is_grouped(self, tmp_path):
        distance, report = self._pair(tmp_path, [], Region.EUROPE, Region.EUROPE)
        assert distance == 0
        assert len(report.pairs) == 1

    def test_2_same_region_approximate_at_4_is_grouped(self, tmp_path):
        distance, report = self._pair(tmp_path, self._D[:4], Region.EUROPE, Region.EUROPE)
        assert distance == 4
        assert len(report.pairs) == 1, "the same-region threshold of 5 must not have changed"

    def test_3_cross_region_exact_is_grouped(self, tmp_path):
        # The protection that must survive: one photograph in both datasets.
        distance, report = self._pair(tmp_path, [], Region.EUROPE, Region.INDIA)
        assert distance == 0
        assert len(report.pairs) == 1

    def test_4_cross_region_approximate_at_1_is_grouped(self, tmp_path):
        # A re-encoded or brightness-shifted copy of one photograph lands here.
        distance, report = self._pair(tmp_path, self._D[:1], Region.EUROPE, Region.INDIA)
        assert distance == 1
        assert len(report.pairs) == 1

    def test_5_cross_region_approximate_at_2_is_grouped(self, tmp_path):
        distance, report = self._pair(tmp_path, self._D[:2], Region.EUROPE, Region.INDIA)
        assert distance == 2
        assert len(report.pairs) == 1

    def test_6_cross_region_at_3_is_not_grouped(self, tmp_path):
        distance, report = self._pair(tmp_path, self._D[:3], Region.EUROPE, Region.INDIA)
        assert distance == 3
        assert report.pairs == []

    def test_7_cross_region_at_5_is_not_grouped(self, tmp_path):
        distance, report = self._pair(tmp_path, self._D[:5], Region.EUROPE, Region.INDIA)
        assert distance == 5
        assert report.pairs == []

    def test_10_the_real_cross_region_false_positive(self, tmp_path):
        """Regression for the pair found in the 3,105-image rebuild.

        `eu-valid-d_license_plate_206` (a German plate crop) sat 5 bits from
        `in-train-video3_1090` (an Indian street scene). Visually unrelated,
        different dimensions, different hashes — a coincidence at the threshold.
        It merged an 81-image cross-region group and pushed Indian validation to
        19.1% against a 15% target.

        Reconstructed structurally rather than by shipping the real images: the
        repository fakes heavy inputs and keeps plate imagery out of git.
        """
        distance, report = self._pair(tmp_path, self._D[:5], Region.EUROPE, Region.INDIA)
        assert distance == DEFAULT_THRESHOLD
        assert report.pairs == [], "the real-data false positive would still merge"

    def test_unknown_region_falls_back_to_the_permissive_bound(self, tmp_path):
        # A box-less background image carries no region. Absence of evidence
        # must not tighten the rule; 9 such records exist in the real dataset
        # and none currently take part in an edge.
        distance, report = self._pair(tmp_path, self._D[:4], Region.UNKNOWN, Region.INDIA)
        assert distance == 4
        assert len(report.pairs) == 1

    def test_two_unknown_regions_also_use_the_permissive_bound(self, tmp_path):
        distance, report = self._pair(tmp_path, self._D[:4], Region.UNKNOWN, Region.UNKNOWN)
        assert distance == 4
        assert len(report.pairs) == 1

    def test_threshold_zero_still_means_exact_only(self, tmp_path):
        # The cross-region bound is clamped to `threshold`, so it can never
        # loosen a caller who asked for exact matches.
        images = tmp_path / "images"
        images.mkdir(exist_ok=True)
        levels_image(images / "left.png", gradient())
        levels_image(images / "right.png", gradient(self._D[:1]))
        records = [
            ImageRecord(
                image_id=n,
                width=360,
                height=320,
                file_name=f"{n}.png",
                boxes=(PlateBox(0.5, 0.5, 0.2, 0.1, region=Region.EUROPE),),
            )
            for n in ("left", "right")
        ]
        assignment = SplitAssignment(
            by_group={"left": Split.TRAIN, "right": Split.TEST}, seed=0, ratios={Split.TRAIN: 1.0}
        )
        assert find_duplicates(records, assignment, images, threshold=0).pairs == []


class TestCleanSubset:
    def test_drops_only_contaminated_images(self, tmp_path):
        records, assignment, images = _setup(
            tmp_path,
            {"tr": (5, Split.TRAIN), "te_dirty": (5, Split.TEST), "te_clean": (7, Split.TEST)},
        )
        report = find_duplicates(records, assignment, images)
        test_records = [r for r in records if assignment.of(r) is Split.TEST]

        clean = clean_subset(test_records, report, Split.TEST)
        assert [r.image_id for r in clean] == ["te_clean"]

"""Ingest adapters."""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.data import DatasetError, Region
from alpr.data.ingest import (
    from_roboflow_export,
    from_yolo_dir,
    image_size,
    parse_yolo_label,
)


class TestParseYoloLabel:
    def test_single_box(self):
        boxes, dropped = parse_yolo_label("0 0.5 0.5 0.2 0.1\n", keep_classes=None)
        assert len(boxes) == 1
        assert boxes[0].cx == 0.5
        assert dropped == 0

    def test_blank_lines_ignored(self):
        boxes, _ = parse_yolo_label("\n0 0.5 0.5 0.2 0.1\n\n", keep_classes=None)
        assert len(boxes) == 1

    def test_extra_fields_tolerated(self):
        # Some exporters append a confidence column.
        boxes, _ = parse_yolo_label("0 0.5 0.5 0.2 0.1 0.98\n", keep_classes=None)
        assert len(boxes) == 1

    def test_drops_unwanted_classes(self):
        text = "0 0.5 0.5 0.2 0.1\n1 0.5 0.5 0.6 0.6\n"
        boxes, dropped = parse_yolo_label(text, keep_classes=frozenset({0}))
        assert len(boxes) == 1
        assert dropped == 1

    def test_too_few_fields_raises_with_line_number(self):
        with pytest.raises(DatasetError, match="line 2: expected 5 fields"):
            parse_yolo_label("0 0.5 0.5 0.2 0.1\n0 0.5 0.5\n", keep_classes=None)

    def test_non_numeric_raises_with_line_number(self):
        with pytest.raises(DatasetError, match="line 1"):
            parse_yolo_label("0 abc 0.5 0.2 0.1\n", keep_classes=None)

    def test_out_of_range_box_raises(self):
        with pytest.raises(DatasetError, match="line 1"):
            parse_yolo_label("0 0.5 0.5 0.0 0.1\n", keep_classes=None)


def _make_yolo_dataset(tmp_path, n=4, label_body="0 0.5 0.5 0.2 0.1\n", size=(640, 480)):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for i in range(n):
        Image.new("RGB", size).save(images / f"img{i}.jpg")
        (labels / f"img{i}.txt").write_text(label_body)
    return images, labels


class TestImageSize:
    def test_reads_dimensions(self, tmp_path):
        path = tmp_path / "a.png"
        Image.new("RGB", (321, 123)).save(path)
        assert image_size(path) == (321, 123)


class TestFromYoloDir:
    def test_ingests_images_and_labels(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="test")
        assert len(report.records) == 4
        assert all(r.width == 640 and r.height == 480 for r in report.records)
        assert all(r.source == "test" for r in report.records)

    def test_defaults_to_sibling_labels_dir(self, tmp_path):
        images, _ = _make_yolo_dataset(tmp_path)
        assert len(from_yolo_dir(images, source="test").records) == 4

    def test_applies_region(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t", region=Region.GERMANY)
        assert all(b.region is Region.GERMANY for r in report.records for b in r.boxes)

    def test_id_prefix_prevents_cross_dataset_collisions(self, tmp_path):
        # Two sources both containing img0.jpg must not collapse to one record.
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t", id_prefix="uc3m-")
        assert all(r.image_id.startswith("uc3m-") for r in report.records)

    def test_missing_labels_are_reported_not_silently_dropped(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img0.txt").unlink()
        report = from_yolo_dir(images, labels, source="t")
        assert report.skipped_no_label == ["img0"]
        assert len(report.records) == 3

    def test_strict_mode_raises_on_bad_label(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img1.txt").write_text("0 0.5 0.5\n")
        with pytest.raises(DatasetError, match="expected 5 fields"):
            from_yolo_dir(images, labels, source="t")

    def test_non_strict_records_bad_labels(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        (labels / "img1.txt").write_text("0 0.5 0.5\n")
        report = from_yolo_dir(images, labels, source="t", strict=False)
        assert len(report.records) == 3
        assert report.skipped_bad_label[0][0] == "img1"
        assert "bad label" in report.summary()

    def test_drops_vehicle_class(self, tmp_path):
        images, labels = _make_yolo_dataset(
            tmp_path, label_body="0 0.5 0.5 0.2 0.1\n1 0.5 0.5 0.8 0.8\n"
        )
        report = from_yolo_dir(images, labels, source="t", keep_classes=frozenset({0}))
        assert report.dropped_boxes == 4
        assert all(len(r.boxes) == 1 for r in report.records)
        assert "dropped boxes" in report.summary()

    def test_each_still_is_its_own_group_by_default(self, tmp_path):
        # Unrelated stills must not be lumped into one group, or the split
        # would put the whole dataset in a single bucket.
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t")
        assert len({r.group_key for r in report.records}) == 4

    def test_frame_grouping_when_requested(self, tmp_path):
        images = tmp_path / "images"
        labels = tmp_path / "labels"
        images.mkdir()
        labels.mkdir()
        for clip in range(2):
            for frame in range(3):
                name = f"clip{clip}_frame_{frame:03d}"
                Image.new("RGB", (100, 100)).save(images / f"{name}.jpg")
                (labels / f"{name}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
        report = from_yolo_dir(images, labels, source="t", group_by_stem=True)
        assert len({r.group_key for r in report.records}) == 2

    def test_background_image_with_empty_label(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path, label_body="")
        report = from_yolo_dir(images, labels, source="t")
        assert len(report.records) == 4
        assert all(r.boxes == () for r in report.records)

    def test_file_name_is_relative_to_images_dir_by_default(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        report = from_yolo_dir(images, labels, source="t")
        assert sorted(r.file_name for r in report.records)[0] == "img0.jpg"

    def test_path_root_makes_paths_composable_across_sources(self, tmp_path):
        # Regression: merging two sources into one manifest and exporting
        # against their shared parent failed, because each file_name was
        # relative to its own images_dir and so resolved to the wrong file.
        raw = tmp_path / "raw"
        for source in ("srcA", "srcB"):
            (raw / source).mkdir(parents=True)
            _make_yolo_dataset(raw / source, n=2)

        records = []
        for source in ("srcA", "srcB"):
            records += from_yolo_dir(
                raw / source / "images",
                source=source,
                id_prefix=f"{source}-",
                path_root=raw,
            ).records

        # Every recorded path resolves to a real file under the shared root.
        assert len(records) == 4
        for record in records:
            assert (raw / record.file_name).exists(), record.file_name
        assert len({r.file_name for r in records}) == 4

    def test_path_root_must_contain_images_dir(self, tmp_path):
        images, labels = _make_yolo_dataset(tmp_path)
        with pytest.raises(DatasetError, match="not a parent of"):
            from_yolo_dir(images, labels, source="t", path_root=tmp_path / "elsewhere")

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="images_dir does not exist"):
            from_yolo_dir(tmp_path / "nope", source="t")

    def test_missing_labels_directory_raises(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        with pytest.raises(DatasetError, match="labels_dir does not exist"):
            from_yolo_dir(images, source="t")


def _make_roboflow_export(tmp_path, name="export", splits=("train", "valid", "test"), n=3):
    """Mimic a Roboflow YOLO export: one directory per split."""
    location = tmp_path / name
    for split in splits:
        images = location / split / "images"
        labels = location / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(n):
            stem = f"{split}_img{i}"
            Image.new("RGB", (640, 480)).save(images / f"{stem}.jpg")
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    return location


# Verbatim filenames from the project's own surviving identifiers
# (`labels/index.json`), minus the `<source>-<dir>-` prefix that ingest adds
# back. Synthetic stand-ins would not have caught the bug these guard: the
# `_jpg.rf.<32-hex>` suffix is exactly what stopped the frame rule matching.
REAL_DAYRIDE = [
    "dayride_type1_001-mp4-t-451_jpg.rf.570aecd2ac67cc8efe361d014d8b4a0f",
    "dayride_type1_001-mp4-t-491_jpg.rf.bca97daf1eb0a40df5a3ef85d6f03599",
    "dayride_type1_001-mp4-t-751_jpg.rf.5c6687a87b9a7a5a3cff5183d1ab1730",
    "dayride_type1_001-mp4-t-1115_jpg.rf.0e19cdbaf4ebcd9807f9821253e88a63",
]
REAL_NIGHTRIDE = [
    "nightride_type3_001-mp4-t-256_jpg.rf.5590acf9db420e97fd1d4fa81ccb14f8",
    "nightride_type3_001-mp4-t-540_jpg.rf.074a05f85482dd46241f83b1e3b07259",
]
REAL_VIDEO3 = [
    "video3_640_jpg.rf.94139d382e35981bc62633d868c43c87",
    "video3_1220_jpg.rf.6425a54377facd894a0311dc4daf412b",
    "video3_2190_jpg.rf.f9a4bf94d32d92ec52483139a3029050",
]
REAL_STILLS = [
    "d_license_plate_188_jpg.rf.487f425dd3a99b3d42ac72be15690f65",
    "pl_license_plate_327_jpg.rf.dcd2b7a1c3c0296310c252682e78e48f",
    "car-wbs-KL54H369_00000_jpeg.rf.ec38c1bf7ae1c0a7d5a550a0c2e59a47",
]


def _export_with(tmp_path, by_split, name="export"):
    """Build a Roboflow-shaped export from explicit `{split: [stem, ...]}`."""
    location = tmp_path / name
    for split, stems in by_split.items():
        images = location / split / "images"
        labels = location / split / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            Image.new("RGB", (640, 480)).save(images / f"{stem}.jpg")
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.1\n")
    return location


class TestFromRoboflowExport:
    def test_pools_every_split(self, tmp_path):
        # Roboflow's own split is discarded — we re-split ourselves, grouped
        # and stratified, so all three directories become one pool.
        location = _make_roboflow_export(tmp_path)
        report = from_roboflow_export(location, source="rf")
        assert len(report.records) == 9

    def test_handles_val_as_well_as_valid(self, tmp_path):
        location = _make_roboflow_export(tmp_path, splits=("train", "val"))
        assert len(from_roboflow_export(location, source="rf").records) == 6

    def test_tolerates_a_missing_split(self, tmp_path):
        location = _make_roboflow_export(tmp_path, splits=("train",))
        assert len(from_roboflow_export(location, source="rf").records) == 3

    def test_video_frames_in_one_directory_share_a_group(self, tmp_path):
        location = _export_with(tmp_path, {"train": REAL_DAYRIDE[:2]})
        records = from_roboflow_export(location, source="rf").records
        assert len({r.group_key for r in records}) == 1

    def test_video_frames_across_directories_share_a_group(self, tmp_path):
        # The leak this whole change exists to close: Roboflow scatters frames
        # of one clip across its own train/valid/test directories, and the
        # image id has to carry the directory to keep stems unique. Grouping on
        # the id therefore split the clip in two.
        location = _export_with(
            tmp_path,
            {"train": REAL_DAYRIDE[:2], "valid": REAL_DAYRIDE[2:3], "test": REAL_DAYRIDE[3:4]},
        )
        records = from_roboflow_export(location, source="rf", id_prefix="eu-").records
        assert len(records) == 4
        assert len({r.group_key for r in records}) == 1
        assert records[0].group_key == "eu-dayride_type1_001"
        # The upstream directory survives for provenance, just not in the group.
        assert {r.meta["roboflow_split"] for r in records} == {"train", "valid", "test"}

    def test_video_n_frames_across_directories_share_a_group(self, tmp_path):
        location = _export_with(tmp_path, {"train": REAL_VIDEO3[:1], "test": REAL_VIDEO3[1:]})
        records = from_roboflow_export(location, source="rf", id_prefix="in-").records
        assert len({r.group_key for r in records}) == 1
        assert records[0].group_key == "in-video3"

    def test_two_clips_do_not_merge(self, tmp_path):
        location = _export_with(
            tmp_path, {"train": REAL_DAYRIDE[:2] + REAL_NIGHTRIDE[:2] + REAL_VIDEO3[:1]}
        )
        records = from_roboflow_export(location, source="rf").records
        assert len({r.group_key for r in records}) == 3

    def test_unrelated_stills_still_get_their_own_group(self, tmp_path):
        # Stills carry no evidence that two of them are related, so ingest does
        # not claim they are. Only filenames that announce themselves as video
        # frames get a shared group.
        location = _export_with(tmp_path, {"train": REAL_STILLS, "test": REAL_STILLS[:1]})
        records = from_roboflow_export(location, source="rf").records
        assert len({r.group_key for r in records}) == len(records)
        assert all(r.group == r.image_id for r in records)

    def test_grouping_can_be_pinned_to_the_image_id(self, tmp_path):
        # The escape hatch that reproduces the old, leak-prone grouping.
        location = _export_with(tmp_path, {"train": REAL_DAYRIDE[:1], "test": REAL_DAYRIDE[1:2]})
        records = from_roboflow_export(location, source="rf", group_by_clip=False).records
        assert all(r.group == r.image_id for r in records)
        assert len({r.group_key for r in records}) == 2

    def test_paths_resolve_under_the_shared_root(self, tmp_path):
        location = _make_roboflow_export(tmp_path)
        report = from_roboflow_export(location, source="rf")
        for record in report.records:
            assert (tmp_path / record.file_name).exists(), record.file_name

    def test_same_stem_in_two_splits_does_not_collide(self, tmp_path):
        # Nothing guarantees stems are unique across train/valid/test. A
        # collision would only surface as a "duplicate image_id" abort from
        # write_manifest — after the whole dataset had been downloaded.
        location = tmp_path / "export"
        for split in ("train", "valid", "test"):
            images = location / split / "images"
            labels = location / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            Image.new("RGB", (100, 100)).save(images / "img_0001.jpg")
            (labels / "img_0001.txt").write_text("0 0.5 0.5 0.2 0.1\n")

        report = from_roboflow_export(location, source="rf")
        assert len(report.records) == 3
        assert len({r.image_id for r in report.records}) == 3

    def test_records_keep_the_upstream_split_as_provenance(self, tmp_path):
        location = _make_roboflow_export(tmp_path)
        report = from_roboflow_export(location, source="rf")
        assert {r.meta["roboflow_split"] for r in report.records} == {"train", "valid", "test"}

    def test_two_exports_compose_without_collision(self, tmp_path):
        a = _make_roboflow_export(tmp_path, name="eu")
        b = _make_roboflow_export(tmp_path, name="in")
        records = [
            *from_roboflow_export(a, source="eu", id_prefix="eu-").records,
            *from_roboflow_export(b, source="in", id_prefix="in-").records,
        ]
        assert len({r.image_id for r in records}) == 18
        assert len({r.file_name for r in records}) == 18
        for record in records:
            assert (tmp_path / record.file_name).exists()

    def test_applies_region_tag(self, tmp_path):
        location = _make_roboflow_export(tmp_path, splits=("train",))
        report = from_roboflow_export(location, source="rf", region=Region.EUROPE)
        assert all(b.region is Region.EUROPE for r in report.records for b in r.boxes)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(DatasetError, match="does not exist"):
            from_roboflow_export(tmp_path / "nope", source="rf")

    def test_export_without_split_dirs_raises(self, tmp_path):
        # A flat YOLO dump is a different shape; fail clearly instead of
        # silently ingesting nothing.
        empty = tmp_path / "flat"
        (empty / "images").mkdir(parents=True)
        with pytest.raises(DatasetError, match="no split directories"):
            from_roboflow_export(empty, source="rf")

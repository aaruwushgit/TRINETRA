"""YOLO export."""

from __future__ import annotations

import pytest

from alpr.data import (
    DatasetError,
    ImageRecord,
    PlateBox,
    Split,
    export_yolo,
    format_label_file,
    split_records,
)


class TestLabelFormatting:
    def test_single_box(self):
        record = ImageRecord("a", 100, 100, boxes=(PlateBox(0.5, 0.5, 0.2, 0.1),))
        assert format_label_file(record) == "0 0.500000 0.500000 0.200000 0.100000\n"

    def test_background_image_gets_an_empty_file(self):
        # Ultralytics reads a zero-byte label as "no objects here", which is
        # exactly what a background image should teach.
        assert format_label_file(ImageRecord("a", 100, 100)) == ""

    def test_multiple_boxes_one_per_line(self):
        record = ImageRecord(
            "a",
            100,
            100,
            boxes=(PlateBox(0.2, 0.2, 0.1, 0.1), PlateBox(0.8, 0.8, 0.1, 0.1)),
        )
        assert len(format_label_file(record).strip().splitlines()) == 2

    def test_clips_out_of_frame_boxes(self):
        # Ultralytics drops images whose labels fall outside [0, 1].
        record = ImageRecord("a", 100, 100, boxes=(PlateBox(0.02, 0.5, 0.2, 0.1),))
        values = [float(v) for v in format_label_file(record).split()[1:]]
        cx, _, w, _ = values
        assert cx - w / 2 >= 0.0

    def test_clipping_can_be_disabled(self):
        record = ImageRecord("a", 100, 100, boxes=(PlateBox(0.02, 0.5, 0.2, 0.1),))
        values = [float(v) for v in format_label_file(record, clip=False).split()[1:]]
        assert values[0] == pytest.approx(0.02)


def _dataset(tmp_path, n_groups=12, frames=4, make_images=True):
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    records = []
    for g in range(n_groups):
        for f in range(frames):
            name = f"clip{g:02d}_frame_{f:03d}"
            if make_images:
                (image_root / f"{name}.jpg").write_bytes(b"\xff\xd8\xff")  # JPEG magic
            records.append(
                ImageRecord(
                    name,
                    640,
                    480,
                    boxes=(PlateBox(0.5, 0.5, 0.1, 0.05),),
                    file_name=f"{name}.jpg",
                )
            )
    return image_root, records


class TestExport:
    def test_builds_the_ultralytics_layout(self, tmp_path):
        image_root, records = _dataset(tmp_path)
        assignment = split_records(records)
        result = export_yolo(records, assignment, image_root, tmp_path / "out")

        for split in Split:
            assert (tmp_path / "out" / "images" / split.value).is_dir()
            assert (tmp_path / "out" / "labels" / split.value).is_dir()
        assert result.data_yaml.exists()
        assert sum(result.counts.values()) == len(records)

    def test_every_image_gets_a_label(self, tmp_path):
        image_root, records = _dataset(tmp_path)
        out = tmp_path / "out"
        export_yolo(records, split_records(records), image_root, out)

        for split in Split:
            images = sorted(p.stem for p in (out / "images" / split.value).iterdir())
            labels = sorted(p.stem for p in (out / "labels" / split.value).iterdir())
            assert images == labels

    def test_symlinks_by_default(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2)
        out = tmp_path / "out"
        export_yolo(records, split_records(records), image_root, out)
        linked = next((out / "images" / "train").iterdir())
        assert linked.is_symlink()
        assert linked.read_bytes() == b"\xff\xd8\xff"

    def test_symlinks_are_relative_so_the_tree_can_move(self, tmp_path):
        import os

        image_root, records = _dataset(tmp_path, n_groups=6, frames=2)
        out = tmp_path / "out"
        export_yolo(records, split_records(records), image_root, out)
        linked = next((out / "images" / "train").iterdir())
        assert not os.path.isabs(os.readlink(linked))

    def test_copy_mode(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2)
        out = tmp_path / "out"
        export_yolo(records, split_records(records), image_root, out, symlink=False)
        copied = next((out / "images" / "train").iterdir())
        assert not copied.is_symlink()
        assert copied.read_bytes() == b"\xff\xd8\xff"

    def test_rerun_overwrites_cleanly(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2)
        out = tmp_path / "out"
        assignment = split_records(records)
        export_yolo(records, assignment, image_root, out)
        export_yolo(records, assignment, image_root, out)  # must not raise
        total = sum(len(list((out / "images" / s.value).iterdir())) for s in Split)
        assert total == len(records)

    def test_missing_image_raises_by_default(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2, make_images=False)
        with pytest.raises(DatasetError, match="not found"):
            export_yolo(records, split_records(records), image_root, tmp_path / "out")

    def test_missing_images_reported_when_allowed(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2, make_images=False)
        result = export_yolo(
            records, split_records(records), image_root, tmp_path / "out", allow_missing=True
        )
        assert len(result.missing_images) == len(records)
        assert "WARNING" in result.summary()

    def test_data_yaml_declares_one_class(self, tmp_path):
        image_root, records = _dataset(tmp_path, n_groups=6, frames=2)
        result = export_yolo(records, split_records(records), image_root, tmp_path / "out")
        text = result.data_yaml.read_text()
        assert "0: license_plate" in text
        assert "train: images/train" in text
        assert "val: images/val" in text
        assert "test: images/test" in text

    def test_resolves_images_without_file_name(self, tmp_path):
        image_root = tmp_path / "images"
        image_root.mkdir()
        records = []
        for g in range(6):
            for f in range(2):
                name = f"clip{g}_frame_{f}"
                (image_root / f"{name}.png").write_bytes(b"\x89PNG")
                records.append(ImageRecord(name, 100, 100, boxes=(PlateBox(0.5, 0.5, 0.1, 0.1),)))
        result = export_yolo(records, split_records(records), image_root, tmp_path / "out")
        assert result.missing_images == []

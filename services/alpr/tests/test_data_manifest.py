"""Manifest I/O."""

from __future__ import annotations

import pytest

from alpr.data import (
    DatasetError,
    ImageRecord,
    PlateBox,
    Region,
    read_manifest,
    write_manifest,
)


def test_round_trip(tmp_path):
    records = [
        ImageRecord(
            f"img{i}",
            1920,
            1080,
            boxes=(PlateBox(0.5, 0.5, 0.1, 0.05, text=f"P{i}", region=Region.INDIA),),
            file_name=f"img{i}.jpg",
            source="test",
        )
        for i in range(5)
    ]
    path = tmp_path / "manifest.jsonl"
    assert write_manifest(records, path) == 5
    assert read_manifest(path) == records


def test_empty_manifest(tmp_path):
    path = tmp_path / "m.jsonl"
    assert write_manifest([], path) == 0
    assert read_manifest(path) == []


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "m.jsonl"
    write_manifest([ImageRecord("a", 10, 10)], path)
    path.write_text(path.read_text() + "\n\n")
    assert len(read_manifest(path)) == 1


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "m.jsonl"
    write_manifest([ImageRecord("a", 10, 10)], path)
    assert path.exists()


def test_malformed_json_names_the_line(tmp_path):
    # With 50k lines, "invalid JSON" without a line number is unactionable.
    path = tmp_path / "m.jsonl"
    path.write_text('{"image_id": "a", "width": 10, "height": 10}\nnot json\n')
    with pytest.raises(DatasetError, match=r":2: malformed JSON"):
        read_manifest(path)


def test_invalid_record_names_the_line(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        '{"image_id": "a", "width": 10, "height": 10}\n'
        '{"image_id": "b", "width": 0, "height": 10}\n'
    )
    with pytest.raises(DatasetError, match=r":2: bad image size"):
        read_manifest(path)


def test_duplicate_ids_rejected_on_write(tmp_path):
    records = [ImageRecord("same", 10, 10), ImageRecord("same", 10, 10)]
    with pytest.raises(DatasetError, match="duplicate image_id"):
        write_manifest(records, tmp_path / "m.jsonl")


def test_duplicate_ids_rejected_on_read(tmp_path):
    path = tmp_path / "m.jsonl"
    line = '{"image_id": "a", "width": 10, "height": 10}\n'
    path.write_text(line * 2)
    with pytest.raises(DatasetError, match="duplicate image_id"):
        read_manifest(path)


def test_failed_write_leaves_previous_manifest_intact(tmp_path):
    # An interrupted write must not destroy a good manifest.
    path = tmp_path / "m.jsonl"
    write_manifest([ImageRecord("good", 10, 10)], path)
    before = path.read_text()

    with pytest.raises(DatasetError):
        write_manifest([ImageRecord("dup", 10, 10), ImageRecord("dup", 10, 10)], path)

    assert path.read_text() == before
    assert not list(tmp_path.glob("*.tmp"))

"""Crop extraction and the labelling page."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from alpr.data.schema import ImageRecord, PlateBox
from alpr.label import (
    build_page,
    extract_crops,
    labelled_crops,
    load_labels,
    sample_records,
)


def dataset(tmp_path, n=12, widths=(0.02, 0.05, 0.10, 0.30)):
    """Records spanning every size bucket, with real images on disk."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    records = []
    for i in range(n):
        name = f"img{i:03d}.png"
        Image.new("RGB", (1000, 1000), (60, 60, 70)).save(images / name)
        width = widths[i % len(widths)]
        records.append(
            ImageRecord(
                image_id=f"img{i:03d}",
                width=1000,
                height=1000,
                boxes=(PlateBox(0.5, 0.5, width, width / 3),),
                file_name=name,
            )
        )
    return images, records


class TestSampleRecords:
    def test_respects_the_limit(self, tmp_path):
        _, records = dataset(tmp_path, n=20)
        assert len(sample_records(records, 8)) == 8

    def test_spreads_across_size_buckets(self, tmp_path):
        # 200 plates drawn at random would be mostly large and easy, and the
        # resulting accuracy would flatter the system.
        _, records = dataset(tmp_path, n=40)
        picked = sample_records(records, 8)
        widths = {r.boxes[i].pixel_size(r.width, r.height)[0] for r, i in picked}
        assert len(widths) >= 3, "should not draw only one size of plate"

    def test_is_deterministic(self, tmp_path):
        _, records = dataset(tmp_path, n=20)
        first = [(r.image_id, i) for r, i in sample_records(records, 6, seed=1)]
        second = [(r.image_id, i) for r, i in sample_records(records, 6, seed=1)]
        assert first == second

    def test_seed_changes_the_sample(self, tmp_path):
        _, records = dataset(tmp_path, n=40)
        a = [(r.image_id, i) for r, i in sample_records(records, 8, seed=1)]
        b = [(r.image_id, i) for r, i in sample_records(records, 8, seed=2)]
        assert a != b

    def test_limit_beyond_the_dataset(self, tmp_path):
        _, records = dataset(tmp_path, n=4)
        assert len(sample_records(records, 100)) == 4

    def test_empty_dataset(self):
        assert sample_records([], 10) == []


class TestExtractCrops:
    def test_writes_crops_and_an_index(self, tmp_path):
        images, records = dataset(tmp_path, n=8)
        out = tmp_path / "labels"
        refs = extract_crops(records, images, out, limit=5)

        assert len(refs) == 5
        assert len(list((out / "crops").iterdir())) == 5
        index = json.loads((out / "index.json").read_text())
        assert len(index) == 5
        assert {"crop_id", "image_id", "file_name", "width_px", "bucket"} <= set(index[0])

    def test_crops_are_padded_beyond_the_box(self, tmp_path):
        # A plate cut exactly to its box loses the quiet space around the
        # characters that both humans and OCR read better with.
        images, records = dataset(tmp_path, n=1, widths=(0.20,))
        out = tmp_path / "labels"
        extract_crops(records, images, out, limit=1, padding=0.5)
        crop = Image.open(next((out / "crops").iterdir()))
        assert crop.width > 200  # 0.20 * 1000 without padding

    def test_skips_records_with_missing_images(self, tmp_path):
        images, records = dataset(tmp_path, n=4)
        for path in images.iterdir():
            path.unlink()
        assert extract_crops(records, images, tmp_path / "labels", limit=4) == []

    def test_crop_ids_identify_the_box(self, tmp_path):
        images, records = dataset(tmp_path, n=3)
        refs = extract_crops(records, images, tmp_path / "labels", limit=3)
        assert all("#" in r.crop_id for r in refs)


class TestBuildPage:
    def test_writes_a_self_contained_page(self, tmp_path):
        images, records = dataset(tmp_path, n=4)
        out = tmp_path / "labels"
        extract_crops(records, images, out, limit=4)
        page = build_page(out)

        html = page.read_text()
        # Crops embedded, so the file works from file:// after being
        # downloaded off a Colab runtime.
        assert "data:image/png;base64," in html
        assert html.count("<input") == 4
        assert "localStorage" in html

    def test_page_has_no_external_references(self, tmp_path):
        images, records = dataset(tmp_path, n=2)
        out = tmp_path / "labels"
        extract_crops(records, images, out, limit=2)
        html = build_page(out).read_text()
        assert "http://" not in html and "https://" not in html

    def test_crop_ids_are_escaped_into_the_page(self, tmp_path):
        images, records = dataset(tmp_path, n=2)
        out = tmp_path / "labels"
        refs = extract_crops(records, images, out, limit=2)
        html = build_page(out).read_text()
        assert refs[0].crop_id in html


class TestLoadLabels:
    def test_reads_a_flat_mapping(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"a#0": "mh12ab1234", "b#0": " daxy123 "}))
        labels = load_labels(path)
        assert labels == {"a#0": "MH12AB1234", "b#0": "DAXY123"}

    def test_drops_blanks(self, tmp_path):
        # A blank means "unreadable", not a plate whose text is empty.
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"a#0": "ABC123", "b#0": "   "}))
        assert list(load_labels(path)) == ["a#0"]

    def test_accepts_a_wrapped_payload(self, tmp_path):
        path = tmp_path / "labels.json"
        path.write_text(json.dumps({"labels": {"a#0": "ABC123"}}))
        assert load_labels(path) == {"a#0": "ABC123"}


class TestLabelledCrops:
    def test_pairs_crops_with_their_text(self, tmp_path):
        images, records = dataset(tmp_path, n=4)
        out = tmp_path / "labels"
        refs = extract_crops(records, images, out, limit=4)

        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps({refs[0].crop_id: "MH12AB1234"}))

        pairs = labelled_crops(out, labels)
        assert len(pairs) == 1
        assert pairs[0][0].exists()
        assert pairs[0][1] == "MH12AB1234"

    def test_unlabelled_crops_are_excluded(self, tmp_path):
        images, records = dataset(tmp_path, n=4)
        out = tmp_path / "labels"
        extract_crops(records, images, out, limit=4)
        labels = tmp_path / "labels.json"
        labels.write_text(json.dumps({}))
        assert labelled_crops(out, labels) == []


@pytest.mark.weights
def test_pairs_feed_the_cer_harness(tmp_path):
    """The point of the whole module: labels in, CER out."""
    from alpr.cer import score
    from alpr.ocr import PlateReader

    images, records = dataset(tmp_path, n=2)
    out = tmp_path / "labels"
    refs = extract_crops(records, images, out, limit=2)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({r.crop_id: "MH12AB1234" for r in refs}))

    reader = PlateReader()
    pairs = [(truth, reader.read_path(path).text) for path, truth in labelled_crops(out, labels)]
    assert score(pairs).samples == 2

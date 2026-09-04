"""Detection matching, sliced metrics, and the failure gallery."""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.data.schema import ImageRecord, PlateBox
from alpr.detect import Detection, iou
from alpr.evaluate import (
    SIZE_BUCKETS,
    Counts,
    bucket_for,
    evaluate_records,
    match,
    render_failure_gallery,
)


def det(x1, y1, x2, y2, confidence=0.9) -> Detection:
    return Detection(x1, y1, x2, y2, confidence)


class TestIou:
    def test_identical_boxes(self):
        assert iou((0.1, 0.1, 0.3, 0.3), (0.1, 0.1, 0.3, 0.3)) == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        assert iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        assert iou((0.0, 0.0, 0.1, 0.1), (0.1, 0.0, 0.2, 0.1)) == 0.0

    def test_half_overlap(self):
        # Two unit-ish boxes sharing half their area: 0.5 / 1.5.
        assert iou((0.0, 0.0, 0.2, 0.1), (0.1, 0.0, 0.3, 0.1)) == pytest.approx(1 / 3)

    def test_degenerate_box(self):
        assert iou((0.1, 0.1, 0.1, 0.1), (0.0, 0.0, 0.2, 0.2)) == 0.0


class TestMatch:
    def test_perfect_detection(self):
        result = match([(0.1, 0.1, 0.3, 0.2)], [det(0.1, 0.1, 0.3, 0.2)])
        assert result.counts.true_positives == 1
        assert result.counts.false_positives == 0
        assert result.counts.false_negatives == 0

    def test_missed_plate(self):
        result = match([(0.1, 0.1, 0.3, 0.2)], [])
        assert result.counts.false_negatives == 1
        assert result.unmatched_truth == [0]

    def test_false_positive(self):
        result = match([], [det(0.5, 0.5, 0.6, 0.6)])
        assert result.counts.false_positives == 1

    def test_poor_overlap_is_not_a_match(self):
        # Barely touching: below the 0.5 IoU threshold, so it is both a miss
        # and a false positive rather than a hit.
        result = match([(0.0, 0.0, 0.2, 0.1)], [det(0.15, 0.0, 0.35, 0.1)])
        assert result.counts.true_positives == 0
        assert result.counts.false_positives == 1
        assert result.counts.false_negatives == 1

    def test_one_truth_box_is_claimed_once(self):
        # Two overlapping predictions on one plate: one hit, one false
        # positive. Counting both as hits would inflate recall past 100%.
        result = match(
            [(0.1, 0.1, 0.3, 0.2)],
            [det(0.1, 0.1, 0.3, 0.2, 0.9), det(0.11, 0.11, 0.31, 0.21, 0.8)],
        )
        assert result.counts.true_positives == 1
        assert result.counts.false_positives == 1

    def test_highest_confidence_claims_first(self):
        # The greedy order matters: the confident prediction should take the
        # box, not whichever happens to come first in the list.
        result = match(
            [(0.1, 0.1, 0.3, 0.2)],
            [det(0.11, 0.11, 0.31, 0.21, 0.4), det(0.1, 0.1, 0.3, 0.2, 0.95)],
        )
        (prediction_index, _, _) = result.matched[0]
        assert prediction_index == 1

    def test_two_plates_two_detections(self):
        result = match(
            [(0.0, 0.0, 0.2, 0.1), (0.6, 0.6, 0.8, 0.7)],
            [det(0.6, 0.6, 0.8, 0.7), det(0.0, 0.0, 0.2, 0.1)],
        )
        assert result.counts.true_positives == 2

    def test_threshold_is_configurable(self):
        truth = [(0.0, 0.0, 0.2, 0.1)]
        loose = match(truth, [det(0.05, 0.0, 0.25, 0.1)], iou_threshold=0.4)
        strict = match(truth, [det(0.05, 0.0, 0.25, 0.1)], iou_threshold=0.9)
        assert loose.counts.true_positives == 1
        assert strict.counts.true_positives == 0


class TestCounts:
    def test_metrics(self):
        counts = Counts(true_positives=8, false_positives=2, false_negatives=2)
        assert counts.precision == pytest.approx(0.8)
        assert counts.recall == pytest.approx(0.8)
        assert counts.f1 == pytest.approx(0.8)

    def test_no_predictions_gives_zero_not_an_error(self):
        counts = Counts(false_negatives=5)
        assert counts.precision == 0.0
        assert counts.recall == 0.0
        assert counts.f1 == 0.0


class TestBuckets:
    @pytest.mark.parametrize(
        ("width", "expected"),
        [
            (10, "tiny (<32px)"),
            (40, "small (32-64px)"),
            (100, "medium (64-128px)"),
            (500, "large (>=128px)"),
        ],
    )
    def test_bucket_boundaries(self, width, expected):
        assert bucket_for(width) == expected

    def test_buckets_are_contiguous(self):
        # A gap would silently drop plates from every slice.
        edges = [(low, high) for _, low, high in SIZE_BUCKETS]
        # Deliberately ragged: each bucket is paired with its successor.
        for (_, high), (low, _) in zip(edges, edges[1:], strict=False):
            assert high == low
        assert edges[0][0] == 0.0
        assert edges[-1][1] == float("inf")


def record(image_id="img", boxes=(), width=1000, height=1000, file_name=None):
    return ImageRecord(
        image_id=image_id,
        width=width,
        height=height,
        boxes=tuple(boxes),
        file_name=file_name,
    )


class TestEvaluateRecords:
    def test_perfect_run(self):
        records = [record(boxes=[PlateBox(0.2, 0.2, 0.2, 0.1)])]
        report = evaluate_records(records, [[det(0.1, 0.15, 0.3, 0.25)]])
        assert report.overall.recall == pytest.approx(1.0)
        assert report.overall.precision == pytest.approx(1.0)

    def test_rejects_misaligned_inputs(self):
        # A silent misalignment produces plausible but meaningless numbers.
        with pytest.raises(ValueError, match="must be aligned"):
            evaluate_records([record()], [[], []])

    def test_slices_by_ground_truth_size(self):
        # A tiny plate missed, a large one found: overall recall hides which.
        records = [
            record("tiny", boxes=[PlateBox(0.5, 0.5, 0.02, 0.01)]),  # 20 px wide
            record("large", boxes=[PlateBox(0.5, 0.5, 0.30, 0.10)]),  # 300 px
        ]
        report = evaluate_records(records, [[], [det(0.35, 0.45, 0.65, 0.55)]])

        assert report.by_size["tiny (<32px)"].recall == 0.0
        assert report.by_size["large (>=128px)"].recall == pytest.approx(1.0)
        assert report.overall.recall == pytest.approx(0.5)

    def test_report_mentions_populated_buckets_only(self):
        records = [record(boxes=[PlateBox(0.5, 0.5, 0.3, 0.1)])]
        text = evaluate_records(records, [[det(0.35, 0.45, 0.65, 0.55)]]).report()
        assert "large (>=128px)" in text
        assert "tiny (<32px)" not in text

    def test_worst_ranks_by_error_count(self):
        records = [
            record("clean", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)]),
            record("bad", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)]),
        ]
        report = evaluate_records(
            records,
            [
                [det(0.4, 0.45, 0.6, 0.55)],  # correct
                [det(0.01, 0.01, 0.05, 0.05), det(0.9, 0.9, 0.95, 0.95)],  # 1 miss + 2 FP
            ],
        )
        assert report.worst(1)[0].image_id == "bad"
        assert report.worst(1)[0].errors == 3

    def test_image_with_no_plates_and_no_detections(self):
        report = evaluate_records([record(boxes=[])], [[]])
        assert report.overall.ground_truth == 0
        assert report.overall.false_positives == 0


class TestFailureGallery:
    def test_renders_only_failing_images(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        for name in ("good", "bad"):
            Image.new("RGB", (640, 480), (30, 30, 30)).save(images / f"{name}.jpg")

        records = [
            record("good", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)], file_name="good.jpg"),
            record("bad", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)], file_name="bad.jpg"),
        ]
        predictions = [[det(0.4, 0.45, 0.6, 0.55)], []]
        report = evaluate_records(records, predictions)

        written = render_failure_gallery(report, records, predictions, images, tmp_path / "gallery")
        assert len(written) == 1
        assert "bad" in written[0].name
        assert written[0].exists()

    def test_missing_source_image_is_skipped(self, tmp_path):
        records = [record("gone", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)], file_name="gone.jpg")]
        report = evaluate_records(records, [[]])
        written = render_failure_gallery(
            report, records, [[]], tmp_path / "images", tmp_path / "gallery"
        )
        assert written == []

    def test_limit_is_respected(self, tmp_path):
        images = tmp_path / "images"
        images.mkdir()
        records, predictions = [], []
        for i in range(5):
            Image.new("RGB", (320, 240), (20, 20, 20)).save(images / f"i{i}.jpg")
            records.append(
                record(f"i{i}", boxes=[PlateBox(0.5, 0.5, 0.2, 0.1)], file_name=f"i{i}.jpg")
            )
            predictions.append([])
        report = evaluate_records(records, predictions)
        written = render_failure_gallery(
            report, records, predictions, images, tmp_path / "g", limit=2
        )
        assert len(written) == 2

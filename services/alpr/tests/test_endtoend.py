"""End-to-end evaluation and failure attribution."""

from __future__ import annotations

import pytest
from PIL import Image

from alpr.data.schema import ImageRecord, PlateBox
from alpr.detect import Detection
from alpr.endtoend import (
    EndToEndReport,
    Outcome,
    PlateOutcome,
    best_detection_for,
    classify,
    evaluate_end_to_end,
)
from alpr.ocr import OcrResult


class TestClassify:
    def test_correct(self):
        assert classify("MH12AB1234", "MH12AB1234", "MH12AB1234") is Outcome.CORRECT

    def test_grammar_repaired_a_bad_read(self):
        # The success case for the grammar: OCR wrong, output right.
        assert classify("MH12AB1234", "MH12A81234", "MH12AB1234") is Outcome.CORRECT

    def test_rejection(self):
        assert classify("MH12AB1234", "ZZZZ", None) is Outcome.GRAMMAR_REJECTED

    def test_ocr_error_the_grammar_could_not_fix(self):
        assert classify("MH12AB1234", "XY99ZZ0000", "XY99ZZ0000") is Outcome.OCR_ERROR

    def test_grammar_corrupted_a_correct_read(self):
        # The category the taxonomy exists for: a correction step that damages
        # correct reads is worse than no correction step, and an aggregate
        # score hides it.
        assert classify("MH12AB1234", "MH12AB1234", "MH12AB1284") is Outcome.GRAMMAR_CORRUPTED


class TestBestDetection:
    def test_finds_the_overlapping_box(self):
        truth = (0.2, 0.5, 0.4, 0.6)
        detections = [Detection(0.6, 0.1, 0.7, 0.2, 0.9), Detection(0.21, 0.5, 0.41, 0.6, 0.8)]
        assert best_detection_for(truth, detections).confidence == 0.8

    def test_returns_none_when_nothing_overlaps(self):
        elsewhere = [Detection(0.8, 0.8, 0.9, 0.9, 0.9)]
        assert best_detection_for((0.2, 0.5, 0.4, 0.6), elsewhere) is None

    def test_prefers_the_best_overlap_not_the_most_confident(self):
        # A confident box somewhere else is not this plate.
        truth = (0.2, 0.5, 0.4, 0.6)
        detections = [
            Detection(0.2, 0.5, 0.4, 0.6, 0.5),  # exact
            Detection(0.25, 0.5, 0.45, 0.6, 0.99),  # confident, offset
        ]
        assert best_detection_for(truth, detections).confidence == 0.5

    def test_empty_detections(self):
        assert best_detection_for((0.2, 0.5, 0.4, 0.6), []) is None


class TestReport:
    def _report(self, outcomes):
        return EndToEndReport(
            outcomes=[
                PlateOutcome(crop_id=str(i), truth="T", outcome=o) for i, o in enumerate(outcomes)
            ]
        )

    def test_accuracy(self):
        report = self._report([Outcome.CORRECT, Outcome.CORRECT, Outcome.OCR_ERROR])
        assert report.accuracy == pytest.approx(2 / 3)

    def test_rejected_reads_lower_recall_but_not_precision(self):
        # Refusing to log a bad read is the grammar working. It costs a plate,
        # but it does not put a wrong plate in the log.
        report = self._report([Outcome.CORRECT, Outcome.GRAMMAR_REJECTED])
        assert report.recall == pytest.approx(0.5)
        assert report.precision == pytest.approx(1.0)

    def test_detection_misses_are_not_counted_as_wrong_logs(self):
        report = self._report([Outcome.CORRECT, Outcome.DETECTION_MISS])
        assert report.precision == pytest.approx(1.0)
        assert report.recall == pytest.approx(0.5)

    def test_ocr_errors_hurt_precision(self):
        report = self._report([Outcome.CORRECT, Outcome.OCR_ERROR])
        assert report.precision == pytest.approx(0.5)

    def test_report_warns_about_corrupted_reads(self):
        text = self._report([Outcome.CORRECT, Outcome.GRAMMAR_CORRUPTED]).report()
        assert "WARNING" in text
        assert "worse than none" in text

    def test_report_omits_the_warning_when_clean(self):
        assert "WARNING" not in self._report([Outcome.CORRECT]).report()

    def test_empty_report_does_not_divide_by_zero(self):
        report = EndToEndReport()
        assert report.accuracy == 0.0
        assert report.precision == 0.0
        report.report()


class _Detector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image, **kwargs):
        return self.detections


class _Reader:
    def __init__(self, text):
        self.text = text

    def read(self, image, detection):
        return OcrResult(self.text, 0.9)


def _setup(tmp_path):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    Image.new("RGB", (1000, 1000), (50, 50, 60)).save(images / "a.png")
    record = ImageRecord(
        image_id="a",
        width=1000,
        height=1000,
        boxes=(PlateBox(0.5, 0.5, 0.2, 0.06),),
        file_name="a.png",
    )
    return images, [record]


class TestEvaluateEndToEnd:
    def test_correct_plate(self, tmp_path):
        images, records = _setup(tmp_path)
        report = evaluate_end_to_end(
            records,
            {"a#0": "MH12AB1234"},
            _Detector([Detection(0.4, 0.47, 0.6, 0.53, 0.9)]),
            _Reader("MH12AB1234"),
            image_root=images,
            parse=lambda text, region=None: type("M", (), {"text": text})(),
        )
        assert report.accuracy == 1.0
        assert report.outcomes[0].outcome is Outcome.CORRECT

    def test_detection_miss(self, tmp_path):
        # Nothing downstream can recover a plate that was never found.
        images, records = _setup(tmp_path)
        report = evaluate_end_to_end(
            records,
            {"a#0": "MH12AB1234"},
            _Detector([]),
            _Reader("MH12AB1234"),
            image_root=images,
            parse=lambda text, region=None: type("M", (), {"text": text})(),
        )
        assert report.outcomes[0].outcome is Outcome.DETECTION_MISS
        assert report.precision == 0.0

    def test_grammar_rejection(self, tmp_path):
        images, records = _setup(tmp_path)
        report = evaluate_end_to_end(
            records,
            {"a#0": "MH12AB1234"},
            _Detector([Detection(0.4, 0.47, 0.6, 0.53, 0.9)]),
            _Reader("!!!"),
            image_root=images,
            parse=lambda text, region=None: None,
        )
        assert report.outcomes[0].outcome is Outcome.GRAMMAR_REJECTED

    def test_missing_image_is_skipped(self, tmp_path):
        images, records = _setup(tmp_path)
        (images / "a.png").unlink()
        report = evaluate_end_to_end(
            records,
            {"a#0": "MH12AB1234"},
            _Detector([]),
            _Reader("X"),
            image_root=images,
        )
        assert report.total == 0

    def test_unknown_crop_id_is_skipped(self, tmp_path):
        images, records = _setup(tmp_path)
        report = evaluate_end_to_end(
            records, {"nope#0": "X"}, _Detector([]), _Reader("X"), image_root=images
        )
        assert report.total == 0

    def test_out_of_range_box_index_is_skipped(self, tmp_path):
        images, records = _setup(tmp_path)
        report = evaluate_end_to_end(
            records, {"a#5": "X"}, _Detector([]), _Reader("X"), image_root=images
        )
        assert report.total == 0

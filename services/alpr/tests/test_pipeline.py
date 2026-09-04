"""End-to-end orchestration.

The detector and reader are stubbed: this tests that the phases compose —
detection feeds tracking, tracking feeds OCR, votes feed the grammar, and only
survivors reach the log — not that YOLO or PaddleOCR work.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from alpr.data.schema import Region
from alpr.detect import Detection
from alpr.excel import read_workbook
from alpr.ocr import OcrResult
from alpr.pipeline import Pipeline, PipelineConfig, PipelineStats
from alpr.sources import Frame, FrameSource


class FakeSource(FrameSource):
    """Replays a scripted sequence of frames."""

    name = "fake.mp4"

    def __init__(self, count: int) -> None:
        self.count = count

    def frames(self):
        for i in range(self.count):
            yield Frame(
                index=i,
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp=1_800_000_000.0 + i,
            )

    def close(self) -> None:
        pass


class FakeDetector:
    """One plate, drifting, for the first `visible_for` frames."""

    def __init__(self, visible_for: int = 8) -> None:
        self.visible_for = visible_for
        self.calls = 0

    def detect(self, image, **kwargs):
        index = self.calls
        self.calls += 1
        if index >= self.visible_for:
            return []
        x = 0.20 + index * 0.005
        return [Detection(x, 0.55, x + 0.12, 0.60, 0.92)]


class FakeReader:
    """Returns scripted OCR text, cycling through the list."""

    def __init__(self, texts) -> None:
        self.texts = list(texts)
        self.calls = 0

    def read(self, image, detection):
        text = self.texts[self.calls % len(self.texts)]
        self.calls += 1
        return OcrResult(text, 0.9)


def run(tmp_path, texts, *, frames=20, **config_kwargs) -> tuple[PipelineStats, list]:
    detector = FakeDetector()
    reader = FakeReader(texts)
    config = PipelineConfig(min_hits=2, max_age=3, **config_kwargs)
    pipeline = Pipeline(detector, reader, config)
    out = tmp_path / "log.xlsx"
    stats = pipeline.run(FakeSource(frames), out)
    rows = read_workbook(out) if out.exists() else []
    return stats, rows


class TestPipeline:
    def test_one_vehicle_becomes_one_row(self, tmp_path):
        stats, rows = run(tmp_path, ["MH12AB1234"], ocr_every=1)
        assert stats.logged == 1
        assert len(rows) == 1
        assert rows[0]["Plate"] == "MH12AB1234"

    def test_noisy_reads_are_voted_into_the_right_plate(self, tmp_path):
        # No single read is repeated enough to win a string vote.
        stats, rows = run(
            tmp_path,
            ["MH12A81234", "MH12AB1Z34", "NH12AB1234", "MH12AB1234"],
            ocr_every=1,
        )
        assert rows[0]["Plate"] == "MH12AB1234"
        assert stats.logged == 1

    def test_ocr_every_reduces_calls(self, tmp_path):
        # The live-framerate lever: fewer reads, same answer.
        dense, _ = run(tmp_path / "a", ["MH12AB1234"], ocr_every=1)
        sparse, rows = run(tmp_path / "b", ["MH12AB1234"], ocr_every=3)
        (tmp_path / "a").mkdir(exist_ok=True)
        assert sparse.ocr_calls < dense.ocr_calls
        assert rows[0]["Plate"] == "MH12AB1234"

    def test_unparseable_plates_are_rejected_not_logged(self, tmp_path):
        # A false positive that survived tracking is what the grammar catches.
        stats, rows = run(tmp_path, ["ZZZZZZ"], ocr_every=1)
        assert stats.rejected >= 1
        assert stats.logged == 0
        assert rows == []

    def test_region_restricts_the_grammar(self, tmp_path):
        stats, _ = run(tmp_path, ["MH12AB1234"], ocr_every=1, region=Region.GERMANY)
        assert stats.logged == 0
        assert stats.rejected >= 1

    def test_confidence_takes_the_weaker_evidence(self, tmp_path):
        # Vote confidence and grammar confidence are different evidence.
        _, rows = run(tmp_path, ["MH12AB1234"], ocr_every=1)
        assert rows[0]["Confidence"] <= 1.0

    def test_stats_are_reported(self, tmp_path):
        stats, _ = run(tmp_path, ["MH12AB1234"], ocr_every=1)
        assert stats.frames == 20
        assert stats.detections == 8
        assert stats.tracks_completed == 1
        assert "throughput" in stats.report()

    def test_max_frames_stops_early(self, tmp_path):
        detector, reader = FakeDetector(), FakeReader(["MH12AB1234"])
        pipeline = Pipeline(detector, reader, PipelineConfig(min_hits=2))
        stats = pipeline.run(FakeSource(50), tmp_path / "log.xlsx", max_frames=5)
        assert stats.frames == 5

    def test_on_frame_callback_receives_each_frame(self, tmp_path):
        seen = []
        detector, reader = FakeDetector(), FakeReader(["MH12AB1234"])
        pipeline = Pipeline(detector, reader, PipelineConfig(min_hits=2))
        pipeline.run(
            FakeSource(6),
            tmp_path / "log.xlsx",
            on_frame=lambda f, d, t, texts: seen.append(f.index),
        )
        assert seen == [0, 1, 2, 3, 4, 5]

    def test_a_track_still_live_at_the_end_is_logged(self, tmp_path):
        # Without finish(), a vehicle in frame on the last frame is lost.
        detector = FakeDetector(visible_for=100)
        pipeline = Pipeline(
            detector, FakeReader(["MH12AB1234"]), PipelineConfig(min_hits=2, ocr_every=1)
        )
        out = tmp_path / "log.xlsx"
        stats = pipeline.run(FakeSource(6), out)
        assert stats.logged == 1

    def test_a_short_pass_with_sparse_ocr_is_counted_not_dropped(self, tmp_path):
        # A vehicle can end its track before accumulating enough reads to
        # vote. That is a real miss, and a vehicle vanishing with nothing
        # recording it is the worst failure this pipeline has.
        detector = FakeDetector(visible_for=100)
        pipeline = Pipeline(
            detector,
            FakeReader(["MH12AB1234"]),
            PipelineConfig(min_hits=2, ocr_every=3, min_reads=2),
        )
        stats = pipeline.run(FakeSource(6), tmp_path / "log.xlsx")
        assert stats.logged == 0
        assert stats.too_few_reads == 1
        assert "too few reads" in stats.report()


class TestConfig:
    def test_rejects_zero_ocr_interval(self):
        with pytest.raises(ValueError, match="ocr_every"):
            PipelineConfig(ocr_every=0)


class StillSource(FakeSource):
    """Unrelated photographs rather than a video sequence."""

    name = "photos"

    @property
    def is_still(self) -> bool:
        return True

    def frames(self):
        for frame in super().frames():
            # A real ImageSource names every frame after its file.
            yield replace(frame, source_name=f"photo{frame.index}.jpg")


class AlwaysDetects:
    """A plate in the same place in every image — the contamination trap."""

    def detect(self, image, **kwargs):
        return [Detection(0.30, 0.55, 0.42, 0.60, 0.92)]


class TestStillImages:
    def test_config_relaxes_for_stills(self):
        config = PipelineConfig(min_hits=3, min_reads=2, ocr_every=3).for_stills()
        # A plate appears once in a still; the video thresholds would confirm
        # no track at all and nothing would ever be logged.
        assert (config.min_hits, config.min_reads, config.ocr_every) == (1, 1, 1)

    def test_config_is_unchanged_for_video(self, tmp_path):
        stats, rows = run(tmp_path, ["MH12AB1234"], ocr_every=1)
        assert stats.logged == 1  # video path still works

    def test_a_single_still_is_logged(self, tmp_path):
        pipeline = Pipeline(AlwaysDetects(), FakeReader(["MH12AB1234"]), PipelineConfig())
        out = tmp_path / "log.xlsx"
        stats = pipeline.run(StillSource(1), out)
        assert stats.logged == 1
        assert read_workbook(out)[0]["Plate"] == "MH12AB1234"

    def test_each_still_is_independent(self, tmp_path):
        # The trap: the same box position in consecutive images. On video that
        # is one car; across unrelated photos it is three different cars, and
        # merging them would vote three plates into one.
        pipeline = Pipeline(
            AlwaysDetects(),
            FakeReader(["AAA111", "BBB222", "CCC333"]),
            PipelineConfig(),
        )
        out = tmp_path / "log.xlsx"
        stats = pipeline.run(StillSource(3), out)

        assert stats.frames == 3
        assert stats.tracks_completed == 3, "images were merged into one track"
        assert stats.logged == 3
        assert {r["Plate"] for r in read_workbook(out)} == {"AAA111", "BBB222", "CCC333"}

    def test_each_row_names_the_photo_it_came_from(self, tmp_path):
        # Provenance is the whole point of a batch run. Logging three plates
        # against "3 images" leaves no way back to the photograph, which is
        # exactly what someone reviewing a batch needs.
        pipeline = Pipeline(
            AlwaysDetects(),
            FakeReader(["AAA111", "BBB222", "CCC333"]),
            PipelineConfig(),
        )
        out = tmp_path / "log.xlsx"
        pipeline.run(StillSource(3), out)

        rows = {r["Plate"]: r["Source"] for r in read_workbook(out)}
        assert rows == {
            "AAA111": "photo0.jpg",
            "BBB222": "photo1.jpg",
            "CCC333": "photo2.jpg",
        }

    def test_video_rows_still_name_the_clip(self, tmp_path):
        # A video's frames carry no per-frame name, so the source name stands.
        _, rows = run(tmp_path, ["MH12AB1234"], ocr_every=1)
        assert rows[0]["Source"] == "fake.mp4"

    def test_video_still_merges_frames_into_one_track(self, tmp_path):
        # The counterpart: on video the same three reads are one vehicle.
        pipeline = Pipeline(
            AlwaysDetects(),
            FakeReader(["MH12AB1234"]),
            PipelineConfig(min_hits=1, min_reads=1, ocr_every=1),
        )
        stats = pipeline.run(FakeSource(3), tmp_path / "log.xlsx")
        assert stats.tracks_completed == 1

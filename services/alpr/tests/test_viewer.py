"""Drawing and recording annotated frames.

All of this is tested without opening a window — a viewer that only breaks on
someone else's machine is the kind that breaks during a demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from alpr.detect import Detection
from alpr.viewer import CONFIRMED, PENDING, Viewer, ViewerError, annotate, draw_box, draw_hud


@dataclass
class FakeTrack:
    track_id: int
    detection: Detection


def frame(width=640, height=480):
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestDrawBox:
    def test_draws_something(self):
        img = frame()
        draw_box(img, Detection(0.2, 0.4, 0.5, 0.6, 0.9), CONFIRMED)
        assert img.any(), "nothing was drawn"

    def test_label_is_drawn(self):
        plain, labelled = frame(), frame()
        detection = Detection(0.2, 0.4, 0.5, 0.6, 0.9)
        draw_box(plain, detection, CONFIRMED)
        draw_box(labelled, detection, CONFIRMED, "#3 MH12AB1234")
        assert labelled.sum() > plain.sum()

    def test_label_stays_inside_a_frame_top_box(self):
        # A plate at the very top would push its label off-screen; it goes
        # below the box instead. Drawing off-canvas would silently vanish.
        img = frame()
        draw_box(img, Detection(0.1, 0.0, 0.4, 0.05, 0.9), CONFIRMED, "#1 ABC123")
        assert img.any()

    def test_handles_a_box_at_the_frame_edge(self):
        img = frame()
        draw_box(img, Detection(0.9, 0.9, 1.0, 1.0, 0.9), CONFIRMED, "#9")
        assert img.any()


class TestHud:
    def test_draws_lines(self):
        img = frame()
        draw_hud(img, ["frame 12", "tracks 3"])
        assert img[:60, :200].any()

    def test_empty_hud_is_a_no_op(self):
        img = frame()
        draw_hud(img, [])
        assert not img.any()


class TestAnnotate:
    def test_confirmed_and_pending_are_drawn_differently(self):
        # Grey for unconfirmed, green for confirmed: the eye should go to the
        # box the pipeline actually believes.
        detections = [Detection(0.1, 0.1, 0.3, 0.2, 0.9), Detection(0.6, 0.6, 0.8, 0.7, 0.9)]
        tracks = [FakeTrack(1, detections[1])]
        img = annotate(frame(), detections, tracks)

        colours = {tuple(px) for row in img for px in row if any(px)}
        assert CONFIRMED in colours
        assert PENDING in colours

    def test_plate_text_is_drawn_when_known(self):
        detection = Detection(0.3, 0.4, 0.6, 0.5, 0.9)
        tracks = [FakeTrack(7, detection)]
        without = annotate(frame(), [detection], tracks, {})
        with_text = annotate(frame(), [detection], tracks, {7: "MH12AB1234"})
        assert with_text.sum() > without.sum()

    def test_no_detections_is_fine(self):
        annotate(frame(), [], [], {})

    def test_returns_the_same_array(self):
        img = frame()
        assert annotate(img, [], []) is img


class TestViewer:
    def test_inactive_by_default(self):
        viewer = Viewer()
        assert viewer.active is False
        assert viewer.push(frame()) is True  # costs nothing, never blocks

    def test_writes_a_video(self, tmp_path):
        out = tmp_path / "annotated.mp4"
        with Viewer(save_path=out, fps=10) as viewer:
            for _ in range(5):
                assert viewer.push(frame()) is True
        assert out.exists()
        assert out.stat().st_size > 0
        assert viewer.frames_written == 5

    def test_creates_the_output_directory(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "v.mp4"
        with Viewer(save_path=out) as viewer:
            viewer.push(frame())
        assert out.exists()

    def test_video_is_readable_back(self, tmp_path):
        import cv2

        out = tmp_path / "v.mp4"
        with Viewer(save_path=out, fps=10) as viewer:
            for _ in range(6):
                viewer.push(frame(320, 240))

        capture = cv2.VideoCapture(str(out))
        ok, image = capture.read()
        capture.release()
        assert ok, "wrote a video that cannot be opened"
        assert image.shape[:2] == (240, 320)

    def test_close_is_idempotent(self, tmp_path):
        viewer = Viewer(save_path=tmp_path / "v.mp4")
        viewer.push(frame())
        viewer.close()
        viewer.close()

    def test_window_failure_explains_the_alternative(self, monkeypatch):
        # The headless build has no windowing, and the error has to point at
        # --save-video rather than just failing.
        import cv2

        def _boom(*args, **kwargs):
            raise cv2.error("no GUI support")

        monkeypatch.setattr(cv2, "namedWindow", _boom)
        with pytest.raises(ViewerError, match="save-video"):
            Viewer(show=True).push(frame())

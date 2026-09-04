"""Frame sources.

The file source is tested against a real video written with OpenCV. Camera and
RTSP tests stub the capture, because CI has neither a webcam nor a network
camera — and the logic worth testing is the frame-dropping and reconnection,
not OpenCV's ability to open a device.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpr.sources import (
    CameraSource,
    RtspSource,
    SourceError,
    VideoFileSource,
    _ThreadedCapture,
    open_source,
)


def write_video(path, frames=10, size=(320, 240), fps=15):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        image = np.full((size[1], size[0], 3), i * 20 % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()
    return path


class TestOpenSource:
    def test_int_is_a_camera(self):
        assert isinstance(open_source(0), CameraSource)

    def test_digit_string_is_a_camera(self):
        assert open_source("1").index == 1

    def test_rtsp_url(self):
        assert isinstance(open_source("rtsp://cam.local/stream"), RtspSource)

    def test_http_url_is_a_stream(self):
        assert isinstance(open_source("http://cam.local/video"), RtspSource)

    def test_path_is_a_file(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4")
        assert isinstance(open_source(str(path)), VideoFileSource)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SourceError, match="not found"):
            open_source(str(tmp_path / "absent.mp4"))


class TestVideoFileSource:
    def test_yields_every_frame(self, tmp_path):
        # A file must not drop frames: that would silently discard data.
        path = write_video(tmp_path / "clip.mp4", frames=10)
        assert len(list(VideoFileSource(path))) == 10

    def test_frames_are_indexed_in_order(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=5)
        assert [f.index for f in VideoFileSource(path)] == [0, 1, 2, 3, 4]

    def test_frame_reports_its_shape(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=2, size=(320, 240))
        assert next(iter(VideoFileSource(path))).shape == (240, 320)

    def test_reads_fps(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=4, fps=15)
        assert VideoFileSource(path).fps == pytest.approx(15, abs=1)

    def test_is_not_live(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=2)
        assert VideoFileSource(path).is_live is False

    def test_context_manager_closes(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=3)
        with VideoFileSource(path) as source:
            next(iter(source))
        assert source._capture is None


class _StubCapture:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, frames=5, fail_after=None):
        self.remaining = frames
        self.fail_after = fail_after
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            return False, None
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.released = True

    def isOpened(self):  # noqa: N802 — matches the OpenCV API
        return True

    def set(self, *_):
        return True


class TestThreadedCapture:
    def test_delivers_frames(self):
        capture = _ThreadedCapture(lambda: _StubCapture(frames=3))
        capture.start()
        try:
            assert capture.read(timeout=2.0) is not None
        finally:
            capture.stop()

    def test_keeps_only_the_newest_frame(self):
        # The latency fix: a slow consumer must see current reality, not a
        # queue of stale frames.
        capture = _ThreadedCapture(lambda: _StubCapture(frames=200))
        capture.start()
        try:
            import time

            time.sleep(0.2)  # let the reader get ahead
            first = capture.read(timeout=2.0)
            assert first is not None
            assert capture.dropped > 0
        finally:
            capture.stop()

    def test_returns_none_when_the_stream_ends(self):
        capture = _ThreadedCapture(lambda: _StubCapture(frames=1))
        capture.start()
        try:
            capture.read(timeout=2.0)
            assert capture.read(timeout=0.3) is None
        finally:
            capture.stop()

    def test_reconnects_when_asked(self):
        opened = []

        def opener():
            capture = _StubCapture(frames=2, fail_after=2)
            opened.append(capture)
            return capture

        capture = _ThreadedCapture(opener, reconnect=True)
        capture.start()
        try:
            import time

            time.sleep(1.6)  # past the first backoff
            assert len(opened) > 1, "should have reopened the stream"
        finally:
            capture.stop()

    def test_stop_releases_the_capture(self):
        stub = _StubCapture(frames=100)
        capture = _ThreadedCapture(lambda: stub)
        capture.start()
        capture.stop()
        assert stub.released


class TestCameraSource:
    def test_is_live(self):
        assert CameraSource(0).is_live is True

    def test_yields_frames_from_a_stubbed_device(self, monkeypatch):
        source = CameraSource(0)
        monkeypatch.setattr(source, "_open", lambda: _StubCapture(frames=3))
        frames = list(source.frames())
        assert len(frames) >= 1
        assert frames[0].image.shape == (10, 10, 3)

    def test_empty_rtsp_url_raises(self):
        with pytest.raises(SourceError, match="empty RTSP url"):
            RtspSource("")

    def test_rtsp_is_live(self):
        assert RtspSource("rtsp://x/y").is_live is True


@pytest.mark.camera
def test_real_webcam():
    """Opt-in: needs a camera and, on macOS, camera permission."""
    with CameraSource(0) as source:
        frame = next(iter(source))
        assert frame.image is not None


class TestImageSource:
    def _photo(self, path, shade=90):
        import cv2

        cv2.imwrite(str(path), np.full((120, 200, 3), shade, dtype=np.uint8))
        return path

    def test_every_frame_names_its_file(self, tmp_path):
        # Without this a batch logs each plate against "3 images" and the
        # photograph it came from is unrecoverable.
        from alpr.sources import ImageSource

        paths = [
            self._photo(tmp_path / f"{n}.jpg", shade=s)
            for n, s in zip("abc", (60, 90, 120), strict=True)
        ]
        source = ImageSource(paths)
        assert [f.source_name for f in source.frames()] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_a_video_frame_has_no_per_frame_name(self, tmp_path):
        # A video is one file for the whole run; the source name stands.
        source = VideoFileSource(write_video(tmp_path / "clip.mp4", frames=3))
        assert all(f.source_name is None for f in source.frames())

    def test_a_single_image_file(self, tmp_path):
        from alpr.sources import ImageSource

        source = ImageSource(self._photo(tmp_path / "a.jpg"))
        frames = list(source.frames())
        assert len(frames) == 1
        assert source.is_still is True

    def test_open_source_recognises_an_image(self, tmp_path):
        from alpr.sources import ImageSource

        path = self._photo(tmp_path / "a.png")
        assert isinstance(open_source(str(path)), ImageSource)

    def test_open_source_recognises_a_directory(self, tmp_path):
        from alpr.sources import ImageSource

        for i in range(3):
            self._photo(tmp_path / f"{i}.jpg")
        source = open_source(str(tmp_path))
        assert isinstance(source, ImageSource)
        assert len(list(source.frames())) == 3

    def test_directory_images_are_ordered(self, tmp_path):
        from alpr.sources import ImageSource

        for i in (2, 0, 1):
            self._photo(tmp_path / f"{i}.jpg")
        source = ImageSource(sorted(tmp_path.glob("*.jpg")))
        assert [f.index for f in source.frames()] == [0, 1, 2]

    def test_video_is_not_treated_as_stills(self, tmp_path):
        path = write_video(tmp_path / "clip.mp4", frames=3)
        assert open_source(str(path)).is_still is False

    def test_undecodable_file_is_skipped_not_fatal(self, tmp_path):
        from alpr.sources import ImageSource

        good = self._photo(tmp_path / "good.jpg")
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image")
        # A directory can hold a file with an image extension that will not
        # decode; skipping beats aborting a batch part-way through.
        assert len(list(ImageSource([good, bad]).frames())) == 1

    def test_missing_image_raises(self, tmp_path):
        from alpr.sources import ImageSource

        with pytest.raises(SourceError, match="not found"):
            ImageSource(tmp_path / "absent.jpg")

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SourceError, match="no images"):
            open_source(str(tmp_path / "empty"))

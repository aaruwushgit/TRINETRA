"""Frame sources: video files, webcams and RTSP streams.

A file and a camera look alike — both hand you frames — but they fail in
opposite directions, and one abstraction that ignores that is wrong for both.

**A file is offline.** Every frame matters; falling behind costs nothing but
time. Reading it is a loop.

**A camera is live, and the enemy is latency.** OpenCV buffers frames behind
the scenes. If detection takes 60 ms while the camera produces a frame every
33 ms, the buffer fills and every frame you read is older than the last —
seconds behind reality within a minute, and the drift never recovers. The fix
is not to go faster. It is to **throw frames away**: a background thread reads
continuously and keeps only the newest frame, so the pipeline always processes
what the camera sees *now* and simply skips what it could not keep up with.

That is why `CameraSource` and `RtspSource` are threaded and `VideoFileSource`
is not. Dropping frames on a file would silently discard data; keeping them on
a camera would silently discard *time*.

RTSP adds one more failure: the network. A stream drops, and a source that
raises on the first hiccup makes an unattended camera useless. `RtspSource`
reconnects with backoff and keeps going.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

DEFAULT_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 30.0
DEFAULT_READ_TIMEOUT = 10.0


class SourceError(RuntimeError):
    """Raised when a source cannot be opened or has failed permanently."""


@dataclass(frozen=True)
class Frame:
    """One frame, with the information needed to timestamp a log row."""

    index: int
    image: Any
    timestamp: float

    # Which file this frame came from, when that differs frame to frame.
    # A video has one name for the whole run and leaves this unset; a folder
    # of stills has a different one per frame, and losing it would log every
    # plate against "24 images" with no way back to the photograph.
    source_name: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return self.image.shape[0], self.image.shape[1]


class FrameSource(ABC):
    """Something that yields frames."""

    name: str = "source"

    @abstractmethod
    def frames(self) -> Iterator[Frame]:
        """Yield frames until the source ends."""

    @abstractmethod
    def close(self) -> None: ...

    @property
    def is_live(self) -> bool:
        """Live sources drop frames to stay current; files do not."""
        return False

    @property
    def is_still(self) -> bool:
        """True when consecutive frames are unrelated images, not a sequence.

        This changes the pipeline's meaning entirely. Tracking and voting exist
        because consecutive video frames show the *same* vehicle; across
        unrelated stills they would link different cars that happen to occupy
        similar positions, and vote their readings together. A still source
        must therefore be processed one image at a time.
        """
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()


class VideoFileSource(FrameSource):
    """Every frame of a video file, in order.

    Deliberately not threaded: dropping frames here would discard data, and
    there is no clock to keep up with.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise SourceError(f"video not found: {self.path}")
        self.name = self.path.name
        self._capture = None

    def _open(self):
        import cv2

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise SourceError(f"cannot open {self.path} — unsupported codec?")
        return capture

    @property
    def fps(self) -> float:
        import cv2

        capture = self._capture or self._open()
        value = capture.get(cv2.CAP_PROP_FPS)
        if self._capture is None:
            capture.release()
        return float(value or 0.0)

    @property
    def frame_count(self) -> int:
        import cv2

        capture = self._capture or self._open()
        value = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if self._capture is None:
            capture.release()
        return int(value or 0)

    def frames(self) -> Iterator[Frame]:
        self._capture = self._open()
        index = 0
        try:
            while True:
                ok, image = self._capture.read()
                if not ok:
                    break
                yield Frame(index=index, image=image, timestamp=time.time())
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class _ThreadedCapture:
    """Reads a live capture continuously, keeping only the newest frame.

    This is the whole latency fix. The reader never blocks the pipeline, and
    the pipeline never processes a frame the camera has already moved past.
    """

    def __init__(self, opener, *, reconnect: bool = False) -> None:
        self._opener = opener
        self._reconnect = reconnect
        self._capture = None
        self._lock = threading.Lock()
        self._latest: Any = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0
        self.reconnects = 0
        self.error: str | None = None

    def start(self) -> None:
        self._capture = self._opener()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        delay = DEFAULT_RECONNECT_DELAY
        while not self._stop.is_set():
            ok, image = (False, None)
            if self._capture is not None:
                ok, image = self._capture.read()

            if not ok:
                if not self._reconnect:
                    self.error = "stream ended"
                    break
                # Exponential backoff: a camera that is down stays down for a
                # while, and hammering it helps nobody.
                self._release()
                if self._stop.wait(delay):
                    break
                try:
                    self._capture = self._opener()
                    self.reconnects += 1
                    delay = DEFAULT_RECONNECT_DELAY
                except Exception:
                    delay = min(delay * 2, MAX_RECONNECT_DELAY)
                continue

            with self._lock:
                if self._latest is not None:
                    # The previous frame was never consumed: the pipeline is
                    # slower than the camera, which is expected and fine.
                    self.dropped += 1
                self._latest = image
                self._seq += 1

    def read(self, timeout: float = DEFAULT_READ_TIMEOUT) -> tuple[int, Any] | None:
        """Take the newest frame, or None if none arrives before `timeout`."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    image, seq = self._latest, self._seq
                    self._latest = None
                    return seq, image
            if self._thread is not None and not self._thread.is_alive():
                return None
            time.sleep(0.002)
        return None

    def _release(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release()


class CameraSource(FrameSource):
    """A locally attached camera.

    Cannot run on Colab: a hosted runtime is a VM in a datacenter with no
    access to your webcam. This is one of the two phases that is local by
    nature.
    """

    def __init__(self, index: int = 0, *, read_timeout: float = DEFAULT_READ_TIMEOUT) -> None:
        self.index = index
        self.name = f"camera:{index}"
        self.read_timeout = read_timeout
        self._capture: _ThreadedCapture | None = None

    @property
    def is_live(self) -> bool:
        return True

    def _open(self):
        import cv2

        capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            raise SourceError(
                f"cannot open camera {self.index}. On macOS the app running "
                "Python needs camera permission in System Settings > Privacy."
            )
        # Ask the driver for the smallest buffer it will give us; the reader
        # thread is the real fix, but this reduces what it has to discard.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def frames(self) -> Iterator[Frame]:
        self._capture = _ThreadedCapture(self._open)
        self._capture.start()
        try:
            while True:
                item = self._capture.read(self.read_timeout)
                if item is None:
                    break
                seq, image = item
                yield Frame(index=seq, image=image, timestamp=time.time())
        finally:
            self.close()

    @property
    def dropped(self) -> int:
        return self._capture.dropped if self._capture else 0

    def close(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None


class RtspSource(CameraSource):
    """A network camera, with reconnection.

    Also unreachable from Colab: an RTSP camera on a LAN is not routable from
    a Google datacenter.
    """

    def __init__(self, url: str, *, read_timeout: float = DEFAULT_READ_TIMEOUT) -> None:
        if not url:
            raise SourceError("empty RTSP url")
        self.url = url
        self.name = url
        self.read_timeout = read_timeout
        self._capture: _ThreadedCapture | None = None

    def _open(self):
        import cv2

        capture = cv2.VideoCapture(self.url)
        if not capture.isOpened():
            raise SourceError(f"cannot open stream: {self.url}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def frames(self) -> Iterator[Frame]:
        self._capture = _ThreadedCapture(self._open, reconnect=True)
        self._capture.start()
        try:
            while True:
                item = self._capture.read(self.read_timeout)
                if item is None:
                    break
                seq, image = item
                yield Frame(index=seq, image=image, timestamp=time.time())
        finally:
            self.close()

    @property
    def reconnects(self) -> int:
        return self._capture.reconnects if self._capture else 0


class ImageSource(FrameSource):
    """Still images: one file, or every image in a directory.

    Each image is an independent scene. The pipeline resets its tracker
    between them, so nothing is carried from one photograph to the next.

    The cost is real and worth stating: a single image gives multi-frame
    voting nothing to work with, so accuracy is raw OCR accuracy rather than
    the voted figure. Stills are a convenience, not the mode this pipeline is
    designed around.
    """

    def __init__(self, paths: Sequence[str | Path] | str | Path) -> None:
        if isinstance(paths, str | Path):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise SourceError(f"image not found: {missing[0]}")
        if not self.paths:
            raise SourceError("no images given")
        self.name = self.paths[0].name if len(self.paths) == 1 else f"{len(self.paths)} images"

    @property
    def is_still(self) -> bool:
        return True

    def frames(self) -> Iterator[Frame]:
        import cv2

        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path))
            if image is None:
                # A directory can hold a file with an image extension that is
                # not decodable; skipping beats aborting a batch.
                continue
            yield Frame(
                index=index,
                image=image,
                timestamp=time.time(),
                source_name=path.name,
            )

    def close(self) -> None:
        pass


def _images_in(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def open_source(spec: str | int | Path, **kwargs) -> FrameSource:
    """Build the right source for a spec.

    - an int, or a digit string -> camera index
    - something starting `rtsp://`, `http://` or `https://` -> network stream
    - anything else -> a video file path
    """
    if isinstance(spec, int):
        return CameraSource(spec, **kwargs)

    text = str(spec)
    if text.isdigit():
        return CameraSource(int(text), **kwargs)
    if text.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return RtspSource(text, **kwargs)

    path = Path(text)
    if path.is_dir():
        images = _images_in(path)
        if not images:
            raise SourceError(f"no images in {path}")
        return ImageSource(images)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return ImageSource(path)

    return VideoFileSource(text)

"""Drawing what the pipeline is doing.

The pipeline is invisible without this. Boxes, track ids and the plate text as
it converges are what make it legible — to someone watching, and to you when
something is wrong.

Two outputs, because they fail differently:

**A window** needs a GUI build of OpenCV. The project depends on
`opencv-python-headless`, which deliberately has no windowing, so `--show`
works only where a full build happens to be installed and says so clearly
when it is not.

**An annotated video file** always works, needs no display at all, and is the
better artifact anyway: it can be shared, replayed and put in a README.

Drawing is a pure function over a frame. It is unit-tested without any window,
because a viewer that only breaks on someone else's machine is the kind that
breaks during a demo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alpr.detect import Detection

# BGR, because that is what OpenCV wants.
CONFIRMED = (80, 220, 100)  # green: a track that cleared min_hits
PENDING = (140, 140, 140)  # grey: seen once, probably a false positive
TEXT_BG = (30, 30, 30)
TEXT_FG = (245, 245, 245)


class ViewerError(RuntimeError):
    """Raised when a window cannot be opened."""


def draw_box(frame, detection: Detection, colour, label: str = "", thickness: int = 2):
    """Draw one detection, with an optional label above it."""
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection.pixel_box(width, height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)

    if label:
        scale = 0.6
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        # Above the box, or inside it when the box is against the top edge.
        top = y1 - text_h - baseline - 4
        if top < 0:
            top = y2 + 4
        cv2.rectangle(
            frame,
            (x1, top),
            (x1 + text_w + 8, top + text_h + baseline + 4),
            TEXT_BG,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            TEXT_FG,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame, lines: Sequence[str]):
    """Draw a status panel in the top-left corner."""
    import cv2

    if not lines:
        return frame

    scale, pad = 0.6, 8
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0] for line in lines]
    box_w = max(w for w, _ in sizes) + pad * 2
    line_h = max(h for _, h in sizes) + 6
    box_h = line_h * len(lines) + pad

    panel = frame[0 : box_h + pad, 0:box_w].copy()
    cv2.rectangle(frame, (0, 0), (box_w, box_h + pad), TEXT_BG, -1)
    # Semi-transparent so the panel never hides a plate underneath it.
    cv2.addWeighted(
        panel, 0.35, frame[0 : box_h + pad, 0:box_w], 0.65, 0, frame[0 : box_h + pad, 0:box_w]
    )

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (pad, pad + line_h * (index + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            TEXT_FG,
            2,
            cv2.LINE_AA,
        )
    return frame


def annotate(
    frame,
    detections: Sequence[Detection],
    tracks: Sequence[Any] = (),
    texts: Mapping[int, str] | None = None,
    hud: Sequence[str] = (),
):
    """Draw a frame's detections, tracks and current plate readings.

    Args:
        frame: BGR image; modified in place and returned.
        detections: everything the detector found this frame.
        tracks: confirmed tracks, which get ids and text.
        texts: `{track_id: plate text so far}` — the vote as it converges,
            which is the part worth watching.
        hud: status lines for the corner panel.
    """
    texts = texts or {}

    # Unconfirmed detections first and in grey, so a confirmed box drawn over
    # one is the thing that stands out.
    tracked = {id(t.detection) for t in tracks}
    for detection in detections:
        if id(detection) not in tracked:
            draw_box(frame, detection, PENDING, thickness=1)

    for track in tracks:
        text = texts.get(track.track_id, "")
        label = f"#{track.track_id}" + (f"  {text}" if text else "")
        draw_box(frame, track.detection, CONFIRMED, label)

    return draw_hud(frame, hud)


class Viewer:
    """Shows and/or records annotated frames.

    Both outputs are optional; with neither, this does nothing and costs
    nothing, which is what the pipeline uses by default.
    """

    def __init__(
        self,
        *,
        show: bool = False,
        save_path: str | Path | None = None,
        fps: float = 20.0,
        window: str = "ALPR",
    ) -> None:
        self.show = show
        self.save_path = Path(save_path) if save_path else None
        self.fps = fps if fps and fps > 0 else 20.0
        self.window = window
        self._writer = None
        self._opened = False
        self.frames_written = 0

    @property
    def active(self) -> bool:
        return self.show or self.save_path is not None

    def _open_window(self) -> None:
        import cv2

        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        except Exception as exc:
            raise ViewerError(
                "cannot open a window — this OpenCV build has no GUI support. "
                "Install the full build (`pip install opencv-python`) or use "
                "--save-video instead, which needs no display."
            ) from exc
        self._opened = True

    def _open_writer(self, frame) -> None:
        import cv2

        height, width = frame.shape[:2]
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.save_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not self._writer.isOpened():
            raise ViewerError(f"cannot write video to {self.save_path}")

    def push(self, frame) -> bool:
        """Display and/or record one annotated frame.

        Returns:
            False when the user asked to quit (q or Esc), True otherwise.
        """
        if not self.active:
            return True

        import cv2

        if self.save_path is not None:
            if self._writer is None:
                self._open_writer(frame)
            self._writer.write(frame)
            self.frames_written += 1

        if self.show:
            if not self._opened:
                self._open_window()
            cv2.imshow(self.window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
        return True

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._opened:
            import cv2

            cv2.destroyWindow(self.window)
            self._opened = False

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

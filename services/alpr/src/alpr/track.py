"""Multi-frame tracking.

A detector runs per frame and has no memory. Without tracking, a car visible
for forty frames produces forty independent detections, and the pipeline has
no way to know they are one vehicle.

Tracking here is IoU-based association rather than a Kalman filter. Plates
move smoothly between frames at any sane framerate, there is only one class,
and vehicles rarely occlude each other's plates — the motion model a full
tracker buys would not earn its complexity. Ultralytics ships ByteTrack, but
using it means driving detection through their tracking API, which would
couple the whole pipeline to that library; `Detection` objects stay ours.

Two parameters carry most of the value:

`min_hits` — a track must be seen this many times before it is confirmed.
This is the single-frame false-positive filter, and it matters because that is
exactly the detector's error profile: 23 false positives against 1 missed
plate, most appearing in one frame only. A real plate persists; a spurious box
on a headlight does not.

`max_age` — how many frames a track survives unmatched before it dies. Plates
disappear briefly behind wipers, sign posts and other cars; killing a track on
the first missed frame would split one vehicle into two logged events.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from alpr.detect import Detection, iou

DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_AGE = 15
DEFAULT_MIN_HITS = 3


@dataclass
class Track:
    """One vehicle's plate followed across frames."""

    track_id: int
    detection: Detection
    first_frame: int
    last_frame: int
    hits: int = 1
    age: int = 0
    history: list[tuple[int, Detection]] = field(default_factory=list)
    confirmed: bool = False
    emitted: bool = False

    @property
    def frames_seen(self) -> int:
        return self.hits

    @property
    def duration(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def best(self) -> Detection:
        """The highest-confidence sighting — the best crop to read."""
        return max((d for _, d in self.history), key=lambda d: d.confidence)

    def update(self, detection: Detection, frame: int) -> None:
        self.detection = detection
        self.last_frame = frame
        self.hits += 1
        self.age = 0
        self.history.append((frame, detection))


class Tracker:
    """Assigns stable ids to plates across frames.

    Association is greedy by IoU: the highest-overlap pair is matched first,
    then the next, each track and detection claimed once. With one class and
    smooth motion this matches what the Hungarian algorithm would produce in
    all but pathological cases, for a fraction of the code.
    """

    def __init__(
        self,
        *,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_age: int = DEFAULT_MAX_AGE,
        min_hits: int = DEFAULT_MIN_HITS,
    ) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if max_age < 0:
            raise ValueError(f"max_age must not be negative, got {max_age}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be at least 1, got {min_hits}")

        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: list[Track] = []
        self._next_id = 1
        self._finished: list[Track] = []

    def update(self, detections: Sequence[Detection], frame: int) -> list[Track]:
        """Advance one frame. Returns the currently confirmed tracks."""
        pairs = self._associate(detections)

        matched_tracks = set()
        matched_detections = set()
        for track_index, detection_index, _ in pairs:
            self.tracks[track_index].update(detections[detection_index], frame)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for index, track in enumerate(self.tracks):
            if index not in matched_tracks:
                track.age += 1
            if track.hits >= self.min_hits:
                track.confirmed = True

        for index, detection in enumerate(detections):
            if index not in matched_detections:
                self._start(detection, frame)

        self._retire()
        return [t for t in self.tracks if t.confirmed]

    def _associate(self, detections: Sequence[Detection]) -> list[tuple[int, int, float]]:
        candidates = []
        for track_index, track in enumerate(self.tracks):
            box = (
                track.detection.x1,
                track.detection.y1,
                track.detection.x2,
                track.detection.y2,
            )
            for detection_index, detection in enumerate(detections):
                overlap = iou(box, (detection.x1, detection.y1, detection.x2, detection.y2))
                if overlap >= self.iou_threshold:
                    candidates.append((track_index, detection_index, overlap))

        # Best overlap first; each track and detection may be claimed once.
        candidates.sort(key=lambda c: c[2], reverse=True)
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        pairs = []
        for track_index, detection_index, overlap in candidates:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            pairs.append((track_index, detection_index, overlap))
        return pairs

    def _start(self, detection: Detection, frame: int) -> Track:
        track = Track(
            track_id=self._next_id,
            detection=detection,
            first_frame=frame,
            last_frame=frame,
            history=[(frame, detection)],
            confirmed=self.min_hits <= 1,
        )
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _retire(self) -> None:
        alive = []
        for track in self.tracks:
            if track.age > self.max_age:
                # Only confirmed tracks are worth reporting. An unconfirmed
                # one never cleared min_hits, which is the false-positive
                # filter doing its job.
                if track.confirmed:
                    self._finished.append(track)
            else:
                alive.append(track)
        self.tracks = alive

    def finish(self) -> list[Track]:
        """End the sequence, retiring every live track.

        Call at end of video: tracks still alive on the last frame have not
        died naturally, and without this their vehicles would never be logged.
        """
        for track in self.tracks:
            if track.confirmed:
                self._finished.append(track)
        self.tracks = []
        return self._finished

    @property
    def finished(self) -> list[Track]:
        return list(self._finished)

    def completed(self) -> Iterator[Track]:
        """Confirmed tracks that have ended and not yet been emitted.

        Emitting once per track is what makes the log one row per vehicle
        rather than one row per frame.
        """
        for track in self._finished:
            if not track.emitted:
                track.emitted = True
                yield track

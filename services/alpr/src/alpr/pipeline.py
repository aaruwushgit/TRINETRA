"""End-to-end orchestration: frames in, Excel rows out.

Every phase meets here. A frame is detected on, detections become tracks,
tracks accumulate OCR reads, reads are voted into one string, the string is
validated against a country grammar, and the survivor becomes one row.

**OCR does not run on every frame, and that is the difference between a demo
and something that holds a live framerate.** Detection is ~30 ms and
recognition ~20 ms per plate; doing both on every frame of a 30 fps stream
with two vehicles in view costs more than the frame budget. But a track does
not need forty reads — voting is decisive with a handful, and each extra read
past that buys almost nothing. `ocr_every` reads each track once every N
frames, which keeps the vote strong and the loop real-time.

The other reason to be selective: the crop worth reading is the biggest and
sharpest one, not whichever frame happens to come next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from alpr.data.schema import Region
from alpr.dedup import DEFAULT_COOLDOWN
from alpr.detect import PlateDetector
from alpr.excel import ExcelLog, PlateEvent
from alpr.ocr import PlateReader
from alpr.plates import format_display, parse
from alpr.sources import FrameSource
from alpr.track import Tracker
from alpr.vote import Read, TrackVoter


@dataclass
class PipelineConfig:
    """Everything that changes how a run behaves."""

    # Read each track once every N frames. 1 reads every frame — accurate,
    # and too slow for a live source.
    ocr_every: int = 3

    # A track must contribute this many reads before its vote is trusted.
    min_reads: int = 2

    # Restrict plate parsing to one country when the site is known; it removes
    # any chance of an Indian reading winning on a German road.
    region: Region | None = None

    # Suppress the same plate for this long — the car circling a car park.
    cooldown: timedelta = DEFAULT_COOLDOWN

    # Tracking.
    min_hits: int = 3
    max_age: int = 15

    # Detection.
    confidence: float = 0.25

    def for_stills(self) -> PipelineConfig:
        """The same settings, adjusted for unrelated still images.

        Tracking and voting assume consecutive frames show the same vehicle.
        A still gives them nothing: a plate appears once, so `min_hits=3`
        would confirm no track at all and nothing would ever be logged. These
        thresholds drop to 1 so a single sighting counts.

        This is a genuine loss, not a free switch. Voting across frames is a
        large part of the pipeline's accuracy, and on one image there is
        nothing to vote on — the result is raw OCR accuracy.
        """
        from dataclasses import replace

        return replace(self, min_hits=1, min_reads=1, ocr_every=1)

    def __post_init__(self) -> None:
        if self.ocr_every < 1:
            raise ValueError(f"ocr_every must be at least 1, got {self.ocr_every}")


@dataclass
class PipelineStats:
    """What a run did."""

    frames: int = 0
    detections: int = 0
    ocr_calls: int = 0
    tracks_completed: int = 0
    logged: int = 0
    rejected: int = 0
    # Tracks that ended with too few OCR reads to vote on. A real vehicle
    # would otherwise disappear here with nothing recording it: raise
    # `ocr_every`'s frequency or lower `min_reads` if this is not near zero.
    too_few_reads: int = 0
    suppressed: int = 0
    dropped_frames: int = 0
    elapsed: float = 0.0
    per_stage: dict[str, float] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.frames / self.elapsed if self.elapsed else 0.0

    def report(self) -> str:
        lines = [
            f"frames            {self.frames}",
            f"detections        {self.detections}",
            f"ocr calls         {self.ocr_calls}",
            f"tracks completed  {self.tracks_completed}",
            f"logged            {self.logged}",
            f"rejected by grammar {self.rejected}",
            f"too few reads to vote {self.too_few_reads}",
            f"duplicates suppressed {self.suppressed}",
            f"elapsed           {self.elapsed:.1f}s",
            f"throughput        {self.fps:.1f} fps",
        ]
        if self.dropped_frames:
            # Not an error on a live source — it is the latency fix working.
            lines.append(f"dropped (live)    {self.dropped_frames}")
        if self.per_stage:
            lines.append("")
            for stage, seconds in sorted(self.per_stage.items(), key=lambda s: -s[1]):
                share = seconds / self.elapsed * 100 if self.elapsed else 0
                lines.append(f"  {stage:<12} {seconds:>6.1f}s  {share:>5.1f}%")
        return "\n".join(lines)


class Pipeline:
    """Runs detection, tracking, OCR, voting, validation and logging."""

    def __init__(
        self,
        detector: PlateDetector,
        reader: PlateReader,
        config: PipelineConfig | None = None,
    ) -> None:
        self.detector = detector
        self.reader = reader
        self.config = config or PipelineConfig()

    def run(
        self,
        source: FrameSource,
        out: str | Path,
        *,
        max_frames: int | None = None,
        on_frame=None,
    ) -> PipelineStats:
        """Process a source into an Excel log.

        Args:
            source: where frames come from.
            out: the workbook to write.
            max_frames: stop early, for smoke tests.
            on_frame: called as `on_frame(frame, detections, tracks, texts)`
                where `texts` maps track id to the vote *so far*. The viewer
                uses it to draw an overlay without this module needing to know
                anything about windows — and the converging vote is the part
                worth watching, so it is passed rather than left inside.

        Returns:
            The run's statistics. Stops early if `on_frame` returns False,
            which is how the viewer's quit key ends a live run.
        """
        from PIL import Image

        stills = getattr(source, "is_still", False)
        config = self.config.for_stills() if stills else self.config

        def fresh():
            return (
                Tracker(min_hits=config.min_hits, max_age=config.max_age),
                TrackVoter(min_reads=config.min_reads),
            )

        tracker, voter = fresh()
        stats = PipelineStats()
        timings: dict[str, float] = {"detect": 0.0, "ocr": 0.0, "log": 0.0}
        # A folder of stills names each frame; a video names the whole run.
        origin = source.name

        started = time.time()
        with ExcelLog(out, cooldown=config.cooldown) as log:
            for frame in source:
                if max_frames is not None and stats.frames >= max_frames:
                    break
                stats.frames += 1
                origin = frame.source_name or source.name

                mark = time.time()
                detections = self.detector.detect(frame.image, confidence=config.confidence)
                timings["detect"] += time.time() - mark
                stats.detections += len(detections)

                tracks = tracker.update(detections, frame.index)

                # BGR from OpenCV; PIL and the reader both expect RGB.
                image = None
                for track in tracks:
                    if frame.index % config.ocr_every:
                        continue
                    if image is None:
                        mark = time.time()
                        image = Image.fromarray(frame.image[:, :, ::-1])
                    result = self.reader.read(image, track.detection)
                    stats.ocr_calls += 1
                    if result.text:
                        voter.add(
                            track.track_id,
                            Read(result.text, result.confidence, frame.index),
                        )
                if image is not None:
                    timings["ocr"] += time.time() - mark

                mark = time.time()
                self._emit(tracker, voter, log, stats, frame.timestamp, origin)
                timings["log"] += time.time() - mark

                if stills:
                    # Each still is its own scene. Closing the tracker here
                    # emits this image's plates and guarantees nothing carries
                    # into the next photograph, which would otherwise link two
                    # different cars that happen to sit in similar positions.
                    tracker.finish()
                    self._emit(tracker, voter, log, stats, frame.timestamp, origin)
                    tracker, voter = fresh()

                if on_frame is not None:
                    texts = {}
                    for track in tracks:
                        voted = voter.result(track.track_id)
                        if voted is not None:
                            texts[track.track_id] = voted.text
                    if on_frame(frame, detections, tracks, texts) is False:
                        break

            tracker.finish()
            self._emit(tracker, voter, log, stats, time.time(), origin)
            stats.suppressed = log.suppressed

        stats.elapsed = time.time() - started
        stats.per_stage = timings
        stats.dropped_frames = getattr(source, "dropped", 0)
        return stats

    def _emit(self, tracker, voter, log, stats, timestamp, source_name) -> None:
        """Vote, validate and log every track that has finished."""
        for track in tracker.completed():
            stats.tracks_completed += 1
            result = voter.pop(track.track_id)
            if result is None:
                # The track ended before it accumulated enough reads — a
                # short pass combined with a sparse `ocr_every`. Counted
                # rather than dropped silently, because a vehicle vanishing
                # with no trace is the worst failure this pipeline has.
                stats.too_few_reads += 1
                continue

            match = parse(result.text, region=self.config.region)
            if match is None:
                # Voted to something no grammar accepts — usually a false
                # positive that survived tracking, which is what the grammar
                # is here to catch.
                stats.rejected += 1
                continue

            logged = log.add(
                PlateEvent(
                    plate=match.text,
                    timestamp=datetime.fromtimestamp(timestamp),
                    region=match.region,
                    display=format_display(match),
                    # Vote confidence and grammar confidence are different
                    # evidence; the weaker one governs.
                    confidence=round(min(result.confidence, match.confidence), 3),
                    edits=match.edits,
                    track_id=track.track_id,
                    frame=track.first_frame,
                    source=source_name,
                )
            )
            if logged:
                stats.logged += 1

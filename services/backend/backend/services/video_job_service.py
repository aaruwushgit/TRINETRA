"""
Video/image ANPR job runner — "upload a file, get plates in the database".

This is the bridge between the offline ALPR pipeline (which is synchronous,
CPU/GPU-bound and thinks in frames) and the API (which is async and thinks in
requests). Three decisions shape the whole file:

**Jobs run on threads, not in the event loop.** A 30-second clip takes ~30
seconds of solid inference. Doing that inside a request handler blocks every
other request on the same worker, including the WebSocket that is supposed to
be reporting progress. So the endpoint returns a job id immediately and the
work happens on a background thread.

**One job at a time.** The executor has a single worker on purpose. The
detector and reader are shared process-wide singletons (loading PaddleOCR
twice is seconds and hundreds of MB), and neither Ultralytics' `predict` nor
PaddleOCR's `predict` promises thread safety on a shared model. Running two
jobs concurrently would also halve each one's framerate on a single GPU, so the
queue is not a limitation being worked around — it is the honest behaviour.
`QUEUED` is a real state a client will see.

**The Excel log is the source of truth for plates, not the on_frame stream.**
`on_frame` sees the vote *in progress* — a plate can appear as "KA02HN182",
then "KA02HN1826" as more reads land. It is perfect for a live UI and wrong for
a database. The workbook the pipeline writes contains only rows that survived
voting, the country grammar and the duplicate cooldown, so that is what gets
ingested. The live stream is decoration; the workbook is the record.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from backend.config import get_settings
from backend.services import compute_monitor
from backend.services.anpr_service import anpr_service

settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where the pipeline writes its workbook + crash journal for each job. Kept on
# disk rather than in a temp dir because the workbook is a real deliverable —
# "give me the spreadsheet of what the camera saw" is a request an operator
# will actually make.
JOB_OUTPUT_DIR = PROJECT_ROOT / "job_outputs"

# One SQLite file per sandbox job. See _SandboxDB.
SANDBOX_DB_DIR = PROJECT_ROOT / "sandbox_dbs"

# Uploads land here; the API module writes them and passes the path in.
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# How many finished jobs to remember. Job records are small (a few KB with the
# rolling plate list capped), so this is about keeping /jobs readable, not
# about memory.
MAX_JOBS = 50

RECENT_PLATES_CAP = 40


class JobState:
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_STATES = {JobState.DONE, JobState.FAILED, JobState.CANCELLED}


# ── helpers ──────────────────────────────────────────────────────────────────

def _naive_utc_now() -> datetime:
    """Naive UTC, matching what every timestamp column in this schema stores."""
    from datetime import timezone

    return datetime.now(timezone.utc).replace(tzinfo=None)


def _local_naive_to_utc(when: datetime) -> datetime:
    """Convert alpr's Excel timestamp (naive *local* time) to naive UTC.

    `alpr.pipeline` writes `datetime.fromtimestamp(...)`, which is local time
    with the offset stripped. Storing that directly would put every event 5.5
    hours off in an IST deployment, and speed-between-cameras alerts divide by
    exactly that difference.
    """
    from datetime import timezone

    if when.tzinfo is not None:
        return when.astimezone(timezone.utc).replace(tzinfo=None)
    return when.astimezone().astimezone(timezone.utc).replace(tzinfo=None)


def _clamp_confidence(value: Any) -> float | None:
    """Coerce a logged confidence into the 0..1 the schema requires.

    alpr already emits 0..1 (verified: a real row logged 0.82), but the column
    is validated `ge=0 le=1` and a percent slipping through would reject the
    whole event. Treating >1 as a percentage is the only sane recovery.
    """
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf > 1.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def _probe_video(path: Path) -> tuple[int, float]:
    """(frame_count, fps) for a video file; (0, 0.0) if not readable.

    Only used for the progress percentage and for spacing event timestamps, so
    a failure here degrades the UI rather than the run.
    """
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return 0, 0.0
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        capture.release()
        return max(0, count), max(0.0, fps)
    except Exception:
        return 0, 0.0


def _resolve_region(region: str | None):
    """Map an API region string onto an alpr Region.

    - None   -> the deployment default from settings (IN for this build)
    - "any"  -> no restriction, every grammar is allowed to match
    - "IN"   -> that grammar only
    """
    if region is None:
        return anpr_service.region
    if str(region).strip().lower() in {"any", "all", ""}:
        return None
    from alpr.data.schema import Region

    return Region(str(region).strip().upper())


# ── sandbox database ─────────────────────────────────────────────────────────

class _SandboxDB:
    """A throwaway SQLite database with the production schema.

    The public "try it yourself" page must not be able to write into the city's
    event table — one uploaded joke video would poison every analytics query and
    every trajectory. Isolation is per session and physical: a separate file,
    its own engine, the same `Base.metadata`. Nothing in `backend.database`
    changes, and the sandbox therefore exercises the *real* ingestion path
    (tracking association, alert evaluation) rather than a stubbed one — the
    behaviour on show is the behaviour that ships.
    """

    def __init__(self, job_id: str) -> None:
        SANDBOX_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.path = SANDBOX_DB_DIR / f"{job_id}.db"
        self.url = f"sqlite:///{self.path}"

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from backend.database import Base

        # Import the models so create_all sees every table. Same list as
        # backend.database.init_db.
        from backend.models import (  # noqa: F401
            alert,
            camera,
            traffic_snapshot,
            trajectory,
            vehicle_event,
        )

        self.engine = create_engine(self.url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def seed_camera(self, camera_id: str) -> None:
        """Make sure the target camera exists in the sandbox.

        `process_event_payload` refuses events from an unregistered camera, and
        rightly so. A sandbox user has not registered anything, so the real
        camera's definition is copied across when it exists (giving true
        coordinates) and a neutral placeholder is created when it does not.
        This is the difference between the public page working on first click
        and it returning "0 events, camera not found".
        """
        from backend.models.camera import Camera

        session = self.SessionLocal()
        try:
            if session.query(Camera).filter(Camera.camera_id == camera_id).first():
                return

            source = None
            try:
                from backend.database import SessionLocal as MainSession

                main = MainSession()
                try:
                    # Read-only touch of the main DB. Copying the real lat/lng
                    # is what makes the sandbox map look right.
                    source = main.query(Camera).filter(Camera.camera_id == camera_id).first()
                    values = (
                        {
                            "camera_id": source.camera_id,
                            "name": source.name,
                            "location": source.location,
                            "latitude": source.latitude,
                            "longitude": source.longitude,
                            "road": source.road,
                            "direction": source.direction,
                            "camera_type": source.camera_type,
                            "speed_limit_kmh": source.speed_limit_kmh,
                        }
                        if source
                        else None
                    )
                finally:
                    main.close()
            except Exception:
                values = None

            if values is None:
                values = {
                    "camera_id": camera_id,
                    "name": f"Sandbox camera {camera_id}",
                    "location": "Sandbox",
                    "latitude": 0.0,
                    "longitude": 0.0,
                    "road": None,
                    "direction": None,
                    "camera_type": "ANPR",
                }
            values["deployment"] = "sandbox"
            session.add(Camera(**values))
            session.commit()
        finally:
            session.close()

    def dispose(self) -> None:
        try:
            self.engine.dispose()
        except Exception:
            pass


# ── job record ───────────────────────────────────────────────────────────────

@dataclass
class Job:
    job_id: str
    kind: str  # "video" | "image"
    camera_id: str
    source_path: str
    sandbox: bool = False

    ocr_every: int = 3
    min_reads: int = 2
    max_frames: int | None = None
    region: str | None = None

    state: str = JobState.QUEUED
    error: str | None = None

    created_at: datetime = field(default_factory=_naive_utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    # progress
    frames_processed: int = 0
    total_frames: int = 0
    source_fps: float = 0.0
    detections: int = 0
    ocr_calls: int = 0
    fps: float = 0.0  # instantaneous processing throughput

    unique_plates: set[str] = field(default_factory=set)
    recent_plates: deque = field(default_factory=lambda: deque(maxlen=RECENT_PLATES_CAP))

    # results
    plates: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] | None = None
    stats_report: str | None = None
    compute: dict[str, Any] | None = None
    events_inserted: int = 0
    events_skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)
    workbook_path: str | None = None
    sandbox_db_path: str | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _sampler: Any = field(default=None, repr=False)

    # -- progress accounting (called from the pipeline thread) -------------

    def note_frame(self, frames: int, detections: int, texts: dict[int, str]) -> None:
        with self._lock:
            self.frames_processed = frames
            self.detections += detections
            for track_id, text in texts.items():
                if not text:
                    continue
                key = f"{track_id}:{text}"
                if key in self._seen_votes:
                    continue
                self._seen_votes.add(key)
                self.recent_plates.append(
                    {
                        "plate": text,
                        "track_id": track_id,
                        "frame": frames,
                        "timestamp": _naive_utc_now().isoformat(),
                        "provisional": True,
                    }
                )

    def __post_init__(self) -> None:
        # Not a dataclass field: it is bookkeeping for note_frame's dedupe and
        # never belongs in the JSON a client sees.
        self._seen_votes: set[str] = set()
        self._fps_mark = time.time()
        self._fps_frames = 0

    def tick_fps(self, frames: int) -> None:
        """Recompute instantaneous throughput about twice a second.

        A running average over the whole job hides the model-load stall at the
        start and would show a misleadingly low number for the first minute.
        """
        now = time.time()
        window = now - self._fps_mark
        if window >= 0.5:
            with self._lock:
                self.fps = round((frames - self._fps_frames) / window, 2)
            self._fps_mark = now
            self._fps_frames = frames

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self) -> None:
        self._cancel.set()

    # -- serialization -----------------------------------------------------

    def progress_percent(self) -> float | None:
        """None when the total is unknown, which is honest for a live source."""
        with self._lock:
            total = self.total_frames
            if self.max_frames:
                total = min(total, self.max_frames) if total else self.max_frames
            if not total:
                return None
            return round(min(100.0, self.frames_processed / total * 100.0), 1)

    def to_dict(self, *, include_plates: bool = True) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "job_id": self.job_id,
                "kind": self.kind,
                "state": self.state,
                "camera_id": self.camera_id,
                "sandbox": self.sandbox,
                "source": Path(self.source_path).name,
                "error": self.error,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None,
                "progress": {
                    "frames_processed": self.frames_processed,
                    "total_frames": self.total_frames or None,
                    "percent": None,  # filled below, outside the lock
                    "fps": self.fps,
                    "detections": self.detections,
                    "ocr_calls": self.ocr_calls,
                    "unique_plates": len(self.unique_plates),
                },
                "config": {
                    "ocr_every": self.ocr_every,
                    "min_reads": self.min_reads,
                    "max_frames": self.max_frames,
                    "region": self.region,
                    "source_fps": self.source_fps,
                },
                "recent_plates": list(self.recent_plates),
                "stats": self.stats,
                "stats_report": self.stats_report,
                "compute": self.compute,
                "ingestion": {
                    "events_inserted": self.events_inserted,
                    "events_skipped": self.events_skipped,
                    "skip_reasons": self.skip_reasons[:10],
                },
                "workbook_path": self.workbook_path,
                "sandbox_db_path": self.sandbox_db_path,
            }
            if include_plates:
                data["plates"] = list(self.plates)
            plate_count = len(self.plates)

        data["progress"]["percent"] = self.progress_percent()
        data["plate_count"] = plate_count

        # A running job has no final compute summary yet, but the panel should
        # not sit empty for the whole run — report the live window instead.
        if self.compute is None and self._sampler is not None:
            try:
                data["compute"] = self._sampler.summary()
            except Exception:
                pass
        return data


# ── registry ─────────────────────────────────────────────────────────────────

_jobs: OrderedDict[str, Job] = OrderedDict()
_registry_lock = threading.Lock()

# Single worker: see the module docstring. `thread_name_prefix` makes a stuck
# job obvious in a stack dump.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anpr-job")


def _register(job: Job) -> None:
    with _registry_lock:
        _jobs[job.job_id] = job
        if len(_jobs) > MAX_JOBS:
            # Only ever evict finished jobs. Dropping a RUNNING job's record
            # would leave a thread writing into an object no client can read.
            for job_id, existing in list(_jobs.items()):
                if len(_jobs) <= MAX_JOBS:
                    break
                if existing.state in TERMINAL_STATES:
                    del _jobs[job_id]


def get_job(job_id: str) -> dict[str, Any] | None:
    with _registry_lock:
        job = _jobs.get(job_id)
    return job.to_dict() if job else None


def get_job_plates(job_id: str) -> dict[str, Any] | None:
    """Final plates when the job is done, provisional votes while it runs."""
    with _registry_lock:
        job = _jobs.get(job_id)
    if job is None:
        return None
    with job._lock:
        final = list(job.plates)
        provisional = list(job.recent_plates)
        state = job.state
    return {
        "job_id": job_id,
        "state": state,
        "count": len(final),
        "plates": final,
        "provisional": provisional if state not in TERMINAL_STATES else [],
    }


def list_jobs(limit: int = 25) -> list[dict[str, Any]]:
    with _registry_lock:
        jobs = list(_jobs.values())
    jobs.reverse()  # newest first
    out = []
    for job in jobs[:limit]:
        summary = job.to_dict(include_plates=False)
        summary.pop("recent_plates", None)
        summary.pop("stats_report", None)
        out.append(summary)
    return out


def cancel_job(job_id: str) -> bool:
    """Ask a job to stop at the next frame boundary.

    Cooperative rather than forced: the pipeline's `on_frame` hook can end a run
    cleanly, which flushes the workbook and still ingests everything read up to
    that point. Killing the thread would lose all of it.
    """
    with _registry_lock:
        job = _jobs.get(job_id)
    if job is None or job.state in TERMINAL_STATES:
        return False
    job.cancel()
    return True


def is_running(job_id: str) -> bool:
    with _registry_lock:
        job = _jobs.get(job_id)
    return bool(job and job.state not in TERMINAL_STATES)


# ── submission ───────────────────────────────────────────────────────────────

def submit_video_job(
    file_path: str | Path,
    camera_id: str,
    *,
    job_id: str | None = None,
    ocr_every: int = 3,
    min_reads: int = 2,
    max_frames: int | None = None,
    region: str | None = None,
    sandbox: bool = False,
) -> str:
    """Queue a video for ANPR + ingestion. Returns the job id immediately."""
    return _submit(
        kind="video",
        file_path=file_path,
        camera_id=camera_id,
        job_id=job_id,
        ocr_every=ocr_every,
        min_reads=min_reads,
        max_frames=max_frames,
        region=region,
        sandbox=sandbox,
    )


def submit_image_job(
    file_path: str | Path,
    camera_id: str,
    *,
    job_id: str | None = None,
    region: str | None = None,
    sandbox: bool = False,
) -> str:
    """Queue a still photo (or a directory of photos).

    Voting thresholds are forced to 1 via `PipelineConfig.for_stills()`. That is
    a real accuracy loss, not a free switch — a single image gives cross-frame
    voting nothing to work with, so what comes out is raw OCR accuracy.
    """
    return _submit(
        kind="image",
        file_path=file_path,
        camera_id=camera_id,
        job_id=job_id,
        ocr_every=1,
        min_reads=1,
        max_frames=None,
        region=region,
        sandbox=sandbox,
    )


def _submit(
    *,
    kind: str,
    file_path: str | Path,
    camera_id: str,
    job_id: str | None,
    ocr_every: int,
    min_reads: int,
    max_frames: int | None,
    region: str | None,
    sandbox: bool,
) -> str:
    path = Path(file_path)
    # Validated here, before a job id is handed out: "the file does not exist"
    # is a 400 on the request, not a job that fails ten seconds later.
    if not path.exists():
        raise FileNotFoundError(f"source not found: {path}")
    if not camera_id:
        raise ValueError("camera_id is required — events cannot be ingested without a camera")
    if ocr_every < 1:
        raise ValueError(f"ocr_every must be >= 1, got {ocr_every}")
    if max_frames is not None and max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames}")
    # Fail fast on a bad region string rather than inside the worker thread.
    _resolve_region(region)

    job = Job(
        job_id=job_id or str(uuid.uuid4()),
        kind=kind,
        camera_id=camera_id,
        source_path=str(path),
        sandbox=sandbox,
        ocr_every=ocr_every,
        min_reads=min_reads,
        max_frames=max_frames,
        region=region,
    )

    if kind == "video":
        job.total_frames, job.source_fps = _probe_video(path)

    _register(job)
    _executor.submit(_run_job, job)
    return job.job_id


# ── execution ────────────────────────────────────────────────────────────────

def _run_job(job: Job) -> None:
    """The whole job, on a worker thread. Never raises."""
    sampler = compute_monitor.ComputeSampler()
    job._sampler = sampler

    with job._lock:
        job.state = JobState.RUNNING
        job.started_at = _naive_utc_now()

    sampler.start()
    try:
        stats, rows, workbook = _run_pipeline(job)

        with job._lock:
            job.stats = _stats_dict(stats)
            job.stats_report = stats.report()
            job.ocr_calls = stats.ocr_calls
            job.detections = stats.detections
            job.frames_processed = stats.frames
            job.workbook_path = str(workbook) if workbook else None
            if stats.elapsed:
                job.fps = round(stats.frames / stats.elapsed, 2)

        plates = _rows_to_plates(job, rows)
        with job._lock:
            job.plates = plates
            job.unique_plates = {p["plate"] for p in plates if p.get("plate")}
            # Replace the provisional votes with the rows that actually
            # survived voting + grammar + dedupe, so a client polling at the
            # end sees the record rather than the guesses.
            job.recent_plates.clear()
            for plate in plates[-RECENT_PLATES_CAP:]:
                job.recent_plates.append({**plate, "provisional": False})

        inserted, skipped, reasons, sandbox_path = _ingest(job, plates)
        with job._lock:
            job.events_inserted = inserted
            job.events_skipped = skipped
            job.skip_reasons = reasons
            job.sandbox_db_path = sandbox_path
            job.state = JobState.CANCELLED if job.cancelled else JobState.DONE
    except Exception as exc:
        # The traceback goes to the log for whoever is debugging; the client
        # gets one actionable line. A raw traceback in an API response is both
        # useless to a UI and a disclosure risk.
        traceback.print_exc()
        with job._lock:
            job.state = JobState.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            summary = sampler.stop()
        except Exception:
            summary = None
        with job._lock:
            job.compute = summary
            job.finished_at = _naive_utc_now()
        job._sampler = None


def _run_pipeline(job: Job):
    """Run the ALPR pipeline over the job's source. Returns (stats, rows, path)."""
    from alpr.excel import read_workbook
    from alpr.sources import open_source

    JOB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workbook = JOB_OUTPUT_DIR / f"{job.job_id}.xlsx"
    journal = workbook.with_suffix(workbook.suffix + ".jsonl")
    # A fresh job must start from a clean log. ExcelLog deliberately *recovers*
    # from an existing journal, which is right after a crash and wrong here.
    for stale in (workbook, journal):
        try:
            stale.unlink(missing_ok=True)
        except OSError:
            pass

    config = replace(
        anpr_service.base_config(),
        ocr_every=job.ocr_every,
        min_reads=job.min_reads,
        region=_resolve_region(job.region),
    )
    if job.kind == "image":
        config = config.for_stills()

    pipeline = anpr_service.build_pipeline(config)
    source = open_source(job.source_path)

    def on_frame(frame, detections, tracks, texts) -> bool:
        # Runs once per frame on the hot path, so it does bookkeeping only.
        # frame.index is 0-based and, for a file, is the true frame number —
        # which is what makes the progress percentage against
        # CAP_PROP_FRAME_COUNT meaningful.
        frames = frame.index + 1
        job.note_frame(frames, len(detections), texts)
        job.tick_fps(frames)
        # Returning False is the pipeline's documented way to stop a run early,
        # and it stops it *cleanly* — the workbook is flushed on the way out.
        return not job.cancelled

    try:
        stats = pipeline.run(source, workbook, max_frames=job.max_frames, on_frame=on_frame)
    finally:
        try:
            source.close()
        except Exception:
            pass

    rows = read_workbook(workbook) if workbook.exists() else []
    return stats, rows, workbook if workbook.exists() else None


def _stats_dict(stats) -> dict[str, Any]:
    return {
        "frames": stats.frames,
        "detections": stats.detections,
        "ocr_calls": stats.ocr_calls,
        "tracks_completed": stats.tracks_completed,
        "logged": stats.logged,
        "rejected_by_grammar": stats.rejected,
        "too_few_reads": stats.too_few_reads,
        "duplicates_suppressed": stats.suppressed,
        "dropped_frames": stats.dropped_frames,
        "elapsed_s": round(stats.elapsed, 2),
        "fps": round(stats.fps, 2),
        "per_stage_s": {k: round(v, 2) for k, v in (stats.per_stage or {}).items()},
    }


def _rows_to_plates(job: Job, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn workbook rows into the plate records the API and the DB both use.

    **Timestamps are derived from the frame index, not from the clock.** The
    pipeline stamps each row with `time.time()` at the moment it was *processed*,
    which for a file means "whenever the upload happened", compressed into
    however long inference took. Ingesting that makes two cars 20 seconds apart
    in the video look 0.3 seconds apart in the database, and the
    checkpoint-to-checkpoint speed alert divides by that gap. Reconstructing
    `job_start + frame/fps` restores the video's own timeline, so trajectories
    and speeds are chronologically sane.
    """
    base = job.started_at or job.created_at
    fps = job.source_fps if job.source_fps and job.source_fps > 0 else 0.0

    plates: list[dict[str, Any]] = []
    for row in rows:
        plate = row.get("Plate")
        if not plate:
            continue

        frame = row.get("Frame")
        frame = int(frame) if isinstance(frame, (int, float)) else None

        if fps and frame is not None:
            timestamp = base + timedelta(seconds=frame / fps)
        elif isinstance(row.get("Timestamp"), datetime):
            timestamp = _local_naive_to_utc(row["Timestamp"])
        else:
            timestamp = base

        track_id = row.get("Track")
        track_id = int(track_id) if isinstance(track_id, (int, float)) else None

        plates.append(
            {
                "plate": str(plate),
                "display": row.get("Formatted") or None,
                "region": row.get("Region") or None,
                "confidence": _clamp_confidence(row.get("Confidence")),
                "ocr_fixes": row.get("OCR fixes"),
                "track_id": track_id,
                "frame": frame,
                "timestamp": timestamp.isoformat(),
                "video_offset_s": round(frame / fps, 2) if fps and frame is not None else None,
            }
        )
    return plates


# SQLite serializes writers, and this project shares one dev.db between the
# API, the seeding scripts and this job runner. A bulk generator holding the
# write lock for a few seconds would otherwise make a whole video's plates fail
# with "database is locked" — which is contention, not an error worth losing
# data over.
INSERT_RETRIES = 4
INSERT_BACKOFF = 0.5
BUSY_TIMEOUT_MS = 15000


def _set_busy_timeout(session) -> None:
    """Let SQLite wait for the write lock instead of failing instantly.

    Set per connection rather than on the engine, because the engine lives in
    backend/database.py and is shared by every other feature.
    """
    try:
        bind = session.get_bind()
        if bind is None or "sqlite" not in str(bind.dialect.name):
            return
        from sqlalchemy import text

        session.execute(text(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}"))
    except Exception:
        # Telemetry-grade concern: if the pragma will not apply, the retry loop
        # below is still there.
        pass


def _insert_with_retry(fn, payload: dict, session):
    """Call the ingest function, retrying only on lock contention.

    Deliberately narrow: a locked database is worth waiting for, while a
    constraint violation or an unknown camera will fail identically every time
    and retrying it just delays the report.
    """
    delay = INSERT_BACKOFF
    last: Exception | None = None
    for attempt in range(INSERT_RETRIES):
        try:
            return fn(payload, session)
        except Exception as exc:
            if "database is locked" not in str(exc).lower():
                raise
            last = exc
            session.rollback()
            if attempt < INSERT_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
    raise last  # type: ignore[misc]


def _short_error(exc: Exception) -> str:
    """One line, not a wall of SQL.

    A failed insert's exception carries the full statement and every bound
    parameter. Forty of those in a JSON response is unreadable and echoes plate
    data back into the payload for no reason.
    """
    text_value = str(exc).strip().splitlines()
    head = text_value[0] if text_value else ""
    return f"{type(exc).__name__}: {head[:200]}"


def _ingest(
    job: Job, plates: list[dict[str, Any]]
) -> tuple[int, int, list[str], str | None]:
    """Persist plates as VehicleEvents. Returns (inserted, skipped, reasons, db_path).

    Reuses `kafka_consumer.process_event_payload` rather than inserting rows
    directly. That function is where plate normalization, global-vehicle
    association and alert evaluation live; a second copy of that logic here
    would drift, and the first symptom would be a vehicle that shows on the map
    but never triggers a blacklist alert.
    """
    from backend.services.kafka_consumer import process_event_payload

    if not plates:
        return 0, 0, [], None

    sandbox_db: _SandboxDB | None = None
    if job.sandbox:
        sandbox_db = _SandboxDB(job.job_id)
        sandbox_db.seed_camera(job.camera_id)
        session_factory = sandbox_db.SessionLocal
    else:
        from backend.database import SessionLocal

        session_factory = SessionLocal

    inserted = 0
    skipped = 0
    reasons: list[str] = []

    session = session_factory()
    try:
        _set_busy_timeout(session)
        for plate in plates:
            payload = {
                "camera_id": job.camera_id,
                "timestamp": plate["timestamp"],
                # Namespaced with the camera id because alpr track ids restart
                # at 1 for every run — "T5" alone collides across cameras and
                # across jobs, and this column is used to group a single
                # camera's sightings.
                "local_track_id": (
                    f"{job.camera_id}_T{plate['track_id']}"
                    if plate.get("track_id") is not None
                    else None
                ),
                "plate": plate["plate"],
                "plate_confidence": plate.get("confidence"),
                # Left to process_event_payload, which falls back to the
                # camera's own coordinates and direction.
                "latitude": None,
                "longitude": None,
                "direction": None,
                # The detector finds plates, not vehicle classes. "car" is the
                # honest default for a plate-bearing vehicle on an Indian road;
                # a classifier would fill this properly.
                "vehicle_type": "car",
                "vehicle_color": None,
                "speed": None,
            }
            try:
                result = _insert_with_retry(process_event_payload, payload, session)
            except Exception as exc:
                # One bad row must not abandon the rest of the video. Rolling
                # back is required: SQLAlchemy leaves the session unusable
                # after a failed flush.
                session.rollback()
                skipped += 1
                reasons.append(f"{plate['plate']}: {_short_error(exc)}")
                continue

            if isinstance(result, dict) and result.get("error"):
                skipped += 1
                reasons.append(f"{plate['plate']}: {result['error']}")
            else:
                inserted += 1
    finally:
        session.close()
        if sandbox_db is not None:
            sandbox_db.dispose()

    # Deduplicate the reason list — 40 plates rejected for one missing camera
    # is one problem, not forty.
    seen: set[str] = set()
    unique_reasons: list[str] = []
    for reason in reasons:
        tail = reason.split(": ", 1)[-1]
        if tail in seen:
            continue
        seen.add(tail)
        unique_reasons.append(reason)

    return inserted, skipped, unique_reasons, (str(sandbox_db.path) if sandbox_db else None)


def sandbox_events(job_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Read back what a sandbox job wrote, from its own database file."""
    path = SANDBOX_DB_DIR / f"{job_id}.db"
    if not path.exists():
        return []

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from backend.models.vehicle_event import VehicleEvent

    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    factory = sessionmaker(bind=engine)
    session = factory()
    try:
        events = (
            session.query(VehicleEvent)
            .order_by(VehicleEvent.timestamp.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "event_id": e.event_id,
                "camera_id": e.camera_id,
                "local_track_id": e.local_track_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "plate": e.plate,
                "plate_confidence": e.plate_confidence,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "vehicle_type": e.vehicle_type,
                "global_vehicle_id": e.global_vehicle_id,
            }
            for e in events
        ]
    finally:
        session.close()
        engine.dispose()

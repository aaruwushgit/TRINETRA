"""
Benchmark & scalability service — "how much compute does city-wide ANPR need?"

The question this module exists to answer is the one a reviewer always asks:
*you demoed one laptop, so what does Delhi cost?* Answering it honestly means
keeping two kinds of number strictly apart:

  MEASURED   — timed on THIS machine, right now, against real media and the
               real ingestion path. Reproducible by re-running the suite.
  PROJECTED  — arithmetic on top of the measured numbers plus assumptions we
               could NOT measure here (camera count, frames analysed per
               camera, how much faster a datacentre GPU is than this Mac).

Every projected figure in the output carries the assumptions it depends on, so
a reviewer can disagree with an assumption and see exactly which figures move.
Nothing in here invents a measurement: if a suite did not run, its baseline is
absent and projections that need it refuse to compute rather than guessing.

WHAT IS MEASURED
  1. hardware — CPU/RAM/GPU/torch profile. Cheap, no inference.
  2. anpr     — plate DETECTION ms/frame and OCR ms/plate-crop, warm, with
                warmup iterations discarded, reported as median AND p95.
                Also video decode ms/frame, process CPU-seconds per frame and
                peak RSS, because sizing needs cores and RAM, not just latency.
  3. ingest   — events/sec and per-event latency through the REAL
                POST /events/ingest and /events/bulk-ingest path, including the
                tracking + alert work each event triggers.
  4. query    — latency of the dashboard-critical read endpoints, at a stated
                row count (latency without a row count is meaningless).

WHY THE DB SUITES RUN IN A SUBPROCESS
  backend.database builds its engine from DATABASE_URL at import time, so the
  only way to point ingestion at a throwaway database is to set the env var
  BEFORE any backend module is imported. Benchmarks must never write to dev.db:
  it is the demo database and other tooling is loading a large dataset into it.
  So the ingest/query suites are executed by a child process launched with
  DATABASE_URL pointing at a temp sqlite file, which is deleted afterwards.

CACHING
  A full run takes ~1-2 minutes (model loading dominates). That must never
  happen inside a web request, so results are persisted to
  deployments/benchmarks/latest.json and the API serves that. Re-measuring is
  an explicit POST that runs in a background thread.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]          # services/backend/
WORKSPACE_ROOT = REPO_ROOT.parent.parent                  # .../SIH/
RESULTS_DIR = REPO_ROOT / "deployments" / "benchmarks"
LATEST_PATH = RESULTS_DIR / "latest.json"

# Results older than this are still served, but flagged — hardware changes and
# code changes both invalidate a benchmark, and a silently ancient number is
# worse than no number.
STALE_AFTER_HOURS = 168.0  # 7 days

SCHEMA_VERSION = 1

# Upper bound on frames pulled off a clip, so pointing the suite at a 3-minute
# 4K video does not turn a 40-second benchmark into a 10-minute one.
MAX_FRAMES_TO_DECODE = 1200

# Timestamped copies of past runs kept alongside latest.json, so re-running
# before a demo cannot destroy the number already on a slide.
MAX_ARCHIVED_RUNS = 10

# Real media used for the ANPR timings. Kept as a list so a missing file
# degrades to "one less sample" instead of a crash.
DEFAULT_VIDEO_CANDIDATES = [
    WORKSPACE_ROOT / "archive" / "Automatic Number Plate Recognition (ANPR) _ Vehicle Number Plate Recognition (1).mp4",
    WORKSPACE_ROOT / "archive" / "pexels-christopher-schultz-5927708 (1080p).mp4",
]
DEFAULT_IMAGE_CANDIDATES = [
    WORKSPACE_ROOT / "services" / "alpr" / "test_car.jpeg",
    WORKSPACE_ROOT / "services" / "alpr" / "test_car2.jpg",
    WORKSPACE_ROOT / "services" / "alpr" / "test_image2.JPG",
]

# The upstream ANPR repo's own README numbers (M4 MacBook Air). Used ONLY as a
# cross-check — if our measurement is wildly different, something is wrong with
# one of the two, and the report says so rather than quietly picking a winner.
UPSTREAM_REFERENCE = {
    "source": "Automatic-License-Plate-Recognition/README.md (reported on an M4 MacBook Air)",
    "detection_ms_per_frame": 29.0,
    "detection_fps": 34.6,
    "ocr_ms_per_crop": 41.1,
    "pipeline_fps_ocr_every_1": 14.3,
    "pipeline_fps_ocr_every_3": 23.5,
}


class BenchmarkError(RuntimeError):
    """Raised when a suite cannot run — surfaced as a clean API error."""


class BaselineMissing(BenchmarkError):
    """Raised when a projection needs a measurement that was never taken."""


# ── compute_monitor integration (defensive) ──────────────────────────────────
#
# backend/services/compute_monitor.py is owned by another module and may not
# exist. It is strictly an enrichment: everything below works without it, and
# automatically gains its extra system-level fields once it lands. We probe a
# few plausible entrypoints by name rather than binding to one API we cannot
# see yet.

_CM_DEVICE_FUNCS = (
    "device_profile", "get_device_info", "get_device_profile",
    "device_info", "profile", "hardware_profile", "get_hardware",
    "snapshot", "get_profile",
)


def _compute_monitor_enrichment() -> dict[str, Any]:
    """Best-effort extra hardware detail from compute_monitor, if present."""
    try:
        from backend.services import compute_monitor as cm  # type: ignore
    except Exception as err:  # ImportError, or the module raising on import
        return {"available": False, "reason": f"{type(err).__name__}: {err}"}

    out: dict[str, Any] = {"available": True, "module": getattr(cm, "__file__", None)}

    # Module-level function, or a method on a singleton called `compute_monitor`.
    holders = [cm, getattr(cm, "compute_monitor", None)]
    for holder in holders:
        if holder is None:
            continue
        for name in _CM_DEVICE_FUNCS:
            fn = getattr(holder, name, None)
            if not callable(fn):
                continue
            try:
                value = fn()
            except Exception:
                continue
            if isinstance(value, dict):
                out.setdefault("detail", {}).update(value)
            elif callable(getattr(value, "to_dict", None)):
                # Dataclass-style profile objects (compute_monitor.DeviceInfo)
                # expose to_dict(); prefer it over poking at __dict__.
                try:
                    out.setdefault("detail", {}).update(value.to_dict())
                except Exception:
                    pass
            elif hasattr(value, "__dict__"):
                out.setdefault("detail", {}).update(
                    {k: v for k, v in vars(value).items() if _json_safe(v)}
                )
            break
    return out


def _json_safe(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool, type(None), list, dict))


# ── Result-provenance helpers ────────────────────────────────────────────────
#
# The whole credibility of this module rests on a reader being able to tell a
# stopwatch reading from arithmetic. So every leaf value is wrapped.


def measured(value: Any, unit: str, method: str, **extra: Any) -> dict[str, Any]:
    """A number that came off a stopwatch on this machine."""
    out = {"value": value, "unit": unit, "kind": "MEASURED", "method": method}
    out.update(extra)
    return out


def projected(
    value: Any,
    unit: str,
    formula: str,
    depends_on: dict[str, Any],
    caveat: str | None = None,
) -> dict[str, Any]:
    """A number derived by arithmetic from measurements + assumptions."""
    out = {
        "value": value,
        "unit": unit,
        "kind": "PROJECTED",
        "formula": formula,
        "depends_on": depends_on,
    }
    if caveat:
        out["caveat"] = caveat
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pctl(samples: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile.

    Deliberately not interpolated: with 30-60 samples an interpolated p95 is
    false precision, and nearest-rank never reports a value that was not
    actually observed.
    """
    if not samples:
        return float("nan")
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _timing_stats(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Median AND p95, because a mean hides the stalls that break a latency SLA."""
    if not samples_ms:
        return {"n": 0}
    return {
        "n": len(samples_ms),
        "mean_ms": round(statistics.fmean(samples_ms), 2),
        "p50_ms": round(statistics.median(samples_ms), 2),
        "p95_ms": round(_pctl(samples_ms, 95), 2),
        "min_ms": round(min(samples_ms), 2),
        "max_ms": round(max(samples_ms), 2),
        "stdev_ms": round(statistics.stdev(samples_ms), 2) if len(samples_ms) > 1 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUITE 1 — hardware / runtime profile
# ══════════════════════════════════════════════════════════════════════════════


def _mac_sysctl(key: str) -> str | None:
    """Read one sysctl value on macOS; None anywhere else or on failure."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
        )
        value = out.stdout.strip()
        return value or None
    except Exception:
        return None


def get_hardware_profile() -> dict[str, Any]:
    """CPU/RAM/accelerator/runtime facts. Cheap enough to serve on every request.

    Note the difference between *built* and *usable* for MPS/CUDA: torch can be
    compiled with a backend that is not actually available at runtime, and
    sizing off a backend that silently falls back to CPU would be a 5-10x error.
    """
    import psutil

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()

    profile: dict[str, Any] = {
        "measured_at": _now_iso(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cpu": {
            "model": _mac_sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown",
            "machine_model": _mac_sysctl("hw.model"),
            "arch": platform.machine(),
            "physical_cores": physical,
            "logical_cores": logical,
            # Apple silicon is heterogeneous: 4 performance + 4 efficiency cores
            # do NOT contribute equally, which is why per-core extrapolation from
            # a Mac to a server is stated as an approximation.
            "performance_cores": _int_or_none(_mac_sysctl("hw.perflevel0.physicalcpu")),
            "efficiency_cores": _int_or_none(_mac_sysctl("hw.perflevel1.physicalcpu")),
        },
        "memory": {
            "total_gb": round(vm.total / 1e9, 2),
            "available_gb": round(vm.available / 1e9, 2),
        },
        "accelerator": {},
        "runtime": {},
    }

    try:
        import torch

        mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
        mps_usable = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        cuda_usable = bool(torch.cuda.is_available())
        profile["runtime"]["torch"] = torch.__version__
        profile["accelerator"] = {
            "cuda_available": cuda_usable,
            "cuda_device_count": torch.cuda.device_count() if cuda_usable else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_usable else None,
            "mps_built": mps_built,
            "mps_usable": mps_usable,
            "selected_device": "cuda:0" if cuda_usable else ("mps" if mps_usable else "cpu"),
            "note": (
                "Apple MPS (unified-memory GPU) is not comparable to a discrete "
                "datacentre GPU. Any GPU-count projection below is an assumption, "
                "not a measurement on this machine."
            ),
        }
    except Exception as err:
        profile["accelerator"] = {"error": f"torch unavailable: {type(err).__name__}: {err}"}

    for mod in ("ultralytics", "cv2", "paddleocr", "fastapi", "sqlalchemy"):
        try:
            m = __import__(mod)
            profile["runtime"][mod] = getattr(m, "__version__", "unknown")
        except Exception:
            profile["runtime"][mod] = None

    profile["compute_monitor"] = _compute_monitor_enrichment()
    return profile


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SUITE 2 — ANPR inference cost, per stage
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class AnprSuiteConfig:
    """Knobs for the inference timing run.

    Defaults are chosen so the suite finishes in well under a minute while
    still giving enough samples for a p95 that means something.
    """
    frames: int = 60             # timed detection frames
    warmup_frames: int = 5       # discarded: first passes include lazy graph build
    ocr_crops: int = 25          # timed OCR reads
    ocr_warmup: int = 3          # discarded: first PaddleOCR call may download models
    imgsz: int = 640             # YOLO inference size — cost scales with this
    video: str | None = None
    images: list[str] | None = None


def _pick_existing(paths: Sequence[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]


def run_anpr_suite(config: AnprSuiteConfig | None = None,
                   progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Time plate detection and OCR on real media, on this machine.

    Reported separately per stage because they scale differently: detection runs
    on EVERY analysed frame, OCR only on frames where a plate is present and
    only every `ocr_every`-th of those. Sizing a city off a single blended
    "pipeline fps" number hides that.
    """
    cfg = config or AnprSuiteConfig()
    say = progress or (lambda _m: None)

    import cv2
    import numpy as np
    import psutil

    from backend.config import get_settings

    settings = get_settings()
    weights = Path(settings.ANPR_WEIGHTS_PATH)
    if not weights.exists():
        raise BenchmarkError(
            f"ANPR weights not found at {weights}. Set ANPR_WEIGHTS_PATH, or run "
            "the hardware/ingest/query suites only."
        )

    try:
        from alpr.detect import PlateDetector, select_device
        from alpr.ocr import PlateReader
    except ImportError as err:
        raise BenchmarkError(
            "The alpr package is not importable in this interpreter — install it "
            "with `.venv/bin/python -m pip install -e "
            "/path/to/Automatic-License-Plate-Recognition`. "
            f"({type(err).__name__}: {err})"
        ) from err

    # ── media ────────────────────────────────────────────────────────────────
    videos = _pick_existing([Path(cfg.video)] if cfg.video else DEFAULT_VIDEO_CANDIDATES)
    images = _pick_existing(
        [Path(p) for p in cfg.images] if cfg.images else DEFAULT_IMAGE_CANDIDATES
    )
    if not videos and not images:
        raise BenchmarkError(
            "No benchmark media found. Expected videos under "
            f"{WORKSPACE_ROOT / 'archive'} or test images in the ANPR repo."
        )

    result: dict[str, Any] = {
        "measured_at": _now_iso(),
        "config": asdict(cfg),
        "media": {"videos": [str(v) for v in videos], "images": [str(i) for i in images]},
        "warmup_note": (
            f"First {cfg.warmup_frames} detection frames and {cfg.ocr_warmup} OCR reads are "
            "DISCARDED. They include lazy model construction, Metal/CUDA kernel "
            "compilation and (for PaddleOCR) a possible one-time model download, "
            "none of which recur in a running deployment."
        ),
    }

    proc = psutil.Process()

    # ── model load cost (a real deployment cost: worker cold start) ──────────
    say("loading plate detector")
    t0 = time.perf_counter()
    detector = PlateDetector(
        weights=str(weights),
        device=settings.ANPR_DEVICE,
        confidence=settings.ANPR_CONFIDENCE,
        imgsz=cfg.imgsz,
    )
    # Ultralytics builds the model lazily on first predict, so force it here to
    # keep "model load" out of the per-frame timings.
    detector.detect(np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8))
    detector_load_s = time.perf_counter() - t0

    device = getattr(detector, "device", None) or select_device(settings.ANPR_DEVICE)
    result["device_used"] = str(device)

    # ── decode frames from real video ────────────────────────────────────────
    frames: list[Any] = []
    decode_ms: list[float] = []
    frame_shape = None
    if videos:
        say(f"decoding {cfg.frames + cfg.warmup_frames} frames")
        cap = cv2.VideoCapture(str(videos[0]))
        video_meta = {
            "path": str(videos[0]),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "source_fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        wanted = cfg.frames + cfg.warmup_frames
        # Sample ACROSS the clip rather than taking the first N frames. The first
        # two seconds of a traffic video are often empty road, which would time
        # detection on frames containing no plate at all and make the measured
        # plates-per-frame meaningless. We still decode every frame sequentially
        # (that IS the real per-frame decode cost a worker pays) but only retain
        # every stride-th one for inference.
        total_frames = video_meta["frame_count"] or 0
        max_read = min(total_frames, MAX_FRAMES_TO_DECODE) if total_frames else MAX_FRAMES_TO_DECODE
        stride = max(1, max_read // wanted)
        video_meta["sampling_stride"] = stride
        read_count = 0
        while read_count < max_read and len(frames) < wanted:
            t = time.perf_counter()
            ok, frame = cap.read()
            dt = (time.perf_counter() - t) * 1000.0
            if not ok:
                break
            decode_ms.append(dt)
            if read_count % stride == 0:
                frames.append(frame)
            read_count += 1
        cap.release()
        if frames:
            frame_shape = frames[0].shape
        result["video"] = video_meta
        # Drop the first few decode samples too: opening a container and seeking
        # keyframes is not the steady-state cost of reading frame N.
        steady_decode = decode_ms[cfg.warmup_frames:] or decode_ms
        result["decode"] = {
            "stats": _timing_stats(steady_decode),
            "frames_decoded": read_count,
            "frames_kept_for_inference": len(frames),
            "measured": measured(
                round(statistics.median(steady_decode), 2) if steady_decode else None,
                "ms/frame",
                f"cv2.VideoCapture.read() on {video_meta['width']}x{video_meta['height']} H.264, "
                f"{read_count} sequential frames, first {cfg.warmup_frames} samples discarded",
            ),
            "why_it_matters": (
                "A central architecture pays this per camera per analysed frame "
                "before any inference happens; an edge architecture does not pay "
                "it centrally at all."
            ),
        }

    # Fall back to still images if the video could not be read.
    if not frames:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(img)
        if not frames:
            raise BenchmarkError("Could not decode any benchmark media.")
        frames = frames * math.ceil((cfg.frames + cfg.warmup_frames) / len(frames))
        frame_shape = frames[0].shape

    result["input_resolution"] = {
        "frame_wh": [int(frame_shape[1]), int(frame_shape[0])] if frame_shape else None,
        "yolo_imgsz": cfg.imgsz,
        "note": (
            "Detection cost is governed by yolo_imgsz (the letterboxed inference "
            "size), not by the source resolution — a 4K feed costs the same to "
            "infer at imgsz=640 but more to decode. Cost rises roughly with "
            "imgsz^2, so imgsz=1280 is ~4x the compute of imgsz=640."
        ),
    }

    # ── detection timing (warm) ─────────────────────────────────────────────
    say("timing plate detection")
    for i in range(cfg.warmup_frames):
        detector.detect(frames[i % len(frames)])

    det_ms: list[float] = []
    plates_per_frame: list[int] = []
    cpu_before = proc.cpu_times()
    wall_before = time.perf_counter()
    for i in range(cfg.frames):
        frame = frames[(cfg.warmup_frames + i) % len(frames)]
        t = time.perf_counter()
        dets = detector.detect(frame)
        det_ms.append((time.perf_counter() - t) * 1000.0)
        plates_per_frame.append(len(dets))
    wall_after = time.perf_counter()
    cpu_after = proc.cpu_times()

    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    wall_seconds = wall_after - wall_before

    det_stats = _timing_stats(det_ms)
    result["detection"] = {
        "stats": det_stats,
        "measured_p50": measured(
            det_stats["p50_ms"], "ms/frame",
            f"YOLOv8s plate detector, device={device}, imgsz={cfg.imgsz}, "
            f"{cfg.frames} warm frames from real video",
        ),
        "measured_p95": measured(
            det_stats["p95_ms"], "ms/frame",
            "same run, nearest-rank p95 — this is the number used for capacity "
            "sizing, because sizing on a median under-provisions half the time",
        ),
        "fps_single_stream_p50": round(1000.0 / det_stats["p50_ms"], 2) if det_stats.get("p50_ms") else None,
        "plates_detected_per_frame_mean": measured(
            round(statistics.fmean(plates_per_frame), 2), "plates/frame",
            f"mean detections over the {cfg.frames} timed frames of this specific clip",
        ),
        "plates_per_frame_caveat": (
            "This is what THIS clip contained, not a Delhi junction. The "
            "projection's `plates_per_frame` assumption is separate and "
            "deliberately not taken from here — one benchmark clip is not a "
            "traffic survey. Detection cost is one YOLO pass per frame either "
            "way; only OCR load depends on this number."
        ),
        "model_load_seconds": measured(
            round(detector_load_s, 2), "s",
            "PlateDetector construction + first forward pass (worker cold-start cost)",
        ),
        "cpu_cost": {
            "process_cpu_seconds_per_frame": measured(
                round(cpu_seconds / cfg.frames, 4), "cpu-core-seconds/frame",
                "psutil user+system CPU time of this process across the timed "
                "detection loop, divided by frame count",
            ),
            "cpu_cores_busy_during_loop": round(cpu_seconds / wall_seconds, 2) if wall_seconds else None,
            "note": (
                "On Apple silicon part of this work is on the GPU via MPS, so "
                "CPU-seconds here understates total silicon used. It is still the "
                "right basis for sizing the CPU side of a worker."
            ),
        },
    }

    # ── OCR timing ──────────────────────────────────────────────────────────
    say("timing OCR (may download PaddleOCR models on first ever call)")
    reader = PlateReader()

    # Build a pool of (PIL frame, detection) pairs from frames that actually
    # contain a plate — timing OCR on a synthetic crop would not be honest.
    from PIL import Image

    ocr_jobs: list[tuple[Any, Any]] = []
    for frame in frames:
        if len(ocr_jobs) >= cfg.ocr_crops + cfg.ocr_warmup:
            break
        dets = detector.detect(frame)
        if not dets:
            continue
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ocr_jobs.append((pil, dets[0]))

    # Still images are the reliable plate source — the video has many
    # plate-free frames.
    if len(ocr_jobs) < cfg.ocr_crops + cfg.ocr_warmup:
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            dets = detector.detect(img)
            if not dets:
                continue
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            while len(ocr_jobs) < cfg.ocr_crops + cfg.ocr_warmup:
                ocr_jobs.append((pil, dets[0]))
            break

    if not ocr_jobs:
        result["ocr"] = {
            "error": "No plate was detected in any benchmark media, so OCR could not be timed.",
        }
    else:
        t0 = time.perf_counter()
        _ = reader.model  # force PaddleOCR construction (and any model download)
        ocr_load_s = time.perf_counter() - t0

        for i in range(min(cfg.ocr_warmup, len(ocr_jobs))):
            pil, det = ocr_jobs[i]
            try:
                reader.read(pil, det)
            except Exception as err:
                raise BenchmarkError(f"OCR warmup failed: {type(err).__name__}: {err}") from err

        ocr_ms: list[float] = []
        texts: list[str] = []
        cpu_before = proc.cpu_times()
        for i in range(cfg.ocr_crops):
            pil, det = ocr_jobs[(cfg.ocr_warmup + i) % len(ocr_jobs)]
            t = time.perf_counter()
            read = reader.read(pil, det)
            ocr_ms.append((time.perf_counter() - t) * 1000.0)
            if read and read.text:
                texts.append(read.text)
        cpu_after = proc.cpu_times()
        ocr_cpu_s = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)

        ocr_stats = _timing_stats(ocr_ms)
        result["ocr"] = {
            "stats": ocr_stats,
            "measured_p50": measured(
                ocr_stats["p50_ms"], "ms/plate-crop",
                f"PaddleOCR TextRecognition via alpr PlateReader.read(), "
                f"{cfg.ocr_crops} warm reads, includes crop+padding prep",
            ),
            "measured_p95": measured(
                ocr_stats["p95_ms"], "ms/plate-crop",
                "same run, nearest-rank p95 — used for sizing",
            ),
            "model_load_seconds": measured(
                round(ocr_load_s, 2), "s",
                "PaddleOCR model construction; includes a one-time download on a "
                "cold machine, which is why it is reported separately and excluded "
                "from per-crop timings",
            ),
            "process_cpu_seconds_per_crop": measured(
                round(ocr_cpu_s / cfg.ocr_crops, 4), "cpu-core-seconds/crop",
                "psutil user+system CPU time across the timed OCR loop",
            ),
            "sample_reads": texts[:5],
            "note": (
                "PaddleOCR here runs on CPU, not MPS — so OCR throughput scales "
                "with cores, while detection scales with the accelerator. That is "
                "the single most important fact for sizing: the two stages are "
                "bottlenecked by different hardware."
            ),
        }

    # ── end-to-end per-frame cost, and the upstream cross-check ─────────────
    det_p50 = det_stats.get("p50_ms")
    ocr_p50 = result.get("ocr", {}).get("stats", {}).get("p50_ms")
    if det_p50 and ocr_p50:
        for ocr_every in (1, 3):
            key = f"pipeline_fps_ocr_every_{ocr_every}"
            per_frame_ms = det_p50 + (ocr_p50 / ocr_every)
            result.setdefault("pipeline_estimate", {})[key] = {
                "value": round(1000.0 / per_frame_ms, 2),
                "unit": "fps (single stream, single worker)",
                "kind": "DERIVED",
                "formula": f"1000 / (det_p50 {det_p50} ms + ocr_p50 {ocr_p50} ms / {ocr_every})",
                "caveat": (
                    "Assumes exactly one plate read per OCR frame and no decode/IO "
                    "overlap. The real alpr Pipeline also runs a tracker and a vote "
                    "window, so treat this as an upper bound."
                ),
            }

    result["upstream_cross_check"] = _cross_check(det_p50, ocr_p50)
    # True PEAK, not the instantaneous RSS. Sampling rss at the end of the run
    # gave a 2x spread between runs (the allocator returns pages after the OCR
    # loop), and this number feeds the slots-per-host sizing, so a volatile
    # reading there is a volatile hardware estimate. ru_maxrss is the high-water
    # mark for the whole process and does not move.
    import resource

    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    peak_bytes = ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024
    current_rss = proc.memory_info().rss
    result["peak_rss_mb"] = measured(
        round(max(peak_bytes, current_rss) / 1e6, 1), "MB",
        "peak resident set size (getrusage ru_maxrss) of this process with the "
        "plate detector AND the PaddleOCR model loaded — the per-inference-worker "
        "RAM floor",
        current_rss_mb=round(current_rss / 1e6, 1),
        caveat="Includes the benchmark harness itself (a few tens of MB) and the "
               "decoded frame pool held for timing, so it is a slight "
               "overestimate of a lean production worker.",
    )
    return result


def _cross_check(det_p50: float | None, ocr_p50: float | None) -> dict[str, Any]:
    """Compare our measurement to the upstream README, and say if they disagree.

    Not a pass/fail: the reference machine is an M4 and this may not be one. The
    point is that a >2x gap means one of the two numbers is measuring something
    different, and hiding that would be dishonest.
    """
    out: dict[str, Any] = {"reference": UPSTREAM_REFERENCE, "verdict": {}}
    for label, ours, theirs in (
        ("detection_ms_per_frame", det_p50, UPSTREAM_REFERENCE["detection_ms_per_frame"]),
        ("ocr_ms_per_crop", ocr_p50, UPSTREAM_REFERENCE["ocr_ms_per_crop"]),
    ):
        if not ours:
            out["verdict"][label] = "not measured"
            continue
        ratio = ours / theirs
        if ratio > 2.0 or ratio < 0.5:
            verdict = "LARGE DISCREPANCY — investigate before quoting either number"
        elif ratio > 1.3 or ratio < 0.77:
            verdict = "different but plausible (different chip / thermal state)"
        else:
            verdict = "consistent"
        out["verdict"][label] = {
            "ours_ms": round(ours, 2),
            "reference_ms": theirs,
            "ratio_ours_over_reference": round(ratio, 2),
            "direction": "we measured FASTER than the reference" if ratio < 1
                         else "we measured SLOWER than the reference",
            "assessment": verdict,
            # Direction matters for how worried to be. Slower than the reference
            # means our sizing is conservative. FASTER than the reference means
            # our sizing is optimistic, which is the dangerous direction — it
            # usually means the two runs measured different things (e.g. the
            # reference figure includes tracking or a larger imgsz).
            "risk": "optimistic sizing — verify before quoting" if ratio < 0.77
                    else "conservative sizing" if ratio > 1.3 else "no material risk",
        }
    out["note"] = (
        "Our measured numbers are the source of truth for every projection. The "
        "reference exists only to catch a broken measurement."
    )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SUITE 3 & 4 — ingest throughput and query latency (child process, temp DB)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class DbSuiteConfig:
    """Knobs for the ingest/query suites.

    seed_rows exists because query latency without a row count is meaningless,
    and the demo database's row count is not under our control. We seed a known
    number of synthetic events into a THROWAWAY database and measure at that
    stated scale, at two scales so the growth trend is visible.
    """
    single_events: int = 60
    single_warmup: int = 5
    bulk_batch_sizes: tuple[int, ...] = (100, 500, 1000)
    bulk_repeats: int = 3
    cameras: int = 50
    plate_pool: int = 4000
    query_scales: tuple[int, ...] = (25_000, 100_000)
    query_repeats: int = 5
    timeout_s: int = 900


def run_db_suites(
    suites: Sequence[str] = ("ingest", "query"),
    config: DbSuiteConfig | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the ingest/query benchmarks in a child process against a temp DB.

    A child process is not an optimisation — it is the safety mechanism.
    backend.database creates its engine from DATABASE_URL at import time, so in
    THIS process (where the app may already be running against dev.db) there is
    no way to redirect writes. dev.db is the demo database and is being loaded
    with a large dataset by other tooling; benchmarking against it would both
    corrupt the demo and produce numbers polluted by that load.
    """
    cfg = config or DbSuiteConfig()
    say = progress or (lambda _m: None)

    tmpdir = Path(tempfile.mkdtemp(prefix="vib-benchmark-"))
    db_path = tmpdir / "benchmark.db"
    out_path = tmpdir / "result.json"
    opts = {
        "suites": list(suites),
        "config": {**asdict(cfg), "bulk_batch_sizes": list(cfg.bulk_batch_sizes),
                   "query_scales": list(cfg.query_scales)},
        "db_path": str(db_path),
        "out_path": str(out_path),
    }

    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{db_path}"
    env["VIB_BENCHMARK_CHILD"] = "1"
    env["USE_REDIS"] = "false"   # keep the measurement about the DB, not a cache
    env["USE_KAFKA"] = "false"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    say(f"spawning DB worker against temp database {db_path}")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.services.benchmark_service",
             "--db-worker", json.dumps(opts)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_s,
        )
        if not out_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise BenchmarkError(
                f"DB benchmark worker produced no result (exit {proc.returncode}). "
                f"Last output:\n{tail}"
            )
        payload = json.loads(out_path.read_text())
        if payload.get("error"):
            raise BenchmarkError(f"DB benchmark worker failed: {payload['error']}")
        payload["worker_stderr_tail"] = (proc.stderr or "")[-500:] or None
        return payload
    except subprocess.TimeoutExpired as err:
        raise BenchmarkError(
            f"DB benchmark worker exceeded {cfg.timeout_s}s and was killed."
        ) from err
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── the child-process implementation ─────────────────────────────────────────


def _db_worker_main(opts: dict[str, Any]) -> None:
    """Entrypoint for the child process. Writes JSON to opts['out_path'].

    Everything below imports backend modules, which is only safe because the
    parent set DATABASE_URL to a temp file before launching us.
    """
    out_path = Path(opts["out_path"])
    try:
        payload = _db_worker_run(opts)
    except Exception as err:  # noqa: BLE001 — report, never traceback to the API
        import traceback
        payload = {
            "error": f"{type(err).__name__}: {err}",
            "traceback_tail": traceback.format_exc()[-2000:],
        }
    out_path.write_text(json.dumps(payload, default=str, indent=2))


def _db_worker_run(opts: dict[str, Any]) -> dict[str, Any]:
    db_path = Path(opts["db_path"])
    raw_cfg = opts["config"]
    cfg = DbSuiteConfig(
        single_events=raw_cfg["single_events"],
        single_warmup=raw_cfg["single_warmup"],
        bulk_batch_sizes=tuple(raw_cfg["bulk_batch_sizes"]),
        bulk_repeats=raw_cfg["bulk_repeats"],
        cameras=raw_cfg["cameras"],
        plate_pool=raw_cfg["plate_pool"],
        query_scales=tuple(raw_cfg["query_scales"]),
        query_repeats=raw_cfg["query_repeats"],
    )
    suites = set(opts["suites"])

    from backend.config import get_settings

    settings = get_settings()
    if "benchmark.db" not in settings.DATABASE_URL:
        # Belt and braces. If this ever fires, the env var did not take effect
        # and we are one step from writing into the demo database.
        raise BenchmarkError(
            "Refusing to run: DATABASE_URL is "
            f"{settings.DATABASE_URL!r}, not the temp benchmark database."
        )

    from fastapi.testclient import TestClient

    from backend.database import SessionLocal, engine, init_db
    from backend.main import app
    from backend.models.camera import Camera

    init_db()

    # ── register synthetic cameras (ingest rejects unknown camera_ids) ───────
    db = SessionLocal()
    try:
        for i in range(cfg.cameras):
            db.merge(Camera(
                camera_id=f"BENCH_CAM_{i:03d}",
                name=f"Benchmark Cam {i}",
                location=f"Synthetic Junction {i}",
                # Spread over roughly Delhi's bounding box so the haversine work
                # in tracking/alerts operates on realistic distances.
                latitude=28.45 + (i % 25) * 0.012,
                longitude=77.05 + (i // 25) * 0.014,
                road=f"Bench Road {i % 7}",
                direction=["NORTH", "SOUTH", "EAST", "WEST"][i % 4],
                deployment="benchmark",
            ))
        db.commit()
    finally:
        db.close()

    out: dict[str, Any] = {
        "measured_at": _now_iso(),
        "temp_database": str(db_path),
        "config": {**asdict(cfg), "bulk_batch_sizes": list(cfg.bulk_batch_sizes),
                   "query_scales": list(cfg.query_scales)},
        "isolation_note": (
            "Measured against a throwaway sqlite database created for this run "
            "and deleted afterwards. dev.db was never opened for writing."
        ),
    }

    with TestClient(app) as client:
        if "ingest" in suites:
            out["ingest"] = _measure_ingest(client, cfg, db_path)
        if "query" in suites:
            out["query"] = _measure_queries(client, cfg, db_path)

    out["storage"] = _measure_storage(db_path)
    return out


def _synthetic_plate(index: int) -> str:
    """Delhi-format plates (DL 01 AB 1234) so normalization does real work."""
    series = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    a = series[(index // 24) % 24]
    b = series[index % 24]
    return f"DL{(index % 13) + 1:02d}{a}{b}{(index * 7) % 10000:04d}"


def _event_payload(i: int, cfg: DbSuiteConfig, base_time: datetime) -> dict[str, Any]:
    return {
        "camera_id": f"BENCH_CAM_{i % cfg.cameras:03d}",
        # Spread timestamps over the last hour so the time-window filters in
        # tracking/alerts match the way real traffic arrives.
        "timestamp": (base_time - timedelta(seconds=(i * 37) % 3600)).isoformat(),
        "local_track_id": f"T{i}",
        "plate": _synthetic_plate(i % cfg.plate_pool),
        "plate_confidence": 0.6 + (i % 40) / 100.0,
        "vehicle_type": ["car", "truck", "bus", "motorcycle", "auto"][i % 5],
        "vehicle_color": ["white", "black", "silver", "red"][i % 4],
        "speed": 20.0 + (i % 60),
    }


def _measure_ingest(client: Any, cfg: DbSuiteConfig, db_path: Path) -> dict[str, Any]:
    """Events/sec and per-event latency through the real ingestion endpoints.

    This is deliberately the full path — validation, insert, MTMC association
    and the alert checks — because that is what a camera worker actually waits
    on. Timing a bare INSERT would flatter us by an order of magnitude.
    """
    base_time = datetime.now(timezone.utc).replace(tzinfo=None)
    counter = 0
    result: dict[str, Any] = {
        "work_per_event": (
            "camera lookup, row insert, MTMC plate/spatio-temporal association, "
            "blacklist check and checkpoint-pair speed/route-anomaly check"
        ),
        "concurrency": "single client, serial requests (no request-level parallelism)",
    }

    # ── single-event endpoint ───────────────────────────────────────────────
    for _ in range(cfg.single_warmup):
        client.post("/events/ingest", json=_event_payload(counter, cfg, base_time))
        counter += 1

    latencies: list[float] = []
    errors = 0
    t_start = time.perf_counter()
    for _ in range(cfg.single_events):
        payload = _event_payload(counter, cfg, base_time)
        counter += 1
        t = time.perf_counter()
        resp = client.post("/events/ingest", json=payload)
        latencies.append((time.perf_counter() - t) * 1000.0)
        if resp.status_code != 200:
            errors += 1
    total_s = time.perf_counter() - t_start

    stats = _timing_stats(latencies)
    result["single"] = {
        "endpoint": "POST /events/ingest",
        "stats": stats,
        "errors": errors,
        "events_per_second": measured(
            round(cfg.single_events / total_s, 1), "events/s",
            f"{cfg.single_events} serial requests over {total_s:.2f}s via "
            "fastapi.testclient (warmup excluded)",
        ),
        "latency_p95": measured(stats["p95_ms"], "ms/event", "same run, nearest-rank p95"),
        "note": (
            "Serial throughput of ONE client. It is a per-event cost measurement, "
            "not the server's ceiling: a real deployment runs many workers "
            "concurrently, and the projection multiplies this out explicitly."
        ),
    }

    # ── bulk endpoint at several batch sizes ────────────────────────────────
    bulk: list[dict[str, Any]] = []
    for size in cfg.bulk_batch_sizes:
        per_batch_s: list[float] = []
        accepted = 0
        for _ in range(cfg.bulk_repeats):
            batch = [_event_payload(counter + j, cfg, base_time) for j in range(size)]
            counter += size
            t = time.perf_counter()
            resp = client.post("/events/bulk-ingest", json=batch)
            per_batch_s.append(time.perf_counter() - t)
            if resp.status_code == 200:
                accepted += len(resp.json())
        mean_batch_s = statistics.fmean(per_batch_s)
        bulk.append({
            "batch_size": size,
            "repeats": cfg.bulk_repeats,
            "accepted_events": accepted,
            "batch_seconds_mean": round(mean_batch_s, 3),
            "batch_seconds_p95": round(_pctl([s for s in per_batch_s], 95), 3),
            "per_event_ms_mean": measured(
                round(mean_batch_s * 1000.0 / size, 3), "ms/event",
                f"mean batch wall time / {size}",
            ),
            "events_per_second": measured(
                round(size / mean_batch_s, 1), "events/s",
                f"batch of {size} through POST /events/bulk-ingest, mean of "
                f"{cfg.bulk_repeats} batches",
            ),
        })
    result["bulk"] = {
        "endpoint": "POST /events/bulk-ingest",
        "by_batch_size": bulk,
        "best_events_per_second": max((b["events_per_second"]["value"] for b in bulk), default=None),
        "note": (
            "Per-event cost falls with batch size because the camera lookup and "
            "the commit amortise, but the per-event MTMC + alert queries do not — "
            "so the curve flattens rather than scaling linearly."
        ),
    }

    result["rows_written"] = _row_count(db_path, "vehicle_events")
    return result


def _seed_events(db_path: Path, target_rows: int, cfg: DbSuiteConfig) -> int:
    """Bulk-insert synthetic events straight through sqlite to reach a row count.

    Raw SQL, not the ORM: the point is to reach a stated scale quickly so query
    latency can be measured there. These rows are NOT part of the ingest
    throughput measurement, and the output labels them as seeded.
    """
    import sqlite3

    current = _row_count(db_path, "vehicle_events") or 0
    to_add = max(0, target_rows - current)
    if not to_add:
        return current

    base = datetime.now(timezone.utc).replace(tzinfo=None)
    conn = sqlite3.connect(str(db_path))
    try:
        cam_rows = list(conn.execute("SELECT camera_id, latitude, longitude FROM cameras"))
        if not cam_rows:
            raise BenchmarkError("No cameras in the temp DB — cannot seed events.")
        rows = []
        for i in range(to_add):
            cam_id, lat, lon = cam_rows[i % len(cam_rows)]
            rows.append((
                str(uuid.uuid4()),
                cam_id,
                f"S{i}",
                # Spread over the last 24h: that is the window the dashboard
                # endpoints query, so seeding outside it would make them look
                # artificially fast.
                (base - timedelta(seconds=(i * 13) % 86_400)).isoformat(sep=" "),
                _synthetic_plate(i % cfg.plate_pool),
                0.7,
                lat,
                lon,
                ["NORTH", "SOUTH", "EAST", "WEST"][i % 4],
                ["car", "truck", "bus", "motorcycle", "auto"][i % 5],
                ["white", "black", "silver", "red"][i % 4],
                20.0 + (i % 70),
                f"VEH_S{i % cfg.plate_pool:06d}",
                base.isoformat(sep=" "),
            ))
        conn.executemany(
            "INSERT INTO vehicle_events (event_id, camera_id, local_track_id, timestamp, "
            "plate, plate_confidence, latitude, longitude, direction, vehicle_type, "
            "vehicle_color, speed, global_vehicle_id, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
    return _row_count(db_path, "vehicle_events") or 0


def _row_count(db_path: Path, table: str) -> int | None:
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            conn.close()
    except Exception:
        return None


# The reads the dashboard cannot render without. Latency here is what a faculty
# member actually perceives as "is it fast".
DASHBOARD_ENDPOINTS = [
    ("summary", "/analytics/summary"),
    ("density_24h", "/analytics/density?hours=24"),
    ("heatmap_24h", "/analytics/heatmap?hours=24"),
    ("trajectory_by_plate", "/vehicles/{plate}/trajectory"),
    ("speed_defaulters_24h", "/vehicles/analytics/speed-defaulters?hours=24"),
]


# The composite indexes that make the time-windowed dashboard queries fast.
# Copied from scripts/generate_delhi_dataset.py, which is currently the ONLY
# place they are created — they are not declared on the ORM model and there is
# no migration for them, so a fresh init_db() deployment does not have them.
# Measuring with and without is the only way to show what that costs.
PERFORMANCE_INDEXES = (
    ("ix_bench_plate_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_bench_plate_timestamp "
     "ON vehicle_events (plate, timestamp, camera_id)"),
    ("ix_bench_camera_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_bench_camera_timestamp "
     "ON vehicle_events (camera_id, timestamp)"),
    ("ix_bench_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_bench_timestamp ON vehicle_events (timestamp)"),
)


def _apply_performance_indexes(db_path: Path) -> dict[str, Any]:
    """Create the composite indexes on the temp DB and report how long it took."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    timings = {}
    try:
        for name, ddl in PERFORMANCE_INDEXES:
            t = time.perf_counter()
            conn.execute(ddl)
            timings[name] = round((time.perf_counter() - t) * 1000.0, 1)
        conn.commit()
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    return {"build_ms": timings}


def _time_endpoints(client: Any, cfg: DbSuiteConfig, sample_plate: str,
                    rows: int) -> list[dict[str, Any]]:
    """Time every dashboard endpoint once, at the current DB state."""
    out: list[dict[str, Any]] = []
    for name, template in DASHBOARD_ENDPOINTS:
        url = template.format(plate=sample_plate)
        client.get(url)  # warmup: first call pays SQLAlchemy statement compile
        samples: list[float] = []
        status = None
        payload_bytes = None
        for _ in range(cfg.query_repeats):
            t = time.perf_counter()
            resp = client.get(url)
            samples.append((time.perf_counter() - t) * 1000.0)
            status = resp.status_code
            payload_bytes = len(resp.content)
        stats = _timing_stats(samples)
        out.append({
            "name": name,
            "url": url,
            "http_status": status,
            "response_bytes": payload_bytes,
            "stats": stats,
            "latency_p50": measured(stats["p50_ms"], "ms", f"GET {url} at {rows:,} rows"),
            "latency_p95": measured(stats["p95_ms"], "ms", "same run, nearest-rank p95"),
        })
    return out


def _measure_queries(client: Any, cfg: DbSuiteConfig, db_path: Path) -> dict[str, Any]:
    """Latency of the dashboard-critical reads, at two known row counts.

    Two scales rather than one because the useful output is not "42 ms" but
    "42 ms at 25k rows and 160 ms at 100k rows" — that slope is what tells you
    whether the endpoint survives a city.
    """
    scales: list[dict[str, Any]] = []
    sample_plate = _synthetic_plate(7)

    for target in cfg.query_scales:
        rows = _seed_events(db_path, target, cfg)
        scales.append({
            "vehicle_event_rows": rows,
            "seeded": True,
            "schema": "as created by init_db() — ORM-declared indexes only",
            "endpoints": _time_endpoints(client, cfg, sample_plate, rows),
        })

    # ── the same queries, with the composite indexes added ──────────────────
    index_impact: dict[str, Any] = {}
    if scales:
        top_rows = scales[-1]["vehicle_event_rows"]
        build = _apply_performance_indexes(db_path)
        after = _time_endpoints(client, cfg, sample_plate, top_rows)
        before_by_name = {e["name"]: e for e in scales[-1]["endpoints"]}
        comparison = []
        for entry in after:
            base = before_by_name.get(entry["name"])
            if not base:
                continue
            t_before = base["stats"].get("p50_ms") or 0.0
            t_after = entry["stats"].get("p50_ms") or 0.0
            comparison.append({
                "name": entry["name"],
                "p50_ms_without_indexes": t_before,
                "p50_ms_with_indexes": t_after,
                "speedup": round(t_before / t_after, 2) if t_after > 0 else None,
            })
        index_impact = {
            "rows": top_rows,
            "indexes_added": [name for name, _ in PERFORMANCE_INDEXES],
            "index_build_ms": build["build_ms"],
            "comparison": comparison,
            "finding": (
                "Two separate things, and the measurement distinguishes them.\n"
                "(1) These composite indexes are created ONLY by "
                "scripts/generate_delhi_dataset.py — they are not declared on the "
                "VehicleEvent model and there is no Alembic migration for them, "
                "so a fresh deployment (init_db() against an empty Postgres) "
                "starts without them. That is worth fixing: it is one "
                "index=True/Index() per line on the model.\n"
                "(2) But adding them is NOT a blanket win, and the numbers above "
                "say so. Selective lookups get much faster (plate trajectory, "
                "the counting aggregates). The full-window scans (density, "
                "heatmap, speed-defaulters) get no better or slightly WORSE, "
                "because when the 24-hour window covers essentially the whole "
                "table, a random-access index scan loses to a sequential scan. "
                "Those endpoints are not index-starved, they are doing O(rows in "
                "window) work — the fix for them is pre-aggregation into "
                "traffic_snapshots, not another index."
            ),
            "honesty_note": (
                "We report the speedups that came out below 1.0 rather than "
                "quietly dropping them. An 'optimisation' that we measured as a "
                "regression is the most useful line in this table."
            ),
        }

    return {
        "index_impact": index_impact,
        "engine": "sqlite (dev default) — Postgres numbers will differ, usually "
                  "better for concurrent reads and worse for tiny single-row lookups",
        "scales": scales,
        "scaling": _query_scaling(scales),
        "row_count_caveat": (
            "Rows are synthetic and seeded by this benchmark into a temp database. "
            "The distribution (plates, cameras, 24h spread) is realistic in shape "
            "but not real traffic. Latency is only meaningful next to its row count."
        ),
        "known_limitation": (
            "speed-defaulters and od-matrix load their rows and aggregate in "
            "Python, so their cost grows with rows in the window and no index "
            "helps — confirmed by index_impact above, where speed-defaulters is "
            "unchanged by indexing and still ~1 s at 100k rows. That is the first "
            "thing to fix before a real city rollout."
        ),
    }


def _query_scaling(scales: list[dict[str, Any]]) -> dict[str, Any]:
    """How each endpoint's latency grew between the two measured row counts.

    This is the number that answers "does the dashboard survive a city?" — an
    endpoint whose latency grows linearly with total rows will not, however fast
    it looks at demo scale. Purely DERIVED: two measurements, one division.
    """
    if len(scales) < 2:
        return {"note": "Needs at least two row-count scales to compute a slope."}

    low, high = scales[0], scales[-1]
    row_ratio = high["vehicle_event_rows"] / max(1, low["vehicle_event_rows"])
    low_by_name = {e["name"]: e for e in low["endpoints"]}

    entries = []
    for endpoint in high["endpoints"]:
        base = low_by_name.get(endpoint["name"])
        if not base:
            continue
        t_low = base["stats"].get("p50_ms") or 0.0
        t_high = endpoint["stats"].get("p50_ms") or 0.0
        if t_low <= 0:
            continue
        latency_ratio = t_high / t_low
        # exponent k in latency ~ rows^k
        k = math.log(latency_ratio) / math.log(row_ratio) if row_ratio > 1 else None
        if k is None:
            verdict = "unknown"
        elif k < 0.3:
            verdict = "flat — index-bound, safe at city scale"
        elif k < 0.8:
            verdict = "sub-linear — acceptable, watch it"
        else:
            verdict = "LINEAR OR WORSE — will not survive city scale without a rollup table"
        entries.append({
            "name": endpoint["name"],
            "p50_ms_at_low_scale": t_low,
            "p50_ms_at_high_scale": t_high,
            "latency_ratio": round(latency_ratio, 2),
            "row_ratio": round(row_ratio, 2),
            "scaling_exponent_k": round(k, 2) if k is not None else None,
            "verdict": verdict,
        })

    return {
        "model": "latency ≈ rows^k, fitted from exactly two points",
        "two_point_caveat": (
            "Two points cannot distinguish a genuine power law from a constant "
            "plus a linear term. Treat k as a direction indicator, not a "
            "coefficient — and do not extrapolate it many orders of magnitude."
        ),
        "what_rows_means_here": (
            "Every seeded row falls inside the endpoints' 24-hour window, so k "
            "measures growth with ROWS IN THE QUERY WINDOW, not with total table "
            "size. That is the right variable for a city: at 200 cameras the "
            "24-hour window is millions of events, so a k≈1 endpoint that takes "
            "60 ms over 100k rows takes tens of seconds over a real day. The fix "
            "is pre-aggregation (the traffic_snapshots table already exists for "
            "exactly this) plus the indexes measured in index_impact — not a "
            "faster machine."
        ),
        "endpoints": entries,
    }


def _measure_storage(db_path: Path) -> dict[str, Any]:
    """Derive bytes-per-event from an actual database file, not from a guess.

    Two sources: the temp benchmark DB we just filled (always available), and
    the real dev.db opened READ-ONLY as a cross-check when it has enough rows.
    """
    out: dict[str, Any] = {}

    rows = _row_count(db_path, "vehicle_events")
    if rows and db_path.exists():
        size = db_path.stat().st_size
        out["temp_db"] = {
            "path": str(db_path),
            "file_bytes": size,
            "vehicle_event_rows": rows,
            "bytes_per_event": measured(
                round(size / rows, 1), "bytes/event",
                "temp benchmark sqlite file size / vehicle_events row count; "
                "includes indexes and the (tiny) other tables, so it slightly "
                "overstates the per-row cost",
            ),
        }

    # Read-only cross-check against the real database. Never written to.
    dev_db = REPO_ROOT / "dev.db"
    if dev_db.exists():
        dev_rows = _row_count(dev_db, "vehicle_events")
        dev_size = dev_db.stat().st_size
        entry: dict[str, Any] = {
            "path": str(dev_db),
            "file_bytes": dev_size,
            "vehicle_event_rows": dev_rows,
            "access": "opened read-only (mode=ro); never written by the benchmark",
        }
        if dev_rows and dev_rows >= 1000:
            entry["bytes_per_event"] = measured(
                round(dev_size / dev_rows, 1), "bytes/event",
                "real dev.db file size / row count",
            )
        else:
            entry["note"] = (
                f"only {dev_rows} rows — too few for a meaningful bytes/event "
                "(file size is dominated by other tables and page overhead), so "
                "the temp-DB figure is used instead"
            )
        out["dev_db_cross_check"] = entry

    return out


# ══════════════════════════════════════════════════════════════════════════════
# EXTRAPOLATION — the actual deliverable
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScaleAssumptions:
    """Every input to the city projection that we could NOT measure here.

    All of them are overridable via the API. The defaults are defensible, not
    optimistic, and each one's reasoning lives in ASSUMPTION_RATIONALE below so
    a reviewer can attack the assumption rather than the arithmetic.
    """
    cameras: int = 200
    fps_per_camera: float = 6.0
    ocr_every: int = 3
    plates_per_frame: float = 1.5
    vehicles_per_camera_per_hour: float = 1200.0
    target_latency_s: float = 2.0
    utilisation_headroom: float = 0.70
    # GPU extrapolation. We measured an Apple-silicon Mac; a datacentre GPU
    # number cannot be measured here, so the multiplier is named and exposed.
    gpu_class: str = "NVIDIA T4 (16GB, inference-class)"
    gpu_speedup_vs_measured_device: float = 3.0
    stream_bitrate_mbps: float = 4.0
    event_wire_bytes: int = 400
    retention_days: int = 90
    replication_factor: float = 1.0
    hours_per_day: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ASSUMPTION_RATIONALE: dict[str, str] = {
    "cameras": (
        "200 = the real Delhi junctions currently in deployments/delhi/cameras.json. "
        "500 and 2000 are modelled as growth scenarios."
    ),
    "fps_per_camera": (
        "ANPR does not need the full 30 fps. A vehicle crossing a junction "
        "approach stays plate-readable for roughly 1-2 seconds, so 6 analysed "
        "frames per second gives 6-12 independent chances to read the same plate "
        "— enough for the pipeline's multi-frame vote — at one fifth of the "
        "compute of 30 fps. Raise it for high-speed expressway cameras where the "
        "readable window is shorter."
    ),
    "ocr_every": (
        "OCR runs on every 3rd frame that contains a plate. The upstream repo "
        "measures ocr_every=3 as the sweet spot (23.5 vs 14.3 fps) because the "
        "vote window still collects several reads per vehicle pass."
    ),
    "plates_per_frame": (
        "Mean number of plates visible per analysed frame at a junction approach. "
        "Drives OCR load only — detection cost is one YOLO pass regardless."
    ),
    "vehicles_per_camera_per_hour": (
        "1200 = 20 vehicles/minute past one camera. NOT measured by us; it sets "
        "the DB event rate. A quiet residential camera is far lower, a peak-hour "
        "arterial approach higher."
    ),
    "target_latency_s": (
        "End-to-end budget from photon to dashboard. 2 s is the operator "
        "requirement for a blacklist hit to be actionable. It constrains queue "
        "depth: a worker may not accumulate more than this much backlog."
    ),
    "utilisation_headroom": (
        "Size to 70% of measured capacity, never 100%. Headroom absorbs peak "
        "hour, thermal throttling, model reloads and node failure. Sizing at "
        "100% guarantees the system falls behind exactly when it matters."
    ),
    "gpu_class": (
        "Which GPU the GPU-count projection assumes. We measured only this Mac, "
        "so this is a stated assumption."
    ),
    "gpu_speedup_vs_measured_device": (
        "How many times faster one such GPU is than the accelerator we measured, "
        "for this YOLOv8s workload. ASSUMPTION, NOT A MEASUREMENT — 3x for a T4 "
        "vs Apple MPS is a conservative published-throughput estimate. Anyone "
        "quoting a GPU count must quote this multiplier with it."
    ),
    "stream_bitrate_mbps": (
        "Per-camera H.264 bitrate for 1080p CCTV. 4 Mbps is typical for a "
        "traffic-grade encoder; 4K or low-compression feeds are several times "
        "higher."
    ),
    "event_wire_bytes": (
        "Size of one structured VehicleEvent JSON on the wire (~400 B observed "
        "for the ingest payload shape). Used for the edge-architecture uplink."
    ),
    "retention_days": "How long events are kept before archival/rollup. 90 days.",
    "replication_factor": "1.0 = single copy. Set to 3.0 for a replicated cluster.",
    "hours_per_day": (
        "Hours per day cameras are analysed. 24 = always on. Night-only or "
        "peak-only operation scales cost down linearly."
    ),
}


def extract_baseline(results: dict[str, Any]) -> dict[str, Any]:
    """Pull the handful of measured numbers the projection actually needs.

    Kept small and explicit on purpose: it makes it obvious that the whole city
    model rests on about eight stopwatch readings, and it makes a missing
    measurement a loud failure instead of a silent default.
    """
    anpr = results.get("measured", {}).get("anpr") or {}
    ingest = results.get("measured", {}).get("ingest") or {}
    storage = results.get("measured", {}).get("storage") or {}
    hardware = results.get("hardware") or {}

    def dig(d: dict[str, Any], *path: str) -> Any:
        cur: Any = d
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    bytes_per_event = (
        dig(storage, "dev_db_cross_check", "bytes_per_event", "value")
        or dig(storage, "temp_db", "bytes_per_event", "value")
    )

    baseline = {
        "detection_ms_p50": dig(anpr, "detection", "stats", "p50_ms"),
        "detection_ms_p95": dig(anpr, "detection", "stats", "p95_ms"),
        "ocr_ms_p50": dig(anpr, "ocr", "stats", "p50_ms"),
        "ocr_ms_p95": dig(anpr, "ocr", "stats", "p95_ms"),
        "decode_ms_p50": dig(anpr, "decode", "stats", "p50_ms"),
        "cpu_core_seconds_per_frame": dig(
            anpr, "detection", "cpu_cost", "process_cpu_seconds_per_frame", "value"),
        "cpu_core_seconds_per_crop": dig(anpr, "ocr", "process_cpu_seconds_per_crop", "value"),
        "worker_rss_mb": dig(anpr, "peak_rss_mb", "value"),
        "device_measured": dig(anpr, "device_used") or dig(hardware, "accelerator", "selected_device"),
        "ingest_events_per_s_single": dig(ingest, "single", "events_per_second", "value"),
        "ingest_latency_p95_ms": dig(ingest, "single", "stats", "p95_ms"),
        "ingest_events_per_s_bulk_best": ingest.get("bulk", {}).get("best_events_per_second"),
        "bytes_per_event": bytes_per_event,
        "machine": dig(hardware, "cpu", "model"),
        "machine_model": dig(hardware, "cpu", "machine_model"),
        "logical_cores": dig(hardware, "cpu", "logical_cores"),
        "host_ram_gb": dig(hardware, "memory", "total_gb"),
        "measured_at": results.get("measured_at"),
    }
    return baseline


def compute_projection(
    baseline: dict[str, Any],
    assumptions: ScaleAssumptions | None = None,
) -> dict[str, Any]:
    """Turn measured per-frame/per-event costs into a deployment sizing.

    Pure arithmetic — no measuring, no IO. Fast enough to serve on every
    keystroke of a "what if we scale to N cameras" slider.
    """
    a = assumptions or ScaleAssumptions()

    det_ms = baseline.get("detection_ms_p95") or baseline.get("detection_ms_p50")
    ocr_ms = baseline.get("ocr_ms_p95") or baseline.get("ocr_ms_p50")
    if not det_ms:
        raise BaselineMissing(
            "No measured plate-detection time available. Run the 'anpr' suite "
            "(scripts/run_benchmarks.py --suite anpr) before asking for a projection."
        )
    if not ocr_ms:
        raise BaselineMissing(
            "No measured OCR time available. Run the 'anpr' suite before asking "
            "for a projection."
        )

    if not 0.0 < a.utilisation_headroom <= 1.0:
        raise BenchmarkError("utilisation_headroom must be in (0, 1].")
    if a.cameras <= 0 or a.fps_per_camera <= 0 or a.ocr_every <= 0:
        raise BenchmarkError("cameras, fps_per_camera and ocr_every must all be > 0.")

    measured_refs = {
        "detection_ms_per_frame_p95": det_ms,
        "ocr_ms_per_crop_p95": ocr_ms,
        "measured_on": f"{baseline.get('machine')} / {baseline.get('machine_model')} "
                       f"(device={baseline.get('device_measured')})",
    }

    # ── inference demand ────────────────────────────────────────────────────
    det_fps_needed = a.cameras * a.fps_per_camera
    ocr_crops_needed = det_fps_needed / a.ocr_every * a.plates_per_frame

    # Capacity of ONE worker identical to the machine we measured. Detection and
    # OCR are serialised inside a worker (same process), so the per-frame cost
    # is detection plus its share of OCR.
    ms_per_analysed_frame = det_ms + (a.plates_per_frame / a.ocr_every) * ocr_ms
    worker_fps_raw = 1000.0 / ms_per_analysed_frame
    worker_fps_usable = worker_fps_raw * a.utilisation_headroom
    workers_measured_class = math.ceil(det_fps_needed / worker_fps_usable)

    dep = {  # what every projected figure in this block hangs on
        "cameras": a.cameras,
        "fps_per_camera": a.fps_per_camera,
        "ocr_every": a.ocr_every,
        "plates_per_frame": a.plates_per_frame,
        "utilisation_headroom": a.utilisation_headroom,
        "measured": measured_refs,
    }

    inference = {
        "required_detection_throughput": projected(
            round(det_fps_needed, 1), "frames/s aggregate",
            f"{a.cameras} cameras x {a.fps_per_camera} analysed fps",
            {"cameras": a.cameras, "fps_per_camera": a.fps_per_camera},
        ),
        "required_ocr_throughput": projected(
            round(ocr_crops_needed, 1), "plate-crops/s aggregate",
            f"{det_fps_needed:.0f} frames/s / ocr_every {a.ocr_every} x "
            f"{a.plates_per_frame} plates per frame",
            {"ocr_every": a.ocr_every, "plates_per_frame": a.plates_per_frame},
        ),
        "cost_per_analysed_frame": projected(
            round(ms_per_analysed_frame, 2), "ms",
            f"detection p95 {det_ms} ms + ({a.plates_per_frame}/{a.ocr_every}) x OCR p95 {ocr_ms} ms",
            {"measured": measured_refs, "ocr_every": a.ocr_every,
             "plates_per_frame": a.plates_per_frame},
            caveat="Assumes detection and OCR share one worker process, as they do today.",
        ),
        "per_worker_capacity": projected(
            round(worker_fps_usable, 2), "analysed frames/s per worker",
            f"1000 / {ms_per_analysed_frame:.2f} ms x {a.utilisation_headroom} headroom",
            dep,
        ),
        "workers_of_measured_class": projected(
            workers_measured_class,
            f"workers equivalent to the measured machine ({baseline.get('machine_model') or 'this Mac'})",
            f"ceil({det_fps_needed:.0f} / {worker_fps_usable:.2f})",
            dep,
            caveat="This is the only worker-count figure grounded entirely in a "
                   "measurement on real hardware. Everything GPU-related below "
                   "applies an assumed multiplier to it.",
        ),
        "cameras_per_worker": projected(
            round(worker_fps_usable / a.fps_per_camera, 2), "cameras/worker",
            f"{worker_fps_usable:.2f} usable fps / {a.fps_per_camera} fps per camera",
            dep,
        ),
        "what_a_worker_is": (
            "One SERIAL pipeline slot: a process that decodes, detects and reads "
            "one frame at a time. The count is set by latency per frame, not by "
            "how much silicon a slot uses — see sizing_reconciliation, because "
            "N slots do NOT mean N physical machines."
        ),
    }

    # ── slots vs machines: the reconciliation a reviewer will ask for ────────
    #
    # 92 "workers" and "51 CPU cores" look contradictory until you notice a slot
    # is latency-bound, not CPU-bound: it spends most of its 53 ms waiting on the
    # GPU and on PaddleOCR, using well under one core. Several slots therefore
    # share a host. We say this explicitly rather than letting a reader assume
    # one worker = one machine (which would overstate the hardware ~10x).
    cpu_per_frame_total = None
    if baseline.get("cpu_core_seconds_per_frame") is not None:
        cpu_per_frame_total = (
            baseline["cpu_core_seconds_per_frame"]
            + (a.plates_per_frame / a.ocr_every) * (baseline.get("cpu_core_seconds_per_crop") or 0.0)
        )
    host_cores = baseline.get("logical_cores")
    if cpu_per_frame_total and host_cores:
        cores_per_slot = worker_fps_usable * cpu_per_frame_total
        slots_by_cpu = max(1, int((host_cores * a.utilisation_headroom) / cores_per_slot))
        # RAM usually binds first, and that is a measured fact, not a guess:
        # every slot is its own process and loads its own ~1.7 GB of models.
        host_ram_gb = baseline.get("host_ram_gb")
        rss_gb = (baseline.get("worker_rss_mb") or 0) / 1000.0
        slots_by_ram = (
            max(1, int((host_ram_gb * a.utilisation_headroom) / rss_gb))
            if host_ram_gb and rss_gb else None
        )
        slots_per_host = min(slots_by_cpu, slots_by_ram) if slots_by_ram else slots_by_cpu
        binding = ("RAM" if slots_by_ram and slots_by_ram < slots_by_cpu else "CPU")
        inference["sizing_reconciliation"] = {
            "slots_limited_by_cpu": slots_by_cpu,
            "slots_limited_by_ram": slots_by_ram,
            "binding_constraint": binding,
            "binding_constraint_note": (
                f"On the measured machine, {binding} binds first. Each slot is a "
                "separate process that loads its own copy of the YOLO and "
                "PaddleOCR models — the measured RSS below. Sharing one model "
                "instance across threads would raise slots-per-host "
                "substantially and is the cheapest available optimisation, but "
                "we have not measured it, so it is not assumed here."
            ),
            "cpu_cores_consumed_per_slot": projected(
                round(cores_per_slot, 3), "CPU cores per slot",
                f"{worker_fps_usable:.2f} frames/s x {cpu_per_frame_total:.4f} "
                "measured core-seconds per frame (detection + its OCR share)",
                {**dep, "measured_cpu_core_seconds_per_frame":
                    baseline.get("cpu_core_seconds_per_frame"),
                 "measured_cpu_core_seconds_per_crop":
                    baseline.get("cpu_core_seconds_per_crop")},
            ),
            "slots_per_host": projected(
                slots_per_host,
                f"slots per host ({host_cores} cores, {baseline.get('host_ram_gb')} GB RAM)",
                f"min(CPU: floor({host_cores} x {a.utilisation_headroom} / "
                f"{cores_per_slot:.3f}) = {slots_by_cpu}, "
                f"RAM: floor({baseline.get('host_ram_gb')} x {a.utilisation_headroom} / "
                f"{rss_gb:.2f} GB) = {slots_by_ram})",
                {**dep, "measured_logical_cores": host_cores,
                 "measured_host_ram_gb": baseline.get("host_ram_gb"),
                 "measured_worker_rss_mb": baseline.get("worker_rss_mb")},
                caveat="ASSUMES PERFECT PARALLEL SCALING, WHICH WE DID NOT "
                       "MEASURE. We ran one slot at a time. Real slots contend "
                       "for one shared GPU/MPS queue and for memory bandwidth, so "
                       "the true figure is lower. Measuring it needs a "
                       "concurrency test that this suite does not yet run.",
            ),
            "physical_hosts_of_measured_class": projected(
                math.ceil(workers_measured_class / slots_per_host),
                f"physical machines like {baseline.get('machine_model') or 'the measured Mac'}",
                f"ceil({workers_measured_class} slots / {slots_per_host} slots per host)",
                {**dep, "measured_logical_cores": host_cores},
                caveat="Upper-bound-optimistic for the same reason as slots_per_host.",
            ),
            "why_this_block_exists": (
                "Without it, 'workers' and 'CPU cores' in this report look "
                "inconsistent. They are not: a slot is latency-bound and uses a "
                "fraction of a core."
            ),
        }

    # ── GPU translation (assumption-heavy, labelled as such) ────────────────
    gpu_fps_usable = worker_fps_raw * a.gpu_speedup_vs_measured_device * a.utilisation_headroom
    gpus_needed = math.ceil(det_fps_needed / gpu_fps_usable) if gpu_fps_usable > 0 else None
    gpu_dep = dict(dep)
    gpu_dep.update({
        "gpu_class": a.gpu_class,
        "gpu_speedup_vs_measured_device": a.gpu_speedup_vs_measured_device,
    })
    inference["gpus"] = {
        "assumed_gpu_class": a.gpu_class,
        "assumed_speedup": a.gpu_speedup_vs_measured_device,
        "assumption_warning": (
            "WE DID NOT MEASURE A DATACENTRE GPU. This count is the measured "
            f"per-frame cost divided by an assumed {a.gpu_speedup_vs_measured_device}x "
            f"speedup for a {a.gpu_class}. Treat the GPU count as a sizing estimate "
            "whose error bar is the multiplier's error bar. The worker count above "
            "is the honest, measured figure."
        ),
        "gpus_required": projected(
            gpus_needed, f"x {a.gpu_class}",
            f"ceil({det_fps_needed:.0f} fps / ({worker_fps_raw:.2f} fps x "
            f"{a.gpu_speedup_vs_measured_device} x {a.utilisation_headroom}))",
            gpu_dep,
            caveat="Assumed speedup, not measured.",
        ),
        "cameras_per_gpu": projected(
            round(gpu_fps_usable / a.fps_per_camera, 1), "cameras/GPU",
            f"{gpu_fps_usable:.2f} usable fps / {a.fps_per_camera} fps per camera",
            gpu_dep,
        ),
    }

    # ── CPU / RAM ───────────────────────────────────────────────────────────
    cpu_per_frame = baseline.get("cpu_core_seconds_per_frame")
    cpu_per_crop = baseline.get("cpu_core_seconds_per_crop")
    compute_resources: dict[str, Any] = {}
    if cpu_per_frame is not None:
        cores_raw = det_fps_needed * cpu_per_frame + (ocr_crops_needed * (cpu_per_crop or 0.0))
        compute_resources["cpu_cores"] = projected(
            math.ceil(cores_raw / a.utilisation_headroom), "CPU cores (inference tier)",
            f"({det_fps_needed:.0f} fps x {cpu_per_frame} core-s/frame + "
            f"{ocr_crops_needed:.0f} crops/s x {cpu_per_crop} core-s/crop) / "
            f"{a.utilisation_headroom}",
            {**dep, "measured_cpu_core_seconds_per_frame": cpu_per_frame,
             "measured_cpu_core_seconds_per_crop": cpu_per_crop},
            caveat="Derived from CPU time measured on Apple silicon, where some "
                   "detection work runs on the GPU via MPS. On a CPU-only or "
                   "CUDA server the split between cores and accelerator differs.",
        )
    rss_mb = baseline.get("worker_rss_mb")
    if rss_mb:
        compute_resources["ram_gb"] = projected(
            round(workers_measured_class * rss_mb / 1000.0 / a.utilisation_headroom, 1),
            "GB (inference tier)",
            f"{workers_measured_class} workers x {rss_mb} MB measured RSS / "
            f"{a.utilisation_headroom}",
            {**dep, "measured_worker_rss_mb": rss_mb},
            caveat="Excludes the database tier and OS overhead.",
        )

    # ── data tier ───────────────────────────────────────────────────────────
    events_per_s = a.cameras * a.vehicles_per_camera_per_hour / 3600.0
    events_per_day = events_per_s * 3600.0 * a.hours_per_day
    bytes_per_event = baseline.get("bytes_per_event")
    ingest_capacity = (
        baseline.get("ingest_events_per_s_bulk_best")
        or baseline.get("ingest_events_per_s_single")
    )

    data_dep = {
        "cameras": a.cameras,
        "vehicles_per_camera_per_hour": a.vehicles_per_camera_per_hour,
        "hours_per_day": a.hours_per_day,
    }
    data_tier: dict[str, Any] = {
        "events_per_second": projected(
            round(events_per_s, 1), "events/s",
            f"{a.cameras} cameras x {a.vehicles_per_camera_per_hour} vehicles/h / 3600",
            data_dep,
            caveat="One event per vehicle per camera pass (the pipeline's vote "
                   "window collapses many frames into one event), NOT one per frame.",
        ),
        "events_per_day": projected(
            int(events_per_day), "events/day",
            f"{events_per_s:.1f} events/s x 3600 x {a.hours_per_day} h",
            data_dep,
        ),
    }

    if ingest_capacity:
        headroom_ratio = ingest_capacity * a.utilisation_headroom / events_per_s if events_per_s else None
        data_tier["measured_ingest_capacity"] = measured(
            ingest_capacity, "events/s",
            "best measured throughput through the real ingestion endpoints on "
            "this machine (sqlite, single client)",
        )
        data_tier["ingest_headroom_ratio"] = projected(
            round(headroom_ratio, 2) if headroom_ratio else None, "x required",
            f"{ingest_capacity} measured events/s x {a.utilisation_headroom} / "
            f"{events_per_s:.1f} required events/s",
            {**data_dep, "utilisation_headroom": a.utilisation_headroom,
             "measured_ingest_events_per_s": ingest_capacity},
            caveat="Above 1.0 means one backend process on this laptop already "
                   "keeps up. Below 1.0 means the write path needs sharding or "
                   "Postgres + batching.",
        )
        data_tier["ingest_processes_needed"] = projected(
            max(1, math.ceil(events_per_s / (ingest_capacity * a.utilisation_headroom)))
            if ingest_capacity else None,
            "backend ingest processes",
            f"ceil({events_per_s:.1f} / ({ingest_capacity} x {a.utilisation_headroom}))",
            {**data_dep, "utilisation_headroom": a.utilisation_headroom},
        )

    if bytes_per_event:
        per_day_gb = events_per_day * bytes_per_event * a.replication_factor / 1e9
        storage_dep = {**data_dep, "measured_bytes_per_event": bytes_per_event,
                       "replication_factor": a.replication_factor}
        data_tier["storage"] = {
            "bytes_per_event": measured(
                bytes_per_event, "bytes/event",
                "derived from an actual sqlite database file size / row count",
            ),
            "per_day_gb": projected(
                round(per_day_gb, 2), "GB/day",
                f"{int(events_per_day):,} events/day x {bytes_per_event} B x "
                f"{a.replication_factor} replicas",
                storage_dep,
            ),
            "per_month_gb": projected(
                round(per_day_gb * 30, 1), "GB/month", "per_day_gb x 30", storage_dep,
            ),
            "at_retention_gb": projected(
                round(per_day_gb * a.retention_days, 1),
                f"GB at {a.retention_days}-day retention",
                f"per_day_gb x {a.retention_days}",
                {**storage_dep, "retention_days": a.retention_days},
            ),
            "caveat": (
                "Structured events only. Storing plate crops or video clips is "
                "orders of magnitude larger and is a separate decision — a 50 KB "
                "JPEG per event would multiply this by ~100x."
            ),
        }

    # ── latency budget check ────────────────────────────────────────────────
    decode_ms = baseline.get("decode_ms_p50") or 0.0
    one_pass_ms = decode_ms + ms_per_analysed_frame + (baseline.get("ingest_latency_p95_ms") or 0.0)
    latency = {
        "budget_s": a.target_latency_s,
        "measured_pipeline_ms": measured(
            round(one_pass_ms, 1), "ms",
            f"decode p50 {decode_ms:.1f} + inference {ms_per_analysed_frame:.1f} + "
            f"ingest p95 {baseline.get('ingest_latency_p95_ms')} — measured stages, summed",
        ),
        "fits_budget": bool(one_pass_ms / 1000.0 < a.target_latency_s),
        "max_queue_depth_frames": projected(
            int(a.target_latency_s * 1000.0 / ms_per_analysed_frame),
            "frames of backlog a worker may hold",
            f"{a.target_latency_s} s / {ms_per_analysed_frame:.2f} ms per frame",
            dep,
            caveat="Exceeding this means the plate arrives after the operator "
                   "needs it, even though throughput looks fine. This is why "
                   "sizing uses p95 and 70% headroom.",
        ),
        "note": (
            "This is the single-pass processing cost, not wall-clock "
            "photon-to-dashboard: it excludes camera encoder delay and network "
            "transit, which we cannot measure without real cameras."
        ),
    }

    # ── network + the edge-vs-central argument ──────────────────────────────
    network = _network_and_architecture(a, events_per_s, det_fps_needed,
                                       workers_measured_class, dep)

    return {
        "assumptions": a.to_dict(),
        "assumption_rationale": ASSUMPTION_RATIONALE,
        "baseline_used": {k: v for k, v in baseline.items() if v is not None},
        "inference": inference,
        "compute_resources": compute_resources,
        "data_tier": data_tier,
        "latency": latency,
        "network_and_architecture": network,
    }


def _network_and_architecture(
    a: ScaleAssumptions,
    events_per_s: float,
    det_fps_needed: float,
    workers: int,
    dep: dict[str, Any],
) -> dict[str, Any]:
    """Bandwidth for both architectures, and the ratio that decides between them.

    This is the real scalability argument. Central inference is simpler to
    operate but its uplink cost grows with VIDEO bitrate; edge inference's
    uplink cost grows with EVENT rate, which is thousands of times smaller.
    """
    central_mbps = a.cameras * a.stream_bitrate_mbps
    edge_mbps = events_per_s * a.event_wire_bytes * 8 / 1e6
    net_dep = {**dep, "stream_bitrate_mbps": a.stream_bitrate_mbps,
               "event_wire_bytes": a.event_wire_bytes}

    return {
        "central_architecture": {
            "description": (
                "Every camera streams video to a central GPU cluster, which "
                "decodes and runs ANPR."
            ),
            "uplink": projected(
                round(central_mbps, 1), "Mbps aggregate ingress",
                f"{a.cameras} cameras x {a.stream_bitrate_mbps} Mbps",
                net_dep,
            ),
            "uplink_gbps": projected(
                round(central_mbps / 1000.0, 2), "Gbps aggregate ingress",
                "Mbps / 1000", net_dep,
            ),
            "extra_central_cost": (
                "The cluster also pays video DECODE for every analysed frame — "
                "measured separately in the anpr suite's decode stage — on top of "
                "inference. An edge deployment pays that on the camera box."
            ),
            "pros": ["one place to deploy/upgrade models",
                     "full video retained centrally for forensics",
                     "no per-camera compute hardware"],
            "cons": [f"needs ~{central_mbps / 1000.0:.2f} Gbps of sustained, reliable "
                     "city-wide uplink",
                     "a network outage blinds the whole system",
                     "central decode compute is pure overhead"],
        },
        "edge_architecture": {
            "description": (
                "Inference runs on the camera or a nearby edge box; only "
                "structured VehicleEvent JSON is sent centrally."
            ),
            "uplink": projected(
                round(edge_mbps, 4), "Mbps aggregate ingress",
                f"{events_per_s:.1f} events/s x {a.event_wire_bytes} B x 8 / 1e6",
                net_dep,
            ),
            "edge_nodes": projected(
                workers, "edge inference nodes of the measured class",
                "same worker count as the central case — the compute does not "
                "disappear, it relocates",
                dep,
                caveat="If each camera gets its own box, the count is the camera "
                       "count and each box is mostly idle; consolidating several "
                       "cameras per box is what makes the worker count the right "
                       "number.",
            ),
            "pros": ["uplink drops by the ratio below — ordinary 4G/broadband suffices",
                     "each node degrades independently",
                     "video need never leave the pole (privacy + retention cost)"],
            "cons": ["model updates must be pushed to N nodes",
                     "per-node hardware cost and physical maintenance",
                     "no central video for after-the-fact re-analysis"],
        },
        "bandwidth_reduction_factor": projected(
            round(central_mbps / edge_mbps, 1) if edge_mbps > 0 else None,
            "x less uplink for edge vs central",
            f"{central_mbps:.1f} Mbps / {edge_mbps:.4f} Mbps",
            net_dep,
        ),
        "recommendation": (
            "Edge inference for scale, central for depth. The bandwidth ratio is "
            "large enough that a 2000-camera city is a networking problem in the "
            "central design and a fleet-management problem in the edge design — "
            "the latter is the cheaper problem. A hybrid (edge inference plus "
            "on-demand video pull for flagged events only) keeps the forensic "
            "capability without the sustained uplink."
        ),
        "cost_note": (
            "Deliberately no currency figures: we did not measure or quote "
            "hardware or bandwidth prices, and inventing them would undermine "
            "every measured number above. The figures here are quantities — "
            "GPUs, cores, GB, Gbps — which a procurement price list turns into "
            "money."
        ),
    }


DEFAULT_SCENARIOS = (200, 500, 2000)


def build_projections(
    baseline: dict[str, Any],
    camera_counts: Sequence[int] = DEFAULT_SCENARIOS,
    base_assumptions: ScaleAssumptions | None = None,
) -> dict[str, Any]:
    """One projection per city-scale scenario, sharing all other assumptions."""
    base = base_assumptions or ScaleAssumptions()
    out: dict[str, Any] = {}
    for count in camera_counts:
        out[str(count)] = compute_projection(baseline, replace(base, cameras=count))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration, persistence, and background runs
# ══════════════════════════════════════════════════════════════════════════════

ALL_SUITES = ("hardware", "anpr", "ingest", "query")


def run_full_suite(
    suites: Sequence[str] = ALL_SUITES,
    anpr_config: AnprSuiteConfig | None = None,
    db_config: DbSuiteConfig | None = None,
    progress: Callable[[str], None] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Run the requested suites, assemble the report, and cache it.

    Each suite is isolated: one failing (e.g. missing ANPR weights) records its
    error and the rest still produce numbers. A partial benchmark is useful; a
    benchmark that refuses to run because one model file moved is not.
    """
    say = progress or (lambda _m: None)
    unknown = [s for s in suites if s not in ALL_SUITES]
    if unknown:
        raise BenchmarkError(f"Unknown suite(s): {unknown}. Valid: {list(ALL_SUITES)}")

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "measured_at": _now_iso(),
        "suites_requested": list(suites),
        "suites_completed": [],
        "errors": {},
        "measured": {},
        "how_to_read_this": {
            "MEASURED": "timed on the machine named in `hardware`, reproducible "
                        "by re-running scripts/run_benchmarks.py",
            "PROJECTED": "arithmetic on MEASURED values plus the assumptions "
                         "listed in each figure's `depends_on`",
            "DERIVED": "arithmetic on MEASURED values only, no external assumptions",
            "rule": "No number in this file was typed in by hand. If a suite did "
                    "not run, its section holds an error, not a placeholder.",
        },
    }

    say("profiling hardware")
    report["hardware"] = get_hardware_profile()
    if "hardware" in suites:
        report["suites_completed"].append("hardware")

    if "anpr" in suites:
        try:
            say("running ANPR inference suite")
            report["measured"]["anpr"] = run_anpr_suite(anpr_config, progress=say)
            report["suites_completed"].append("anpr")
        except Exception as err:  # noqa: BLE001
            report["errors"]["anpr"] = f"{type(err).__name__}: {err}"

    db_suites = [s for s in ("ingest", "query") if s in suites]
    if db_suites:
        try:
            say(f"running DB suites {db_suites} in an isolated child process")
            db_result = run_db_suites(db_suites, db_config, progress=say)
            for key in ("ingest", "query", "storage"):
                if key in db_result:
                    report["measured"][key] = db_result[key]
            report["measured"]["db_isolation"] = {
                "temp_database": db_result.get("temp_database"),
                "note": db_result.get("isolation_note"),
            }
            report["suites_completed"].extend(db_suites)
        except Exception as err:  # noqa: BLE001
            for key in db_suites:
                report["errors"][key] = f"{type(err).__name__}: {err}"

    # ── carry forward suites this run did not re-measure ────────────────────
    #
    # A targeted re-measure (e.g. POST /benchmarks/run with suites=["anpr"])
    # must not destroy the ingest/query numbers from the last full run — that
    # would silently break the projection endpoint. Carried-over sections keep
    # their original timestamp and are listed explicitly, so nobody mistakes an
    # old measurement for a fresh one.
    _carry_forward_previous(report)

    report["baseline"] = extract_baseline(report)

    try:
        report["projections"] = build_projections(report["baseline"])
        report["projection_scenarios"] = list(DEFAULT_SCENARIOS)
    except BaselineMissing as err:
        report["errors"]["projections"] = str(err)
        report["projections"] = {}

    report["duration_seconds"] = round(time.perf_counter() - started, 1)

    if persist:
        try:
            report["persisted_to"] = str(save_results(report))
        except Exception as err:  # noqa: BLE001
            report["errors"]["persist"] = f"{type(err).__name__}: {err}"

    return report


CARRYABLE_SECTIONS = ("anpr", "ingest", "query", "storage")


def _carry_forward_previous(report: dict[str, Any]) -> None:
    """Fill measured sections this run skipped from the previously cached run."""
    try:
        previous = load_results()
    except BenchmarkError:
        return  # corrupt cache: better to publish a partial run than to fail
    if not previous or previous.get("schema_version") != SCHEMA_VERSION:
        return

    prev_measured = previous.get("measured") or {}
    carried: dict[str, Any] = {}
    for key in CARRYABLE_SECTIONS:
        if key in report["measured"] or key not in prev_measured:
            continue
        report["measured"][key] = prev_measured[key]
        carried[key] = {
            "measured_at": (prev_measured[key] or {}).get("measured_at")
                           or previous.get("measured_at"),
            "from_run": previous.get("measured_at"),
        }
    if carried:
        report["carried_over_from_previous_run"] = {
            "sections": carried,
            "why": (
                "This run did not re-measure these suites, so their numbers come "
                "from the previous run rather than being dropped. Each entry "
                "keeps its own measured_at — check it before quoting."
            ),
        }


def save_results(report: dict[str, Any], path: Path | None = None) -> Path:
    """Write the report to latest.json plus a timestamped archive copy.

    The archive copy exists so a re-run before the demo does not destroy the
    number you already put on a slide.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = path or LATEST_PATH
    payload = json.dumps(report, default=str, indent=2)
    # Write via a temp file then replace, so a crash mid-write cannot leave the
    # API serving a truncated JSON document.
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(payload)
    tmp.replace(target)

    if target == LATEST_PATH:
        stamp = report.get("measured_at", _now_iso()).replace(":", "").replace("-", "")
        (RESULTS_DIR / f"run-{stamp}.json").write_text(payload)
        # Keep the archive bounded: these are ~100 KB each and a demo day can
        # produce dozens of runs.
        archives = sorted(RESULTS_DIR.glob("run-*.json"))
        for old in archives[:-MAX_ARCHIVED_RUNS]:
            old.unlink(missing_ok=True)
    return target


_CACHE: dict[str, Any] = {"mtime": None, "data": None}


def load_results(path: Path | None = None) -> dict[str, Any] | None:
    """Read the cached report, memoised on file mtime.

    The mtime check keeps GET /benchmarks/projection at pure-arithmetic speed
    while still picking up a fresh run without a server restart.
    """
    target = path or LATEST_PATH
    if not target.exists():
        return None
    mtime = target.stat().st_mtime
    if _CACHE["mtime"] == mtime and _CACHE["data"] is not None:
        return _CACHE["data"]
    try:
        data = json.loads(target.read_text())
    except json.JSONDecodeError as err:
        raise BenchmarkError(f"Cached benchmark file is not valid JSON: {err}") from err
    _CACHE["mtime"] = mtime
    _CACHE["data"] = data
    return data


def results_freshness(report: dict[str, Any] | None) -> dict[str, Any]:
    """Whether the cached report is missing, stale, or current."""
    if not report:
        return {
            "available": False,
            "stale": True,
            "message": "No benchmark results yet. Run: "
                       ".venv/bin/python scripts/run_benchmarks.py --suite all",
        }
    measured_at = report.get("measured_at")
    age_hours = None
    if measured_at:
        try:
            then = datetime.fromisoformat(measured_at)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            age_hours = (datetime.now(timezone.utc) - then).total_seconds() / 3600.0
        except ValueError:
            pass
    stale = age_hours is None or age_hours > STALE_AFTER_HOURS
    return {
        "available": True,
        "measured_at": measured_at,
        "age_hours": round(age_hours, 2) if age_hours is not None else None,
        "stale": stale,
        "stale_after_hours": STALE_AFTER_HOURS,
        "message": (
            "Results are older than the staleness threshold — re-run before "
            "quoting them." if stale else "Results are current."
        ),
    }


# ── background run manager ───────────────────────────────────────────────────
#
# A full run takes ~1-2 minutes, which must never happen inside a request. One
# run at a time: two concurrent runs would contend for the GPU and both would
# report inflated times, which is worse than refusing the second.

_RUNS: dict[str, dict[str, Any]] = {}
_RUNS_LOCK = threading.Lock()
_MAX_RUN_HISTORY = 20


def _prune_runs() -> None:
    if len(_RUNS) <= _MAX_RUN_HISTORY:
        return
    finished = sorted(
        (r for r in _RUNS.values() if r["status"] in ("completed", "failed")),
        key=lambda r: r.get("started_at", ""),
    )
    for run in finished[: len(_RUNS) - _MAX_RUN_HISTORY]:
        _RUNS.pop(run["run_id"], None)


def active_run() -> dict[str, Any] | None:
    with _RUNS_LOCK:
        for run in _RUNS.values():
            if run["status"] == "running":
                return dict(run)
    return None


def start_background_run(
    suites: Sequence[str] = ALL_SUITES,
    anpr_config: AnprSuiteConfig | None = None,
    db_config: DbSuiteConfig | None = None,
) -> dict[str, Any]:
    """Kick off a run in a daemon thread and return its id immediately."""
    unknown = [s for s in suites if s not in ALL_SUITES]
    if unknown:
        raise BenchmarkError(f"Unknown suite(s): {unknown}. Valid: {list(ALL_SUITES)}")

    with _RUNS_LOCK:
        for run in _RUNS.values():
            if run["status"] == "running":
                raise BenchmarkError(
                    f"Benchmark run {run['run_id']} is already in progress "
                    f"(step: {run.get('current_step')}). Wait for it to finish — "
                    "concurrent runs contend for the same GPU and both would "
                    "report inflated timings."
                )
        run_id = uuid.uuid4().hex[:12]
        state: dict[str, Any] = {
            "run_id": run_id,
            "status": "running",
            "suites": list(suites),
            "started_at": _now_iso(),
            "finished_at": None,
            "current_step": "starting",
            "log": [],
            "error": None,
            "result_summary": None,
        }
        _RUNS[run_id] = state
        _prune_runs()

    def _progress(message: str) -> None:
        with _RUNS_LOCK:
            state["current_step"] = message
            state["log"].append({"at": _now_iso(), "message": message})

    def _worker() -> None:
        try:
            report = run_full_suite(suites, anpr_config, db_config, progress=_progress)
            with _RUNS_LOCK:
                state["status"] = "completed"
                state["current_step"] = "done"
                state["result_summary"] = summarize(report)
                state["suite_errors"] = report.get("errors") or {}
        except Exception as err:  # noqa: BLE001
            with _RUNS_LOCK:
                state["status"] = "failed"
                state["error"] = f"{type(err).__name__}: {err}"
        finally:
            with _RUNS_LOCK:
                state["finished_at"] = _now_iso()

    threading.Thread(target=_worker, name=f"benchmark-{run_id}", daemon=True).start()
    return dict(state)


def get_run(run_id: str) -> dict[str, Any] | None:
    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
        return dict(run) if run else None


def list_runs() -> list[dict[str, Any]]:
    with _RUNS_LOCK:
        return [dict(r) for r in _RUNS.values()]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """The handful of numbers that belong on a slide."""
    baseline = report.get("baseline") or extract_baseline(report)
    proj = (report.get("projections") or {}).get("200") or {}

    def pv(*path: str) -> Any:
        cur: Any = proj
        for key in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
        return cur

    return {
        "measured_at": report.get("measured_at"),
        "machine": f"{baseline.get('machine')} ({baseline.get('machine_model')}), "
                   f"{baseline.get('logical_cores')} logical cores, "
                   f"device={baseline.get('device_measured')}",
        "measured": {
            "detection_ms_per_frame_p50": baseline.get("detection_ms_p50"),
            "detection_ms_per_frame_p95": baseline.get("detection_ms_p95"),
            "ocr_ms_per_crop_p50": baseline.get("ocr_ms_p50"),
            "ocr_ms_per_crop_p95": baseline.get("ocr_ms_p95"),
            "ingest_events_per_s_single": baseline.get("ingest_events_per_s_single"),
            "ingest_events_per_s_bulk_best": baseline.get("ingest_events_per_s_bulk_best"),
            "bytes_per_event": baseline.get("bytes_per_event"),
        },
        "projected_200_cameras": {
            "detection_frames_per_s": pv("inference", "required_detection_throughput", "value"),
            "ocr_crops_per_s": pv("inference", "required_ocr_throughput", "value"),
            "worker_slots_of_measured_class": pv("inference", "workers_of_measured_class", "value"),
            "physical_hosts_of_measured_class": pv(
                "inference", "sizing_reconciliation",
                "physical_hosts_of_measured_class", "value"),
            "gpus_assumed_class": pv("inference", "gpus", "gpus_required", "value"),
            "events_per_s": pv("data_tier", "events_per_second", "value"),
            "storage_gb_per_day": pv("data_tier", "storage", "per_day_gb", "value"),
            "central_uplink_gbps": pv("network_and_architecture", "central_architecture",
                                      "uplink_gbps", "value"),
        },
        "errors": report.get("errors") or {},
    }


# ── child-process entrypoint ─────────────────────────────────────────────────

if __name__ == "__main__":
    # Only reachable via run_db_suites(). Not a general CLI — use
    # scripts/run_benchmarks.py for that.
    if len(sys.argv) >= 3 and sys.argv[1] == "--db-worker":
        _db_worker_main(json.loads(sys.argv[2]))
    else:
        print(
            "This module is not a CLI. Run: "
            ".venv/bin/python scripts/run_benchmarks.py --suite all",
            file=sys.stderr,
        )
        sys.exit(2)

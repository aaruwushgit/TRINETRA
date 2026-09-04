"""
Benchmarks router — measured compute cost, and what a city-scale deployment needs.

The design constraint that shapes this whole file: **a benchmark must never run
inside a web request.** A full measurement takes 1-2 minutes (model loading
dominates) and pegs the GPU, so:

  GET  /benchmarks/              serves the last PERSISTED run, instantly.
  GET  /benchmarks/hardware      live, but cheap — no inference.
  POST /benchmarks/run           starts a background thread, returns a run id.
  GET  /benchmarks/run/{id}      progress for that run.
  GET  /benchmarks/projection    pure arithmetic on the cached measured
                                 baselines with caller-supplied assumptions.
                                 This is the interactive "what if we scale to N
                                 cameras" endpoint — it must stay sub-10ms.

Errors are returned as structured JSON with a `hint`, never as a traceback: the
most likely failure here is "you haven't run the suite yet", and that deserves a
copy-pasteable command rather than a 500.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.services import benchmark_service as bench

router = APIRouter(prefix="/benchmarks", tags=["Benchmarks"])


# ── request models ───────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    """Which suites to (re-)measure, and how hard to push them."""

    suites: list[str] = Field(
        default_factory=lambda: list(bench.ALL_SUITES),
        description="Any of: hardware, anpr, ingest, query.",
    )
    # Exposed so a pre-demo run can take more samples (tighter p95) while an
    # ad-hoc re-measure stays quick.
    anpr_frames: int | None = Field(default=None, ge=10, le=1000)
    anpr_ocr_crops: int | None = Field(default=None, ge=5, le=500)
    anpr_imgsz: int | None = Field(default=None, ge=320, le=1920)
    ingest_single_events: int | None = Field(default=None, ge=10, le=2000)
    query_scales: list[int] | None = Field(default=None)


def _error(code: int, message: str, hint: str | None = None) -> HTTPException:
    """One shape for every failure, so the frontend can render it generically."""
    payload: dict[str, Any] = {"error": message}
    if hint:
        payload["hint"] = hint
    return HTTPException(status_code=code, detail=payload)


def _load_or_404() -> dict[str, Any]:
    try:
        report = bench.load_results()
    except bench.BenchmarkError as err:
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            str(err),
            "Delete deployments/benchmarks/latest.json and re-run the suite.",
        ) from err
    if not report:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            "No benchmark results have been recorded yet.",
            "Run: .venv/bin/python scripts/run_benchmarks.py --suite all "
            "(or POST /benchmarks/run).",
        )
    return report


# ── read endpoints ───────────────────────────────────────────────────────────


@router.get("/")
def get_last_results() -> dict[str, Any]:
    """The last persisted full benchmark run, with freshness metadata.

    Served straight from deployments/benchmarks/latest.json (memoised on file
    mtime), so this is fast regardless of how long the measurement took.
    """
    report = _load_or_404()
    return {
        "freshness": bench.results_freshness(report),
        "summary": bench.summarize(report),
        "results": report,
    }


@router.get("/summary")
def get_summary() -> dict[str, Any]:
    """Just the headline numbers — what belongs on a slide."""
    report = _load_or_404()
    return {
        "freshness": bench.results_freshness(report),
        "summary": bench.summarize(report),
    }


@router.get("/hardware")
def get_hardware() -> dict[str, Any]:
    """Live hardware/runtime profile of the machine serving this request.

    Measured on every call because it is cheap (sysctl + psutil + a torch
    availability check) and because a cached hardware profile is actively
    misleading if the API moved to another box.
    """
    try:
        return bench.get_hardware_profile()
    except Exception as err:  # noqa: BLE001
        raise _error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"Could not profile hardware: {type(err).__name__}: {err}",
        ) from err


@router.get("/assumptions")
def get_assumptions() -> dict[str, Any]:
    """The projection's default assumptions and the reasoning behind each.

    Every one of these is overridable on /benchmarks/projection. Published as an
    endpoint so the UI can render the assumptions next to the numbers they
    produce — a projection without its assumptions on screen is a claim, not an
    estimate.
    """
    return {
        "defaults": bench.ScaleAssumptions().to_dict(),
        "rationale": bench.ASSUMPTION_RATIONALE,
        "scenarios_precomputed": list(bench.DEFAULT_SCENARIOS),
    }


# ── the interactive projection endpoint ──────────────────────────────────────


@router.get("/projection")
def get_projection(
    cameras: int = Query(default=200, ge=1, le=100_000,
                         description="Camera count. 200 = real Delhi junctions today."),
    fps_per_camera: float = Query(default=6.0, gt=0, le=60,
                                  description="Analysed frames/s per camera (NOT the stream fps)."),
    ocr_every: int = Query(default=3, ge=1, le=30,
                           description="Run OCR on every Nth plate-bearing frame."),
    plates_per_frame: float = Query(default=1.5, gt=0, le=50),
    vehicles_per_camera_per_hour: float = Query(default=1200.0, gt=0, le=100_000),
    target_latency_s: float = Query(default=2.0, gt=0, le=60),
    utilisation_headroom: float = Query(default=0.70, gt=0.0, le=1.0,
                                        description="Size to this fraction of measured capacity."),
    gpu_class: str = Query(default="NVIDIA T4 (16GB, inference-class)"),
    gpu_speedup_vs_measured_device: float = Query(
        default=3.0, gt=0, le=100,
        description="ASSUMPTION: how many times faster that GPU is than the "
                    "device we actually measured. Not a measurement.",
    ),
    stream_bitrate_mbps: float = Query(default=4.0, gt=0, le=200),
    event_wire_bytes: int = Query(default=400, ge=50, le=100_000),
    retention_days: int = Query(default=90, ge=1, le=3650),
    replication_factor: float = Query(default=1.0, ge=1.0, le=10.0),
    hours_per_day: float = Query(default=24.0, gt=0, le=24.0),
) -> dict[str, Any]:
    """Recompute the city-scale sizing from cached measurements + your assumptions.

    Pure arithmetic over ~8 cached stopwatch readings: no models are loaded, no
    database is touched, nothing is re-measured. That is what makes it safe to
    call on every drag of a slider.

    MEASURED values are echoed under `baseline_used`; every PROJECTED figure
    carries a `depends_on` block naming the assumptions it rests on.
    """
    report = _load_or_404()
    baseline = report.get("baseline") or bench.extract_baseline(report)

    assumptions = bench.ScaleAssumptions(
        cameras=cameras,
        fps_per_camera=fps_per_camera,
        ocr_every=ocr_every,
        plates_per_frame=plates_per_frame,
        vehicles_per_camera_per_hour=vehicles_per_camera_per_hour,
        target_latency_s=target_latency_s,
        utilisation_headroom=utilisation_headroom,
        gpu_class=gpu_class,
        gpu_speedup_vs_measured_device=gpu_speedup_vs_measured_device,
        stream_bitrate_mbps=stream_bitrate_mbps,
        event_wire_bytes=event_wire_bytes,
        retention_days=retention_days,
        replication_factor=replication_factor,
        hours_per_day=hours_per_day,
    )

    try:
        projection = bench.compute_projection(baseline, assumptions)
    except bench.BaselineMissing as err:
        # 409: the request is valid, the server just lacks the measurement.
        raise _error(
            status.HTTP_409_CONFLICT,
            str(err),
            "POST /benchmarks/run with suites=[\"anpr\"], or run "
            ".venv/bin/python scripts/run_benchmarks.py --suite anpr",
        ) from err
    except bench.BenchmarkError as err:
        raise _error(status.HTTP_400_BAD_REQUEST, str(err)) from err

    return {
        "measured_at": report.get("measured_at"),
        "freshness": bench.results_freshness(report),
        "projection": projection,
    }


@router.get("/projection/scenarios")
def get_scenarios(
    camera_counts: str = Query(default="200,500,2000",
                               description="Comma-separated camera counts."),
    fps_per_camera: float = Query(default=6.0, gt=0, le=60),
    ocr_every: int = Query(default=3, ge=1, le=30),
    utilisation_headroom: float = Query(default=0.70, gt=0.0, le=1.0),
) -> dict[str, Any]:
    """Several camera counts side by side, sharing every other assumption.

    This is the shape the "200 / 500 / 2000 cameras" comparison table wants.
    """
    try:
        counts = [int(c.strip()) for c in camera_counts.split(",") if c.strip()]
    except ValueError as err:
        raise _error(
            status.HTTP_400_BAD_REQUEST,
            f"camera_counts must be a comma-separated list of integers: {err}",
        ) from err
    if not counts or any(c <= 0 or c > 100_000 for c in counts):
        raise _error(status.HTTP_400_BAD_REQUEST,
                     "Each camera count must be between 1 and 100000.")

    report = _load_or_404()
    baseline = report.get("baseline") or bench.extract_baseline(report)
    base = bench.ScaleAssumptions(
        fps_per_camera=fps_per_camera,
        ocr_every=ocr_every,
        utilisation_headroom=utilisation_headroom,
    )
    try:
        return {
            "measured_at": report.get("measured_at"),
            "baseline_used": {k: v for k, v in baseline.items() if v is not None},
            "scenarios": bench.build_projections(baseline, counts, base),
        }
    except bench.BaselineMissing as err:
        raise _error(status.HTTP_409_CONFLICT, str(err),
                     "Run the 'anpr' suite first.") from err
    except bench.BenchmarkError as err:
        raise _error(status.HTTP_400_BAD_REQUEST, str(err)) from err


# ── measurement triggers ─────────────────────────────────────────────────────


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
def trigger_run(request: RunRequest = Body(default=RunRequest())) -> dict[str, Any]:
    """Start a benchmark run in a background thread; return its id immediately.

    202 Accepted, not 200: the work has been queued, not done. Only one run may
    be in flight — two concurrent runs would contend for the same GPU and both
    would report inflated timings, so a second request is refused with 409 and
    the id of the run already going.
    """
    anpr_cfg = None
    if any(v is not None for v in (request.anpr_frames, request.anpr_ocr_crops, request.anpr_imgsz)):
        defaults = bench.AnprSuiteConfig()
        anpr_cfg = bench.AnprSuiteConfig(
            frames=request.anpr_frames or defaults.frames,
            ocr_crops=request.anpr_ocr_crops or defaults.ocr_crops,
            imgsz=request.anpr_imgsz or defaults.imgsz,
        )

    db_cfg = None
    if request.ingest_single_events is not None or request.query_scales is not None:
        defaults = bench.DbSuiteConfig()
        db_cfg = bench.DbSuiteConfig(
            single_events=request.ingest_single_events or defaults.single_events,
            query_scales=tuple(request.query_scales) if request.query_scales else defaults.query_scales,
        )

    try:
        state = bench.start_background_run(request.suites, anpr_cfg, db_cfg)
    except bench.BenchmarkError as err:
        running = bench.active_run()
        raise _error(
            status.HTTP_409_CONFLICT if running else status.HTTP_400_BAD_REQUEST,
            str(err),
            f"Poll GET /benchmarks/run/{running['run_id']}" if running else None,
        ) from err

    return {
        "run_id": state["run_id"],
        "status": state["status"],
        "suites": state["suites"],
        "started_at": state["started_at"],
        "poll": f"/benchmarks/run/{state['run_id']}",
        "note": (
            "Typically 60-120 s. Results are written to "
            "deployments/benchmarks/latest.json and served by GET /benchmarks/."
        ),
    }


@router.get("/run/{run_id}")
def get_run_status(run_id: str) -> dict[str, Any]:
    """Progress (and, once finished, the headline summary) for one run."""
    state = bench.get_run(run_id)
    if not state:
        raise _error(
            status.HTTP_404_NOT_FOUND,
            f"No benchmark run with id '{run_id}'.",
            "Run ids live in memory only, so they are lost on server restart — "
            "the results themselves survive in GET /benchmarks/.",
        )
    return state


@router.get("/runs")
def list_runs() -> dict[str, Any]:
    """Every run this process has started (in-memory, cleared on restart)."""
    runs = bench.list_runs()
    return {"count": len(runs), "active": bench.active_run(), "runs": runs}

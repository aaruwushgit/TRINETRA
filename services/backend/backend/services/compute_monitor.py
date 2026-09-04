"""
Compute telemetry — "what does this inference actually cost?"

The demo has to answer a jury question that benchmarks alone cannot: *can a
city afford to run this?* Throughput (fps) is only half the answer; the other
half is how much of a machine one camera consumes. So every ANPR run is
wrapped in a sampler that records CPU, RSS and device so the dashboard can
show cost per stream next to accuracy.

Two deliberate choices:

**Sampling on a thread, not around the call.** A single before/after reading of
CPU time tells you the average over the whole run, which hides the thing that
matters — the peak. Model load spikes RSS far above steady-state inference, and
a deployment sized on the average will be OOM-killed by the spike. A 2 Hz
background sample is cheap (psutil reads /proc-equivalent counters) and catches
it.

**Degrading instead of failing.** psutil and torch are both optional here on
purpose. This module is imported by the job runner, and telemetry going missing
must never be the reason a plate does not get logged — a run with no numbers is
worth far more than no run. Every field that cannot be measured is reported as
None rather than zero, because zero is a measurement and None is an absence.
"""
from __future__ import annotations

import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Optional dependencies. Both are present in this project's venv, but the
# import is guarded so a stripped deployment (or a Docker image built without
# torch) still runs the API.
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


# 2 Hz. Fast enough to catch a model-load RSS spike (which lasts seconds),
# slow enough that the sampler itself does not show up in the CPU figure.
SAMPLE_INTERVAL = 0.5

_MB = 1024.0 * 1024.0


# ── Static device description ────────────────────────────────────────────────

@dataclass(frozen=True)
class DeviceInfo:
    """What hardware this process is running inference on."""

    device: str  # "cuda" | "mps" | "cpu"
    device_name: str
    cpu_count: int | None
    cpu_count_logical: int | None
    total_ram_mb: float | None
    machine: str
    platform: str
    python_version: str
    torch_version: str | None
    cv2_version: str | None
    cuda_available: bool
    mps_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "device_name": self.device_name,
            "cpu_count": self.cpu_count,
            "cpu_count_logical": self.cpu_count_logical,
            "total_ram_mb": self.total_ram_mb,
            "machine": self.machine,
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cv2_version": self.cv2_version,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
        }


def _mac_model() -> str | None:
    """The Mac's marketing model string, if this is a Mac and it is cheap.

    Worth having because "M4 Pro" is the single most useful number for
    extrapolating cost, and `platform.machine()` only says "arm64". Guarded and
    time-boxed: a hung sysctl must not delay an API response.
    """
    if platform.system() != "Darwin":
        return None
    try:
        import subprocess

        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        value = out.stdout.strip()
        return value or None
    except Exception:
        return None


def _detect_device() -> tuple[str, str]:
    """(device, human-readable name). Mirrors alpr.detect.select_device order."""
    if torch is None:
        return "cpu", _mac_model() or platform.processor() or "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            # MPS has no device-name API, so fall back to the CPU brand string,
            # which on Apple Silicon names the same chip the GPU lives on.
            return "mps", _mac_model() or "Apple Silicon GPU (MPS)"
    except Exception:
        pass
    return "cpu", _mac_model() or platform.processor() or "cpu"


_device_info_cache: DeviceInfo | None = None


def device_info() -> DeviceInfo:
    """Static hardware description, computed once.

    Cached because `torch.cuda.is_available()` initializes a CUDA context on
    first call and the dashboard polls this endpoint.
    """
    global _device_info_cache
    if _device_info_cache is not None:
        return _device_info_cache

    device, device_name = _detect_device()

    total_ram_mb: float | None = None
    cpu_logical: int | None = None
    cpu_physical: int | None = None
    if psutil is not None:
        try:
            total_ram_mb = round(psutil.virtual_memory().total / _MB, 1)
            cpu_logical = psutil.cpu_count(logical=True)
            cpu_physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
    if cpu_logical is None:
        cpu_logical = os.cpu_count()

    cv2_version: str | None = None
    try:
        import cv2

        cv2_version = cv2.__version__
    except Exception:
        pass

    cuda_ok = False
    mps_ok = False
    if torch is not None:
        try:
            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            pass
        try:
            mps = getattr(torch.backends, "mps", None)
            mps_ok = bool(mps is not None and mps.is_available())
        except Exception:
            pass

    _device_info_cache = DeviceInfo(
        device=device,
        device_name=device_name,
        cpu_count=cpu_physical,
        cpu_count_logical=cpu_logical,
        total_ram_mb=total_ram_mb,
        machine=platform.machine(),
        platform=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        torch_version=getattr(torch, "__version__", None) if torch else None,
        cv2_version=cv2_version,
        cuda_available=cuda_ok,
        mps_available=mps_ok,
    )
    return _device_info_cache


# ── Live sampling ────────────────────────────────────────────────────────────

@dataclass
class Sample:
    """One instant of resource use."""

    t: float
    elapsed: float
    process_cpu_percent: float | None      # can exceed 100 — sums all threads
    process_cpu_percent_per_core: float | None  # normalized to 0..100
    process_rss_mb: float | None
    system_cpu_percent: float | None
    system_memory_percent: float | None
    cuda_allocated_mb: float | None = None
    cuda_max_allocated_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "elapsed": round(self.elapsed, 3),
            "process_cpu_percent": self.process_cpu_percent,
            "process_cpu_percent_per_core": self.process_cpu_percent_per_core,
            "process_rss_mb": self.process_rss_mb,
            "system_cpu_percent": self.system_cpu_percent,
            "system_memory_percent": self.system_memory_percent,
            "cuda_allocated_mb": self.cuda_allocated_mb,
            "cuda_max_allocated_mb": self.cuda_max_allocated_mb,
        }


class ComputeSampler:
    """Samples this process's resource use on a background thread.

    Usable either as a context manager or via explicit start()/stop(), and safe
    to restart: each start() clears the previous run's samples so a reused
    sampler never reports a mixture of two workloads. stop() joins the thread,
    so repeated cycles cannot leak threads even if the caller forgets.

        with ComputeSampler() as s:
            run_the_model()
        s.summary()
    """

    def __init__(self, interval: float = SAMPLE_INTERVAL, max_samples: int = 4000) -> None:
        self.interval = max(0.05, interval)
        # A long video at 2 Hz would otherwise grow without bound; 4000 samples
        # is ~33 minutes, after which the oldest are dropped. Peaks are tracked
        # separately so trimming cannot lose the number that matters.
        self.max_samples = max_samples

        self._lock = threading.Lock()
        self._samples: list[Sample] = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started_at: float | None = None
        self._stopped_at: float | None = None

        self._peak_rss_mb: float | None = None
        self._proc = None
        if psutil is not None:
            try:
                self._proc = psutil.Process(os.getpid())
                # First call to cpu_percent() always returns 0.0 — it needs a
                # prior reading to difference against. Prime it here so the
                # very first real sample is meaningful.
                self._proc.cpu_percent(None)
                psutil.cpu_percent(None)
            except Exception:
                self._proc = None

        self._cores = device_info().cpu_count_logical or os.cpu_count() or 1

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> ComputeSampler:
        if self._thread is not None and self._thread.is_alive():
            return self  # already running; idempotent by design

        with self._lock:
            self._samples = []
            self._peak_rss_mb = None
        self._stop.clear()
        self._started_at = time.time()
        self._stopped_at = None

        if torch is not None and device_info().cuda_available:
            try:
                # Reset so max_memory_allocated describes *this* workload rather
                # than the highest point since the process started.
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        self._thread = threading.Thread(
            target=self._run, name="compute-sampler", daemon=True
        )
        self._thread.start()
        # Take one sample synchronously so a workload shorter than the sample
        # interval still reports numbers instead of an empty summary.
        self._record()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            # Bounded join: the sampler waits on an Event so it exits promptly,
            # but a timeout guarantees stop() cannot hang a request thread.
            thread.join(timeout=self.interval * 4 + 1.0)
        self._thread = None
        self._stopped_at = time.time()
        self._record()
        return self.summary()

    def __enter__(self) -> ComputeSampler:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- sampling ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._record()

    def _record(self) -> None:
        sample = self._take()
        with self._lock:
            self._samples.append(sample)
            if len(self._samples) > self.max_samples:
                del self._samples[: len(self._samples) - self.max_samples]
            if sample.process_rss_mb is not None:
                if self._peak_rss_mb is None or sample.process_rss_mb > self._peak_rss_mb:
                    self._peak_rss_mb = sample.process_rss_mb

    def _take(self) -> Sample:
        now = time.time()
        elapsed = now - self._started_at if self._started_at else 0.0

        cpu_total: float | None = None
        rss: float | None = None
        sys_cpu: float | None = None
        sys_mem: float | None = None

        if self._proc is not None:
            try:
                cpu_total = float(self._proc.cpu_percent(None))
                rss = round(self._proc.memory_info().rss / _MB, 1)
            except Exception:
                pass
        if psutil is not None:
            try:
                sys_cpu = float(psutil.cpu_percent(None))
                sys_mem = float(psutil.virtual_memory().percent)
            except Exception:
                pass

        cuda_alloc: float | None = None
        cuda_peak: float | None = None
        if torch is not None and device_info().cuda_available:
            try:
                cuda_alloc = round(torch.cuda.memory_allocated() / _MB, 1)
                cuda_peak = round(torch.cuda.max_memory_allocated() / _MB, 1)
            except Exception:
                pass

        return Sample(
            t=now,
            elapsed=elapsed,
            process_cpu_percent=round(cpu_total, 1) if cpu_total is not None else None,
            # Both forms are reported because they answer different questions:
            # "how many cores is this using" (total, can be 800%) and "how much
            # of the machine is left" (per-core, 0..100).
            process_cpu_percent_per_core=(
                round(cpu_total / self._cores, 1) if cpu_total is not None else None
            ),
            process_rss_mb=rss,
            system_cpu_percent=round(sys_cpu, 1) if sys_cpu is not None else None,
            system_memory_percent=round(sys_mem, 1) if sys_mem is not None else None,
            cuda_allocated_mb=cuda_alloc,
            cuda_max_allocated_mb=cuda_peak,
        )

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The latest reading plus static device info. JSON-serializable."""
        latest = self._take()
        with self._lock:
            peak = self._peak_rss_mb
        return {
            "device": device_info().to_dict(),
            "current": latest.to_dict(),
            "peak_rss_mb": peak,
            "sampling": self.running,
            "telemetry_available": psutil is not None,
        }

    def summary(self) -> dict[str, Any]:
        """Averages and peaks over the sampled window. JSON-serializable."""
        with self._lock:
            samples = list(self._samples)
            peak_rss = self._peak_rss_mb

        end = self._stopped_at or time.time()
        elapsed = round(end - self._started_at, 3) if self._started_at else 0.0

        def stat(attr: str) -> tuple[float | None, float | None]:
            values = [
                getattr(s, attr) for s in samples if getattr(s, attr) is not None
            ]
            if not values:
                return None, None
            return round(sum(values) / len(values), 1), round(max(values), 1)

        cpu_avg, cpu_peak = stat("process_cpu_percent")
        core_avg, core_peak = stat("process_cpu_percent_per_core")
        rss_avg, _ = stat("process_rss_mb")
        sys_cpu_avg, sys_cpu_peak = stat("system_cpu_percent")
        sys_mem_avg, sys_mem_peak = stat("system_memory_percent")
        _, cuda_peak = stat("cuda_max_allocated_mb")
        cuda_last = samples[-1].cuda_allocated_mb if samples else None

        return {
            "device": device_info().to_dict(),
            "elapsed_s": elapsed,
            "samples": len(samples),
            "telemetry_available": psutil is not None,
            "process_cpu_percent_avg": cpu_avg,
            "process_cpu_percent_peak": cpu_peak,
            "process_cpu_percent_per_core_avg": core_avg,
            "process_cpu_percent_per_core_peak": core_peak,
            "process_rss_mb_avg": rss_avg,
            "process_rss_mb_peak": peak_rss,
            "system_cpu_percent_avg": sys_cpu_avg,
            "system_cpu_percent_peak": sys_cpu_peak,
            "system_memory_percent_avg": sys_mem_avg,
            "system_memory_percent_peak": sys_mem_peak,
            "cuda_allocated_mb": cuda_last,
            "cuda_max_allocated_mb": cuda_peak,
        }


# A process-wide sampler that is never started, used purely for one-off
# snapshot() calls from the dashboard's compute panel. Kept separate from the
# per-job samplers so polling the endpoint cannot disturb a running job's
# measurement window.
_ambient = ComputeSampler()


def snapshot() -> dict[str, Any]:
    """Current device info + live usage, for GET /jobs/system/compute."""
    return _ambient.snapshot()

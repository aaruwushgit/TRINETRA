#!/usr/bin/env python
"""
Attach a YouTube stream as a live ANPR camera — plug and play.

    .venv/bin/python scripts/youtube_camera.py \
        --url "https://www.youtube.com/watch?v=..." \
        --camera-id CAM_YT_01 --lat 28.6139 --lng 77.2090 \
        --name "Connaught Place North" --duration 120

The point of this script is that attaching a new camera requires **zero manual
database setup**. It registers itself with the backend first (treating "already
exists" as success), then streams plates into the same /events/ingest endpoint
the production edge workers use. Onboarding a camera is one command.

Three design notes:

**Plate events are read from the pipeline's write-ahead journal, not from the
live vote.** `on_frame` sees a vote still converging — "KA02HN182" one frame,
"KA02HN1826" three frames later — and posting those would put both into the
database as two vehicles. The pipeline appends each *accepted* event (voted,
grammar-validated, duplicate-suppressed) to a JSONL journal the instant it
happens, so tailing that file gives live delivery with batch-quality data.

**Timestamps are sent as naive UTC.** The ingest endpoint strips a timezone
offset without converting it, so sending an IST-aware timestamp would store
local time labelled as UTC and put every speed calculation 5.5 hours out.

**Several format selectors are tried in order.** A live stream only offers HLS;
a normal video offers progressive MP4. OpenCV's FFmpeg backend can open either,
but not every yt-dlp format (notably DASH-only audio/video splits), so the
script probes candidates and reports clearly when none open.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Run from a checkout without installing the backend package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Ordered by "most likely to be openable by OpenCV" rather than by quality.
# 1080p is pointless for plate reading at this distance and costs decode time,
# so 720p is the cap.
#
# **These ask for `bestvideo`, not `best`.** YouTube has stopped serving muxed
# progressive formats for most videos — every video-bearing format now reports
# `acodec: none`, so the obvious selectors (`best`, `best[ext=mp4]`,
# format `18`) resolve to *nothing* and yt-dlp raises "Requested format is not
# available". That is fine here: ANPR does not need audio, and a video-only
# track is a single URL rather than a DASH pair, which is exactly what OpenCV
# can open.
#
# `vcodec^=avc1` is preferred over VP9/AV1 because H.264 is the codec every
# OpenCV FFmpeg build decodes without surprises.
FORMAT_CANDIDATES = [
    # Live streams offer HLS only.
    "bestvideo[protocol^=m3u8][vcodec^=avc1][height<=720]",
    # Normal videos: DASH MP4, H.264.
    "bestvideo[protocol^=https][ext=mp4][vcodec^=avc1][height<=720]",
    "bestvideo[protocol^=m3u8][height<=720]",
    "bestvideo[ext=mp4][vcodec^=avc1]",
    # Older uploads still carry a real progressive format; try it before
    # falling back to a codec OpenCV may not decode.
    "best[ext=mp4][acodec!=none]",
    "bestvideo[ext=mp4]",
    "bestvideo",
    "best",
]

# Where node lives when it is not on PATH. nvm installs are the common case on
# a developer Mac and are invisible to a process started outside a login shell
# — which is exactly how the backend runs.
NODE_SEARCH_GLOBS = [
    "~/.nvm/versions/node/*/bin/node",
    "/opt/homebrew/bin/node",
    "/usr/local/bin/node",
]


def _js_runtimes() -> dict:
    """Locate a JavaScript runtime for yt-dlp.

    **YouTube extraction now requires one.** Without a JS runtime yt-dlp cannot
    solve the player challenge, returns an empty format list, and every selector
    fails with the misleading "Requested format is not available" — which reads
    like a bad selector and is actually a missing dependency. yt-dlp only
    enables `deno` by default, so node has to be found and passed explicitly.

    Returns the `js_runtimes` mapping yt-dlp's Python API expects
    (`{runtime: {"path": ...}}`), empty if nothing was found.
    """
    import glob
    import shutil

    runtimes: dict[str, dict] = {}

    for name in ("deno", "bun", "quickjs"):
        found = shutil.which(name)
        if found:
            runtimes[name] = {"path": found}

    node = shutil.which("node")
    if not node:
        for pattern in NODE_SEARCH_GLOBS:
            matches = sorted(glob.glob(os.path.expanduser(pattern)))
            if matches:
                node = matches[-1]  # highest version
                break
    if node:
        runtimes["node"] = {"path": node}

    return runtimes

# How often to drain the journal and print telemetry. Every frame would spend
# more time on syscalls than on inference.
JOURNAL_POLL_FRAMES = 15
TELEMETRY_EVERY_S = 5.0


class Fatal(RuntimeError):
    """An error worth exiting on, with a message the operator can act on."""


# ── stream resolution ────────────────────────────────────────────────────────

def resolve_stream(url: str, *, verbose: bool = False) -> tuple[str, dict]:
    """Resolve a YouTube URL to a direct media URL OpenCV can open.

    Returns (media_url, info). Raises Fatal with an actionable message when no
    candidate format can be opened.
    """
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise Fatal(
            "yt-dlp is not installed in this environment.\n"
            "  .venv/bin/python -m pip install yt-dlp"
        ) from exc

    import cv2

    runtimes = _js_runtimes()
    if not runtimes:
        raise Fatal(
            "No JavaScript runtime found, and YouTube extraction now requires one.\n"
            "Install either:\n"
            "  brew install deno      (yt-dlp's preferred runtime)\n"
            "  brew install node\n"
            "Without one, yt-dlp resolves an empty format list and every format "
            "fails with the misleading 'Requested format is not available'."
        )
    if verbose:
        detail = ", ".join(f"{name}={cfg['path']}" for name, cfg in runtimes.items())
        print(f"   JS runtimes: {detail}")

    last_error: str | None = None
    tried: list[str] = []

    for selector in FORMAT_CANDIDATES:
        options = {
            "quiet": not verbose,
            "no_warnings": not verbose,
            "format": selector,
            "noplaylist": True,
            # A live stream's manifest must not be downloaded to a file.
            "skip_download": True,
            "js_runtimes": runtimes,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            message = str(exc).strip().splitlines()[-1] if str(exc) else type(exc).__name__
            last_error = f"{selector}: {message}"
            tried.append(selector)
            if verbose:
                print(f"   format '{selector}' -> {message}")
            # A network or auth failure will fail identically for every
            # selector, so bail out rather than repeating it six times.
            lowered = message.lower()
            if any(
                token in lowered
                for token in ("temporary failure", "name resolution", "unable to connect",
                              "connection refused", "network is unreachable")
            ):
                raise Fatal(
                    f"Cannot reach YouTube: {message}\n"
                    "Check network connectivity, then retry."
                ) from exc
            if "sign in" in lowered or "private" in lowered or "unavailable" in lowered:
                raise Fatal(f"YouTube refused this video: {message}") from exc
            continue

        media_url = info.get("url")
        if not media_url and info.get("requested_formats"):
            # DASH split into separate video and audio streams. OpenCV cannot
            # mux those, so skip rather than hand it a URL that will fail.
            last_error = f"{selector}: DASH split format, not openable by OpenCV"
            tried.append(selector)
            continue
        if not media_url:
            last_error = f"{selector}: yt-dlp returned no direct URL"
            tried.append(selector)
            continue

        # Probe before committing: an unopenable URL discovered here is a clear
        # message, and discovered later is a confusing "stream ended".
        capture = cv2.VideoCapture(media_url)
        opened = capture.isOpened()
        ok = False
        if opened:
            ok, _ = capture.read()
        capture.release()

        if ok:
            print(
                f"✅ Stream resolved via format '{selector}' "
                f"({info.get('width')}x{info.get('height')}, "
                f"{'LIVE' if info.get('is_live') else 'VOD'})"
            )
            return media_url, info

        last_error = f"{selector}: OpenCV could not read frames from the resolved URL"
        tried.append(selector)
        if verbose:
            print(f"   format '{selector}' -> opened={opened}, first read failed")

    raise Fatal(
        "Could not resolve a stream OpenCV can open.\n"
        f"  tried: {', '.join(tried) or 'nothing'}\n"
        f"  last error: {last_error}\n"
        "Try a different video, or check that your OpenCV build has FFmpeg support "
        "(python -c \"import cv2; print(cv2.getBuildInformation())\" | grep FFMPEG)."
    )


# ── backend client ───────────────────────────────────────────────────────────

class Backend:
    """Thin HTTP client for the two endpoints this script needs."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        import requests

        self.base = base_url.rstrip("/")
        self.timeout = timeout
        # One Session so the ingest POSTs reuse a TCP connection — at a few
        # plates a second, handshaking each time is most of the cost.
        self.session = requests.Session()
        self.posted = 0
        self.failed = 0

    def register_camera(self, payload: dict) -> None:
        """Register the camera, treating 409 (already exists) as success."""
        import requests

        # Trailing slash on purpose: the route is declared as "/" under the
        # /cameras prefix, and a POST to the unslashed path takes a redirect.
        try:
            response = self.session.post(
                f"{self.base}/cameras/", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise Fatal(
                f"Cannot reach the backend at {self.base}: {exc}\n"
                "Start it with: .venv/bin/python -m uvicorn backend.main:app --reload"
            ) from exc

        if response.status_code in (200, 201):
            print(f"✅ Registered camera {payload['camera_id']}")
        elif response.status_code == 409:
            print(f"ℹ️  Camera {payload['camera_id']} already registered — reusing it")
        else:
            raise Fatal(
                f"Camera registration failed ({response.status_code}): "
                f"{_detail(response)}"
            )

    def ingest(self, payload: dict) -> bool:
        import requests

        try:
            response = self.session.post(
                f"{self.base}/events/ingest", json=payload, timeout=self.timeout
            )
        except requests.RequestException as exc:
            # A live camera must not die because the backend blipped. The plate
            # is already durable in the pipeline's journal.
            self.failed += 1
            print(f"⚠️  ingest failed (network): {exc}")
            return False

        if response.status_code in (200, 201):
            self.posted += 1
            return True
        self.failed += 1
        print(f"⚠️  ingest rejected ({response.status_code}): {_detail(response)}")
        return False


def _detail(response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", body))
    except Exception:
        return response.text[:300]


# ── journal tailing ──────────────────────────────────────────────────────────

class JournalTail:
    """Streams newly appended events out of the pipeline's JSONL journal.

    Tracks a byte offset and only ever reads forward, so draining costs one
    seek and one read of whatever arrived since last time.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._partial = ""

    def drain(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return []

        if not chunk:
            return []

        text = self._partial + chunk
        lines = text.split("\n")
        # The last element is either "" (clean boundary) or a torn line that
        # will be completed by the next append.
        self._partial = lines.pop()

        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events


def _build_vod_source():
    """A FrameSource for a finite remote video: every frame, in order.

    Built lazily so importing this module does not require alpr.

    `open_source` sends any https URL to `RtspSource`, which is threaded and
    reconnects with backoff forever. Both behaviours are correct for a live
    camera and wrong for a finite video:

    - **Reconnecting** turns "the video ended" into "re-open and play it again",
      so the run only stops when `--duration` fires, having spent most of its
      wall clock blocked on read timeouts.
    - **Frame dropping** discards data. A live feed must stay at *now*, so
      throwing away frames it could not keep up with is the right trade. An
      offline video has no clock to keep up with, and dropping frames there
      just loses vehicles.

    So a VOD is read the way a file is read: synchronously, every frame. The one
    concession to the network is tolerating a few consecutive failed reads —
    an HLS segment boundary produces a transient failure that a plain
    `if not ok: break` mistakes for the end of the video, which is what cut a
    23-second clip off after 27 frames.
    """
    import time as _time

    from alpr.sources import Frame, FrameSource, SourceError

    class _VodStream(FrameSource):
        # Enough to ride out a segment boundary, few enough that a genuinely
        # finished video is not waited on for long.
        MAX_CONSECUTIVE_MISSES = 5
        RETRY_DELAY = 0.25

        def __init__(self, url: str) -> None:
            self.url = url
            self.name = url
            self._capture = None
            self.dropped = 0  # read by PipelineStats; nothing is dropped here

        def frames(self):
            import cv2

            self._capture = cv2.VideoCapture(self.url)
            if not self._capture.isOpened():
                raise SourceError(f"cannot open stream: {self.url}")

            fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
            started = _time.time()
            index = 0
            misses = 0
            try:
                while True:
                    ok, image = self._capture.read()
                    if not ok:
                        misses += 1
                        if misses > self.MAX_CONSECUTIVE_MISSES:
                            break
                        _time.sleep(self.RETRY_DELAY)
                        continue
                    misses = 0
                    # Timestamps follow the *video's* timeline, not the clock.
                    # Decoding runs faster than real time, so wall-clock stamps
                    # would compress vehicles 20 seconds apart into 2 — and the
                    # camera-to-camera speed alert divides by exactly that gap.
                    yield Frame(
                        index=index,
                        image=image,
                        timestamp=started + index / fps if fps > 0 else _time.time(),
                    )
                    index += 1
            finally:
                self.close()

        def close(self) -> None:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    return _VodStream


def _open_stream(media_url: str, *, is_live: bool):
    """Open the resolved URL with the behaviour its kind deserves."""
    from alpr.sources import open_source

    if is_live:
        return open_source(media_url)
    return _build_vod_source()(media_url)


def _to_naive_utc_iso(value: str) -> str:
    """alpr journals local-naive ISO timestamps; the API wants naive UTC."""
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
    if when.tzinfo is None:
        when = when.astimezone()  # attach the local offset
    return when.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


# ── main run ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    from backend.services.anpr_service import anpr_service
    from backend.services.compute_monitor import ComputeSampler, device_info

    print(f"🔎 Resolving {args.url}")
    media_url, info = resolve_stream(args.url, verbose=args.verbose)

    backend = Backend(args.backend_url)
    backend.register_camera(
        {
            "camera_id": args.camera_id,
            "name": args.name or (info.get("title") or args.camera_id)[:80],
            "location": args.name or (info.get("title") or "YouTube live feed")[:80],
            "latitude": args.lat,
            "longitude": args.lng,
            "road": args.road,
            "direction": args.direction,
            "speed_limit_kmh": args.speed_limit,
        }
    )

    print(f"🧠 Loading models (device: {device_info().device})...")
    warm = anpr_service.warmup()
    print(f"   ready: {warm}")

    journal_dir = Path(__file__).resolve().parent.parent / "job_outputs"
    journal_dir.mkdir(parents=True, exist_ok=True)
    workbook = journal_dir / f"live_{args.camera_id}_{int(time.time())}.xlsx"
    tail = JournalTail(workbook.with_suffix(workbook.suffix + ".jsonl"))

    from dataclasses import replace

    config = replace(
        anpr_service.base_config(),
        ocr_every=args.ocr_every,
        min_reads=args.min_reads,
        min_hits=3,   # a live feed has real motion; require a confirmed track
        max_age=15,
    )
    pipeline = anpr_service.build_pipeline(config)

    # Either way the reader is threaded and drops stale frames, which is what
    # keeps the feed at "now" instead of drifting seconds behind reality.
    is_live = bool(info.get("is_live"))
    source = _open_stream(media_url, is_live=is_live)
    if not is_live:
        print("ℹ️  Not a live stream — the run will end when the video does.")

    stop_at = time.time() + args.duration if args.duration else None
    sampler = ComputeSampler().start()

    stopping = {"flag": False}

    def _handle_sigint(signum, frame):
        # Cooperative: let the pipeline finish the current frame and flush the
        # workbook. A hard exit here would lose the last un-flushed rows.
        if stopping["flag"]:
            print("\n⏹  Forcing exit.")
            raise KeyboardInterrupt
        stopping["flag"] = True
        print("\n⏹  Stopping after this frame (Ctrl-C again to force)...")

    previous_sigint = signal.signal(signal.SIGINT, _handle_sigint)

    state = {"last_telemetry": time.time(), "frames_at_mark": 0, "plates": 0}

    def deliver() -> None:
        for event in tail.drain():
            plate = event.get("plate")
            if not plate:
                continue
            state["plates"] += 1
            ok = backend.ingest(
                {
                    "camera_id": args.camera_id,
                    "timestamp": _to_naive_utc_iso(event.get("timestamp", "")),
                    "local_track_id": (
                        f"{args.camera_id}_T{event['track_id']}"
                        if event.get("track_id") is not None
                        else None
                    ),
                    "plate": plate,
                    "plate_confidence": _clamp(event.get("confidence")),
                    "latitude": args.lat,
                    "longitude": args.lng,
                    "direction": args.direction,
                    "vehicle_type": "car",
                    "speed": None,
                }
            )
            marker = "→ DB" if ok else "→ FAILED"
            print(f"🚗 {plate:<12} conf={event.get('confidence')} {marker}")

    def on_frame(frame, detections, tracks, texts) -> bool:
        if frame.index % JOURNAL_POLL_FRAMES == 0:
            deliver()

        now = time.time()
        if now - state["last_telemetry"] >= TELEMETRY_EVERY_S:
            window = now - state["last_telemetry"]
            fps = (frame.index - state["frames_at_mark"]) / window if window else 0.0
            snap = sampler.snapshot()["current"]
            print(
                f"📊 {fps:5.1f} fps | cpu {snap['process_cpu_percent']}% "
                f"({snap['process_cpu_percent_per_core']}%/core) | "
                f"rss {snap['process_rss_mb']} MB | "
                f"tracks {len(tracks)} | plates {state['plates']}"
            )
            state["last_telemetry"] = now
            state["frames_at_mark"] = frame.index

        if stopping["flag"]:
            return False
        if stop_at and now >= stop_at:
            print(f"⏱  Reached --duration {args.duration}s.")
            return False
        return True

    exit_code = 0
    try:
        stats = pipeline.run(source, workbook, max_frames=args.max_frames, on_frame=on_frame)
        # Anything logged between the last poll and the flush.
        deliver()
        print("\n" + stats.report())
        if stats.frames == 0:
            print(
                "\n⚠️  No frames were read. The stream ended immediately or the URL expired "
                "(YouTube media URLs are short-lived) — re-run to resolve a fresh one."
            )
            exit_code = 1
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        exit_code = 130
    except Exception as exc:
        print(f"\n❌ Run failed: {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        try:
            source.close()
        except Exception:
            pass
        summary = sampler.stop()
        print(
            "\n💻 compute: "
            f"{summary['elapsed_s']}s | cpu avg {summary['process_cpu_percent_avg']}% "
            f"(peak {summary['process_cpu_percent_peak']}%) | "
            f"rss avg {summary['process_rss_mb_avg']} MB "
            f"(peak {summary['process_rss_mb_peak']} MB) | "
            f"device {summary['device']['device']} ({summary['device']['device_name']})"
        )
        print(
            f"📤 posted {backend.posted} event(s), {backend.failed} failure(s). "
            f"Workbook: {workbook}"
        )

    return exit_code


def _clamp(value) -> float | None:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf > 1.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach a YouTube video/live stream as an ANPR camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="YouTube video or live stream URL")
    parser.add_argument("--camera-id", required=True, help="Camera id to register and ingest under")
    parser.add_argument("--lat", type=float, default=28.6139, help="Camera latitude")
    parser.add_argument("--lng", type=float, default=77.2090, help="Camera longitude")
    parser.add_argument("--name", default=None, help="Human name (defaults to the video title)")
    parser.add_argument("--road", default=None, help="Road name, for route-anomaly analytics")
    parser.add_argument("--direction", default=None, help="e.g. NORTHBOUND")
    parser.add_argument("--speed-limit", type=float, default=60.0, help="Speed limit in km/h")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND, help="Backend base URL")
    parser.add_argument(
        "--ocr-every",
        type=int,
        default=3,
        help="Read each track once every N frames (1 is accurate and too slow for live)",
    )
    parser.add_argument(
        "--min-reads", type=int, default=2, help="Reads a track needs before its vote is trusted"
    )
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--verbose", action="store_true", help="Show yt-dlp format probing detail")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.ocr_every < 1:
        print("❌ --ocr-every must be at least 1")
        return 2
    try:
        return run(args)
    except Fatal as exc:
        print(f"❌ {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹  Interrupted before the stream started.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

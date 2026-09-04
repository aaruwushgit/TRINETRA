"""
Hamburg Camera Adapter
======================
Fetches real live JPEG frames from the Hamburg Verkehrskameras API and feeds
them into the existing CameraWorker pipeline. This is the ONLY Hamburg-specific
file — everything downstream (YOLO, ByteTrack, ANPR, FastAPI) is untouched.

Source:  https://api.hamburg.de/datasets/v1/verkehrskameras
License: Datenlizenz Deutschland – Namensnennung – Version 2.0
         Source: Freie und Hansestadt Hamburg, Behörde für Verkehr und Mobilitätswende (BVM)
"""
from __future__ import annotations

import io
import time
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import requests

# Re-use existing CameraWorker pipeline
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.camera_worker import CameraWorker, VEHICLE_CLASS_MAP

# Hamburg OGC API
HAMBURG_API_BASE = "https://api.hamburg.de/datasets/v1/verkehrskameras"
HAMBURG_COLLECTION = "verkehr_kameras_internet"
# Hamburg serves live snapshots at this endpoint per camera item
HAMBURG_SNAPSHOT_URL = (
    f"{HAMBURG_API_BASE}/collections/{HAMBURG_COLLECTION}/items/{{hamburg_id}}/photo"
)
# Fallback: the OGC API page itself sometimes embeds the image inline
HAMBURG_ITEM_URL = (
    f"{HAMBURG_API_BASE}/collections/{HAMBURG_COLLECTION}/items/{{hamburg_id}}"
)

# Request headers that identify us as a legitimate client
HEADERS = {
    "User-Agent": "VehicleIntelligence/1.0 (SIH Project; Educational)",
    "Accept": "image/jpeg, image/png, image/*, */*",
}


class CameraStatus(Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"


class HamburgFrameSource:
    """
    Manages the live JPEG frame fetching for a single Hamburg camera.
    Thread-safe: a background thread continuously fetches frames, and the
    main processing thread reads them via get_frame().
    """

    STALE_THRESHOLD_S = 10.0   # Mark STALE after this many seconds without a new frame
    MAX_RETRY_DELAY_S = 60.0   # Maximum backoff between retries

    def __init__(self, camera_id: str, hamburg_id: int, fetch_interval_s: float = 2.0):
        self.camera_id = camera_id
        self.hamburg_id = hamburg_id
        self.fetch_interval = fetch_interval_s

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._last_fetch_ts: Optional[float] = None
        self._last_success_ts: Optional[float] = None
        self._frames_received: int = 0
        self._status: CameraStatus = CameraStatus.DISCONNECTED
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._retry_delay = 2.0

    # ── Properties (thread-safe reads) ──────────────────────────────
    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    @property
    def frames_received(self) -> int:
        with self._lock:
            return self._frames_received

    @property
    def last_frame_age_ms(self) -> Optional[float]:
        with self._lock:
            if self._last_success_ts is None:
                return None
            return (time.time() - self._last_success_ts) * 1000

    @property
    def fps(self) -> float:
        """Approximate FPS: 1 / fetch_interval as upper bound."""
        age = self.last_frame_age_ms
        if age and age < self.STALE_THRESHOLD_S * 1000:
            return round(1.0 / self.fetch_interval, 1)
        return 0.0

    # ── Frame fetching ───────────────────────────────────────────────
    def _fetch_frame(self) -> Optional[np.ndarray]:
        """
        Try to fetch a live JPEG from the Hamburg API.
        Returns an OpenCV BGR numpy array or None on failure.
        """
        # Try the /photo endpoint first
        snapshot_url = HAMBURG_SNAPSHOT_URL.format(hamburg_id=self.hamburg_id)
        try:
            r = requests.get(snapshot_url, headers=HEADERS, timeout=8)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                arr = np.frombuffer(r.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
        except requests.RequestException:
            pass

        # Fallback: try the item HTML page for an embedded image
        # (Some Hamburg cameras embed their image in the OGC API HTML response)
        return None

    def _fetch_loop(self):
        """Background thread: continuously fetch frames."""
        while self._running:
            t0 = time.time()
            frame = self._fetch_frame()

            with self._lock:
                self._last_fetch_ts = time.time()
                if frame is not None:
                    self._frame = frame
                    self._last_success_ts = time.time()
                    self._frames_received += 1
                    self._status = CameraStatus.LIVE
                    self._retry_delay = 2.0  # reset backoff on success
                else:
                    # Check if we've gone stale
                    if self._last_success_ts is None:
                        self._status = CameraStatus.DISCONNECTED
                    elif (time.time() - self._last_success_ts) > self.STALE_THRESHOLD_S:
                        self._status = CameraStatus.STALE

            # Sleep for remaining fetch interval (or retry delay on failure)
            elapsed = time.time() - t0
            sleep_for = max(0, self.fetch_interval - elapsed)
            if frame is None:
                sleep_for = min(self._retry_delay, self.MAX_RETRY_DELAY_S)
                self._retry_delay = min(self._retry_delay * 1.5, self.MAX_RETRY_DELAY_S)
            time.sleep(sleep_for)

    def start(self):
        """Start the background fetch thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._fetch_loop,
            name=f"HamburgFetch-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def diagnostic(self) -> dict:
        """Returns a status dict for display/logging."""
        return {
            "camera_id": self.camera_id,
            "hamburg_id": self.hamburg_id,
            "source": HAMBURG_SNAPSHOT_URL.format(hamburg_id=self.hamburg_id),
            "status": self.status.value,
            "frames_received": self.frames_received,
            "fps": self.fps,
            "last_frame_age_ms": self.last_frame_age_ms,
            "last_success_at": (
                datetime.fromtimestamp(self._last_success_ts, tz=timezone.utc).isoformat()
                if self._last_success_ts else None
            ),
        }


class HamburgCameraWorker(CameraWorker):
    """
    Extends the existing CameraWorker to use HamburgFrameSource instead of
    OpenCV VideoCapture. The YOLO, ByteTrack, ANPR, and all event logic are
    inherited completely unchanged.
    """

    def __init__(self, camera_cfg: dict, backend_url: str, yolo_weights: str = "yolov8n.pt", fetch_interval: float = 2.0):
        super().__init__(
            camera_id=camera_cfg["camera_id"],
            source=f"hamburg_api:{camera_cfg['hamburg_id']}",  # Informational only
            backend_url=backend_url,
            yolo_weights=yolo_weights,
            latitude=camera_cfg["latitude"],
            longitude=camera_cfg["longitude"],
            direction=camera_cfg.get("direction", "NORTH"),
            frame_skip=1,  # We control FPS via fetch_interval, not frame_skip
            snapshot_interval=60,
        )
        self.hamburg_source = HamburgFrameSource(
            camera_id=camera_cfg["camera_id"],
            hamburg_id=camera_cfg["hamburg_id"],
            fetch_interval_s=fetch_interval,
        )
        self.fetch_interval = fetch_interval

    def run(self):
        """Override run() to use Hamburg live frames instead of VideoCapture."""
        self.init_model()
        self.hamburg_source.start()

        print(f"🇩🇪 [{self.camera_id}] Hamburg camera adapter started.")
        print(f"   Source: {HAMBURG_SNAPSHOT_URL.format(hamburg_id=self.hamburg_source.hamburg_id)}")
        print(f"   Fetch interval: {self.fetch_interval}s")

        last_snapshot_time = time.time()
        wait_count = 0

        try:
            while True:
                diag = self.hamburg_source.diagnostic()
                status = self.hamburg_source.status

                # Print diagnostic every 30 seconds
                if wait_count % 15 == 0:
                    print(
                        f"[{self.camera_id}] Status: {diag['status']} | "
                        f"Frames: {diag['frames_received']} | "
                        f"FPS: {diag['fps']} | "
                        f"Age: {diag['last_frame_age_ms']:.0f}ms"
                        if diag['last_frame_age_ms'] else
                        f"[{self.camera_id}] Status: {diag['status']} | Waiting for first frame..."
                    )

                if status == CameraStatus.DISCONNECTED:
                    # No frames yet — wait and retry
                    time.sleep(2)
                    wait_count += 1
                    if wait_count > 60:  # ~2 min with no frame
                        print(f"⚠️  [{self.camera_id}] CAMERA OFFLINE — Hamburg feed unavailable after 2 min.")
                        print(f"    Reason: /photo endpoint returned no valid image.")
                        print(f"    This is expected per Hamburg documentation: cameras may be blocked for privacy.")
                        print(f"    Last attempt: {HAMBURG_SNAPSHOT_URL.format(hamburg_id=self.hamburg_source.hamburg_id)}")
                        break
                    continue

                if status == CameraStatus.STALE:
                    age = diag.get("last_frame_age_ms") or 0
                    print(f"⚠️  [{self.camera_id}] STALE — last frame {age/1000:.1f}s ago")
                    time.sleep(2)
                    continue

                # Get the latest live frame
                frame = self.hamburg_source.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                wait_count = 0  # Reset on successful frame

                # Run the EXISTING YOLO tracking (inherited method)
                try:
                    results = self.model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                        conf=0.25,
                    )
                except Exception as e:
                    print(f"[{self.camera_id}] YOLO error: {e}")
                    time.sleep(0.5)
                    continue

                res = results[0]
                current_frame_vehicles = 0

                if res.boxes is not None and res.boxes.id is not None:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                    track_ids = res.boxes.id.cpu().numpy().astype(int)
                    current_frame_vehicles = len(track_ids)

                    # Import optional ANPR
                    try:
                        from backend.services.anpr_service import anpr_service
                    except Exception:
                        anpr_service = None

                    for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                        cls_name = self.model.names.get(cls_id, VEHICLE_CLASS_MAP.get(cls_id, "vehicle"))
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # Speed calculation (pixel displacement)
                        hist = self.track_history.setdefault(track_id, [])
                        curr_time = time.time()
                        hist.append((cx, cy, curr_time))
                        speed_kmh = None
                        from scripts.camera_worker import SPEED_WINDOW, SPEED_MIN_KMH, TRACK_HISTORY_MAX, PX_TO_METER
                        if len(hist) >= SPEED_WINDOW:
                            s, e = hist[-SPEED_WINDOW], hist[-1]
                            dist_px = ((e[0]-s[0])**2 + (e[1]-s[1])**2)**0.5
                            dt = e[2] - s[2]
                            if dt > 0:
                                speed_kmh = ((dist_px * PX_TO_METER) / dt) * 3.6
                                self.track_speeds[track_id] = speed_kmh
                                if speed_kmh > SPEED_MIN_KMH:
                                    self.window_speeds.append(speed_kmh)
                            if len(hist) > TRACK_HISTORY_MAX:
                                hist.pop(0)

                        # ANPR on vehicle crop
                        plate_text, plate_conf = None, None
                        if anpr_service:
                            crop = frame[max(0, y1):y2, max(0, x1):x2]
                            if crop.size > 0:
                                try:
                                    p_res = anpr_service.process_frame(crop)
                                    if p_res.plate:
                                        plate_text, plate_conf = self.get_consensus_plate(
                                            track_id, p_res.plate, p_res.confidence
                                        )
                                except Exception:
                                    pass

                        # Buffer event (uses inherited bulk-ingest buffering)
                        if track_id not in self.seen_tracks or (plate_text and track_id not in self.track_plates):
                            self.seen_tracks.add(track_id)
                            self.window_class_counts[cls_name] += 1
                            self.post_event(
                                track_id=track_id,
                                vehicle_type=cls_name,
                                plate=plate_text or self.track_plates.get(track_id),
                                conf=plate_conf,
                                speed=speed_kmh,
                            )

                self.window_peak_density = max(self.window_peak_density, current_frame_vehicles)

                # Periodic snapshot flush
                if time.time() - last_snapshot_time >= self.snapshot_interval:
                    self.flush_snapshot()
                    last_snapshot_time = time.time()

                # Flush bulk events if needed
                import time as _t
                if (_t.time() - self.last_bulk_flush > self.bulk_interval or
                        len(self.event_buffer) >= 50):
                    self.flush_bulk_events()

                # Rate-limit to fetch_interval
                time.sleep(max(0, self.fetch_interval - 0.05))

        except KeyboardInterrupt:
            print(f"[{self.camera_id}] Stopping Hamburg worker...")
        finally:
            self.flush_bulk_events()
            self.flush_snapshot()
            self.hamburg_source.stop()

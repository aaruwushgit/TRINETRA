"""
Camera Worker — Processes a single camera stream with YOLOv8 + ByteTrack + ANPR.

Reads frames, tracks vehicles, calculates speed via pixel displacement,
attempts license plate OCR via ANPR service, and ingests events and traffic
snapshots into the central backend.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import requests

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Optional ANPR import
try:
    from backend.services.anpr_service import anpr_service
except Exception as e:
    anpr_service = None

# Default vehicle classes (COCO or custom traffic model)
VEHICLE_CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck"
}

# Speed estimation constants
PX_TO_METER = 0.05
SPEED_WINDOW = 5
TRACK_HISTORY_MAX = 30
SPEED_MIN_KMH = 3.0


class CameraWorker:
    def __init__(
        self,
        camera_id: str,
        source: str,
        backend_url: str = "http://localhost:8000",
        yolo_weights: str = "yolov8n.pt",
        latitude: float | None = None,
        longitude: float | None = None,
        direction: str | None = None,
        frame_skip: int = 2,
        snapshot_interval: int = 60,
        name: str | None = None,
        road: str | None = None,
        location: str | None = None,
        speed_limit_kmh: float = 60.0,
    ):
        self.camera_id = camera_id
        self.source = source
        self.backend_url = backend_url.rstrip("/")
        self.yolo_weights = yolo_weights
        self.latitude = latitude
        self.longitude = longitude
        self.direction = direction
        self.frame_skip = max(1, frame_skip)
        self.snapshot_interval = snapshot_interval
        self.name = name or camera_id
        self.road = road
        self.location = location or self.name
        self.speed_limit_kmh = speed_limit_kmh

        self.model = None
        self.seen_tracks: set[int] = set()
        self.track_plates: dict[int, str] = {}
        self.track_ocr_history: dict[int, list[tuple[str, float]]] = {}  # 5-frame OCR voting buffer
        self.track_history: dict[int, list[tuple[int, int, float]]] = {}
        self.track_speeds: dict[int, float] = {}

        # Snapshot aggregation state
        self.window_start = datetime.now(timezone.utc)
        self.window_class_counts: Counter = Counter()
        self.window_speeds: list[float] = []
        self.window_peak_density = 0

        # Bulk ingestion buffer
        self.event_buffer: list[dict] = []
        self.last_bulk_flush = time.time()
        self.bulk_interval = 2.0  # flush every 2 seconds

    def init_model(self):
        print(f"[{self.camera_id}] Loading YOLO model: {self.yolo_weights}")
        self.model = YOLO(self.yolo_weights)

    def self_register(self):
        """
        Plug-and-play camera onboarding: register this camera's metadata with
        the central backend if it isn't already known, so an operator never
        has to manually call POST /cameras before plugging in a new feed.
        Idempotent — a 409 (already registered) is treated as success.
        """
        if self.latitude is None or self.longitude is None:
            print(f"[{self.camera_id}] ⚠️ No --lat/--lng provided, skipping self-registration "
                  f"(camera must already be registered via POST /cameras).")
            return

        payload = {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road": self.road,
            "direction": self.direction or "NORTH",
            "speed_limit_kmh": self.speed_limit_kmh,
        }
        try:
            res = requests.post(f"{self.backend_url}/cameras", json=payload, timeout=5)
            if res.status_code == 201:
                print(f"✅ [{self.camera_id}] Self-registered with backend ({self.backend_url}).")
            elif res.status_code == 409:
                print(f"[{self.camera_id}] Already registered with backend.")
            else:
                print(f"⚠️ [{self.camera_id}] Self-registration returned {res.status_code}: {res.text}")
        except requests.RequestException as e:
            print(f"⚠️ [{self.camera_id}] Could not reach backend to self-register: {e}")

    def get_consensus_plate(self, track_id: int, new_plate: str | None, conf: float | None) -> tuple[str | None, float | None]:
        """
        Temporal Voting Consensus Engine:
        Maintains a rolling 5-frame OCR buffer per track ID.
        Returns the majority-voted plate string with average confidence.
        """
        if not new_plate:
            return self.track_plates.get(track_id), None

        history = self.track_ocr_history.setdefault(track_id, [])
        clean_p = new_plate.upper().replace(" ", "")
        history.append((clean_p, conf or 0.8))
        if len(history) > 5:
            history.pop(0)

        # Count frequencies
        plate_counts = Counter(p for p, _ in history)
        best_plate, count = plate_counts.most_common(1)[0]

        # Calculate average confidence for the winning plate
        matched_confs = [c for p, c in history if p == best_plate]
        avg_conf = sum(matched_confs) / len(matched_confs)

        self.track_plates[track_id] = best_plate
        return best_plate, round(avg_conf, 2)

    def run(self):
        self.self_register()
        self.init_model()
        source_val = int(self.source) if self.source.isdigit() else self.source
        retry_delay = 2

        while True:
            cap = cv2.VideoCapture(source_val)
            # Reduce buffer size to minimize RTSP latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                print(f"⚠️ [{self.camera_id}] Video source unavailable: {self.source}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(30, retry_delay * 2)
                continue

            print(f"✅ [{self.camera_id}] Connected. Processing stream...")
            retry_delay = 2
            frame_idx = 0
            last_snapshot_time = time.time()

            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        # If video file ended, loop back for demo streams
                        if isinstance(source_val, str) and os.path.isfile(source_val):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        print(f"⚠️ [{self.camera_id}] Stream disconnected or frame lost.")
                        break

                    frame_idx += 1
                    if frame_idx % self.frame_skip != 0:
                        continue

                    # Run YOLO tracking with ByteTrack
                    results = self.model.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        verbose=False,
                        conf=0.25,
                    )

                    res = results[0]
                    current_frame_vehicles = 0

                    if res.boxes is not None and res.boxes.id is not None:
                        boxes = res.boxes.xyxy.cpu().numpy()
                        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                        track_ids = res.boxes.id.cpu().numpy().astype(int)
                        current_frame_vehicles = len(track_ids)

                        for box, cls_id, track_id in zip(boxes, cls_ids, track_ids):
                            cls_name = self.model.names.get(cls_id, VEHICLE_CLASS_MAP.get(cls_id, "vehicle"))
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            curr_time = time.time()

                            # Speed calculation via trajectory displacement
                            hist = self.track_history.setdefault(track_id, [])
                            hist.append((cx, cy, curr_time))

                            speed_kmh = None
                            if len(hist) >= SPEED_WINDOW:
                                s, e = hist[-SPEED_WINDOW], hist[-1]
                                dist_px = ((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5
                                dt = e[2] - s[2]
                                if dt > 0:
                                    speed_kmh = ((dist_px * PX_TO_METER) / dt) * 3.6
                                    self.track_speeds[track_id] = speed_kmh
                                    if speed_kmh > SPEED_MIN_KMH:
                                        self.window_speeds.append(speed_kmh)

                                if len(hist) > TRACK_HISTORY_MAX:
                                    hist.pop(0)

                            # Run ANPR OCR on vehicle crop
                            plate_text = None
                            plate_conf = None
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

                            # Ingest event if new track or updated consensus plate
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

                    # Periodic snapshot check
                    if time.time() - last_snapshot_time >= self.snapshot_interval:
                        self.flush_snapshot()
                        last_snapshot_time = time.time()

            except KeyboardInterrupt:
                print(f"[{self.camera_id}] Stopping stream worker...")
                break
            finally:
                self.flush_snapshot()
                cap.release()

            # If it was a single video file (not RTSP), finish
            if isinstance(source_val, str) and not source_val.startswith("rtsp://") and not source_val.startswith("http://"):
                break

    def post_event(self, track_id: int, vehicle_type: str, plate: str | None, conf: float | None, speed: float | None):
        """Buffer vehicle events for bulk ingestion."""
        payload = {
            "camera_id": self.camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_track_id": f"{self.camera_id}_T{track_id}",
            "plate": plate,
            "plate_confidence": conf,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "direction": self.direction,
            "vehicle_type": vehicle_type,
            "speed": round(speed, 1) if speed else None,
        }
        self.event_buffer.append(payload)

        # Flush if interval reached or buffer large
        if time.time() - self.last_bulk_flush > self.bulk_interval or len(self.event_buffer) >= 50:
            self.flush_bulk_events()

    def flush_bulk_events(self):
        """Flush the buffered events to the backend."""
        if not self.event_buffer:
            return

        batch = list(self.event_buffer)
        self.event_buffer.clear()
        self.last_bulk_flush = time.time()

        try:
            res = requests.post(f"{self.backend_url}/events/bulk-ingest", json=batch, timeout=3)
            if res.status_code != 200:
                print(f"[{self.camera_id}] ⚠️ Bulk ingest failed: {res.text}")
        except requests.RequestException as e:
            print(f"[{self.camera_id}] ⚠️ Backend connection error during bulk ingest: {e}")
            # Could re-append to buffer, but omitting for simplicity/memory safety

    def flush_snapshot(self):
        """Aggregate and send traffic snapshot."""
        now = datetime.now(timezone.utc)
        duration = (now - self.window_start).total_seconds()
        if duration <= 0:
            return

        avg_speed = sum(self.window_speeds) / len(self.window_speeds) if self.window_speeds else None
        vol = sum(self.window_class_counts.values())

        c_level = "LOW"
        if vol > 50 or (avg_speed and avg_speed < 15):
            c_level = "HIGH"
        elif vol > 20 or (avg_speed and avg_speed < 30):
            c_level = "MEDIUM"

        payload = {
            "camera_id": self.camera_id,
            "window_start": self.window_start.isoformat(),
            "window_end": now.isoformat(),
            "vehicle_count": vol,
            "avg_speed": round(avg_speed, 1) if avg_speed else None,
            "peak_density": self.window_peak_density,
            "class_counts": dict(self.window_class_counts),
            "congestion_level": c_level,
        }

        try:
            requests.post(f"{self.backend_url}/analytics/traffic-snapshot", json=payload, timeout=2)
        except requests.RequestException as e:
            print(f"[{self.camera_id}] ⚠️ Could not send snapshot: {e}")

        # Reset state
        self.window_start = now
        self.window_class_counts.clear()
        self.window_speeds.clear()
        self.window_peak_density = 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera AI Worker")
    parser.add_argument("--camera-id", required=True, help="Camera identifier (e.g. CAM001)")
    parser.add_argument("--source", required=True, help="RTSP URL, video file path, or webcam index")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights path")
    parser.add_argument("--lat", type=float, default=None, help="Latitude")
    parser.add_argument("--lng", type=float, default=None, help="Longitude")
    parser.add_argument("--direction", default="NORTH", help="Direction (NORTH, SOUTH, etc.)")
    parser.add_argument("--frame-skip", type=int, default=2, help="Process every Nth frame")
    parser.add_argument("--snapshot-interval", type=int, default=30, help="Traffic snapshot interval in seconds")
    parser.add_argument("--name", default=None, help="Human-readable camera name (for self-registration)")
    parser.add_argument("--road", default=None, help="Road/corridor name (for self-registration)")
    parser.add_argument("--location", default=None, help="Location label (for self-registration)")
    parser.add_argument("--speed-limit", type=float, default=60.0, help="Legal speed limit (km/h) for this road segment")

    args = parser.parse_args()
    worker = CameraWorker(
        camera_id=args.camera_id,
        source=args.source,
        backend_url=args.backend_url,
        yolo_weights=args.weights,
        latitude=args.lat,
        longitude=args.lng,
        direction=args.direction,
        frame_skip=args.frame_skip,
        snapshot_interval=args.snapshot_interval,
        name=args.name,
        road=args.road,
        location=args.location,
        speed_limit_kmh=args.speed_limit,
    )
    worker.run()


if __name__ == "__main__":
    main()

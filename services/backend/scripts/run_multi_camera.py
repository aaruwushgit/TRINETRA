"""
Multi-Camera Stream Orchestrator.

Spawns independent worker processes for each configured camera stream.
Processes run simultaneously and push detections, ANPR plates, and
traffic intelligence snapshots into the central backend.

Usage:
  python scripts/run_multi_camera.py --config scripts/cameras_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from multiprocessing import Process
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.camera_worker import CameraWorker


def start_worker(camera_cfg: dict, backend_url: str, weights_path: str):
    """Entrypoint executed inside each spawned process."""
    source = camera_cfg["source"]
    # If relative path, resolve relative to this script directory
    if not source.startswith("rtsp://") and not source.startswith("http://") and not source.isdigit():
        resolved_source = (BASE_DIR / "scripts" / source).resolve()
        if resolved_source.exists():
            source = str(resolved_source)

    worker = CameraWorker(
        camera_id=camera_cfg["camera_id"],
        source=source,
        backend_url=backend_url,
        yolo_weights=weights_path,
        latitude=camera_cfg.get("latitude"),
        longitude=camera_cfg.get("longitude"),
        direction=camera_cfg.get("direction", "NORTH"),
        frame_skip=camera_cfg.get("frame_skip", 2),
        snapshot_interval=camera_cfg.get("snapshot_interval", 30),
        name=camera_cfg.get("name"),
        road=camera_cfg.get("road"),
        location=camera_cfg.get("location"),
        speed_limit_kmh=camera_cfg.get("speed_limit_kmh", 60.0),
    )
    worker.run()


def main():
    parser = argparse.ArgumentParser(description="Multi-Camera Stream Runner")
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "scripts" / "cameras_config.json"),
        help="Path to camera JSON config",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Central FastAPI backend URL",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Path to YOLOv8 weights file",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        cameras = json.load(f)

    print(f"🚀 Starting Multi-Camera Pipeline with {len(cameras)} streams...")
    print(f"   Backend: {args.backend_url}")
    print(f"   Model Weights: {args.weights}\n")

    processes: list[Process] = []

    for cam in cameras:
        p = Process(
            target=start_worker,
            args=(cam, args.backend_url, args.weights),
            name=f"Worker-{cam['camera_id']}",
        )
        p.daemon = True
        p.start()
        processes.append(p)
        print(f"  ▶️  Spawned worker for {cam['camera_id']} ({cam.get('name', 'Camera')}) [PID: {p.pid}]")

    def handle_sigint(signum, frame):
        print("\n🛑 Stopping all camera stream workers...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        print("✅ All camera processes terminated.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # Keep orchestrator alive
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

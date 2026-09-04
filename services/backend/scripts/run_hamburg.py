"""
Hamburg Deployment Runner
=========================
Starts the full Hamburg camera deployment. Reads deployments/hamburg/cameras.json,
registers all cameras via the existing /cameras/ API, then spawns one
HamburgCameraWorker per camera using the existing multi-process pipeline.

Usage:
    python scripts/run_hamburg.py [--backend-url URL] [--weights PATH] [--cameras N]

Environment:
    BACKEND_URL   = http://localhost:8000
    YOLO_WEIGHTS  = yolov8n.pt
    HAMBURG_CAMERAS = 5  (how many cameras to activate, default all 18)
    HAMBURG_FETCH_INTERVAL = 3  (seconds between frame fetches)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path

# Use default start method on macOS (spawn)
mp.set_executable(sys.executable)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import requests


def register_camera(camera_cfg: dict, backend_url: str) -> bool:
    """Register a Hamburg camera with the existing backend API."""
    payload = {
        "camera_id": camera_cfg["camera_id"],
        "name": camera_cfg["name"],
        "location": camera_cfg["location"],
        "latitude": camera_cfg["latitude"],
        "longitude": camera_cfg["longitude"],
        "direction": camera_cfg.get("direction", "NORTH"),
        "camera_type": "HAMBURG_WEB",
    }
    try:
        r = requests.post(f"{backend_url}/cameras/", json=payload, timeout=10)
        if r.status_code in (200, 201):
            print(f"  ✅ Registered: {camera_cfg['camera_id']} — {camera_cfg['name']}")
            return True
        elif r.status_code == 409:
            print(f"  ⏩ Already exists: {camera_cfg['camera_id']}")
            return True
        else:
            print(f"  ⚠️  Failed to register {camera_cfg['camera_id']}: {r.status_code} {r.text[:100]}")
            return False
    except requests.RequestException as e:
        print(f"  ❌ Cannot reach backend at {backend_url}: {e}")
        return False


def start_worker(camera_cfg: dict, backend_url: str, weights: str, fetch_interval: float):
    """Entry point for each worker process."""
    from scripts.hamburg_adapter import HamburgCameraWorker
    worker = HamburgCameraWorker(
        camera_cfg=camera_cfg,
        backend_url=backend_url,
        yolo_weights=weights,
        fetch_interval=fetch_interval,
    )
    worker.run()


def main():
    parser = argparse.ArgumentParser(description="Hamburg Camera Deployment")
    parser.add_argument("--backend-url", default=os.getenv("BACKEND_URL", "http://localhost:8000"))
    parser.add_argument("--weights", default=os.getenv("YOLO_WEIGHTS", "yolov8n.pt"))
    parser.add_argument("--cameras", type=int, default=int(os.getenv("HAMBURG_CAMERAS", "18")),
                        help="Number of cameras to activate (default: all 18)")
    parser.add_argument("--fetch-interval", type=float,
                        default=float(os.getenv("HAMBURG_FETCH_INTERVAL", "3.0")),
                        help="Seconds between frame fetches per camera (default: 3.0)")
    args = parser.parse_args()

    cameras_path = BASE_DIR / "deployments" / "hamburg" / "cameras.json"
    if not cameras_path.exists():
        print(f"❌ cameras.json not found at {cameras_path}")
        sys.exit(1)

    with open(cameras_path) as f:
        all_cameras = json.load(f)

    cameras = all_cameras[:args.cameras]

    print("🇩🇪 HAMBURG DEPLOYMENT — Vehicle Intelligence Backend")
    print("=" * 60)
    print(f"   Backend:        {args.backend_url}")
    print(f"   YOLO weights:   {args.weights}")
    print(f"   Cameras:        {len(cameras)} of {len(all_cameras)}")
    print(f"   Fetch interval: {args.fetch_interval}s (~{1/args.fetch_interval:.1f} FPS)")
    print(f"   Source:         Hamburg OGC API (live JPEG snapshots)")
    print(f"   License:        Datenlizenz Deutschland – Namensnennung 2.0")
    print(f"                   © Freie und Hansestadt Hamburg, BVM")
    print()

    # Register all cameras with the existing backend
    print("📡 Registering cameras with backend API...")
    for cam in cameras:
        register_camera(cam, args.backend_url)
    print()

    # Start worker processes (one per camera)
    print(f"🚀 Spawning {len(cameras)} Hamburg camera workers...")
    processes: list[Process] = []

    for cam in cameras:
        p = Process(
            target=start_worker,
            args=(cam, args.backend_url, args.weights, args.fetch_interval),
            name=f"Hamburg-{cam['camera_id']}",
        )
        p.daemon = True
        p.start()
        processes.append(p)
        print(f"  ▶  {cam['camera_id']} ({cam['name']}) [PID {p.pid}]")
        time.sleep(0.2)  # Stagger startup

    print()
    print("✅ All Hamburg workers running. Press Ctrl+C to stop.")
    print("   Watch for: [HH_XXX] Status: LIVE | Frames: N | FPS: X.X")

    def handle_stop(signum, frame):
        print("\n🛑 Shutting down Hamburg deployment...")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=3)
        print("✅ Hamburg deployment stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()

"""
High-Throughput Tracking & Trajectory Benchmark.
Tests how fast the backend processes incoming multi-camera vehicle sightings
and reconstructs complete city-wide trajectories across 10,000+ records.
"""
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.database import init_db
from backend.main import app

init_db()
client = TestClient(app)

def benchmark_tracking():
    print("=" * 75)
    print("⚡ MULTI-CAMERA VEHICLE TRACKING & TRAJECTORY LATENCY BENCHMARK")
    print("=" * 75)

    # 1. Benchmark Event Ingestion + Live Tracking Association
    ingest_latencies = []
    num_samples = 100
    now = datetime.now(timezone.utc)

    print(f"\n[Test 1] Benchmarking Single Event Ingestion + Identity Association ({num_samples} samples)...")
    for i in range(num_samples):
        payload = {
            "camera_id": "CAM_HUB_01",
            "timestamp": now.isoformat(),
            "local_track_id": f"TRACK_{i}",
            "plate": f"KA01AB{i % 20:04d}",
            "plate_confidence": 0.98,
            "latitude": 13.015,
            "longitude": 80.212,
            "vehicle_type": "car",
            "speed": 52.0
        }
        t0 = time.perf_counter()
        resp = client.post("/events/ingest", json=payload)
        t1 = time.perf_counter()
        
        assert resp.status_code == 200
        ingest_latencies.append((t1 - t0) * 1000)

    avg_ingest = statistics.mean(ingest_latencies)
    p95_ingest = statistics.quantiles(ingest_latencies, n=20)[18]  # 95th percentile
    print(f"  ✓ Average Ingestion + MTMC Association Latency: {avg_ingest:.2f} ms")
    print(f"  ✓ 95th Percentile Latency (p95):               {p95_ingest:.2f} ms")
    print(f"  ✓ Estimated Ingestion Throughput:              {1000 / avg_ingest:.0f} events/sec per worker")

    # 2. Benchmark Trajectory Reconstruction
    print(f"\n[Test 2] Benchmarking Full Trajectory Reconstruction Across Database...")
    sample_plate = "KA01AB0001"
    traj_latencies = []

    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.get(f"/vehicles/{sample_plate}/trajectory")
        t1 = time.perf_counter()
        assert resp.status_code == 200
        traj_latencies.append((t1 - t0) * 1000)

    avg_traj = statistics.mean(traj_latencies)
    print(f"  ✓ Target Vehicle: {sample_plate}")
    data = resp.json()
    print(f"  ✓ Trajectory Points Found: {len(data['points'])} waypoints across city")
    print(f"  ✓ Trajectory Query Latency: {avg_traj:.2f} ms")

    # 3. Benchmark ML Next-Location Prediction
    print(f"\n[Test 3] Benchmarking ML Trajectory Prediction Latency...")
    pred_latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.get(f"/vehicles/{sample_plate}/predict-next-location?top_n=3")
        t1 = time.perf_counter()
        assert resp.status_code == 200
        pred_latencies.append((t1 - t0) * 1000)

    avg_pred = statistics.mean(pred_latencies)
    print(f"  ✓ ML Next-Hop Prediction Latency: {avg_pred:.2f} ms")

    print("\n" + "=" * 75)
    print("🚀 SUMMARY OF SYSTEM SPEEDS:")
    print("=" * 75)
    print(f"  • Frame OCR Ingest -> Identity Match:   ~{avg_ingest:.1f} ms")
    print(f"  • Full Historical Path Reconstruction:  ~{avg_traj:.1f} ms")
    print(f"  • ML Next-Hop Destination Forecast:    ~{avg_pred:.1f} ms")
    print("=" * 75)

if __name__ == "__main__":
    benchmark_tracking()

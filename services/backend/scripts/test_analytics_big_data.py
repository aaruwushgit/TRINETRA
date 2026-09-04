"""
Big Data Synthetic Benchmark for Analytics Engine.

Generates 5,000 synthetic vehicle event records across 10 camera junctions
spanning various traffic corridors, rush hours, and vehicle classes, then
benchmarks and validates all 8 analytics endpoints.
"""
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.database import SessionLocal, init_db
from backend.main import app
from backend.models.camera import Camera
from backend.models.vehicle_event import VehicleEvent

client = TestClient(app)

# 10 Camera Network across Major Hubs
SYNTHETIC_CAMERAS = [
    {"camera_id": f"CAM_HUB_{i:02d}", "name": f"Junction Corridor #{i}", "location": "SmartCity", "latitude": 13.0000 + (i * 0.015), "longitude": 80.2000 + (i * 0.012), "road": f"Corridor-{i}", "direction": "NORTH" if i % 2 == 0 else "SOUTH"}
    for i in range(1, 11)
]

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "auto"]
VEHICLE_CLASS_WEIGHTS = [0.60, 0.20, 0.08, 0.07, 0.05]

def seed_big_dataset(num_records: int = 5000):
    print(f"📦 Generating & seeding {num_records} synthetic vehicle detections across 10 camera corridors...")
    init_db()
    db = SessionLocal()
    try:
        # Register 10 cameras
        for cam in SYNTHETIC_CAMERAS:
            existing = db.query(Camera).filter(Camera.camera_id == cam["camera_id"]).first()
            if not existing:
                db.add(Camera(**cam, is_active=True))
        db.commit()

        # Generate realistic multi-camera trips for 500 unique vehicles
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        events_to_insert = []
        
        plates = [f"KA01AB{i:04d}" for i in range(1, 501)]

        for _ in range(num_records):
            plate = random.choice(plates)
            cam = random.choice(SYNTHETIC_CAMERAS)
            # Spread across last 12 hours
            time_offset_sec = random.randint(0, 12 * 3600)
            event_time = now - timedelta(seconds=time_offset_sec)
            
            v_type = random.choices(VEHICLE_CLASSES, weights=VEHICLE_CLASS_WEIGHTS)[0]
            # Speed varies by corridor (rush hour congestion vs clear)
            base_speed = random.uniform(18.0, 75.0)
            if cam["camera_id"] in ["CAM_HUB_01", "CAM_HUB_02"]:  # Congested nodes
                base_speed = random.uniform(8.0, 22.0)

            ev = VehicleEvent(
                camera_id=cam["camera_id"],
                local_track_id=f"{cam['camera_id']}_T{random.randint(100, 999)}",
                timestamp=event_time,
                plate=plate,
                plate_confidence=round(random.uniform(0.85, 0.99), 2),
                latitude=cam["latitude"],
                longitude=cam["longitude"],
                direction=cam["direction"],
                vehicle_type=v_type,
                speed=round(base_speed, 1),
                global_vehicle_id=f"VEH_SYNTH_{plate[-4:]}",
            )
            events_to_insert.append(ev)

        # Bulk save
        db.bulk_save_objects(events_to_insert)
        db.commit()
        print(f"✅ Successfully seeded {num_records} records in PostgreSQL/SQLite database.")
    finally:
        db.close()


def benchmark_analytics():
    print("\n" + "=" * 75)
    print("📊 BENCHMARKING ALL 8 ANALYTICS ENDPOINTS ON 5,000+ RECORDS")
    print("=" * 75)

    endpoints = [
        ("Summary Dashboard KPIs", "/analytics/summary"),
        ("Camera Traffic Density (24h)", "/analytics/density?hours=24"),
        ("GIS Heatmap Points (24h)", "/analytics/heatmap?hours=24"),
        ("Average Speed per Corridor", "/analytics/speed?hours=24"),
        ("Live Congestion Reports", "/analytics/congestion?minutes=60"),
        ("Origin-Destination (OD) Matrix", "/analytics/od-matrix?hours=24"),
        ("Time-Series Traffic Flow (15m)", "/analytics/flow?hours=12&bucket_minutes=15"),
        ("Snapshot Feed", "/analytics/snapshots?limit=50"),
    ]

    for label, url in endpoints:
        start = time.perf_counter()
        resp = client.get(url)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert resp.status_code == 200, f"Endpoint {url} failed with {resp.status_code}"
        data = resp.json()
        count = len(data) if isinstance(data, list) else len(data.keys())
        
        print(f"  ✓ {label:<35} -> Latency: {elapsed_ms:6.2f} ms | Output: {count} items")

    print("\n" + "=" * 75)
    print("📈 SAMPLE REAL-TIME ANALYTICS OUTPUT PREVIEWS:")
    print("=" * 75)

    # 1. Summary
    summary = client.get("/analytics/summary").json()
    print(f"\n1️⃣  Summary KPIs:\n{json.dumps(summary, indent=4)}")

    # 2. Congestion
    congestion = client.get("/analytics/congestion?minutes=60").json()
    print(f"\n2️⃣  Corridor Congestion Sample (Top 3 Nodes):")
    for c in congestion[:3]:
        print(f"    -> {c['camera_id']} ({c.get('road')}): Volume: {c['vehicle_count']} vehicles | Avg Speed: {c['avg_speed_kmh']} km/h | Status: 🚦 {c['congestion_level']}")

    # 3. OD Matrix
    od = client.get("/analytics/od-matrix?hours=24").json()
    print(f"\n3️⃣  Origin-Destination Route Matrix Sample (Top 3 Corridors):")
    for row in od[:3]:
        print(f"    -> Route {row['origin_camera_id']} ➔ {row['destination_camera_id']}: {row['trip_count']} trips | Avg Travel Time: {row['avg_duration_minutes']} mins")

    # 4. GIS Heatmap
    heatmap = client.get("/analytics/heatmap?hours=24").json()
    print(f"\n4️⃣  GIS Heatmap Spatial Density Points (First 3):")
    for pt in heatmap[:3]:
        print(f"    -> Node {pt['camera_id']}: GPS ({pt['latitude']}, {pt['longitude']}) with {pt['weight']} hits")

    print("\n" + "=" * 75)
    print("🎉 BIG DATA ANALYTICS VERIFICATION COMPLETED WITH 100% SUCCESS!")
    print("=" * 75)


if __name__ == "__main__":
    seed_big_dataset(5000)
    benchmark_analytics()

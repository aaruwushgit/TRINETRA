"""
Comprehensive automated tests for Multi-Camera Ingestion, Heatmap, Speed, Congestion, and OD Analytics.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.main import app
from backend.models.alert import Blacklist
from backend.models.camera import Camera
from backend.models.vehicle_event import VehicleEvent

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()
    db = SessionLocal()
    try:
        # Ensure test cameras exist
        cams = [
            Camera(
                camera_id="CAM_TEST_1",
                name="Test Cam 1",
                location="Junction A",
                latitude=13.0827,
                longitude=80.2099,
                road="Route 1",
                direction="NORTH",
            ),
            Camera(
                camera_id="CAM_TEST_2",
                name="Test Cam 2",
                location="Junction B",
                latitude=13.0569,
                longitude=80.2425,
                road="Route 2",
                direction="EAST",
            ),
        ]
        for c in cams:
            existing = db.query(Camera).filter(Camera.camera_id == c.camera_id).first()
            if not existing:
                db.add(c)
        # Clean all events so test suite is 100% idempotent
        db.query(VehicleEvent).delete(synchronize_session=False)
        # Add a blacklisted plate for alert testing
        db.merge(Blacklist(plate="TN09AL1234", reason="Stolen vehicle test"))
        db.commit()
    finally:
        db.close()


def test_health():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_event_ingest_and_trajectory():
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(minutes=15)).isoformat()
    t2 = now.isoformat()

    # Camera 1 sighting
    r1 = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_1",
            "timestamp": t1,
            "local_track_id": "CAM1_T10",
            "plate": "TN09AB9999",
            "plate_confidence": 0.95,
            "latitude": 13.0827,
            "longitude": 80.2099,
            "vehicle_type": "car",
            "speed": 45.5,
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["event_id"] is not None
    assert d1["global_vehicle_id"] is not None
    assert d1["alert_fired"] is False

    # Camera 2 sighting (same car moving along route)
    r2 = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_2",
            "timestamp": t2,
            "local_track_id": "CAM2_T44",
            "plate": "TN09AB9999",
            "plate_confidence": 0.92,
            "latitude": 13.0569,
            "longitude": 80.2425,
            "vehicle_type": "car",
            "speed": 38.0,
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    # Same global_vehicle_id assigned across different cameras
    assert d2["global_vehicle_id"] == d1["global_vehicle_id"]

    # Trajectory query
    traj_res = client.get("/vehicles/TN09AB9999/trajectory")
    assert traj_res.status_code == 200
    traj = traj_res.json()
    assert len(traj["points"]) == 2
    assert traj["points"][0]["camera_id"] == "CAM_TEST_1"
    assert traj["points"][1]["camera_id"] == "CAM_TEST_2"


def test_blacklist_alert():
    # Ingest blacklisted car
    res = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plate": "TN09AL1234",
            "plate_confidence": 0.98,
            "vehicle_type": "car",
            "speed": 60.0,
        },
    )
    assert res.status_code == 200
    assert res.json()["alert_fired"] is True

    # Check alert list
    alerts_res = client.get("/alerts")
    assert alerts_res.status_code == 200
    alerts = alerts_res.json()
    assert any(a["vehicle_id"] == "TN09AL1234" for a in alerts)


def test_analytics_endpoints():
    # 1. Summary
    res = client.get("/analytics/summary")
    assert res.status_code == 200
    assert "total_detections" in res.json()

    # 2. Heatmap
    res = client.get("/analytics/heatmap")
    assert res.status_code == 200
    points = res.json()
    assert isinstance(points, list)
    if points:
        assert "latitude" in points[0]
        assert "longitude" in points[0]
        assert "weight" in points[0]

    # 3. Speed
    res = client.get("/analytics/speed")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # 4. Congestion
    res = client.get("/analytics/congestion")
    assert res.status_code == 200
    cong = res.json()
    assert isinstance(cong, list)
    assert any("congestion_level" in item for item in cong)

    # 5. OD Matrix
    res = client.get("/analytics/od-matrix")
    assert res.status_code == 200
    od = res.json()
    assert isinstance(od, list)

    # 6. Flow
    res = client.get("/analytics/flow")
    assert res.status_code == 200

    # 7. Traffic Snapshot ingestion
    now = datetime.now(timezone.utc)
    snap_res = client.post(
        "/analytics/traffic-snapshot",
        json={
            "camera_id": "CAM_TEST_1",
            "window_start": (now - timedelta(seconds=60)).isoformat(),
            "window_end": now.isoformat(),
            "vehicle_count": 25,
            "avg_speed": 34.2,
            "peak_density": 8,
            "class_counts": {"car": 18, "motorcycle": 5, "bus": 2},
            "congestion_level": "MEDIUM",
        },
    )
    assert snap_res.status_code == 201
    assert snap_res.json()["snapshot_id"] is not None

    # Retrieve snapshots
    list_snap = client.get("/analytics/snapshots?camera_id=CAM_TEST_1")
    assert list_snap.status_code == 200
    assert len(list_snap.json()) >= 1


def test_route_anomaly_alert():
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(minutes=2)).isoformat()
    t2 = now.isoformat()

    # Sighting 1: Cam 1
    client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_1",
            "timestamp": t1,
            "plate": "TN09ANOMALY",
            "latitude": 13.0827,
            "longitude": 80.2099,
        },
    )

    # Sighting 2: Cam 2 (20 km away in 2 minutes = 600 km/h anomaly)
    r2 = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_2",
            "timestamp": t2,
            "plate": "TN09ANOMALY",
            "latitude": 13.2500,  # far away
            "longitude": 80.4000,
        },
    )
    assert r2.status_code == 200
    assert r2.json()["alert_fired"] is True


def test_websocket_endpoints():
    with client.websocket_connect("/ws/alerts") as ws:
        data = ws.receive_json()
        assert "message" in data or "data" in data

    with client.websocket_connect("/ws/stats") as ws:
        data = ws.receive_json()
        assert "message" in data or "data" in data


def test_auth_flow():
    # Login
    res = client.post("/auth/login", json={"username": "admin@traffic.gov.in", "password": "admin123"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    token = data["access_token"]

    # Profile with Bearer token
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "admin@traffic.gov.in"
    assert me_res.json()["role"] == "admin"


def test_clahe_preprocessing():
    import numpy as np
    from backend.services.anpr_service import anpr_service

    # Create dummy dark frame
    fake_crop = np.zeros((50, 150, 3), dtype=np.uint8)
    enhanced = anpr_service.preprocess_plate_crop(fake_crop)
    assert enhanced is not None
    assert enhanced.shape[0] >= 40
    assert enhanced.shape[1] >= 100


def test_consensus_ocr_voting():
    from scripts.camera_worker import CameraWorker

    worker = CameraWorker(camera_id="CAM_TEST_1", source="0")
    # Feed 4 readings of TN09AB1111 and 1 noisy reading TN09AB1118
    worker.get_consensus_plate(101, "TN09AB1111", 0.90)
    worker.get_consensus_plate(101, "TN09AB1111", 0.92)
    worker.get_consensus_plate(101, "TN09AB1118", 0.70)
    best_plate, conf = worker.get_consensus_plate(101, "TN09AB1111", 0.95)

    # Majority consensus should win
    assert best_plate == "TN09AB1111"
    assert conf > 0.85


def test_spatio_temporal_reid():
    now = datetime.now(timezone.utc)
    t1 = (now - timedelta(minutes=10)).isoformat()
    t2 = now.isoformat()

    # Cam 1: Vehicle without readable plate (truck at Junction A)
    r1 = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_1",
            "timestamp": t1,
            "latitude": 13.0827,
            "longitude": 80.2099,
            "vehicle_type": "truck",
        },
    )
    assert r1.status_code == 200
    gid1 = r1.json()["global_vehicle_id"]
    assert gid1 is not None
    assert gid1.startswith("VEH_")

    # Cam 2: Same truck type ~4km away 10 mins later (feasible ~24 km/h)
    r2 = client.post(
        "/events/ingest",
        json={
            "camera_id": "CAM_TEST_2",
            "timestamp": t2,
            "latitude": 13.0569,
            "longitude": 80.2425,
            "vehicle_type": "truck",
        },
    )
    assert r2.status_code == 200
    gid2 = r2.json()["global_vehicle_id"]
    assert gid2 is not None
    assert gid2.startswith("VEH_")
    # Re-ID should match: same global vehicle identity assigned across cameras
    assert gid2 == gid1


def test_poi_prediction():
    """Predict next camera for a known vehicle with multi-camera sightings."""
    now = datetime.now(timezone.utc)

    # Seed two sightings for the same plate across two cameras
    client.post("/events/ingest", json={
        "camera_id": "CAM_TEST_1",
        "timestamp": (now - timedelta(minutes=15)).isoformat(),
        "plate": "TN09PREDICT",
        "plate_confidence": 0.95,
        "latitude": 13.0827,
        "longitude": 80.2099,
        "speed": 45.0,
        "vehicle_type": "car",
    })
    client.post("/events/ingest", json={
        "camera_id": "CAM_TEST_2",
        "timestamp": (now - timedelta(minutes=5)).isoformat(),
        "plate": "TN09PREDICT",
        "plate_confidence": 0.93,
        "latitude": 13.0569,
        "longitude": 80.2425,
        "speed": 42.0,
        "vehicle_type": "car",
    })

    r = client.get("/vehicles/TN09PREDICT/predict-next-location?top_n=2")
    assert r.status_code == 200
    data = r.json()

    assert data["plate"] == "TN09PREDICT"
    assert "last_sighting" in data
    assert data["last_sighting"]["camera_id"] == "CAM_TEST_2"
    assert isinstance(data["predicted_destinations"], list)
    assert len(data["predicted_destinations"]) >= 1

    # Each candidate must have required fields
    for dest in data["predicted_destinations"]:
        assert "camera_id" in dest
        assert "probability" in dest
        assert "eta_minutes" in dest
        assert "interception_priority" in dest
        assert dest["probability"] > 0

    # Probabilities must sum to ~1.0
    total_prob = sum(d["probability"] for d in data["predicted_destinations"])
    assert abs(total_prob - 1.0) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



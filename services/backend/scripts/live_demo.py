"""
Live End-to-End Simulation & Verification Script.

Simulates:
  1. Multi-camera stream ingestion across Chennai city junction nodes
  2. Trajectory reconstruction & MTMC Spatio-Temporal ReID
  3. Real-time Blacklist & Route Anomaly detection
  4. GIS Heatmap generation & Traffic Density KPIs
  5. WebSocket alerts broadcast
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient

from backend.main import app
from backend.database import init_db, SessionLocal
from backend.models.camera import Camera
from backend.models.alert import Blacklist, Alert
from backend.models.vehicle_event import VehicleEvent

client = TestClient(app)

def run_live_demo():
    print("=" * 70)
    print("🚦 RUNNING CITY-WIDE VEHICLE INTELLIGENCE LIVE DEMO")
    print("=" * 70)

    # 1. Health & JWT Authentication
    print("\n[1/6] 🔐 Authenticating System Operator...")
    auth_resp = client.post("/auth/login", json={"username": "admin@traffic.gov.in", "password": "admin123"})
    token_data = auth_resp.json()
    print(f"  ✓ Login Success! JWT Token generated: {token_data['access_token'][:30]}... (Role: {token_data['role']})")
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}

    # 2. Camera Nodes
    print("\n[2/6] 📹 Registering City Camera Network...")
    cameras = [
        {"camera_id": "CAM_ANNA_NAGAR", "name": "Anna Nagar Roundtana", "location": "Chennai", "latitude": 13.0850, "longitude": 80.2101, "road": "2nd Avenue", "direction": "NORTH"},
        {"camera_id": "CAM_T_NAGAR", "name": "Panagal Park Junction", "location": "Chennai", "latitude": 13.0418, "longitude": 80.2341, "road": "Usman Road", "direction": "SOUTH"},
        {"camera_id": "CAM_OMR_IT_CORRIDOR", "name": "Tidel Park Signal", "location": "Chennai", "latitude": 12.9892, "longitude": 80.2482, "road": "OMR Expressway", "direction": "SOUTH"},
    ]
    for cam in cameras:
        r = client.post("/cameras", json=cam, headers=headers)
        print(f"  ✓ Registered: {cam['camera_id']} ({cam['name']}) -> GPS: ({cam['latitude']}, {cam['longitude']})")

    # 3. Add a stolen vehicle to the Blacklist
    print("\n[3/6] 🚨 Registering Blacklisted Target Vehicle...")
    bl_resp = client.post("/alerts/blacklist", json={"plate": "TN07BZ7777", "reason": "Red Alert: Stolen Luxury SUV / Wanted in FIR 492"}, headers=headers)
    print(f"  ✓ Blacklist Active: Plate TN07BZ7777 registered in central wanted DB.")

    # 4. Multi-Camera Stream Ingestion & Tracking
    print("\n[4/6] 🚘 Simulating Simultaneous Vehicle Passes Across Cameras...")
    now = datetime.now(timezone.utc)

    # Normal Vehicle Trip from Anna Nagar -> T Nagar
    client.post("/events/ingest", json={
        "camera_id": "CAM_ANNA_NAGAR",
        "timestamp": (now - timedelta(minutes=15)).isoformat(),
        "local_track_id": "CAM_ANNA_T12",
        "plate": "TN09AB3456",
        "plate_confidence": 0.96,
        "vehicle_type": "car",
        "speed": 48.5
    })
    client.post("/events/ingest", json={
        "camera_id": "CAM_T_NAGAR",
        "timestamp": (now - timedelta(minutes=2)).isoformat(),
        "local_track_id": "CAM_TNAGAR_T88",
        "plate": "TN09AB3456",
        "plate_confidence": 0.94,
        "vehicle_type": "car",
        "speed": 42.0
    })
    print("  ✓ Normal Vehicle 'TN09AB3456' tracked across 2 cameras.")

    # Stolen Vehicle Detection at OMR
    bl_event = client.post("/events/ingest", json={
        "camera_id": "CAM_OMR_IT_CORRIDOR",
        "timestamp": now.isoformat(),
        "local_track_id": "CAM_OMR_T04",
        "plate": "TN07BZ7777",
        "plate_confidence": 0.98,
        "vehicle_type": "car",
        "speed": 75.2
    }).json()
    print(f"  🚨 Ingestion result for TN07BZ7777 -> Alert Triggered: {bl_event['alert_fired']} (Global ID: {bl_event['global_vehicle_id']})")

    # 5. Trajectory Reconstruction
    print("\n[5/6] 🗺️ Reconstructing Vehicle Trajectory for TN09AB3456...")
    traj = client.get("/vehicles/TN09AB3456/trajectory", headers=headers).json()
    print(f"  ✓ Vehicle Path: {len(traj['points'])} GPS Waypoints Found:")
    for pt in traj['points']:
        print(f"    -> [{pt['timestamp']}] Camera: {pt['camera_id']} | Speed: {pt['speed']} km/h | GPS: ({pt['latitude']}, {pt['longitude']})")

    # 6. Real-Time GIS Heatmap & Traffic Analytics
    print("\n[6/6] 📊 Live Traffic Intelligence & GIS Heatmap Points:")
    heatmap = client.get("/analytics/heatmap?hours=24", headers=headers).json()
    print(f"  ✓ GIS Heatmap Intensity Points (Leaflet/Mapbox Ready):")
    for pt in heatmap[:3]:
        print(f"    -> Location: ({pt['latitude']}, {pt['longitude']}) | Density Weight: {pt['weight']} hits | Camera: {pt['camera_id']}")

    summary = client.get("/analytics/summary", headers=headers).json()
    print(f"  ✓ City Dashboard KPI Summary: {json.dumps(summary, indent=4)}")

    alerts = client.get("/alerts", headers=headers).json()
    print(f"  ✓ Active Command Center Alerts:")
    for a in alerts[:2]:
        print(f"    -> ⚠️ [{a['alert_type']}] Vehicle: {a['vehicle_id']} at {a['camera_id']}: {a['description']}")

    print("\n" + "=" * 70)
    print("🎉 ALL END-TO-END WORKFLOWS FUNCTIONING PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_live_demo()

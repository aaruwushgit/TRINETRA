"""
Seed script: adds sample cameras to the database.
Run once after first startup:
  python scripts/seed_cameras.py
"""
import sys
from pathlib import Path

# Make sure the backend package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import SessionLocal, init_db
from backend.models.camera import Camera

SAMPLE_CAMERAS = [
    {
        "camera_id": "CAM_001",
        "name": "Anna Nagar Junction Cam",
        "location": "Anna Nagar, Chennai",
        "latitude": 13.0827,
        "longitude": 80.2099,
        "road": "Anna Nagar Main Road",
        "direction": "NORTH",
        "camera_type": "ANPR",
    },
    {
        "camera_id": "CAM_002",
        "name": "Nungambakkam Cam",
        "location": "Nungambakkam, Chennai",
        "latitude": 13.0569,
        "longitude": 80.2425,
        "road": "Nungambakkam High Road",
        "direction": "EAST",
        "camera_type": "ANPR",
    },
    {
        "camera_id": "CAM_003",
        "name": "T Nagar Cam",
        "location": "T Nagar, Chennai",
        "latitude": 13.0418,
        "longitude": 80.2341,
        "road": "Usman Road",
        "direction": "SOUTH",
        "camera_type": "ANPR",
    },
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        for data in SAMPLE_CAMERAS:
            existing = db.query(Camera).filter(Camera.camera_id == data["camera_id"]).first()
            if not existing:
                db.add(Camera(**data))
                print(f"  ✅ Added camera: {data['camera_id']} — {data['name']}")
            else:
                print(f"  ⚠️  Already exists: {data['camera_id']}, skipping.")
        db.commit()
        print("\nDone. Camera seeding complete.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()

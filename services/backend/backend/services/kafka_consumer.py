"""
Kafka Consumer Service — Scalable worker that consumes events from Kafka topics
and ingests them directly into PostgreSQL/SQLite and Redis.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from backend.config import get_settings
from backend.database import SessionLocal, init_db
from backend.models.camera import Camera
from backend.models.traffic_snapshot import TrafficSnapshot
from backend.models.vehicle_event import VehicleEvent
from backend.services.alert_service import alert_service
from backend.services.redis_service import redis_service
from backend.services.tracking_service import tracking_service

settings = get_settings()


def process_event_payload(payload: dict, db) -> dict:
    """Internal ingestion logic used by Kafka consumer and REST endpoint."""
    camera = db.query(Camera).filter(Camera.camera_id == payload["camera_id"]).first()
    if not camera:
        return {"error": f"Camera {payload['camera_id']} not found"}

    ts = payload.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)

    # Normalize at the point of entry, same as the /events/ingest schema validator,
    # so plates from Kafka workers match exactly against tracking/alert lookups.
    plate = payload.get("plate")
    if plate:
        plate = plate.upper().replace(" ", "").replace("-", "").strip() or None

    event = VehicleEvent(
        camera_id=payload["camera_id"],
        local_track_id=payload.get("local_track_id"),
        timestamp=ts or datetime.utcnow(),
        plate=plate,
        plate_confidence=payload.get("plate_confidence"),
        latitude=payload.get("latitude") or camera.latitude,
        longitude=payload.get("longitude") or camera.longitude,
        direction=payload.get("direction") or camera.direction,
        vehicle_type=payload.get("vehicle_type"),
        vehicle_color=payload.get("vehicle_color"),
        speed=payload.get("speed"),
    )
    db.add(event)
    db.flush()

    global_id = tracking_service.associate_event(event, db)
    event.global_vehicle_id = global_id
    db.commit()
    db.refresh(event)

    # Check blacklist & route anomaly alerts
    alert = alert_service.check_and_fire(event, db)
    if alert:
        redis_service.publish_alert({
            "alert_id": alert.alert_id,
            "vehicle_id": alert.vehicle_id,
            "camera_id": alert.camera_id,
            "alert_type": alert.alert_type,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat(),
        })

    return {"event_id": event.event_id, "global_vehicle_id": global_id}


def process_snapshot_payload(payload: dict, db):
    w_start = payload.get("window_start")
    w_end = payload.get("window_end")
    if isinstance(w_start, str):
        w_start = datetime.fromisoformat(w_start.replace("Z", "+00:00")).replace(tzinfo=None)
    if isinstance(w_end, str):
        w_end = datetime.fromisoformat(w_end.replace("Z", "+00:00")).replace(tzinfo=None)

    snapshot = TrafficSnapshot(
        camera_id=payload["camera_id"],
        window_start=w_start or datetime.utcnow(),
        window_end=w_end or datetime.utcnow(),
        vehicle_count=payload.get("vehicle_count", 0),
        avg_speed=payload.get("avg_speed"),
        peak_density=payload.get("peak_density", 0),
        class_counts=payload.get("class_counts", {}),
        congestion_level=payload.get("congestion_level", "LOW"),
    )
    db.add(snapshot)
    db.commit()

    redis_service.publish_stats({
        "camera_id": payload["camera_id"],
        "vehicle_count": payload.get("vehicle_count"),
        "congestion_level": payload.get("congestion_level"),
        "avg_speed": payload.get("avg_speed"),
    })


def run_kafka_consumer_loop():
    """Main Kafka consumer loop."""
    init_db()
    from kafka import KafkaConsumer

    print(f"📡 Connecting to Kafka broker: {settings.KAFKA_BROKER}...")
    consumer = KafkaConsumer(
        settings.KAFKA_TOPIC_EVENTS,
        settings.KAFKA_TOPIC_SNAPSHOTS,
        bootstrap_servers=settings.KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="vehicle-intelligence-consumers",
        auto_offset_reset="latest",
    )
    print("✅ Kafka Consumer ready. Listening for stream messages...")

    for msg in consumer:
        db = SessionLocal()
        try:
            if msg.topic == settings.KAFKA_TOPIC_EVENTS:
                process_event_payload(msg.value, db)
            elif msg.topic == settings.KAFKA_TOPIC_SNAPSHOTS:
                process_snapshot_payload(msg.value, db)
        except Exception as err:
            print(f"Error processing Kafka message: {err}")
        finally:
            db.close()

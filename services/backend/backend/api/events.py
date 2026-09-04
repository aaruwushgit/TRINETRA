"""
Events router — the main ingestion pipeline.

Two endpoints:
  POST /events/ingest       — AI workers push structured VehicleEvent JSON
  POST /events/analyze-frame — Upload an image, get back plate text (tests ANPR)
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.vehicle_event import VehicleEvent
from backend.models.camera import Camera
from backend.schemas.vehicle import (
    FrameAnalysisResponse,
    IngestEventRequest,
    IngestEventResponse,
    VehicleEventOut,
)
from backend.services.alert_service import alert_service
from backend.services.anpr_service import anpr_service
from backend.services.tracking_service import tracking_service

router = APIRouter(prefix="/events", tags=["Events"])


@router.get("/recent", response_model=list[VehicleEventOut])
def list_recent_events(
    camera_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Most recent detection events across the network — backs the dashboard live feed."""
    q = db.query(VehicleEvent).order_by(VehicleEvent.timestamp.desc())
    if camera_id:
        q = q.filter(VehicleEvent.camera_id == camera_id)
    return q.limit(limit).all()


@router.post("/ingest", response_model=IngestEventResponse)
def ingest_event(payload: IngestEventRequest, db: Session = Depends(get_db)):
    """
    Primary ingestion endpoint. AI workers call this after processing a frame.

    Flow:
      1. Validate camera exists
      2. Persist the VehicleEvent
      3. Call tracking service → get/create global_vehicle_id
      4. Check blacklist → fire alert if needed
      5. Return event_id + global_vehicle_id
    """
    # 1. Validate camera
    camera = db.query(Camera).filter(Camera.camera_id == payload.camera_id).first()
    if not camera:
        raise HTTPException(
            status_code=404,
            detail=f"Camera '{payload.camera_id}' not registered. Add it via POST /cameras first.",
        )

    # 2. Persist event
    event = VehicleEvent(
        camera_id=payload.camera_id,
        local_track_id=payload.local_track_id,
        timestamp=payload.timestamp.replace(tzinfo=None) if payload.timestamp.tzinfo else payload.timestamp,
        plate=payload.plate,
        plate_confidence=payload.plate_confidence,
        latitude=payload.latitude or camera.latitude,
        longitude=payload.longitude or camera.longitude,
        direction=payload.direction or camera.direction,
        vehicle_type=payload.vehicle_type,
        vehicle_color=payload.vehicle_color,
        speed=payload.speed,
    )
    db.add(event)
    db.flush()  # get event_id without committing yet

    # 3. Associate → global vehicle ID
    global_id = tracking_service.associate_event(event, db)
    event.global_vehicle_id = global_id
    db.commit()
    db.refresh(event)

    # 4. Check blacklist
    alert = alert_service.check_and_fire(event, db)

    return IngestEventResponse(
        event_id=event.event_id,
        global_vehicle_id=global_id,
        alert_fired=alert is not None,
    )


@router.post("/bulk-ingest", response_model=list[IngestEventResponse])
def bulk_ingest_events(payloads: list[IngestEventRequest], db: Session = Depends(get_db)):
    """
    High-throughput bulk ingestion for edge workers.
    Reduces database transactions by processing events in batches.
    """
    if not payloads:
        return []

    # 1. Pre-fetch required cameras to minimize queries
    camera_ids = {p.camera_id for p in payloads}
    cameras = {c.camera_id: c for c in db.query(Camera).filter(Camera.camera_id.in_(camera_ids)).all()}

    responses = []
    events_to_add = []

    # 2. Build Event objects
    for p in payloads:
        cam = cameras.get(p.camera_id)
        if not cam:
            continue # Skip invalid cameras in bulk mode

        event = VehicleEvent(
            camera_id=p.camera_id,
            local_track_id=p.local_track_id,
            timestamp=p.timestamp.replace(tzinfo=None) if p.timestamp.tzinfo else p.timestamp,
            plate=p.plate,
            plate_confidence=p.plate_confidence,
            latitude=p.latitude or cam.latitude,
            longitude=p.longitude or cam.longitude,
            direction=p.direction or cam.direction,
            vehicle_type=p.vehicle_type,
            vehicle_color=p.vehicle_color,
            speed=p.speed,
        )
        events_to_add.append(event)

    if not events_to_add:
        return []

    # 3. Bulk insert to DB
    db.add_all(events_to_add)
    db.flush()

    # 4. Post-process (Tracking & Alerts)
    for event in events_to_add:
        global_id = tracking_service.associate_event(event, db)
        event.global_vehicle_id = global_id
        alert = alert_service.check_and_fire(event, db)
        
        responses.append(
            IngestEventResponse(
                event_id=event.event_id,
                global_vehicle_id=global_id,
                alert_fired=alert is not None,
            )
        )

    db.commit()
    return responses


@router.post("/analyze-frame", response_model=FrameAnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Upload an image and get ANPR result back. Useful for testing the ANPR service
    without having to set up a camera feed.

    Does NOT save anything to the database — call /ingest to persist.
    """
    import io

    import cv2
    import numpy as np

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = anpr_service.process_frame(frame)
    return FrameAnalysisResponse(plate=result.plate, confidence=result.confidence)

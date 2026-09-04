"""
Vehicles router — look up vehicles, get trajectories, predict next location.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database import get_db
from backend.models.camera import Camera
from backend.models.vehicle_event import VehicleEvent
from backend.schemas.vehicle import TrajectoryPoint, TrajectoryResponse, VehicleEventOut
from backend.services.prediction_service import prediction_service

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])

settings = get_settings()


@router.get("/analytics/speed-defaulters")
def get_speed_defaulters(
    speed_limit: float | None = Query(default=None, description="Overrides each road segment's own limit with one flat value"),
    hours: int = Query(default=24, ge=1, le=168, description="Look back this many hours (bounds the query for city-scale event volumes)"),
    db: Session = Depends(get_db),
):
    """
    Checkpoint-pair speed report: for every vehicle, compares consecutive
    sightings at different cameras and flags any hop whose implied speed
    exceeds the applicable limit — by default the stricter of the two
    cameras' own road-segment speed_limit_kmh (a ring road and a
    residential lane don't share one number); pass `speed_limit` to
    override with one flat value instead. Bounded to the last `hours`
    (default 24h) so this stays a single indexed range query instead of a
    full-table scan as the camera network grows — see /vehicles/analytics
    for the always-on, real-time version of this same check that fires
    SPEED_VIOLATION alerts at ingest time.

    Hops whose implied speed exceeds MAX_PLAUSIBLE_SPEED_KMH are excluded —
    those are tracking/timestamp errors (already surfaced as ROUTE_ANOMALY
    alerts), not real defaulters.
    """
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    camera_limits = {c.camera_id: c.speed_limit_kmh for c in db.query(Camera).all()}

    events = (
        db.query(VehicleEvent)
        .filter(
            VehicleEvent.latitude.isnot(None),
            VehicleEvent.longitude.isnot(None),
            VehicleEvent.timestamp >= since,
        )
        .order_by(VehicleEvent.plate, VehicleEvent.timestamp)
        .all()
    )

    from collections import defaultdict
    from backend.services.alert_service import haversine_distance_km

    by_plate = defaultdict(list)
    for e in events:
        p = e.plate or e.global_vehicle_id
        if p:
            by_plate[p].append(e)

    defaulters = []

    for vehicle_key, p_events in by_plate.items():
        for i in range(1, len(p_events)):
            prev, curr = p_events[i-1], p_events[i]
            if prev.camera_id == curr.camera_id:
                continue

            t_prev = prev.timestamp.replace(tzinfo=None) if prev.timestamp.tzinfo else prev.timestamp
            t_curr = curr.timestamp.replace(tzinfo=None) if curr.timestamp.tzinfo else curr.timestamp

            dt_hours = (t_curr - t_prev).total_seconds() / 3600.0
            if dt_hours <= 0:
                continue

            dist_km = haversine_distance_km(prev.latitude, prev.longitude, curr.latitude, curr.longitude)
            if dist_km <= 0.1:
                continue

            avg_speed_kmh = round(dist_km / dt_hours, 1)
            if avg_speed_kmh > settings.MAX_PLAUSIBLE_SPEED_KMH:
                # Physically implausible camera-to-camera hop (bad match /
                # clock skew) — not a real speeding vehicle, skip it here.
                continue

            radar_speed = curr.speed or 0.0
            max_effective_speed = max(avg_speed_kmh, radar_speed)

            if speed_limit is not None:
                limit = speed_limit
            else:
                pair_limits = [camera_limits.get(prev.camera_id), camera_limits.get(curr.camera_id)]
                pair_limits = [l for l in pair_limits if l]
                limit = min(pair_limits) if pair_limits else settings.DEFAULT_SPEED_LIMIT_KMH

            if max_effective_speed > limit:
                excess = round(max_effective_speed - limit, 1)
                fine_level = "CRITICAL (+30km/h)" if excess >= 30 else "HIGH (+15km/h)" if excess >= 15 else "MODERATE"
                defaulters.append({
                    "plate": curr.plate or "UNREADABLE",
                    "global_vehicle_id": curr.global_vehicle_id,
                    "from_camera": prev.camera_id,
                    "to_camera": curr.camera_id,
                    "distance_km": round(dist_km, 2),
                    "time_elapsed_mins": round(dt_hours * 60.0, 1),
                    "avg_speed_kmh": avg_speed_kmh,
                    "radar_speed_kmh": radar_speed,
                    "effective_speed_kmh": max_effective_speed,
                    "speed_limit_kmh": limit,
                    "excess_speed_kmh": excess,
                    "fine_category": fine_level,
                    "timestamp": curr.timestamp.isoformat(),
                    "vehicle_type": curr.vehicle_type or "car"
                })

    defaulters.sort(key=lambda x: x["excess_speed_kmh"], reverse=True)
    return {
        "total_defaulters": len(defaulters),
        "speed_limit_kmh": speed_limit if speed_limit is not None else "per-road-segment (see each row)",
        "hours": hours,
        "defaulters": defaulters,
    }


@router.get("/{plate}", response_model=list[VehicleEventOut])
def get_vehicle_history(plate: str, db: Session = Depends(get_db)):
    """Return all events for a given plate number."""
    clean = plate.upper().replace(" ", "")
    events = (
        db.query(VehicleEvent)
        .filter(VehicleEvent.plate == clean)
        .order_by(VehicleEvent.timestamp)
        .all()
    )
    if not events:
        raise HTTPException(status_code=404, detail=f"No events found for plate: {clean}")
    return events


@router.get("/{plate}/trajectory", response_model=TrajectoryResponse)
def get_trajectory(plate: str, db: Session = Depends(get_db)):
    """Reconstruct ordered camera sightings for a given plate."""
    clean = plate.upper().replace(" ", "")
    events = (
        db.query(VehicleEvent)
        .filter(VehicleEvent.plate == clean)
        .order_by(VehicleEvent.timestamp)
        .all()
    )
    if not events:
        raise HTTPException(status_code=404, detail=f"No trajectory found for plate: {clean}")

    points = [
        TrajectoryPoint(
            camera_id=e.camera_id,
            timestamp=e.timestamp,
            latitude=e.latitude,
            longitude=e.longitude,
            direction=e.direction,
            plate_confidence=e.plate_confidence,
            speed=e.speed,
        )
        for e in events
    ]
    return TrajectoryResponse(plate=clean, global_vehicle_id=events[-1].global_vehicle_id, points=points)


@router.get("/{plate}/predict-next-location")
def predict_next_location(plate: str, top_n: int = 3, db: Session = Depends(get_db)):
    """
    Predict the next camera(s) a tracked vehicle/POI will appear at.

    Uses a Markov transition graph built from city-wide historical camera
    flows, weighted by real-time traffic speed on each road segment.
    Returns ranked next-hop candidates with ETAs and interception priority.
    """
    result = prediction_service.predict(plate, db, top_n=top_n)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

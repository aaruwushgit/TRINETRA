"""
Analytics router — traffic density, speed, heatmaps, congestion, and OD patterns.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import case, desc, func
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.camera import Camera
from backend.models.traffic_snapshot import TrafficSnapshot
from backend.models.vehicle_event import VehicleEvent
from backend.schemas.analytics import (
    CongestionReport,
    HeatmapPoint,
    ODMatrixEntry,
    SpeedStat,
    TrafficFlowPoint,
    TrafficSnapshotCreate,
    TrafficSnapshotOut,
)
from backend.services.redis_service import redis_service

router = APIRouter(prefix="/analytics", tags=["Analytics"])


# ── 1. Summary KPIs ──────────────────────────────────────────────────────────

@router.get("/summary")
def analytics_summary(db: Session = Depends(get_db)):
    """High-level summary stats for the dashboard header."""
    cached = redis_service.cache_get("analytics:summary")
    if cached:
        return cached

    total_events = db.query(func.count(VehicleEvent.event_id)).scalar() or 0
    unique_plates = (
        db.query(func.count(func.distinct(VehicleEvent.plate)))
        .filter(VehicleEvent.plate.isnot(None))
        .scalar()
        or 0
    )
    last_hour = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
    last_hour_count = (
        db.query(func.count(VehicleEvent.event_id))
        .filter(VehicleEvent.timestamp >= last_hour)
        .scalar()
        or 0
    )
    avg_speed = (
        db.query(func.avg(VehicleEvent.speed))
        .filter(VehicleEvent.speed.isnot(None), VehicleEvent.timestamp >= last_hour)
        .scalar()
    )

    res = {
        "total_detections": total_events,
        "unique_vehicles": unique_plates,
        "detections_last_hour": last_hour_count,
        "avg_speed_kmh": round(avg_speed, 1) if avg_speed else None,
    }
    redis_service.cache_set("analytics:summary", res, 15)
    return res


# ── 2. Traffic Density ────────────────────────────────────────────────────────

@router.get("/density")
def traffic_density(
    camera_id: str | None = Query(default=None),
    hours: int = Query(default=1, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """Vehicle count per camera in the last N hours."""
    cache_key = f"analytics:density:{camera_id}:{hours}"
    cached = redis_service.cache_get(cache_key)
    if cached is not None:
        return cached

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    q = (
        db.query(
            VehicleEvent.camera_id,
            func.count(VehicleEvent.event_id).label("vehicle_count"),
        )
        .filter(VehicleEvent.timestamp >= since)
    )

    if camera_id:
        q = q.filter(VehicleEvent.camera_id == camera_id)

    results = q.group_by(VehicleEvent.camera_id).all()

    payload = [
        {"camera_id": row.camera_id, "vehicle_count": row.vehicle_count, "hours": hours}
        for row in results
    ]
    redis_service.cache_set(cache_key, payload, 30)
    return payload


# ── 3. Heatmap (GIS Coordinates + Intensity Weights) ──────────────────────────

@router.get("/heatmap", response_model=list[HeatmapPoint])
def traffic_heatmap(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated detection counts grouped by GPS location.
    Directly consumable by GIS mapping libraries (Leaflet Heatmap, Mapbox, GeoServer).
    """
    cache_key = f"analytics:heatmap:{hours}"
    cached = redis_service.cache_get(cache_key)
    if cached is not None:
        return cached

    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    results = (
        db.query(
            VehicleEvent.latitude,
            VehicleEvent.longitude,
            VehicleEvent.camera_id,
            func.count(VehicleEvent.event_id).label("weight"),
        )
        .filter(
            VehicleEvent.timestamp >= since,
            VehicleEvent.latitude.isnot(None),
            VehicleEvent.longitude.isnot(None),
        )
        .group_by(VehicleEvent.latitude, VehicleEvent.longitude, VehicleEvent.camera_id)
        .all()
    )

    payload = [
        HeatmapPoint(
            latitude=row.latitude,
            longitude=row.longitude,
            weight=row.weight,
            camera_id=row.camera_id,
        )
        for row in results
    ]
    # Cache the serialized form so a hit does not need to rebuild the models.
    redis_service.cache_set(cache_key, [p.model_dump() for p in payload], 60)
    return payload


# ── 4. Speed Analytics ────────────────────────────────────────────────────────

@router.get("/speed", response_model=list[SpeedStat])
def speed_analytics(
    camera_id: str | None = Query(default=None),
    hours: int = Query(default=1, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """Average vehicle speed (in km/h) per camera node over time window."""
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    q = (
        db.query(
            VehicleEvent.camera_id,
            func.avg(VehicleEvent.speed).label("avg_speed"),
            func.count(VehicleEvent.event_id).label("sample_count"),
        )
        .filter(VehicleEvent.timestamp >= since, VehicleEvent.speed.isnot(None))
    )

    if camera_id:
        q = q.filter(VehicleEvent.camera_id == camera_id)

    results = q.group_by(VehicleEvent.camera_id).all()

    return [
        SpeedStat(
            camera_id=row.camera_id,
            avg_speed_kmh=round(row.avg_speed, 1),
            sample_count=row.sample_count,
        )
        for row in results
    ]


# ── 5. Congestion Level per Camera ────────────────────────────────────────────

@router.get("/congestion", response_model=list[CongestionReport])
def traffic_congestion(
    minutes: int = Query(default=30, ge=5, le=1440),
    db: Session = Depends(get_db),
):
    """
    Computes real-time congestion per active camera.
    Classifies into LOW, MEDIUM, or HIGH based on vehicle volume and average speed.
    """
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=minutes)

    # Fetch active cameras
    cameras = db.query(Camera).filter(Camera.is_active == True).all()
    camera_map = {c.camera_id: c for c in cameras}

    # Fetch volume and speed stats per camera
    stats = (
        db.query(
            VehicleEvent.camera_id,
            func.count(VehicleEvent.event_id).label("vehicle_count"),
            func.avg(VehicleEvent.speed).label("avg_speed"),
        )
        .filter(VehicleEvent.timestamp >= since)
        .group_by(VehicleEvent.camera_id)
        .all()
    )

    reports = []
    stat_cam_ids = set()

    for s in stats:
        stat_cam_ids.add(s.camera_id)
        cam = camera_map.get(s.camera_id)
        avg_spd = round(s.avg_speed, 1) if s.avg_speed is not None else None

        # Congestion heuristic: high count or low average speed (< 20 km/h)
        if s.vehicle_count > 50 or (avg_spd is not None and avg_spd < 15.0):
            level = "HIGH"
        elif s.vehicle_count > 20 or (avg_spd is not None and avg_spd < 30.0):
            level = "MEDIUM"
        else:
            level = "LOW"

        reports.append(
            CongestionReport(
                camera_id=s.camera_id,
                congestion_level=level,
                vehicle_count=s.vehicle_count,
                avg_speed_kmh=avg_spd,
                road=cam.road if cam else None,
            )
        )

    # Fill in zero-activity cameras
    for cam_id, cam in camera_map.items():
        if cam_id not in stat_cam_ids:
            reports.append(
                CongestionReport(
                    camera_id=cam_id,
                    congestion_level="LOW",
                    vehicle_count=0,
                    avg_speed_kmh=None,
                    road=cam.road,
                )
            )

    return reports


# ── 6. Origin-Destination (OD) Matrix ─────────────────────────────────────────

@router.get("/od-matrix", response_model=list[ODMatrixEntry])
def origin_destination_matrix(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """
    Computes Origin-Destination (OD) pairs from sequential camera sightings
    of identifiable vehicles across the city network.
    """
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    # Get events ordered by vehicle and timestamp
    events = (
        db.query(
            VehicleEvent.global_vehicle_id,
            VehicleEvent.plate,
            VehicleEvent.camera_id,
            VehicleEvent.timestamp,
        )
        .filter(
            VehicleEvent.timestamp >= since,
            (VehicleEvent.global_vehicle_id.isnot(None)) | (VehicleEvent.plate.isnot(None)),
        )
        .order_by(
            func.coalesce(VehicleEvent.global_vehicle_id, VehicleEvent.plate),
            VehicleEvent.timestamp.asc(),
        )
        .all()
    )

    pair_counts: dict[tuple[str, str], list[float]] = {}
    last_key: str | None = None
    last_cam: str | None = None
    last_time: datetime | None = None

    for ev in events:
        vkey = ev.global_vehicle_id or ev.plate
        if vkey == last_key and last_cam and last_time and ev.camera_id != last_cam:
            pair = (last_cam, ev.camera_id)
            duration_min = (ev.timestamp - last_time).total_seconds() / 60.0
            if duration_min > 0 and duration_min < 180:  # within 3 hours
                pair_counts.setdefault(pair, []).append(duration_min)

        last_key = vkey
        last_cam = ev.camera_id
        last_time = ev.timestamp

    return [
        ODMatrixEntry(
            origin_camera_id=origin,
            destination_camera_id=destination,
            trip_count=len(durations),
            avg_duration_minutes=round(sum(durations) / len(durations), 1) if durations else None,
        )
        for (origin, destination), durations in pair_counts.items()
    ]


# ── 7. Temporal Traffic Flow (for Chart.js) ───────────────────────────────────

@router.get("/flow")
def temporal_traffic_flow(
    camera_id: str | None = Query(default=None),
    hours: int = Query(default=6, ge=1, le=168),
    bucket_minutes: int = Query(default=15, ge=5, le=60),
    db: Session = Depends(get_db),
):
    """
    Returns time-series bucketed vehicle counts per camera for Chart.js rendering.
    """
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    q = db.query(
        VehicleEvent.camera_id,
        VehicleEvent.timestamp,
    ).filter(VehicleEvent.timestamp >= since)

    if camera_id:
        q = q.filter(VehicleEvent.camera_id == camera_id)

    events = q.order_by(VehicleEvent.timestamp.asc()).all()

    # Bucket events by time window
    bucket_counts: dict[tuple[str, str], int] = {}
    for ev in events:
        # Align timestamp to bucket interval
        minute_bucket = (ev.timestamp.minute // bucket_minutes) * bucket_minutes
        bucket_time = ev.timestamp.replace(minute=minute_bucket, second=0, microsecond=0)
        bucket_str = bucket_time.strftime("%Y-%m-%d %H:%M")
        key = (bucket_str, ev.camera_id)
        bucket_counts[key] = bucket_counts.get(key, 0) + 1

    return [
        {"timestamp": b_time, "camera_id": cam, "vehicle_count": cnt}
        for (b_time, cam), cnt in sorted(bucket_counts.items())
    ]


# ── 8. Traffic Snapshot Ingestion ─────────────────────────────────────────────

@router.get("/road-usage")
def road_usage(
    limit: int = Query(default=150, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Busiest road corridors: directed camera-to-camera segments ranked by trip
    count, returned with real road geometry where the routing cache has it.

    This is what backs the "most used roads" heatmap. It reads the precomputed
    `road_usage` aggregate table when available, because deriving segments by
    scanning consecutive sightings across a city-scale event table (10M+ rows)
    on every dashboard refresh is far too slow. If that table has not been
    built yet it falls back to computing the top segments directly from recent
    events, so the endpoint still answers on a fresh database.
    """
    cameras = {c.camera_id: c for c in db.query(Camera).all()}

    # Optional deps: the aggregate table and the routing cache are built by
    # separate pipeline stages, so treat both as "may not exist yet".
    RoadUsage = None
    try:
        from backend.models.analytics_agg import RoadUsage as _RoadUsage
        RoadUsage = _RoadUsage
    except Exception:  # noqa: BLE001 - aggregate stage not run yet
        RoadUsage = None

    route_geometry: dict[tuple[str, str], list] = {}
    try:
        from backend.models.route_segment import RouteSegment

        for seg in db.query(RouteSegment).all():
            geom = seg.geometry
            if isinstance(geom, list) and len(geom) >= 2:
                route_geometry[(seg.from_camera_id, seg.to_camera_id)] = geom
    except Exception:  # noqa: BLE001 - routing cache not built yet
        route_geometry = {}

    rows: list[dict] = []

    if RoadUsage is not None and db.query(RoadUsage).first() is not None:
        agg = (
            db.query(RoadUsage)
            .order_by(desc(RoadUsage.trip_count))
            .limit(limit)
            .all()
        )
        for r in agg:
            rows.append({
                "from_camera_id": r.from_camera_id,
                "to_camera_id": r.to_camera_id,
                "trip_count": r.trip_count,
                "avg_speed_kmh": round(r.avg_speed_kmh, 1) if r.avg_speed_kmh else None,
                "avg_duration_minutes": (
                    round(r.avg_travel_minutes, 1) if getattr(r, "avg_travel_minutes", None) else None
                ),
                "road": getattr(r, "road", None),
            })
    else:
        # Fallback: derive from consecutive sightings in a bounded window.
        since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
        events = (
            db.query(
                VehicleEvent.plate,
                VehicleEvent.global_vehicle_id,
                VehicleEvent.camera_id,
                VehicleEvent.timestamp,
                VehicleEvent.speed,
            )
            .filter(
                VehicleEvent.timestamp >= since,
                (VehicleEvent.global_vehicle_id.isnot(None)) | (VehicleEvent.plate.isnot(None)),
            )
            .order_by(
                func.coalesce(VehicleEvent.global_vehicle_id, VehicleEvent.plate),
                VehicleEvent.timestamp.asc(),
            )
            .limit(400_000)  # hard bound so a huge table cannot stall the dashboard
            .all()
        )

        pair_trips: dict[tuple[str, str], int] = {}
        pair_speeds: dict[tuple[str, str], list[float]] = {}
        last_key = last_cam = None
        for ev in events:
            vkey = ev.global_vehicle_id or ev.plate
            if vkey == last_key and last_cam and ev.camera_id != last_cam:
                pair = (last_cam, ev.camera_id)
                pair_trips[pair] = pair_trips.get(pair, 0) + 1
                if ev.speed:
                    pair_speeds.setdefault(pair, []).append(ev.speed)
            last_key, last_cam = vkey, ev.camera_id

        top = sorted(pair_trips.items(), key=lambda kv: -kv[1])[:limit]
        for (origin, dest), trips in top:
            speeds = pair_speeds.get((origin, dest), [])
            rows.append({
                "from_camera_id": origin,
                "to_camera_id": dest,
                "trip_count": trips,
                "avg_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
                "avg_duration_minutes": None,
                "road": None,
            })

    # Attach coordinates + real road geometry for map rendering
    for row in rows:
        a = cameras.get(row["from_camera_id"])
        b = cameras.get(row["to_camera_id"])
        row["from_latitude"] = a.latitude if a else None
        row["from_longitude"] = a.longitude if a else None
        row["to_latitude"] = b.latitude if b else None
        row["to_longitude"] = b.longitude if b else None
        if not row.get("road"):
            row["road"] = (a.road if a else None) or (b.road if b else None)

        geom = route_geometry.get((row["from_camera_id"], row["to_camera_id"]))
        if geom is None:
            # Road geometry is undirected in practice; reuse the reverse leg.
            reverse = route_geometry.get((row["to_camera_id"], row["from_camera_id"]))
            geom = list(reversed(reverse)) if reverse else None
        row["geometry"] = geom
        row["geometry_source"] = "road" if geom else "straight_line"

    return rows


@router.post("/traffic-snapshot", response_model=TrafficSnapshotOut, status_code=status.HTTP_201_CREATED)
def record_traffic_snapshot(payload: TrafficSnapshotCreate, db: Session = Depends(get_db)):
    """
    AI Workers push aggregated 60-second window snapshots here.
    """
    snapshot = TrafficSnapshot(
        camera_id=payload.camera_id,
        window_start=payload.window_start,
        window_end=payload.window_end,
        vehicle_count=payload.vehicle_count,
        avg_speed=payload.avg_speed,
        peak_density=payload.peak_density,
        class_counts=payload.class_counts,
        congestion_level=payload.congestion_level,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


@router.get("/snapshots", response_model=list[TrafficSnapshotOut])
def list_traffic_snapshots(
    camera_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Retrieve recent traffic intelligence snapshots."""
    q = db.query(TrafficSnapshot).order_by(desc(TrafficSnapshot.window_start))
    if camera_id:
        q = q.filter(TrafficSnapshot.camera_id == camera_id)
    return q.limit(limit).all()

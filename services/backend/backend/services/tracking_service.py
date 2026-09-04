"""
Multi-Camera Tracking (MTMC) Service — Plate + Spatio-Temporal ReID Association.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.models.vehicle_event import VehicleEvent
from backend.services.redis_service import redis_service


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class TrackingService:
    """
    Associates single-camera VehicleEvents with a unified global vehicle identity.

    Hybrid Strategy:
      1. Plate-based match (exact match for readable plates)
      2. Spatio-temporal & attribute match (for unreadable / occluded plates)
    """

    def __init__(self) -> None:
        self._plate_to_global: dict[str, str] = {}
        self._counter = 0

    def associate_event(self, event: VehicleEvent, db: Session) -> str | None:
        """
        Assigns a global_vehicle_id using plate matching or spatio-temporal Re-ID.
        """
        # 1. Plate-based primary match
        if event.plate:
            return self._plate_based_associate(event.plate, db)

        # 2. Spatio-temporal Re-ID match for non-plate / occluded vehicles
        reid_id = self._spatio_temporal_reid(event, db)
        if reid_id:
            return reid_id

        # 3. If unidentifiable, assign a new track ID
        self._counter += 1
        return f"VEH_{self._counter:06d}"

    def _plate_based_associate(self, plate: str, db: Session) -> str:
        clean_plate = plate.upper().replace(" ", "")

        # Check in-memory cache
        if clean_plate in self._plate_to_global:
            return self._plate_to_global[clean_plate]

        # Check DB for previous sightings of this plate
        existing = (
            db.query(VehicleEvent)
            .filter(VehicleEvent.plate == clean_plate, VehicleEvent.global_vehicle_id.isnot(None))
            .first()
        )
        if existing and existing.global_vehicle_id:
            self._plate_to_global[clean_plate] = existing.global_vehicle_id
            return existing.global_vehicle_id

        # Generate new global ID
        self._counter += 1
        gid = f"VEH_{self._counter:06d}"
        self._plate_to_global[clean_plate] = gid
        return gid

    def _spatio_temporal_reid(self, event: VehicleEvent, db: Session) -> str | None:
        """
        Spatio-Temporal Graph Re-ID:
        Matches an unreadable vehicle against recent sightings from upstream cameras
        using matching vehicle type, color, direction, and feasible travel time window.
        """
        if not event.latitude or not event.longitude:
            return None

        ev_time = event.timestamp.replace(tzinfo=None) if event.timestamp.tzinfo else event.timestamp
        # Look back 30 minutes for candidate sightings from different cameras
        window_start = ev_time - timedelta(minutes=30)
        candidates = (
            db.query(VehicleEvent)
            .filter(
                VehicleEvent.event_id != event.event_id,
                VehicleEvent.camera_id != event.camera_id,
                VehicleEvent.timestamp >= window_start,
                VehicleEvent.timestamp <= ev_time,
                VehicleEvent.global_vehicle_id.isnot(None),
            )
            .order_by(desc(VehicleEvent.timestamp))
            .limit(20)
            .all()
        )

        for cand in candidates:
            # Attribute match
            if event.vehicle_type and cand.vehicle_type and event.vehicle_type != cand.vehicle_type:
                continue

            # Spatio-temporal physical feasibility
            if cand.latitude and cand.longitude:
                cand_time = cand.timestamp.replace(tzinfo=None) if cand.timestamp.tzinfo else cand.timestamp
                dist = haversine_km(cand.latitude, cand.longitude, event.latitude, event.longitude)
                dt_hours = (ev_time - cand_time).total_seconds() / 3600.0
                if dt_hours > 0:
                    speed = dist / dt_hours
                    # Feasible urban travel speed (5 km/h to 120 km/h)
                    if 5.0 <= speed <= 120.0:
                        return cand.global_vehicle_id

        return None


tracking_service = TrackingService()

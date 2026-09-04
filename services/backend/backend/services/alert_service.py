"""
Alert service — Blacklist checking, Route Anomaly detection, and Redis pub/sub.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.models.alert import Alert, Blacklist
from backend.models.camera import Camera
from backend.models.vehicle_event import VehicleEvent
from backend.services.redis_service import redis_service

settings = get_settings()


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two GPS points in km."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


class AlertService:
    def check_and_fire(self, event: VehicleEvent, db: Session) -> Alert | None:
        """
        Check if this vehicle event triggers:
          1. Blacklist alert
          2. Route anomaly alert
        """
        vkey = event.plate or event.global_vehicle_id
        if not vkey:
            return None

        clean_plate = event.plate.upper().replace(" ", "") if event.plate else None

        # 1. Blacklist check
        if clean_plate:
            blacklisted = db.query(Blacklist).filter(Blacklist.plate == clean_plate).first()
            if blacklisted:
                alert = Alert(
                    vehicle_id=clean_plate,
                    camera_id=event.camera_id,
                    alert_type="BLACKLIST",
                    description=f"Blacklisted plate {clean_plate} detected. Reason: {blacklisted.reason or 'Unspecified'}",
                    status="ACTIVE",
                    timestamp=event.timestamp or datetime.utcnow(),
                )
                db.add(alert)
                db.commit()
                db.refresh(alert)
                self._publish_to_redis(alert)
                return alert

        # 2. Checkpoint-pair speed check: route anomaly (impossible travel) or
        #    a genuine speed-limit violation between two consecutive cameras.
        speed_alert = self.check_checkpoint_speed(event, db)
        if speed_alert:
            self._publish_to_redis(speed_alert)
            return speed_alert

        return None

    def _find_previous_leg(self, event: VehicleEvent, db: Session):
        """
        Find the vehicle's previous sighting at a *different* camera and
        compute the implied checkpoint-to-checkpoint speed. Shared by both
        the route-anomaly check and the speed-violation check so we only
        run the lookup once per event.

        Returns (prev_event, dist_km, calc_speed_kmh) or None.
        """
        plate_key = event.plate.upper().replace(" ", "") if event.plate else None
        vkey = plate_key or event.global_vehicle_id
        if not vkey or not event.latitude or not event.longitude:
            return None

        ev_ts = event.timestamp.replace(tzinfo=None) if event.timestamp.tzinfo else event.timestamp

        prev = (
            db.query(VehicleEvent)
            .filter(
                VehicleEvent.event_id != event.event_id,
                (VehicleEvent.plate == plate_key) if plate_key else (VehicleEvent.global_vehicle_id == vkey),
                VehicleEvent.latitude.isnot(None),
                VehicleEvent.longitude.isnot(None),
                VehicleEvent.timestamp < ev_ts,
            )
            .order_by(desc(VehicleEvent.timestamp))
            .first()
        )

        if not prev or prev.camera_id == event.camera_id:
            return None

        prev_ts = prev.timestamp.replace(tzinfo=None) if prev.timestamp.tzinfo else prev.timestamp
        time_diff_hours = (ev_ts - prev_ts).total_seconds() / 3600.0
        if time_diff_hours <= 0:
            return None

        dist_km = haversine_distance_km(prev.latitude, prev.longitude, event.latitude, event.longitude)
        calc_speed_kmh = dist_km / time_diff_hours
        return prev, dist_km, calc_speed_kmh, time_diff_hours

    def check_checkpoint_speed(self, event: VehicleEvent, db: Session) -> Alert | None:
        """
        Compare this sighting against the vehicle's previous sighting at a
        different camera ("checkpoint") and classify the implied speed:

          - > MAX_PLAUSIBLE_SPEED_KMH  -> ROUTE_ANOMALY (impossible travel;
            almost certainly a bad plate match or clock skew, not a real car)
          - > DEFAULT_SPEED_LIMIT_KMH  -> SPEED_VIOLATION (a genuine
            checkpoint-pair overspeed defaulter)
          - otherwise                 -> no alert
        """
        leg = self._find_previous_leg(event, db)
        if not leg:
            return None
        prev, dist_km, calc_speed_kmh, time_diff_hours = leg

        vehicle_id = event.plate or event.global_vehicle_id or prev.global_vehicle_id

        if calc_speed_kmh > settings.MAX_PLAUSIBLE_SPEED_KMH and dist_km > 0.5:
            alert = Alert(
                vehicle_id=vehicle_id,
                camera_id=event.camera_id,
                alert_type="ROUTE_ANOMALY",
                description=(
                    f"Route anomaly: vehicle covered {dist_km:.2f} km between {prev.camera_id} and "
                    f"{event.camera_id} in {time_diff_hours * 60:.1f} mins ({calc_speed_kmh:.1f} km/h) — "
                    "physically implausible, likely a bad match rather than real travel."
                ),
                status="ACTIVE",
                timestamp=event.timestamp or datetime.utcnow(),
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            return alert

        # Genuine defaulter: use whichever is higher — the checkpoint-pair
        # average speed, or an instantaneous radar/tracker speed reported by
        # the camera worker for this sighting. The limit that applies is the
        # more restrictive of the two road segments the vehicle crossed
        # (a ring road and a residential lane don't share one number).
        effective_speed = max(calc_speed_kmh, event.speed or 0.0)
        speed_limit = self._segment_speed_limit(prev.camera_id, event.camera_id, db)
        if effective_speed > speed_limit and dist_km > 0.1:
            excess = effective_speed - speed_limit
            alert = Alert(
                vehicle_id=vehicle_id,
                camera_id=event.camera_id,
                alert_type="SPEED_VIOLATION",
                description=(
                    f"Speed violation: vehicle covered {dist_km:.2f} km between {prev.camera_id} and "
                    f"{event.camera_id} in {time_diff_hours * 60:.1f} mins ({effective_speed:.1f} km/h), "
                    f"{excess:.1f} km/h over the {speed_limit:.0f} km/h limit."
                ),
                status="ACTIVE",
                timestamp=event.timestamp or datetime.utcnow(),
            )
            db.add(alert)
            db.commit()
            db.refresh(alert)
            return alert

        return None

    def _segment_speed_limit(self, camera_id_a: str, camera_id_b: str, db: Session) -> float:
        """The applicable limit for a hop is the stricter of the two cameras'
        road segments. Falls back to the configured city default for any
        camera missing a limit (shouldn't happen post-migration, but keeps
        this from ever raising on stale data)."""
        cams = db.query(Camera).filter(Camera.camera_id.in_([camera_id_a, camera_id_b])).all()
        limits = [c.speed_limit_kmh for c in cams if c.speed_limit_kmh]
        return min(limits) if limits else settings.DEFAULT_SPEED_LIMIT_KMH

    def _publish_to_redis(self, alert: Alert):
        redis_service.publish_alert({
            "alert_id": alert.alert_id,
            "vehicle_id": alert.vehicle_id,
            "camera_id": alert.camera_id,
            "alert_type": alert.alert_type,
            "description": alert.description,
            "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
        })

    def add_to_blacklist(self, plate: str, reason: str | None, db: Session) -> Blacklist:
        clean = plate.upper().replace(" ", "")
        entry = Blacklist(plate=clean, reason=reason, added_at=datetime.utcnow())
        merged = db.merge(entry)
        db.commit()
        db.refresh(merged)
        return merged

    def remove_from_blacklist(self, plate: str, db: Session) -> bool:
        clean = plate.upper().replace(" ", "")
        deleted = db.query(Blacklist).filter(Blacklist.plate == clean).delete()
        db.commit()
        return deleted > 0


alert_service = AlertService()

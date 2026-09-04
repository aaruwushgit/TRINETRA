"""
POI Trajectory Prediction Service — Markov Transition Graph + Traffic Weighting.

Algorithm:
  1. Build a transition count matrix from historical events: transitions[cam_a][cam_b]++
  2. Normalize per source camera -> P(next_cam | current_cam)
  3. Filter candidates by directional cosine similarity (removes U-turns)
  4. Compute ETA using Haversine distance / live average speed on that road segment
  5. Rank by probability, return top-N

Zero external ML dependencies — pure math on existing DB data.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models.camera import Camera
from backend.models.vehicle_event import VehicleEvent


# Cap on the transition-fitting scan. One index range scan over the tail of
# `timestamp`; large enough that a busy camera has hundreds of observed
# departures, small enough to run in well under a second on a 12M-row table.
TRANSITION_SAMPLE = 200_000

# A vehicle's direction of travel is determined by its last few sightings, not
# by the 400 before them.
MAX_PLATE_SIGHTINGS = 50

# The live-speed lookup below is per-camera over the last hour. On a city-scale
# database that is still a large group-by, so it is capped the same way.
SPEED_SAMPLE = 50_000


@dataclass
class NextHop:
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    probability: float
    distance_km: float
    eta_minutes: float
    congestion: str
    interception_priority: str


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing in degrees from point 1 to point 2."""
    d_lon = math.radians(lon2 - lon1)
    x = math.sin(d_lon) * math.cos(math.radians(lat2))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _bearing_similarity(b1: float, b2: float) -> float:
    """1.0 = same direction, -1.0 = opposite. Filters U-turns."""
    diff = abs(b1 - b2)
    if diff > 180:
        diff = 360 - diff
    return 1.0 - (diff / 180.0)


class PredictionService:
    """Lightweight Markov next-hop predictor for POI vehicles."""

    def predict(
        self,
        plate: str,
        db: Session,
        top_n: int = 3,
        lookback_days: int = 30,
    ) -> dict:
        """
        Predict the next camera(s) a POI vehicle will appear at.

        Returns dict with last sighting details and ranked next-hop candidates.
        """
        clean = plate.upper().replace(" ", "")

        # Recent sightings for this plate. DESC + LIMIT so the database trims:
        # a vehicle with a month of history has hundreds of sightings and only
        # the last few determine where it is heading.
        events = (
            db.query(VehicleEvent)
            .filter(
                VehicleEvent.plate == clean,
                VehicleEvent.latitude.isnot(None),
                VehicleEvent.longitude.isnot(None),
            )
            .order_by(VehicleEvent.timestamp.desc())
            .limit(MAX_PLATE_SIGHTINGS)
            .all()
        )
        if not events:
            return {"error": f"No sightings found for plate: {clean}"}
        events.reverse()

        last = events[-1]

        # --- City-wide Markov transition matrix, from a BOUNDED sample -------
        # This used to be "every event in the last 30 days", which on a
        # city-scale database is a 12M-row scan on every single request — the
        # endpoint simply never returned. Two changes make it bounded:
        #
        #   1. Only transitions OUT OF the camera we actually need are counted.
        #      The old code built the whole matrix and then read one row of it.
        #   2. The scan is capped at the most recent TRANSITION_SAMPLE events,
        #      taken as one index range scan over the tail of `timestamp`.
        #
        # Recency is a feature here, not a compromise: "where does traffic go
        # from this junction *now*" should be fitted on now, not averaged with
        # a public holiday three weeks ago.
        window = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=lookback_days)
        recent = (
            db.query(VehicleEvent.global_vehicle_id, VehicleEvent.camera_id, VehicleEvent.timestamp)
            .filter(VehicleEvent.timestamp >= window, VehicleEvent.global_vehicle_id.isnot(None))
            .order_by(VehicleEvent.timestamp.desc())
            .limit(TRANSITION_SAMPLE)
            .all()
        )

        # Rows arrive newest-first and interleaved across vehicles, so group by
        # vehicle in Python. Asking SQL to ORDER BY (global_vehicle_id,
        # timestamp) over the same window costs an external sort of the sample.
        by_vehicle: dict[str, list[tuple]] = defaultdict(list)
        for vid, cam, ts in recent:
            by_vehicle[vid].append((ts, cam))

        src_transitions: dict[str, int] = defaultdict(int)
        for hops in by_vehicle.values():
            if len(hops) < 2:
                continue
            hops.sort()
            prev_cam = hops[0][1]
            for _ts, cam in hops[1:]:
                if prev_cam == last.camera_id and cam != prev_cam:
                    src_transitions[cam] += 1
                prev_cam = cam

        # --- Get all active cameras (candidate next-hops) ---
        cameras: list[Camera] = db.query(Camera).filter(Camera.is_active.is_(True)).all()

        # --- Get live average speeds per camera for congestion estimate ---
        # Averaged over a capped recent sample rather than a GROUP BY across
        # the whole last hour — at city scale that group-by is minutes of work
        # for a number used only to estimate an ETA.
        hour_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)
        speed_rows = (
            db.query(VehicleEvent.camera_id, VehicleEvent.speed)
            .filter(VehicleEvent.timestamp >= hour_ago, VehicleEvent.speed.isnot(None))
            .order_by(VehicleEvent.timestamp.desc())
            .limit(SPEED_SAMPLE)
            .all()
        )
        speed_totals: dict[str, list[float]] = defaultdict(list)
        for cam_id, spd in speed_rows:
            speed_totals[cam_id].append(spd)
        live_speeds: dict[str, float] = {
            cam_id: sum(vals) / len(vals) for cam_id, vals in speed_totals.items() if vals
        }

        # --- Travel direction of POI from last two sightings ---
        poi_bearing: float | None = None
        if len(events) >= 2:
            prev_e = events[-2]
            if prev_e.latitude and prev_e.longitude and last.latitude and last.longitude:
                poi_bearing = _bearing(prev_e.latitude, prev_e.longitude, last.latitude, last.longitude)

        # --- Score each candidate camera ---
        total_transitions = sum(src_transitions.values()) or 1
        results: list[NextHop] = []

        for cam in cameras:
            if cam.camera_id == last.camera_id:
                continue  # skip current camera

            dist = _haversine(last.latitude, last.longitude, cam.latitude, cam.longitude)
            if dist < 0.1 or dist > 50.0:  # ignore same-spot and very distant cameras
                continue

            # Markov probability (transition count / total, or proximity if no history)
            markov_prob = src_transitions.get(cam.camera_id, 0) / total_transitions
            if markov_prob == 0:
                # Proximity fallback: closer = higher base probability
                markov_prob = max(0.01, 1.0 / (1.0 + dist))

            # Direction filter: penalize cameras in opposite direction
            if poi_bearing is not None:
                to_cam_bearing = _bearing(last.latitude, last.longitude, cam.latitude, cam.longitude)
                similarity = _bearing_similarity(poi_bearing, to_cam_bearing)
                if similarity < -0.2:  # strong U-turn -> skip
                    continue
                markov_prob *= max(0.1, similarity)

            # ETA using live traffic speed or last known POI speed
            road_speed = live_speeds.get(cam.camera_id, last.speed or 40.0)
            road_speed = max(5.0, road_speed)  # floor at 5 km/h
            eta_minutes = (dist / road_speed) * 60.0

            # Congestion classification
            if road_speed < 15:
                congestion = "HIGH"
            elif road_speed < 35:
                congestion = "MEDIUM"
            else:
                congestion = "LOW"

            # Interception priority: high probability + low congestion + soon = HIGH
            priority_score = markov_prob / max(1.0, eta_minutes / 10.0)
            interception = "HIGH" if priority_score > 0.05 else "MEDIUM" if priority_score > 0.01 else "LOW"

            results.append(NextHop(
                camera_id=cam.camera_id,
                camera_name=cam.name,
                latitude=cam.latitude,
                longitude=cam.longitude,
                probability=round(markov_prob, 4),
                distance_km=round(dist, 2),
                eta_minutes=round(eta_minutes, 1),
                congestion=congestion,
                interception_priority=interception,
            ))

        # Normalize probabilities first, then round to preserve sum=1.0
        total_prob = sum(r.probability for r in results) or 1.0
        for r in results:
            r.probability = r.probability / total_prob
        results.sort(key=lambda r: r.probability, reverse=True)
        results = results[:top_n]
        # Round after slicing so visible values reflect actual distribution
        top_total = sum(r.probability for r in results) or 1.0
        for r in results:
            r.probability = round(r.probability / top_total, 3)

        return {
            "plate": clean,
            "global_vehicle_id": last.global_vehicle_id,
            "last_sighting": {
                "camera_id": last.camera_id,
                "timestamp": last.timestamp.isoformat(),
                "latitude": last.latitude,
                "longitude": last.longitude,
                "speed_kmh": last.speed,
                "direction": last.direction,
            },
            "predicted_destinations": [
                {
                    "camera_id": r.camera_id,
                    "camera_name": r.camera_name,
                    "latitude": r.latitude,
                    "longitude": r.longitude,
                    "probability": r.probability,
                    "distance_km": r.distance_km,
                    "eta_minutes": r.eta_minutes,
                    "congestion": r.congestion,
                    "interception_priority": r.interception_priority,
                }
                for r in results
            ],
            "suggested_interception": results[0].camera_id if results else None,
        }


prediction_service = PredictionService()

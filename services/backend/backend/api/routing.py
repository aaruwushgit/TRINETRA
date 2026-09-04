"""
Routing router — ROAD-SNAPPED vehicle trajectories for the GIS map.

This is the map-facing counterpart to /vehicles/{plate}/trajectory. That
endpoint returns bare camera sightings, which a client can only join with
straight lines — a vehicle apparently driving through buildings and across the
Yamuna. Everything here returns the ACTUAL road path between consecutive
sightings (OSRM over OpenStreetMap, cached in route_segments) plus direction
arrows, so the drawn line is a road the vehicle could really have taken and it
is obvious which way it was travelling.

The original endpoint is left untouched: other clients depend on its shape, and
"same data, road-accurate" belongs behind its own prefix rather than as a
breaking change to an existing contract.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.vehicle_event import VehicleEvent
from backend.services import routing_service
from backend.services.routing_service import (
    ensure_route_cache_schema,
    get_route,
    resolve_camera,
    snap_trajectory,
)

router = APIRouter(prefix="/routing", tags=["Routing"])

# Create route_segments on import if init_db() has not registered the model yet
# (see ensure_route_cache_schema's docstring — that import is one line in
# backend/database.py, which this module does not own).
ensure_route_cache_schema()

# Defaults chosen so the map stays responsive: a vehicle with a month of
# history across 200 cameras can have thousands of sightings, and each leg
# carries up to ~400 geometry points. 24h / 60 sightings is ~60 legs, which is
# already a dense map layer; both bounds are overridable per request.
DEFAULT_HOURS = 24
DEFAULT_LIMIT = 60


@router.get("/trajectory/{plate}")
def get_road_trajectory(
    plate: str,
    hours: int = Query(
        default=DEFAULT_HOURS, ge=1, le=24 * 365,
        description="Look back this many hours from now (default 24).",
    ),
    limit: int = Query(
        default=DEFAULT_LIMIT, ge=2, le=500,
        description="Keep at most this many of the MOST RECENT sightings (default 60).",
    ),
    allow_fetch: bool = Query(
        default=True,
        description="Allow live OSRM lookups for uncached legs. Set false to force "
                    "cache-only rendering (fast, zero network, fallback lines for misses).",
    ),
    db: Session = Depends(get_db),
):
    """
    The full road-snapped journey for a plate — this is what the map draws.

    Each leg carries:
      * `geometry`: [[lat, lon], ...] following the real road network, already in
        Leaflet order (no flipping needed client-side),
      * `arrows`: anchor points with `bearing_degrees`, for rotating chevrons to
        show the direction of travel,
      * road vs straight-line distance and BOTH implied speeds, because
        crow-flies speed (what the rest of the API reports) systematically
        under-states real travel by the detour ratio (~1.15–1.6x in Delhi),
      * `source` / `is_real_road` — a leg that fell back to a straight line
        because the router was unreachable says so, rather than pretending.

    Bounded by `hours` and `limit` so one query can't return megabytes of
    geometry. `limit` keeps the most RECENT sightings (an investigator cares
    about where the vehicle is now, not where it was 400 sightings ago).
    """
    clean = plate.upper().replace(" ", "").replace("-", "")
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    # Order DESC + limit in SQL so the database does the trimming, then flip to
    # chronological in Python. Sorting ascending and slicing in Python would
    # pull every row for a heavily-sighted vehicle.
    events = (
        db.query(VehicleEvent)
        .filter(VehicleEvent.plate == clean, VehicleEvent.timestamp >= since)
        .order_by(VehicleEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    events.reverse()

    if not events:
        raise HTTPException(
            status_code=404,
            detail=f"No sightings for plate {clean} in the last {hours}h",
        )
    if len(events) < 2:
        # One sighting is a point, not a journey. 200 with an empty leg list is
        # friendlier than a 404 here: the plate genuinely exists and the client
        # can still drop a marker.
        only = events[0]
        return {
            "plate": clean,
            "global_vehicle_id": only.global_vehicle_id,
            "window_hours": hours,
            "limit": limit,
            "legs": [],
            "totals": {"sightings": 1, "legs": 0},
            "note": "Only one sighting in this window — no road path to draw yet.",
        }

    snapped = snap_trajectory(events, db, allow_fetch=allow_fetch)
    totals = snapped["totals"]
    return {
        "plate": clean,
        "global_vehicle_id": events[-1].global_vehicle_id,
        "window_hours": hours,
        "limit": limit,
        "geometry_order": "[latitude, longitude] (Leaflet order)",
        # Top-level mirrors of the headline totals. frontend/index.html reads
        # `total_road_km` off the response root for its "ROAD-SNAPPED — N km on
        # road" badge; duplicating three scalars is cheaper than a frontend
        # change we don't own, and `totals` stays the canonical block.
        "total_road_km": totals.get("total_road_km"),
        "total_straight_line_km": totals.get("total_straight_line_km"),
        "detour_ratio": totals.get("detour_ratio"),
        **snapped,
    }


@router.get("/segment")
def get_road_segment(
    from_camera: str = Query(..., description="Origin camera_id"),
    to_camera: str = Query(..., description="Destination camera_id"),
    refresh: bool = Query(default=False, description="Bypass the cache and re-fetch from OSRM"),
    allow_fetch: bool = Query(default=True, description="Allow a live OSRM lookup on cache miss"),
    db: Session = Depends(get_db),
):
    """
    One road segment between two cameras (cached, or fetched and then cached).

    Useful on its own for drawing a corridor on the map (top-used road pairs,
    predicted next hop) without needing a vehicle.
    """
    for cid in (from_camera, to_camera):
        if resolve_camera(cid, db) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown camera or camera has no coordinates: {cid}",
            )

    try:
        route = get_route(from_camera, to_camera, db, refresh=refresh, allow_fetch=allow_fetch)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    payload = route.to_dict()
    payload["arrows"] = routing_service.arrow_anchors(route.geometry)
    payload["geometry_order"] = "[latitude, longitude] (Leaflet order)"
    return payload


@router.get("/cache/stats")
def get_cache_stats(
    k: int = Query(default=6, ge=1, le=20, description="k used for adjacency coverage"),
    deployment: str | None = Query(
        default=None,
        description="Score coverage against one deployment's cameras only, e.g. 'delhi'. "
                    "Omit to count every known camera (inflates pairs_wanted when this "
                    "database holds more than one deployment).",
    ),
    db: Session = Depends(get_db),
):
    """
    Route-cache health — the answer to "is this geometry real?".

    `median_detour_ratio` is the lie-detector: real road routing across Delhi
    runs ~1.15–1.6x the straight-line distance. A value near 1.0 (or a high
    `fallback_segments` count) means legs are being drawn as chords, not roads.
    `adjacency.coverage_pct` says how much of the k-nearest-neighbour camera
    graph is precomputed, i.e. how much of the demo works with the network off.
    """
    return routing_service.cache_stats(db, k=k, deployment=deployment)

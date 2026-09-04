"""
Routing service — turns camera sightings into REAL ROAD paths.

The problem this solves
──────────────────────
Everywhere else in this codebase, "the path a vehicle took" is a straight line
between two camera coordinates (see backend/api/vehicles.py's
/vehicles/{plate}/trajectory, and the haversine-based speed maths in
alert_service / speed-defaulters). That is fine as a cheap lower bound, but it
is wrong in two visible ways:

  1. On a map it draws the vehicle straight through buildings and across the
     Yamuna. Nobody believes a surveillance product that does that.
  2. It systematically UNDERSTATES speed. Straight-line distance between two
     Delhi junctions is typically only ~60–85% of the real driving distance, so
     "distance/time" computed from haversine under-reports how fast the vehicle
     actually travelled. A car that did 26.5 km of road in 20 minutes (79 km/h,
     speeding) looks like 21.7 km in 20 minutes (65 km/h, barely over) if you
     measure the crow-flies line. Every leg we return therefore reports BOTH
     speeds so the difference is explicit rather than quietly wrong.

What we do instead
──────────────────
For each consecutive pair of sightings we ask a real road-network router (OSRM,
running on OpenStreetMap data) for the driving route, and draw that polyline.

Why OSRM's own routing instead of a hand-rolled graph search over our cameras:
a vehicle seen at A then C may physically have passed B, and it is tempting to
run Dijkstra over a camera adjacency graph to reconstruct A→B→C. That is the
wrong layer. OSRM already searches the *actual road network* (every OSM way,
turn restriction, one-way and flyover) and returns the shortest/fastest driving
path A→C — which is the genuine "most probable road path", and which will pass
through B anyway if B is on it. Our camera adjacency graph (build_adjacency
below) is therefore used only to decide which pairs are worth PRECOMPUTING,
not to route. Hand-rolling path search over ~200 cameras would produce a
coarse polyline of camera-to-camera hops, i.e. straight lines again.

Caching & good citizenship
──────────────────────────
The public OSRM demo server is rate-limited and must not be hammered:
  * requests are serialised through a global minimum-interval throttle,
  * 429/5xx and connection errors retry with exponential backoff,
  * every result is persisted in the route_segments table (see
    backend/models/route_segment.py) and additionally memoised in-process.
After scripts/build_route_cache.py has run, trajectory rendering makes ZERO
network calls — which is the point, because venue Wi-Fi cannot be trusted.

Failure behaviour: if the router is unreachable and the pair is not cached, we
return a 2-point straight line marked source="fallback_straight". A trajectory
request must never 500 because routing is down — but the degradation is
labelled in the payload so it can be shown honestly in the UI, never passed
off as a road.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from backend.models.camera import Camera
from backend.models.route_segment import RouteSegment

# Reuse the existing haversine rather than adding a *fifth* copy to this repo
# (there are already implementations in alert_service, tracking_service,
# prediction_service and scripts/generate_city_dataset.py — that duplication is
# a pre-existing smell, not one we want to make worse).
from backend.services.alert_service import haversine_distance_km as haversine_km

# ── Configuration ─────────────────────────────────────────────────────────────
# Read straight from the environment rather than backend/config.py: routing is
# an add-on module and this keeps it self-contained (and lets tests point the
# base URL at an unreachable host to prove the offline cache path works).
OSRM_BASE_URL = os.getenv("OSRM_BASE_URL", "https://router.project-osrm.org").rstrip("/")
OSRM_TIMEOUT_S = float(os.getenv("OSRM_TIMEOUT_S", "12"))
OSRM_MAX_ATTEMPTS = int(os.getenv("OSRM_MAX_ATTEMPTS", "3"))
# Minimum wall-clock gap between two outbound OSRM calls, process-wide. The
# demo server is a shared free resource; ~6 req/s is already generous.
OSRM_MIN_INTERVAL_S = float(os.getenv("OSRM_MIN_INTERVAL_S", "0.15"))
# Hard kill-switch: ROUTING_OFFLINE=1 makes the service cache-only. Used by the
# offline verification and available as a demo-day panic button.
ROUTING_OFFLINE = os.getenv("ROUTING_OFFLINE", "").strip().lower() in {"1", "true", "yes"}

# Geometry simplification. OSRM returns ~570 points for a 26 km Delhi route;
# a 6-leg journey at full fidelity is a ~200 KB JSON response for detail no
# screen can show. Douglas-Peucker at ~8 m tolerance keeps every visible curve,
# junction and flyover ramp while cutting point counts roughly 2–3x. We do NOT
# decimate harder than this: the whole deliverable is that the line visibly
# follows roads, and over-simplifying turns it back into chords.
GEOMETRY_TOLERANCE_M = float(os.getenv("ROUTE_GEOMETRY_TOLERANCE_M", "8"))
GEOMETRY_MAX_POINTS = int(os.getenv("ROUTE_GEOMETRY_MAX_POINTS", "400"))

# Direction arrows ("which way did it go?") every ~300 m of road.
ARROW_SPACING_M = float(os.getenv("ROUTE_ARROW_SPACING_M", "300"))
ARROW_MAX_PER_LEG = int(os.getenv("ROUTE_ARROW_MAX_PER_LEG", "40"))
# Tangent look-ahead: bearing from a single 5 m polyline segment is jittery, so
# each arrow's heading is measured over at least this much road ahead.
ARROW_TANGENT_LOOKAHEAD_M = 40.0

DEPLOYMENTS_DIR = Path(__file__).resolve().parents[2] / "deployments"

_EARTH_R_M = 6371000.0


# ── Small geo helpers ─────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (thin wrapper over the shared km one)."""
    return haversine_km(lat1, lon1, lat2, lon2) * 1000.0


def bearing_degrees(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compass bearing (0=N, 90=E) FROM point 1 TO point 2.

    Arrow direction correctness hinges on argument order: point 1 must be the
    earlier point on the polyline and point 2 the later one, so the chevron
    points the way the vehicle travelled, not back where it came from.
    """
    d_lon = math.radians(lon2 - lon1)
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass_point(bearing: float) -> str:
    """Human-readable heading, handy for API payloads and eyeballing arrows."""
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((bearing % 360.0) / 22.5 + 0.5) % 16]


def _local_metres_scale(lat: float) -> tuple[float, float]:
    """Metres per degree (lat, lon) near `lat` — good enough for a city-sized box."""
    m_per_deg_lat = math.pi * _EARTH_R_M / 180.0
    return m_per_deg_lat, m_per_deg_lat * math.cos(math.radians(lat))


def decimate_polyline(points: Sequence[Sequence[float]],
                      tolerance_m: float = GEOMETRY_TOLERANCE_M,
                      max_points: int = GEOMETRY_MAX_POINTS) -> list[list[float]]:
    """
    Douglas-Peucker simplification of a [[lat, lon], ...] polyline.

    Implemented iteratively (explicit stack) rather than recursively: OSRM can
    return thousands of points for a long route and CPython's recursion limit
    is a silly way to break a map. If DP alone still leaves more than
    `max_points`, a uniform stride is applied as a backstop — endpoints are
    always preserved so the line still starts and ends at the cameras.
    """
    pts = [[float(p[0]), float(p[1])] for p in points]
    if len(pts) <= 2:
        return pts

    m_lat, m_lon = _local_metres_scale(pts[0][0])
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]

    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        ax, ay = pts[start][1] * m_lon, pts[start][0] * m_lat
        bx, by = pts[end][1] * m_lon, pts[end][0] * m_lat
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy

        worst_i, worst_d = -1, -1.0
        for i in range(start + 1, end):
            px, py = pts[i][1] * m_lon, pts[i][0] * m_lat
            if seg_len2 == 0.0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d > worst_d:
                worst_i, worst_d = i, d

        if worst_d > tolerance_m:
            keep[worst_i] = True
            stack.append((start, worst_i))
            stack.append((worst_i, end))

    simplified = [p for p, k in zip(pts, keep) if k]

    if len(simplified) > max_points:
        stride = math.ceil(len(simplified) / max_points)
        strided = simplified[::stride]
        if strided[-1] != simplified[-1]:
            strided.append(simplified[-1])
        simplified = strided
    return simplified


def polyline_length_m(points: Sequence[Sequence[float]]) -> float:
    """Summed segment length of a [[lat, lon], ...] polyline, in metres."""
    return sum(
        haversine_m(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        for i in range(len(points) - 1)
    )


# ── Camera coordinate resolution ──────────────────────────────────────────────

@dataclass(frozen=True)
class CameraRef:
    """Minimal camera identity+position; decoupled from the ORM on purpose."""
    camera_id: str
    latitude: float
    longitude: float
    name: str | None = None
    road: str | None = None
    deployment: str | None = None


_deployment_index: dict[str, CameraRef] | None = None
_deployment_lock = threading.Lock()


def deployment_cameras(refresh: bool = False) -> dict[str, CameraRef]:
    """
    Camera positions read from deployments/*/cameras.json.

    Why we read the deployment files at all instead of only the cameras table:
    the route cache is built OFFLINE and ahead of time, potentially before the
    Delhi cameras have been seeded into the database (and this dev.db currently
    still holds a different deployment's cameras). Falling back to the JSON
    that generated the deployment means `build_route_cache.py` and
    `/routing/segment` work regardless of seeding order.
    """
    global _deployment_index
    with _deployment_lock:
        if _deployment_index is not None and not refresh:
            return _deployment_index
        index: dict[str, CameraRef] = {}
        if DEPLOYMENTS_DIR.exists():
            for path in sorted(DEPLOYMENTS_DIR.glob("*/cameras.json")):
                try:
                    rows = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    # A malformed deployment file must not take the API down;
                    # the DB is still authoritative for seeded cameras.
                    continue
                for row in rows if isinstance(rows, list) else []:
                    cid = row.get("camera_id")
                    lat, lon = row.get("latitude"), row.get("longitude")
                    if not cid or lat is None or lon is None:
                        continue
                    index[cid] = CameraRef(
                        camera_id=cid,
                        latitude=float(lat),
                        longitude=float(lon),
                        name=row.get("name"),
                        road=row.get("road"),
                        deployment=row.get("deployment") or path.parent.name,
                    )
        _deployment_index = index
        return _deployment_index


def _as_ref(obj: Any) -> CameraRef | None:
    """Coerce a Camera ORM row / CameraRef / dict into a CameraRef."""
    if obj is None:
        return None
    if isinstance(obj, CameraRef):
        return obj
    if isinstance(obj, dict):
        if obj.get("latitude") is None or obj.get("longitude") is None:
            return None
        return CameraRef(
            camera_id=str(obj.get("camera_id")),
            latitude=float(obj["latitude"]),
            longitude=float(obj["longitude"]),
            name=obj.get("name"),
            road=obj.get("road"),
            deployment=obj.get("deployment"),
        )
    lat = getattr(obj, "latitude", None)
    lon = getattr(obj, "longitude", None)
    cid = getattr(obj, "camera_id", None)
    if cid is None or lat is None or lon is None:
        return None
    return CameraRef(
        camera_id=str(cid),
        latitude=float(lat),
        longitude=float(lon),
        name=getattr(obj, "name", None),
        road=getattr(obj, "road", None),
        deployment=getattr(obj, "deployment", None),
    )


def resolve_camera(camera: Any, db: Session | None = None) -> CameraRef | None:
    """
    Resolve anything camera-shaped (id string, ORM row, dict) to a CameraRef.

    Lookup order: object already carries coordinates → cameras table → the
    deployment JSON. Returns None if the camera is unknown or has no position,
    so callers can raise a clean 404 instead of crashing on None arithmetic.
    """
    if not isinstance(camera, str):
        ref = _as_ref(camera)
        if ref is not None:
            return ref
        camera = getattr(camera, "camera_id", None)
        if camera is None:
            return None

    camera_id = str(camera)
    if db is not None and _has_table(db, Camera.__tablename__):
        row = db.query(Camera).filter(Camera.camera_id == camera_id).first()
        ref = _as_ref(row)
        if ref is not None:
            return ref
    return deployment_cameras().get(camera_id)


def resolve_cameras(camera_ids: Iterable[str], db: Session | None = None) -> dict[str, CameraRef]:
    """Batch version of resolve_camera — one DB query for a whole trajectory."""
    ids = [c for c in dict.fromkeys(camera_ids) if c]
    out: dict[str, CameraRef] = {}
    if db is not None and ids and _has_table(db, Camera.__tablename__):
        for row in db.query(Camera).filter(Camera.camera_id.in_(ids)).all():
            ref = _as_ref(row)
            if ref is not None:
                out[ref.camera_id] = ref
    fallback = deployment_cameras()
    for cid in ids:
        if cid not in out and cid in fallback:
            out[cid] = fallback[cid]
    return out


# ── Camera adjacency graph ────────────────────────────────────────────────────

def build_adjacency(cameras: Sequence[CameraRef], k: int = 6) -> dict[str, list[tuple[str, float]]]:
    """
    k-nearest-neighbour graph over cameras, by great-circle distance.

    This is NOT used for routing (OSRM routes on the real road network — see the
    module docstring). Its only job is to answer "which camera pairs are
    plausible consecutive sightings, and therefore worth precomputing?", so the
    cache builder spends ~1200 OSRM calls on hops vehicles actually make rather
    than 200*199 = 39,800 calls on pairs across the whole city.

    O(n^2) on purpose: n=200 cameras is 40k distance computations (milliseconds).
    A KD-tree here would be complexity for no gain, and haversine-on-a-sphere is
    not a Euclidean metric a naive KD-tree handles correctly anyway.
    """
    adj: dict[str, list[tuple[str, float]]] = {}
    for a in cameras:
        dists = [
            (b.camera_id, haversine_km(a.latitude, a.longitude, b.latitude, b.longitude))
            for b in cameras
            if b.camera_id != a.camera_id
        ]
        dists.sort(key=lambda t: t[1])
        adj[a.camera_id] = dists[:k]
    return adj


def adjacent_pairs(cameras: Sequence[CameraRef], k: int = 6) -> list[tuple[str, str]]:
    """
    Ordered (from, to) pairs to precompute: each k-NN edge in BOTH directions.

    Both directions because route_segments is directional — Delhi one-ways and
    medians mean A→B is genuinely not reverse(B→A). We de-duplicate the
    undirected edge first (A's neighbour list and B's often both contain the
    other), then emit the two directed pairs once each.
    """
    adj = build_adjacency(cameras, k=k)
    undirected: set[tuple[str, str]] = set()
    for a, neighbours in adj.items():
        for b, _km in neighbours:
            undirected.add((a, b) if a < b else (b, a))
    pairs: list[tuple[str, str]] = []
    for a, b in sorted(undirected):
        pairs.append((a, b))
        pairs.append((b, a))
    return pairs


# ── OSRM client ───────────────────────────────────────────────────────────────

class OSRMUnavailable(RuntimeError):
    """Raised internally when the router cannot be reached; never escapes get_route."""


_throttle_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    """Process-wide minimum gap between outbound OSRM calls (be a good citizen)."""
    global _last_call_at
    with _throttle_lock:
        wait = OSRM_MIN_INTERVAL_S - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def fetch_osrm_route(a: CameraRef, b: CameraRef,
                     base_url: str | None = None) -> tuple[list[list[float]], float, float]:
    """
    Ask OSRM for the driving route a→b. Returns ([[lat, lon], ...], metres, seconds).

    Note the coordinate order dance: OSRM speaks GeoJSON [lon, lat] both in the
    URL and in the response. We flip the response to [lat, lon] HERE, once, so
    that everything downstream (cache, API, Leaflet) uses one convention.

    Retries with exponential backoff on 429 and 5xx (the demo server throttles
    aggressively) and on connection errors (flaky venue Wi-Fi). Raises
    OSRMUnavailable when it gives up — the caller decides how to degrade.
    """
    if ROUTING_OFFLINE:
        raise OSRMUnavailable("ROUTING_OFFLINE=1 — cache-only mode")

    url = (
        f"{(base_url or OSRM_BASE_URL).rstrip('/')}/route/v1/driving/"
        f"{a.longitude},{a.latitude};{b.longitude},{b.latitude}"
    )
    params = {"overview": "full", "geometries": "geojson", "alternatives": "false", "steps": "false"}

    backoff = 0.5
    last_err = "unknown"
    for attempt in range(1, OSRM_MAX_ATTEMPTS + 1):
        _throttle()
        try:
            resp = requests.get(url, params=params, timeout=OSRM_TIMEOUT_S)
        except requests.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    last_err = "non-JSON 200 response"
                    payload = None
                if payload is not None:
                    if payload.get("code") != "Ok" or not payload.get("routes"):
                        # e.g. "NoRoute" — a real, non-retryable answer.
                        raise OSRMUnavailable(f"OSRM code={payload.get('code')}")
                    route = payload["routes"][0]
                    coords = route.get("geometry", {}).get("coordinates") or []
                    if len(coords) < 2:
                        raise OSRMUnavailable("OSRM returned degenerate geometry")
                    latlon = [[float(c[1]), float(c[0])] for c in coords]  # [lon,lat] → [lat,lon]
                    return latlon, float(route.get("distance", 0.0)), float(route.get("duration", 0.0))
            elif resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}"
            else:
                raise OSRMUnavailable(f"HTTP {resp.status_code}")

        if attempt < OSRM_MAX_ATTEMPTS:
            time.sleep(backoff)
            backoff *= 3.0

    raise OSRMUnavailable(f"OSRM unreachable after {OSRM_MAX_ATTEMPTS} attempts ({last_err})")


# ── Route object + cache ──────────────────────────────────────────────────────

@dataclass
class Route:
    """
    A resolved road route between two cameras.

    Returned instead of the RouteSegment ORM row itself so that the DB-backed
    path, the in-process memo path and the straight-line fallback path all hand
    callers the same detached, JSON-safe object — no session-bound instance that
    expires the moment the request's session closes.
    """
    from_camera_id: str
    to_camera_id: str
    road_distance_m: float
    road_duration_s: float
    straight_line_m: float
    geometry: list[list[float]] = field(default_factory=list)
    source: str = "osrm"
    cached: bool = False
    fetched_at: datetime | None = None

    @property
    def point_count(self) -> int:
        return len(self.geometry)

    @property
    def detour_ratio(self) -> float:
        """
        Road distance / straight-line distance. ~1.0 means it is NOT a road.

        Can land a hair BELOW 1.0 on dead-straight stretches: OSRM snaps the
        camera coordinates to the nearest road node (tens of metres), so the
        routed distance is measured between snapped points while
        `straight_line_m` is measured between the raw camera coordinates. We saw
        1707.6 m of road against a 1712.5 m straight line on Ring Road — a 0.3%
        artefact of endpoint snapping, not a broken route.
        """
        if self.straight_line_m <= 0:
            return 1.0
        return self.road_distance_m / self.straight_line_m

    @property
    def is_real_road(self) -> bool:
        """
        True when this geometry came from the road network router.

        Judged on `source` alone, deliberately NOT on point count. Some real
        Delhi hops legitimately reduce to two points: e.g. Mahatma Gandhi Marg
        (Ring Road) between two of our cameras is a dead-straight 1707.6 m of
        road against a 1712.5 m straight line, so Douglas-Peucker correctly
        drops every intermediate vertex. An earlier version of this property
        also required >2 points and mislabelled those 12 cached segments as
        fallbacks — a straight road is still a road.
        """
        return self.source == "osrm"

    @property
    def geometry_is_straight(self) -> bool:
        """
        Routed geometry that happens to be a straight line (road really is straight).

        Kept separate from `is_real_road` so the UI can distinguish "the road is
        straight here" from "we could not reach the router". Only meaningful when
        `source == 'osrm'`.
        """
        return self.source == "osrm" and len(self.geometry) <= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_camera_id": self.from_camera_id,
            "to_camera_id": self.to_camera_id,
            "road_distance_m": round(self.road_distance_m, 1),
            "road_distance_km": round(self.road_distance_m / 1000.0, 3),
            "road_duration_s": round(self.road_duration_s, 1),
            "straight_line_m": round(self.straight_line_m, 1),
            "detour_ratio": round(self.detour_ratio, 3),
            "point_count": self.point_count,
            "source": self.source,
            "is_real_road": self.is_real_road,
            "geometry_is_straight": self.geometry_is_straight,
            "served_from_cache": self.cached,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
            "geometry": self.geometry,
        }


def _route_from_row(row: RouteSegment) -> Route:
    return Route(
        from_camera_id=row.from_camera_id,
        to_camera_id=row.to_camera_id,
        road_distance_m=row.road_distance_m,
        road_duration_s=row.road_duration_s,
        straight_line_m=row.straight_line_m,
        geometry=[[float(p[0]), float(p[1])] for p in (row.geometry or [])],
        source=row.source,
        cached=True,
        fetched_at=row.fetched_at,
    )


def _straight_line_route(a: CameraRef, b: CameraRef) -> Route:
    """
    Honest degradation: a 2-point line, explicitly labelled fallback_straight.

    Duration is estimated at 30 km/h (a realistic Delhi arterial average) purely
    so downstream ETA arithmetic has a number; it is not a routed duration and
    the source field says so.
    """
    d = haversine_m(a.latitude, a.longitude, b.latitude, b.longitude)
    return Route(
        from_camera_id=a.camera_id,
        to_camera_id=b.camera_id,
        road_distance_m=d,
        road_duration_s=d / (30_000.0 / 3600.0) if d else 0.0,
        straight_line_m=d,
        geometry=[[a.latitude, a.longitude], [b.latitude, b.longitude]],
        source="fallback_straight",
        cached=False,
    )


# In-process memo on top of the DB cache. Rendering one trajectory can revisit
# the same hop (loops, patrol routes) and the analytics screens redraw the same
# popular corridors constantly; this avoids a JSON decode of a 300-point
# geometry per redraw. Bounded LRU so a long-running server cannot grow without
# limit. Keyed by (from, to) — directional, like the table.
_MEMO_MAX = int(os.getenv("ROUTE_MEMO_MAX", "4000"))
_memo: "OrderedDict[tuple[str, str], Route]" = OrderedDict()
_memo_lock = threading.Lock()

# One lock per pair, so two concurrent requests for the same uncached hop make
# ONE OSRM call instead of two (the whole point of being polite to the demo
# server). Guarded by _pair_locks_guard because dict mutation itself races.
_pair_locks: dict[tuple[str, str], threading.Lock] = {}
_pair_locks_guard = threading.Lock()

_stats_lock = threading.Lock()
_runtime_stats = {"memo_hits": 0, "db_hits": 0, "osrm_fetches": 0, "fallbacks": 0}


def _memo_get(key: tuple[str, str]) -> Route | None:
    with _memo_lock:
        route = _memo.get(key)
        if route is not None:
            _memo.move_to_end(key)
        return route


def _memo_put(key: tuple[str, str], route: Route) -> None:
    with _memo_lock:
        _memo[key] = route
        _memo.move_to_end(key)
        while len(_memo) > _MEMO_MAX:
            _memo.popitem(last=False)


def _pair_lock(key: tuple[str, str]) -> threading.Lock:
    with _pair_locks_guard:
        lock = _pair_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _pair_locks[key] = lock
        return lock


def _bump(stat: str) -> None:
    with _stats_lock:
        _runtime_stats[stat] = _runtime_stats.get(stat, 0) + 1


def runtime_stats() -> dict[str, int]:
    """Counters for how routes were served this process (surfaced by the API)."""
    with _stats_lock:
        return dict(_runtime_stats, memo_size=len(_memo))


def clear_memo() -> None:
    """Drop the in-process memo (used by tests to force the DB path)."""
    with _memo_lock:
        _memo.clear()


def get_route(from_camera: Any, to_camera: Any, db: Session,
              *, refresh: bool = False, allow_fetch: bool = True,
              persist: bool = True) -> Route:
    """
    The road path from one camera to another.

    Resolution order: in-process memo → route_segments table → OSRM (persisted
    on success) → straight-line fallback. Never raises for routing failure; an
    unknown/positionless camera raises ValueError so the API layer can 404.

    `allow_fetch=False` forces cache-only behaviour (used by cache-coverage
    checks and by the offline verification).
    """
    a = resolve_camera(from_camera, db)
    b = resolve_camera(to_camera, db)
    if a is None:
        raise ValueError(f"Unknown camera or missing coordinates: {from_camera!r}")
    if b is None:
        raise ValueError(f"Unknown camera or missing coordinates: {to_camera!r}")

    key = (a.camera_id, b.camera_id)

    # Degenerate hop (same camera, or two cameras sharing a position): no route
    # to draw. Returned rather than raised so trajectory rendering is uniform.
    if a.camera_id == b.camera_id:
        return Route(
            from_camera_id=a.camera_id, to_camera_id=b.camera_id,
            road_distance_m=0.0, road_duration_s=0.0, straight_line_m=0.0,
            geometry=[[a.latitude, a.longitude]], source="same_camera", cached=True,
        )

    if not refresh:
        memo = _memo_get(key)
        if memo is not None:
            _bump("memo_hits")
            return memo

        row = (
            db.query(RouteSegment)
            .filter(RouteSegment.from_camera_id == key[0], RouteSegment.to_camera_id == key[1])
            .first()
        )
        if row is not None:
            route = _route_from_row(row)
            # Only memoise genuine road geometry. A cached fallback should get
            # another chance at the real router once the network is back, rather
            # than being pinned in memory as a straight line for the whole
            # process lifetime.
            if route.source == "osrm":
                _memo_put(key, route)
            _bump("db_hits")
            if route.source == "osrm" or not allow_fetch:
                return route

    if not allow_fetch:
        return _straight_line_route(a, b)

    # Serialise concurrent misses on the same pair: the second caller waits and
    # then finds the first caller's freshly-persisted row instead of firing a
    # duplicate request at the rate-limited demo server.
    with _pair_lock(key):
        if not refresh:
            memo = _memo_get(key)
            if memo is not None and memo.source == "osrm":
                _bump("memo_hits")
                return memo

        straight_m = haversine_m(a.latitude, a.longitude, b.latitude, b.longitude)
        try:
            geometry, distance_m, duration_s = fetch_osrm_route(a, b)
            source = "osrm"
            _bump("osrm_fetches")
        except OSRMUnavailable:
            _bump("fallbacks")
            route = _straight_line_route(a, b)
            geometry, distance_m, duration_s, source = (
                route.geometry, route.road_distance_m, route.road_duration_s, route.source,
            )

        if source == "osrm":
            geometry = decimate_polyline(geometry)

        route = Route(
            from_camera_id=key[0], to_camera_id=key[1],
            road_distance_m=distance_m, road_duration_s=duration_s,
            straight_line_m=straight_m, geometry=geometry, source=source,
            cached=False, fetched_at=datetime.utcnow(),
        )

        if persist:
            _persist_route(route, db)
        if source == "osrm":
            _memo_put(key, route)
        return route


def _persist_route(route: Route, db: Session) -> None:
    """
    Upsert the route into route_segments.

    Wrapped in try/except and rolled back on failure: a cache write is an
    optimisation, and a locked SQLite file (very possible while the 12M-event
    dataset loader is running) must not turn a successful map render into a 500.
    """
    try:
        row = (
            db.query(RouteSegment)
            .filter(
                RouteSegment.from_camera_id == route.from_camera_id,
                RouteSegment.to_camera_id == route.to_camera_id,
            )
            .first()
        )
        if row is None:
            row = RouteSegment(
                from_camera_id=route.from_camera_id, to_camera_id=route.to_camera_id
            )
            db.add(row)
        row.road_distance_m = route.road_distance_m
        row.road_duration_s = route.road_duration_s
        row.straight_line_m = route.straight_line_m
        row.geometry = route.geometry
        row.point_count = len(route.geometry)
        row.source = route.source
        row.fetched_at = route.fetched_at or datetime.utcnow()
        db.commit()
    except Exception:
        db.rollback()


def ensure_route_cache_schema() -> None:
    """
    Create the route_segments table if it does not exist yet.

    Stopgap, deliberately: backend/database.py's init_db() is owned elsewhere and
    does not import backend.models.route_segment, so Base.metadata.create_all()
    never sees this table. Rather than silently 500 on every routing request
    until that one-line import lands, the routing module creates its own table
    with checkfirst=True. This is idempotent and touches nothing else; once
    init_db() imports the model, this becomes a no-op.
    """
    from backend.database import engine

    try:
        RouteSegment.__table__.create(bind=engine, checkfirst=True)
    except Exception:
        # Table already exists / concurrent creator / read-only DB — the query
        # path will surface any real problem as a clean error instead.
        pass


# ── Direction arrows ──────────────────────────────────────────────────────────

def arrow_anchors(geometry: Sequence[Sequence[float]],
                  spacing_m: float = ARROW_SPACING_M,
                  max_arrows: int = ARROW_MAX_PER_LEG) -> list[dict[str, Any]]:
    """
    Anchor points for direction chevrons along a [[lat, lon], ...] polyline.

    One anchor roughly every `spacing_m` of road, each carrying the bearing of
    the LOCAL POLYLINE TANGENT measured forward along the direction of travel,
    so the UI can rotate a chevron by `bearing_degrees` and have it point the
    way the vehicle went. Every leg gets at least one arrow (placed mid-leg) —
    a short 400 m hop still needs to show which way it ran.

    The tangent is measured over >= ARROW_TANGENT_LOOKAHEAD_M of road rather
    than a single 5 m OSRM segment, because per-segment bearings on a curve
    jitter by tens of degrees and make the chevrons look broken.
    """
    pts = [[float(p[0]), float(p[1])] for p in geometry]
    if len(pts) < 2:
        return []

    # Cumulative along-track distance, so arrows are evenly spaced in metres of
    # road rather than "every Nth point" (point density varies hugely: dense on
    # curves, sparse on straights, so every-Nth-point clusters arrows on bends).
    cum = [0.0]
    for i in range(len(pts) - 1):
        cum.append(cum[-1] + haversine_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    total = cum[-1]
    if total <= 0:
        return []

    effective_spacing = max(spacing_m, total / max_arrows) if max_arrows > 0 else spacing_m
    if total < effective_spacing:
        targets = [total / 2.0]           # short leg → single mid-leg arrow
    else:
        n = max(1, int(total // effective_spacing))
        # Offset by half a spacing so no arrow lands exactly on a camera marker.
        step = total / n
        targets = [step * (i + 0.5) for i in range(n)]

    anchors: list[dict[str, Any]] = []
    seg = 0
    for target in targets:
        while seg < len(pts) - 2 and cum[seg + 1] < target:
            seg += 1
        seg_len = cum[seg + 1] - cum[seg]
        t = 0.0 if seg_len <= 0 else max(0.0, min(1.0, (target - cum[seg]) / seg_len))
        lat = pts[seg][0] + t * (pts[seg + 1][0] - pts[seg][0])
        lon = pts[seg][1] + t * (pts[seg + 1][1] - pts[seg][1])

        # Look forward along the polyline for a stable tangent.
        ahead = seg + 1
        while ahead < len(pts) - 1 and (cum[ahead] - target) < ARROW_TANGENT_LOOKAHEAD_M:
            ahead += 1
        brg = bearing_degrees(lat, lon, pts[ahead][0], pts[ahead][1])

        anchors.append({
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "bearing_degrees": round(brg, 1),
            "heading": compass_point(brg),
            "distance_along_leg_m": round(target, 1),
        })
    return anchors


# ── Trajectory snapping ───────────────────────────────────────────────────────

def _naive(ts: datetime | None) -> datetime | None:
    """Match the codebase convention: timestamps are stored naive-UTC."""
    if ts is None:
        return None
    return ts.replace(tzinfo=None) if ts.tzinfo else ts


def snap_trajectory(events: Sequence[Any], db: Session,
                    *, allow_fetch: bool = True) -> dict[str, Any]:
    """
    Turn ordered VehicleEvent sightings into a road-snapped journey.

    For each consecutive pair of sightings (a "leg") we return the real road
    polyline, its length and routed duration, direction-arrow anchors, and BOTH
    speed figures:

      * implied_road_speed_kmh   — road distance / observed elapsed time. This
        is the truthful number: the vehicle drove roads, not chords.
      * implied_straight_line_speed_kmh — haversine / elapsed time, i.e. what
        the rest of this codebase reports (alert_service, speed-defaulters).
        Included side by side because it is systematically LOW: on real Delhi
        camera pairs the road is ~1.15–1.6x the straight line, so crow-flies
        speed under-reports by 15–60% and can let a genuine speeder past a
        threshold check. We surface the gap rather than silently "fixing" the
        existing endpoints, which we do not own.

    Consecutive sightings at the SAME camera are not legs (a vehicle idling in
    a junction's field of view produces many rows); they are collapsed and
    counted as dwell so the map doesn't draw zero-length spurs.
    """
    ordered = [e for e in events if e is not None]
    ordered.sort(key=lambda e: (_naive(getattr(e, "timestamp", None)) or datetime.min))

    cam_ids = [getattr(e, "camera_id", None) for e in ordered]
    cams = resolve_cameras([c for c in cam_ids if c], db)

    def _pos(event: Any) -> CameraRef | None:
        """Camera position, falling back to the coordinates on the event row."""
        cid = getattr(event, "camera_id", None)
        ref = cams.get(cid) if cid else None
        if ref is not None:
            return ref
        lat, lon = getattr(event, "latitude", None), getattr(event, "longitude", None)
        if lat is None or lon is None:
            return None
        return CameraRef(camera_id=cid or "UNKNOWN", latitude=float(lat), longitude=float(lon))

    legs: list[dict[str, Any]] = []
    dwell_events = 0
    skipped_unlocatable = 0

    prev = None
    for event in ordered:
        if _pos(event) is None:
            skipped_unlocatable += 1
            continue
        if prev is None:
            prev = event
            continue
        if getattr(event, "camera_id", None) == getattr(prev, "camera_id", None):
            # Same camera again — dwell, not travel. Keep the LATEST sighting as
            # the departure point so the next leg's elapsed time is the actual
            # travel time, not travel + idle time (which would understate speed).
            dwell_events += 1
            prev = event
            continue

        a, b = _pos(prev), _pos(event)
        t0, t1 = _naive(getattr(prev, "timestamp", None)), _naive(getattr(event, "timestamp", None))
        elapsed_s = (t1 - t0).total_seconds() if (t0 and t1) else None

        try:
            route = get_route(a, b, db, allow_fetch=allow_fetch)
        except ValueError:
            skipped_unlocatable += 1
            prev = event
            continue

        road_km = route.road_distance_m / 1000.0
        straight_km = route.straight_line_m / 1000.0
        hours = (elapsed_s / 3600.0) if elapsed_s and elapsed_s > 0 else None
        road_speed = round(road_km / hours, 1) if hours else None
        line_speed = round(straight_km / hours, 1) if hours else None

        geometry = route.geometry
        legs.append({
            "leg_index": len(legs),
            "from_camera_id": route.from_camera_id,
            "from_camera_name": (a.name if a else None),
            "from_road": (a.road if a else None),
            "to_camera_id": route.to_camera_id,
            "to_camera_name": (b.name if b else None),
            "to_road": (b.road if b else None),
            "from_timestamp": t0.isoformat() if t0 else None,
            "to_timestamp": t1.isoformat() if t1 else None,
            "elapsed_s": round(elapsed_s, 1) if elapsed_s is not None else None,
            "elapsed_minutes": round(elapsed_s / 60.0, 2) if elapsed_s is not None else None,

            # ── the deliverable: real road geometry + direction ──
            "geometry": geometry,
            "point_count": len(geometry),
            "arrows": arrow_anchors(geometry),

            "road_distance_m": round(route.road_distance_m, 1),
            "road_distance_km": round(road_km, 3),
            "straight_line_m": round(route.straight_line_m, 1),
            "straight_line_km": round(straight_km, 3),
            "detour_ratio": round(route.detour_ratio, 3),
            "routed_duration_s": round(route.road_duration_s, 1),

            "implied_road_speed_kmh": road_speed,
            # Alias: frontend/index.html's drawTrajectory() popup reads
            # `implied_speed_kmh`. Kept as a synonym of the ROAD speed (the
            # truthful one) so the dashboard shows a number instead of "—"
            # without needing a frontend change we don't own.
            "implied_speed_kmh": road_speed,
            "implied_straight_line_speed_kmh": line_speed,
            "speed_understated_by_kmh": (
                round(road_speed - line_speed, 1)
                if (road_speed is not None and line_speed is not None) else None
            ),

            "source": route.source,
            "is_real_road": route.is_real_road,
            "geometry_is_straight": route.geometry_is_straight,
            "served_from_cache": route.cached,
        })
        prev = event

    real_legs = [l for l in legs if l["is_real_road"]]
    total_road_km = sum(l["road_distance_km"] for l in legs)
    total_line_km = sum(l["straight_line_km"] for l in legs)
    elapsed_list = [l["elapsed_s"] for l in legs if l["elapsed_s"]]
    total_elapsed_s = sum(elapsed_list) if elapsed_list else 0.0

    first_ts = _naive(getattr(ordered[0], "timestamp", None)) if ordered else None
    last_ts = _naive(getattr(ordered[-1], "timestamp", None)) if ordered else None

    return {
        "legs": legs,
        "totals": {
            "sightings": len(ordered),
            "legs": len(legs),
            "dwell_sightings_collapsed": dwell_events,
            "unlocatable_sightings_skipped": skipped_unlocatable,
            "real_road_legs": len(real_legs),
            "fallback_legs": len(legs) - len(real_legs),
            "total_road_km": round(total_road_km, 3),
            "total_straight_line_km": round(total_line_km, 3),
            # The single headline number for "why road routing matters".
            "detour_ratio": round(total_road_km / total_line_km, 3) if total_line_km > 0 else None,
            "extra_km_vs_straight_line": round(total_road_km - total_line_km, 3),
            "total_travel_seconds": round(total_elapsed_s, 1),
            "avg_road_speed_kmh": (
                round(total_road_km / (total_elapsed_s / 3600.0), 1) if total_elapsed_s > 0 else None
            ),
            "avg_straight_line_speed_kmh": (
                round(total_line_km / (total_elapsed_s / 3600.0), 1) if total_elapsed_s > 0 else None
            ),
            "first_seen": first_ts.isoformat() if first_ts else None,
            "last_seen": last_ts.isoformat() if last_ts else None,
            "journey_span_seconds": (
                round((last_ts - first_ts).total_seconds(), 1) if (first_ts and last_ts) else None
            ),
            "total_geometry_points": sum(l["point_count"] for l in legs),
        },
    }


# ── Cache introspection (for the demo's "is this real?" question) ──────────────

# Below this straight-line separation the detour ratio stops being a meaningful
# quality signal. Two cameras 80 m apart on opposite carriageways of NH-48 are a
# legitimate 3 km drive apart (you must go to the next U-turn), giving a ratio of
# ~30x. Those are real road routes, but they'd swamp the mean and hide a genuine
# regression, so the headline statistics are computed over hops long enough for
# the ratio to mean "how much does the road wander" rather than "is there a
# median in the way".
RATIO_SAMPLE_MIN_STRAIGHT_M = 500.0


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def _has_table(db: Session, name: str) -> bool:
    """Cheap existence check — see camera_universe for why this is needed."""
    try:
        return sa_inspect(db.get_bind()).has_table(name)
    except Exception:
        return False


def camera_universe(db: Session | None, deployment: str | None = None) -> list[CameraRef]:
    """
    Every camera we can position: deployment JSON first, topped up from the DB.

    JSON first because the route cache is built offline and may exist before the
    cameras are seeded (this dev.db currently still holds a different
    deployment's cameras).
    """
    cams: dict[str, CameraRef] = {}
    for ref in deployment_cameras().values():
        if deployment and ref.deployment != deployment:
            continue
        cams[ref.camera_id] = ref
    # The cameras table can legitimately be absent (a brand-new DATABASE_URL
    # pointed at an empty file, which is exactly how the cache builder is run in
    # CI / on a fresh machine). The deployment JSON alone is enough, so treat a
    # missing table as "no extra cameras" rather than an error.
    if db is not None and _has_table(db, Camera.__tablename__):
        try:
            for row in db.query(Camera).all():
                ref = _as_ref(row)
                if ref is None or ref.camera_id in cams:
                    continue
                if deployment and (getattr(row, "deployment", None) or "") != deployment:
                    continue
                cams[ref.camera_id] = ref
        except Exception:
            db.rollback()
    return list(cams.values())


def cache_stats(db: Session, k: int = 6, deployment: str | None = None) -> dict[str, Any]:
    """
    What is in the route cache, and how well does it cover the adjacency graph.

    Reports the detour ratio because that is the cheapest lie-detector we have:
    genuine road routing over Delhi runs ~1.15–1.6x the straight line on hops of
    any real length. If this reads ~1.0, the geometry is straight lines wearing a
    road costume. `median_detour_ratio` is the number to trust — see
    RATIO_SAMPLE_MIN_STRAIGHT_M for why the mean over ALL pairs is misleading.

    Pass `deployment` (e.g. "delhi") to score coverage against just that city's
    adjacency graph; otherwise every known camera counts, which inflates
    `pairs_wanted` with cameras from other deployments sharing this database.
    """
    rows = db.query(RouteSegment).all()
    osrm = [r for r in rows if r.source == "osrm"]
    fallback = [r for r in rows if r.source != "osrm"]

    sample = [r for r in osrm if r.straight_line_m >= RATIO_SAMPLE_MIN_STRAIGHT_M]
    ratios = sorted(r.road_distance_m / r.straight_line_m for r in sample)
    all_ratios = [r.road_distance_m / r.straight_line_m for r in osrm if r.straight_line_m > 0]
    points = [r.point_count for r in osrm]
    median = _percentile(ratios, 0.5) if ratios else None

    cams = camera_universe(db, deployment)
    wanted = set(adjacent_pairs(cams, k=k)) if len(cams) >= 2 else set()
    have = {(r.from_camera_id, r.to_camera_id) for r in rows}
    covered = wanted & have

    return {
        "segments_cached": len(rows),
        "osrm_segments": len(osrm),
        "fallback_segments": len(fallback),
        # Routed segments whose geometry is only 2 points because the road
        # really is straight there. NOT fallbacks — see Route.is_real_road.
        "osrm_segments_with_straight_geometry": sum(1 for r in osrm if r.point_count <= 2),
        "detour_ratio_sample": {
            "note": (
                f"Computed over the {len(sample)} OSRM segments whose straight-line "
                f"separation is >= {RATIO_SAMPLE_MIN_STRAIGHT_M:.0f} m. Very short "
                "camera pairs (opposite carriageways of one highway) legitimately "
                "route many kilometres to the next U-turn and would distort the mean."
            ),
            "segments_in_sample": len(sample),
            "median": round(median, 3) if median else None,
            "mean": round(sum(ratios) / len(ratios), 3) if ratios else None,
            "p10": round(_percentile(ratios, 0.10), 3) if ratios else None,
            "p90": round(_percentile(ratios, 0.90), 3) if ratios else None,
        },
        # Kept for completeness / debugging, but do not judge quality by these.
        "mean_detour_ratio_all_pairs": round(sum(all_ratios) / len(all_ratios), 3) if all_ratios else None,
        "median_detour_ratio": round(median, 3) if median else None,
        "mean_point_count": round(sum(points) / len(points), 1) if points else None,
        "max_point_count": max(points) if points else None,
        # The headline "is this real?" verdict: real roads wander, straight lines don't.
        "geometry_looks_real": bool(median) and median > 1.05,
        "adjacency": {
            "k": k,
            "deployment": deployment,
            "cameras_known": len(cams),
            "pairs_wanted": len(wanted),
            "pairs_cached": len(covered),
            "coverage_pct": round(100.0 * len(covered) / len(wanted), 1) if wanted else None,
        },
        "runtime": runtime_stats(),
        "osrm_base_url": OSRM_BASE_URL,
        "offline_mode": ROUTING_OFFLINE,
    }

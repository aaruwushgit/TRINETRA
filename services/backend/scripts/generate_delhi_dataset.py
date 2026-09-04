"""
Delhi city-scale ANPR dataset generator — one month, 200 real junctions.

What this produces
------------------
A month of physically coherent ANPR detections over the 200 real Delhi traffic
signals in deployments/delhi/cameras.json (thinned from 1,276 OpenStreetMap
`highway=traffic_signals` nodes), plus the precomputed analytics rollups in
backend/models/analytics_agg.py, loaded straight into SQLite at ~1M rows/min.

Defaults: 500,000 unique vehicles, 30 days, ~12M events. All configurable.

This generates *structured event records only* — no video, no YOLO, no OCR.
It is the output a real ANPR pipeline would have produced, so that everything
downstream (tracking, trajectory reconstruction, alerts, analytics, next-hop
prediction) can be exercised at genuine city scale.

Three design decisions carry most of the weight
-----------------------------------------------

1. Timestamps are *derived*, never drawn independently.
   scripts/test_analytics_big_data.py gave every event an independent random
   timestamp inside a window. Two sightings of one plate at junctions 17km
   apart could then land seconds apart, implying ~15,000 km/h — which silently
   destroyed the speed-defaulter logic and dodged the question the platform
   actually has to answer. Here each vehicle walks a camera adjacency graph hop
   by hop, and every timestamp is computed forward from the previous one using
   that hop's real haversine distance, a speed that respects both cameras'
   posted limits and the time-of-day congestion, and a signal dwell. The
   approach is the corrected one from scripts/generate_city_dataset.py; the
   difference is that the graph here comes from real OSM junction coordinates
   rather than a synthetic grid, and the load path is bulk SQL rather than HTTP.

2. Vehicles have *behaviour*, not just identity.
   A dataset where 500k vehicles each drive every day is both wrong and
   unusably large. Each vehicle gets a home junction, a small set of
   destinations that define its corridor, and an activity profile: commuters
   run 18-24 of 30 days, occasional vehicles 2-6, and a thin tail of
   commercial/taxi vehicles run nearly daily with many trips. The population is
   therefore dominated by rarely-seen vehicles, which is also what real
   fixed-camera coverage looks like.

3. Loading goes through raw executemany, not the ORM or HTTP.
   12M ORM objects is hours and many GB of RAM; 12M rows over
   POST /events/bulk-ingest is worse, because each one runs a tracking lookup
   and an alert check. Rows are streamed from a generator into
   `executemany` batches on a single connection with load-tuned PRAGMAs, and
   the indexes are built afterwards. The live feeder
   (scripts/live_event_feeder.py) is what exercises the real ingestion path.

Tradeoffs worth knowing
-----------------------
* Bulk-loaded rows bypass tracking_service and alert_service. global_vehicle_id
  is assigned deterministically instead (see GLOBAL_ID_PREFIX), and historical
  BLACKLIST alerts are backfilled from an indexed lookup at the end. Live
  alerting is the feeder's job.
* Rows are inserted vehicle-major (all of one vehicle's month, then the next),
  not in global timestamp order. Sorting 12M rows by time would cost minutes
  and ~2GB of RAM for no query benefit, since every read path goes through an
  index. Side benefit: a plate's whole trajectory lands on contiguous pages.
* Stored timestamps are true naive UTC, aligned to real wall-clock "now", so
  the dashboard's "last hour" windows work. The diurnal *shape* is placed in
  Delhi local time, so peaks appear at 08-11 and 17-21 IST — i.e. at 02:30-05:30
  and 11:30-15:30 UTC in the stored values. The verification histogram prints
  both clocks.

Usage
-----
  .venv/bin/python scripts/generate_delhi_dataset.py                  # full demo scale
  .venv/bin/python scripts/generate_delhi_dataset.py --vehicles 50000 --target-events 800000
  .venv/bin/python scripts/generate_delhi_dataset.py --db-url sqlite:///./scratch.db
  .venv/bin/python scripts/generate_delhi_dataset.py --skip-verify    # load only

NOTE: tests/test_pipeline.py has an autouse fixture that deletes every row in
vehicle_events against the same dev.db. Run the test suite BEFORE generating,
or it will wipe the dataset.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

MANIFEST = BASE_DIR / "deployments" / "delhi" / "cameras.json"
DEPLOYMENT_TAG = "delhi"

# IST is UTC+5:30. Stored timestamps are UTC (so "last hour" queries against
# datetime.utcnow() behave), but the rush-hour shape belongs to local time.
IST_OFFSET = 5 * 3600 + 1800

# Bulk-loaded global_vehicle_ids are prefixed so they can never collide with
# tracking_service's own "VEH_%06d" counter, which restarts from 1 in every
# fresh process. Without the prefix, the first brand-new plate the live feeder
# or the REST API sees would be handed VEH_000001 — an id already owned by a
# different bulk vehicle — and trajectory-by-global-id would silently merge two
# unrelated cars.
GLOBAL_ID_PREFIX = "VEH_D"

# ── Graph shape ──────────────────────────────────────────────────────────────
KNN_NEIGHBOURS = 5
KNN_MAX_KM = 9.0  # beyond this two signals are not a single road segment

# ── Fleet composition ────────────────────────────────────────────────────────
# Sampled first, so the global mix is exact rather than an emergent side effect
# of the profile mix.
VEHICLE_TYPES = ("car", "motorcycle", "auto", "bus", "truck", "taxi")
VEHICLE_TYPE_WEIGHTS = (0.55, 0.25, 0.08, 0.04, 0.05, 0.03)

# Free-flow speed as a fraction of the posted limit, and an absolute ceiling.
# The ceiling is what keeps a habitual speeder on the one motorway camera
# (120 km/h limit) from being simulated at 200 km/h — which would read as
# broken data rather than as dangerous driving.
TYPE_SPEED_FACTOR = {
    "car": 1.00, "motorcycle": 1.05, "auto": 0.75,
    "bus": 0.72, "truck": 0.70, "taxi": 1.00,
}
TYPE_SPEED_CAP = {
    "car": 135.0, "motorcycle": 105.0, "auto": 65.0,
    "bus": 85.0, "truck": 90.0, "taxi": 120.0,
}

VEHICLE_COLORS = ("white", "silver", "grey", "black", "red", "blue", "brown", "green")
VEHICLE_COLOR_WEIGHTS = (0.31, 0.19, 0.14, 0.13, 0.08, 0.07, 0.05, 0.03)

# P(vehicle of this type gets each behaviour profile). Tuned so the population
# is ~94% rarely-seen vehicles, which is both realistic for fixed-camera
# coverage and what keeps the event budget achievable — a fleet of 500k daily
# commuters would be ~180M events.
PROFILE_BY_TYPE = {
    "car":        (("commuter", 0.055), ("occasional", 1.0)),
    "motorcycle": (("commuter", 0.030), ("occasional", 1.0)),
    "taxi":       (("fleet", 0.150), ("commuter", 0.200), ("occasional", 1.0)),
    "auto":       (("fleet", 0.060), ("occasional", 1.0)),
    "bus":        (("commercial", 0.050), ("occasional", 1.0)),
    "truck":      (("commercial", 0.060), ("occasional", 1.0)),
}


@dataclass(frozen=True)
class Profile:
    """
    days_lo/hi   — active days out of the month (nobody drives past every camera
                   every day; this is the main knob on realism *and* volume)
    trips_lo/hi  — trips on an active day
    hops_lo/hi   — camera hops per trip before the vehicle leaves the covered
                   corridor
    dests        — how many recurring destinations define the corridor
    weekend      — P(active) multiplier on Sat/Sun
    """
    days_lo: int
    days_hi: int
    trips_lo: int
    trips_hi: int
    hops_lo: int
    hops_hi: int
    dests: int
    weekend: float


PROFILES = {
    # 18-24 of 30 days, out-and-back on a fixed corridor: the classic commuter.
    "commuter":   Profile(18, 24, 2, 3, 3, 8, 1, 0.35),
    # Taxis and autos: not quite daily, but many short trips when they run.
    "fleet":      Profile(20, 28, 3, 6, 3, 8, 3, 0.85),
    # Goods vehicles and buses: nearly every day, long routes, high mileage.
    "commercial": Profile(26, 30, 3, 6, 5, 11, 3, 0.75),
    # The long tail — a car that crosses a monitored junction a handful of
    # times a month. ~94% of the fleet.
    "occasional": Profile(2, 6, 1, 2, 2, 5, 2, 1.0),
}

# ── Rhythm ───────────────────────────────────────────────────────────────────
# Trip departure weights by IST hour-of-day. Delhi's evening peak is broader
# and heavier than its morning peak.
HOUR_WEIGHTS = (
    0.28, 0.18, 0.13, 0.12, 0.18, 0.35, 0.65, 1.15,
    1.75, 1.95, 1.55, 1.25, 1.15, 1.10, 1.05, 1.15,
    1.35, 1.75, 2.00, 1.85, 1.40, 1.00, 0.70, 0.45,
)
# Goods vehicles are barred from much of Delhi during both peaks, so their
# departures pile up either side of the restriction windows.
TRUCK_HOUR_WEIGHTS = (
    2.00, 2.00, 1.90, 1.70, 1.50, 1.20, 0.80, 0.15,
    0.10, 0.10, 0.12, 0.50, 0.70, 0.70, 0.70, 0.70,
    0.50, 0.12, 0.10, 0.10, 0.12, 0.90, 1.60, 1.90,
)
# Realised speed as a fraction of free-flow, by IST hour. This is the single
# biggest driver of how the data *looks*: it produces 14 km/h evening-peak
# segments and 60 km/h 3am segments on the same road.
CONGESTION = (
    1.00, 1.02, 1.05, 1.05, 1.00, 0.95, 0.85, 0.62,
    0.45, 0.40, 0.48, 0.60, 0.68, 0.70, 0.70, 0.66,
    0.58, 0.44, 0.36, 0.38, 0.50, 0.68, 0.82, 0.92,
)
WEEKEND_VOLUME = {5: 0.75, 6: 0.55}  # Sat, Sun — Mon=0

# Not every vehicle passing a junction is read. Missing a hop is realistic
# (occlusion, plate angle, no camera on that approach) and it is also the
# honest stress test for the trajectory logic: a gap has to imply a plausible
# two-hop speed, not a teleport.
DETECTION_RATE = 0.85

SPEEDER_RATE = 0.03  # habitual speeders, recorded as ground truth
SPEEDER_BIAS = (1.35, 1.75)
NORMAL_BIAS = (0.82, 1.08)

# Above this gap, two consecutive sightings are separate journeys, not a leg of
# one trip. Shared with the RoadUsage rollup so "parked overnight" never lands
# in avg_travel_minutes.
MAX_SEGMENT_GAP_MINUTES = 180

POI_PLATES = [f"DL01CP{i:04d}" for i in range(1, 11)]
POI_REASONS = [
    "Stolen vehicle — FIR 214/2026, Vasant Vihar PS",
    "Suspect vehicle — inter-state theft ring",
    "Wanted — hit and run, Ring Road",
    "Stolen vehicle — FIR 907/2026, Rohini PS",
    "Suspect vehicle — narcotics surveillance",
    "Wanted — chain snatching, South Delhi",
    "Stolen commercial vehicle — cargo theft",
    "Suspect vehicle — extortion case, Shahdara",
    "Wanted — vehicle used in armed robbery",
    "Stolen vehicle — FIR 1188/2026, Dwarka PS",
]

INSERT_EVENT_SQL = (
    "INSERT INTO vehicle_events "
    "(event_id, camera_id, local_track_id, timestamp, plate, plate_confidence, "
    " latitude, longitude, direction, vehicle_type, vehicle_color, speed, "
    " global_vehicle_id, created_at) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)


# ─────────────────────────────────────────────────────────────────────────────
# Camera network
# ─────────────────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class CameraNet:
    """
    The camera manifest turned into an adjacency graph plus flat, index-addressed
    arrays.

    Everything in the hot loop addresses cameras by integer index into parallel
    lists rather than by camera_id into dicts. At 12M events even a dict lookup
    per field is measurable, and the string ids are only needed at row-build
    time.
    """

    def __init__(self, cameras: list[dict]):
        self.cameras = cameras
        self.n = len(cameras)
        self.ids = [c["camera_id"] for c in cameras]
        self.lat = [c["latitude"] for c in cameras]
        self.lon = [c["longitude"] for c in cameras]
        self.direction = [c.get("direction") or "NORTH" for c in cameras]
        self.limit = [float(c.get("speed_limit_kmh") or 60.0) for c in cameras]
        self.road = [c.get("road") or c["name"] for c in cameras]
        self.road_class = [c.get("road_class") or "secondary" for c in cameras]

        # adj[i] -> list of (neighbour_index, distance_km)
        self.adj: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        self._build_knn()
        self._connect_components()
        self.next_hop = self._build_next_hop()

        # Trunk-ish junctions, used to home trucks onto the corridors they are
        # actually allowed on.
        self.freight_nodes = [
            i for i in range(self.n)
            if self.road_class[i] in ("motorway", "trunk", "primary")
        ] or list(range(self.n))

    def _link(self, a: int, b: int, dist: float) -> None:
        if any(n == b for n, _ in self.adj[a]):
            return
        self.adj[a].append((b, dist))
        self.adj[b].append((a, dist))

    def _build_knn(self) -> None:
        """
        k-nearest-neighbour edges. Two traffic signals a couple of km apart with
        nothing between them are, for ANPR purposes, the two ends of one road
        segment — which is exactly the edge we want. The KNN_MAX_KM cut stops
        the sparse outer-NCT cameras from being wired to each other across
        half the city.
        """
        n, lat, lon = self.n, self.lat, self.lon
        for i in range(n):
            dists = sorted(
                ((haversine_km(lat[i], lon[i], lat[j], lon[j]), j) for j in range(n) if j != i)
            )
            linked = 0
            for d, j in dists:
                if linked >= KNN_NEIGHBOURS:
                    break
                if d > KNN_MAX_KM and linked >= 2:
                    break  # always keep at least 2 edges so no node is stranded
                self._link(i, j, d)
                linked += 1

    def _connect_components(self) -> None:
        """
        A disconnected graph would silently strand whole neighbourhoods: routes
        into them would be impossible and their cameras would report almost no
        traffic. Stitch components together by their closest cross-component
        pair until one component remains.
        """
        parent = list(range(self.n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(self.n):
            for j, _ in self.adj[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

        while True:
            groups: dict[int, list[int]] = {}
            for i in range(self.n):
                groups.setdefault(find(i), []).append(i)
            if len(groups) <= 1:
                self.components_joined = getattr(self, "components_joined", 0)
                return
            keys = list(groups)
            base = groups[keys[0]]
            others = [i for k in keys[1:] for i in groups[k]]
            best = min(
                ((haversine_km(self.lat[a], self.lon[a], self.lat[b], self.lon[b]), a, b)
                 for a in base for b in others)
            )
            self._link(best[1], best[2], best[0])
            parent[find(best[1])] = find(best[2])
            self.components_joined = getattr(self, "components_joined", 0) + 1

    def _build_next_hop(self) -> list[list[int]]:
        """
        next_hop[target][here] = the neighbour of `here` on the shortest
        (distance-weighted) path to `target`.

        One Dijkstra per target over 200 nodes is instant and turns routing in
        the hot loop into a single list index. Without it, 500k vehicles x many
        trips of path-finding would dominate runtime, and random walks instead
        of routed trips would destroy the corridor structure the OD matrix and
        next-hop prediction are supposed to discover.
        """
        n = self.n
        matrix: list[list[int]] = []
        for target in range(n):
            dist = [math.inf] * n
            nxt = [-1] * n
            dist[target] = 0.0
            pq = [(0.0, target)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, w in self.adj[u]:
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        nxt[v] = u  # graph is undirected, so parent == next hop
                        heapq.heappush(pq, (nd, v))
            matrix.append(nxt)
        return matrix

    def edge_count(self) -> int:
        return sum(len(a) for a in self.adj) // 2


def load_cameras(limit: int | None = None) -> list[dict]:
    if not MANIFEST.exists():
        raise SystemExit(f"Camera manifest not found: {MANIFEST}")
    cams = json.loads(MANIFEST.read_text())
    if limit:
        cams = cams[:limit]
    return cams


# ─────────────────────────────────────────────────────────────────────────────
# Plates
# ─────────────────────────────────────────────────────────────────────────────

# DL is the bulk of Delhi traffic; the NCR states supply the daily commuter
# inflow. Stored uppercase with no separators to match the normaliser in
# kafka_consumer.process_event_payload and the /events schema validator.
STATE_WEIGHTS = (("DL", 0.62), ("HR", 0.14), ("UP", 0.14), ("RJ", 0.06), ("PB", 0.04))
STATE_RTO_MAX = {"DL": 13, "HR": 99, "UP": 96, "RJ": 58, "PB": 92}
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted, as on real plates


def make_plates(count: int, rng: random.Random) -> list[str]:
    """
    `count` unique plates in the real Indian format: state code, RTO number,
    a one or two letter series, then four digits (e.g. DL1CAB1234 -> DL1CAB1234).

    Uniqueness by rejection against a set rather than by construction: the
    keyspace is ~80M for DL alone, so collisions are rare and the set costs a
    few tens of MB for the whole fleet — cheap next to guaranteeing that
    "unique vehicles" in the dashboard is a true count.
    """
    states, cum = [], []
    acc = 0.0
    for code, w in STATE_WEIGHTS:
        acc += w
        states.append(code)
        cum.append(acc)

    seen: set[str] = set()
    out: list[str] = []
    rand = rng.random
    randint = rng.randint
    while len(out) < count:
        r = rand()
        si = 0
        while r > cum[si]:
            si += 1
        code = states[si]
        rto = randint(1, STATE_RTO_MAX[code])
        series = LETTERS[randint(0, 23)]
        if rand() < 0.75:  # most modern plates carry a two-letter series
            series += LETTERS[randint(0, 23)]
        plate = f"{code}{rto:02d}{series}{randint(1000, 9999)}"
        if plate not in seen:
            seen.add(plate)
            out.append(plate)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour simulation
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_picker(weights, size: int = 2048) -> list:
    """
    Flatten a weight vector into a lookup table so sampling in the hot loop is
    one random index instead of a random.choices() call (which rebuilds
    cumulative weights and allocates a list every time).
    """
    total = sum(weights)
    table: list[int] = []
    for idx, w in enumerate(weights):
        table.extend([idx] * max(1, round(w / total * size)))
    return table


@dataclass(slots=True)
class Vehicle:
    """slots=True is not cosmetic here: 500k instances with a __dict__ each is
    ~400MB of population state before a single event is generated."""

    plate: str
    global_id: str
    vtype: str
    color: str
    profile: str
    home: int
    dests: tuple[int, ...]
    bias: float
    is_speeder: bool


def assign_vehicles(
    plates: list[str], net: CameraNet, rng: random.Random
) -> tuple[list[Vehicle], set[str]]:
    """
    Give every plate a home junction, a corridor, a behaviour profile and a
    driving style. Done up front (not lazily) because the live feeder and the
    verification pass both want the same population, and 500k lightweight
    objects is ~250MB — affordable, unlike 12M event dicts.
    """
    type_picker = _weighted_picker(VEHICLE_TYPE_WEIGHTS)
    color_picker = _weighted_picker(VEHICLE_COLOR_WEIGHTS)
    rand = rng.random
    randint = rng.randint
    uniform = rng.uniform

    n = net.n
    freight = net.freight_nodes

    # Home junctions are weighted towards the busier road classes so the
    # heatmap has real hotspots instead of a uniform wash of 200 equal dots.
    home_weights = [
        {"motorway": 1.0, "trunk": 2.6, "primary": 2.2, "secondary": 1.4, "tertiary": 0.7}
        .get(net.road_class[i], 1.0)
        for i in range(n)
    ]
    home_picker = _weighted_picker(home_weights, size=4096)

    vehicles: list[Vehicle] = []
    speeders: set[str] = set()

    for i, plate in enumerate(plates):
        vtype = VEHICLE_TYPES[type_picker[randint(0, len(type_picker) - 1)]]
        # The ten persons-of-interest are forced onto the highest-mileage
        # profile: a watchlist plate that only shows up twice in a month gives
        # the demo nothing to point at.
        if plate in POI_PLATES:
            vtype = "car" if i % 3 else "motorcycle"
            profile = "commercial"
        else:
            profile = "occasional"
            r = rand()
            for name, threshold in PROFILE_BY_TYPE[vtype]:
                if r < threshold:
                    profile = name
                    break

        pool = freight if vtype == "truck" else None
        if pool:
            home = pool[randint(0, len(pool) - 1)]
        else:
            home = home_picker[randint(0, len(home_picker) - 1)]

        cfg = PROFILES[profile]
        dests = []
        for _ in range(cfg.dests):
            d = pool[randint(0, len(pool) - 1)] if pool else randint(0, n - 1)
            if d != home:
                dests.append(d)
        if not dests:
            dests.append((home + 1) % n)

        is_speeder = rand() < SPEEDER_RATE
        if is_speeder:
            speeders.add(plate)
        lo, hi = SPEEDER_BIAS if is_speeder else NORMAL_BIAS

        vehicles.append(Vehicle(
            plate=plate,
            global_id=f"{GLOBAL_ID_PREFIX}{i:07d}",
            vtype=vtype,
            color=VEHICLE_COLORS[color_picker[randint(0, len(color_picker) - 1)]],
            profile=profile,
            home=home,
            dests=tuple(dests),
            bias=uniform(lo, hi),
            is_speeder=is_speeder,
        ))
    return vehicles, speeders


class Simulator:
    """
    Turns a Vehicle into a chronologically ordered stream of
    (camera_index, utc_epoch_seconds, speed_kmh) sightings for the whole month.

    The vehicle's position threads from trip to trip: a trip starts wherever the
    last one ended. Restarting each trip from an unrelated junction would
    manufacture teleports between trips that have nothing to do with real
    speeding and would swamp the alert counts with noise — the same bug the
    reference generator calls out.
    """

    def __init__(self, net: CameraNet, days: int, end_epoch: int, rng: random.Random,
                 trip_scale: float = 1.0):
        self.net = net
        self.days = days
        self.end_epoch = end_epoch
        self.rng = rng
        self.trip_scale = trip_scale

        self.hour_picker = _weighted_picker(HOUR_WEIGHTS)
        self.truck_hour_picker = _weighted_picker(TRUCK_HOUR_WEIGHTS)

        # Signal dwell bounds per hour, derived from congestion: a junction that
        # is moving at 36% of free-flow is also the one where you sit through
        # three cycles of the light.
        self.dwell_lo = [3.0 + 22.0 * (1.0 - c) for c in CONGESTION]
        self.dwell_hi = [12.0 + 115.0 * (1.0 - c) for c in CONGESTION]

        # Day 0 starts `days-1` local midnights back, so the window ends at
        # "now" partway through today.
        ist_now = end_epoch + IST_OFFSET
        self.day0_ist_midnight = ((ist_now // 86400) - (days - 1)) * 86400
        self.dow0 = int((self.day0_ist_midnight // 86400 + 3) % 7)  # epoch day 0 = Thursday

    def active_days(self, cfg: Profile, rand, randint) -> list[int]:
        """
        Pick which days of the month this vehicle actually drives. Weekend days
        are dropped with profile-specific probability, which is what produces
        the weekly rhythm — commuters nearly vanish on Sunday, autos barely
        notice.
        """
        want = randint(cfg.days_lo, min(cfg.days_hi, self.days))
        pool = list(range(self.days))
        self.rng.shuffle(pool)
        chosen = []
        for d in pool:
            if len(chosen) >= want:
                break
            dow = (self.dow0 + d) % 7
            if dow in WEEKEND_VOLUME:
                if rand() > cfg.weekend * WEEKEND_VOLUME[dow] / 0.75:
                    continue
            chosen.append(d)
        if not chosen:
            chosen = [randint(0, self.days - 1)]
        return sorted(chosen)

    def simulate(self, veh: Vehicle):
        net = self.net
        rand = self.rng.random
        randint = self.rng.randint
        cfg = PROFILES[veh.profile]

        lat, lon, limit, adj, next_hop = net.lat, net.lon, net.limit, net.adj, net.next_hop
        type_factor = TYPE_SPEED_FACTOR[veh.vtype]
        type_cap = TYPE_SPEED_CAP[veh.vtype]
        free_flow_bias = veh.bias * type_factor
        hour_picker = self.truck_hour_picker if veh.vtype == "truck" else self.hour_picker
        hp_max = len(hour_picker) - 1
        dwell_lo, dwell_hi = self.dwell_lo, self.dwell_hi
        end_epoch = self.end_epoch
        scale = self.trip_scale

        current = veh.home
        targets = (veh.home,) + veh.dests
        first_trip = True
        emitted = 0
        # End of the previous trip, in IST seconds. A vehicle cannot begin its
        # next trip before it finished the last one — departure times are drawn
        # from the diurnal distribution independently, so two of them can easily
        # land 20 minutes apart when the first trip takes an hour. Left
        # unclamped, the vehicle jumps backwards in time between trips and the
        # implied camera-to-camera speed for that pair goes to six figures.
        # This single clamp is the difference between "physically coherent" and
        # the 15,000 km/h class of bug.
        prev_trip_end = 0

        for day in self.active_days(cfg, rand, randint):
            ist_midnight = self.day0_ist_midnight + day * 86400
            dow = (self.dow0 + day) % 7
            volume = WEEKEND_VOLUME.get(dow, 1.0)

            want = randint(cfg.trips_lo, cfg.trips_hi) * scale * volume
            trips = int(want) + (1 if rand() < want - int(want) else 0)
            if first_trip:
                trips = max(1, trips)  # every vehicle is seen at least once

            # Departure times sorted so the vehicle's day runs forward in time.
            departures = sorted(
                ist_midnight + hour_picker[randint(0, hp_max)] * 3600 + randint(0, 3599)
                for _ in range(trips)
            )

            for depart_ist in departures:
                first_trip = False
                # Prefer going somewhere other than where we already are.
                target = targets[randint(0, len(targets) - 1)]
                if target == current:
                    target = targets[(randint(0, len(targets) - 1) + 1) % len(targets)]
                if target == current:
                    continue

                hops = randint(cfg.hops_lo, cfg.hops_hi)
                t_ist = depart_ist
                if t_ist < prev_trip_end:
                    # Parked for 5-50 minutes, then set off again.
                    t_ist = prev_trip_end + 300 + randint(0, 2700)
                node = current

                for hop in range(hops + 1):
                    t_utc = t_ist - IST_OFFSET
                    if t_utc > end_epoch:
                        break
                    hour = (t_ist % 86400) // 3600
                    cong = CONGESTION[hour]

                    nxt = next_hop[target][node]
                    if nxt < 0 or node == target:
                        # Arrived (or unreachable): emit the arrival sighting and
                        # end the trip here.
                        speed = _hop_speed(limit[node], cong, free_flow_bias, type_cap, rand)
                        if rand() < DETECTION_RATE:
                            emitted += 1
                            yield node, t_utc, speed
                        break

                    # A fraction of hops wander off the shortest path — real
                    # drivers detour, and a perfectly optimal fleet would make
                    # the OD matrix suspiciously clean.
                    if rand() < 0.12:
                        cand = adj[node]
                        nxt = cand[randint(0, len(cand) - 1)][0]

                    dist = 0.0
                    for v, w in adj[node]:
                        if v == nxt:
                            dist = w
                            break
                    if dist <= 0.0:
                        dist = haversine_km(lat[node], lon[node], lat[nxt], lon[nxt])

                    hop_limit = limit[node] if limit[node] < limit[nxt] else limit[nxt]
                    speed = _hop_speed(hop_limit, cong, free_flow_bias, type_cap, rand)

                    if rand() < DETECTION_RATE:
                        emitted += 1
                        yield node, t_utc, speed

                    dwell = dwell_lo[hour] + (dwell_hi[hour] - dwell_lo[hour]) * rand()
                    # At least one whole second per hop. Two manifest junctions
                    # can be 10m apart (the OSM thinning keeps both approaches
                    # of some intersections), and a sub-second hop would divide
                    # by ~zero in every implied-speed calculation downstream.
                    step = int(dist / speed * 3600.0 + dwell)
                    t_ist += step if step > 0 else 1
                    node = nxt

                current = node
                prev_trip_end = t_ist

        if not emitted:
            # Every plate has to appear at least once, or the dashboard's
            # "unique vehicles" figure silently undershoots --vehicles. A
            # vehicle whose only trip fell past the end of the window, or whose
            # only sightings were dropped by the detection filter, gets one
            # sighting at its home junction.
            t_utc = self.end_epoch - randint(3600, self.days * 86400 - 1)
            hour = ((t_utc + IST_OFFSET) % 86400) // 3600
            yield veh.home, t_utc, _hop_speed(
                limit[veh.home], CONGESTION[hour], free_flow_bias, type_cap, rand)


def _hop_speed(limit_kmh: float, congestion: float, bias: float, cap: float, rand) -> float:
    """
    Realised speed on one hop.

    Congestion is applied *before* the driver's bias, so a habitual speeder
    stuck in the 18:00 peak still crawls — which is both true and the reason
    speed violations concentrate at night in the generated data.
    """
    v = limit_kmh * congestion * bias * (0.90 + 0.20 * rand())
    if v > cap:
        v = cap
    elif v < 6.0:
        v = 6.0
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Row streaming
# ─────────────────────────────────────────────────────────────────────────────

def build_time_tables(start_epoch: int, end_epoch: int) -> tuple[dict, list, list]:
    """
    Timestamp strings, precomputed.

    SQLAlchemy's SQLite DateTime is text ("YYYY-MM-DD HH:MM:SS.ffffff"), so the
    loader has to produce that string 12M times. datetime.strftime costs ~1.5us
    a call — nearly a minute of pure formatting. Splitting it into a per-day
    table (31 entries) and a per-second-of-day table (86,400 entries, ~6MB)
    makes each row a two-string concatenation instead, at ~0.15us.
    """
    day_str = {}
    d = start_epoch // 86400
    while d <= end_epoch // 86400 + 1:
        day_str[d] = datetime.fromtimestamp(d * 86400, timezone.utc).strftime("%Y-%m-%d ")
        d += 1
    tod = [f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}" for s in range(86400)]
    # Sub-second offset, fixed *per vehicle* rather than per event. Real
    # detections aren't aligned to whole seconds, but varying the fraction
    # between two sightings of the same vehicle is actively dangerous: a hop
    # from x.999999 to x+1.000000 is a 1-microsecond gap, which over 3km reads
    # as 10^10 km/h. A constant per-vehicle phase keeps every within-vehicle
    # delta an exact whole number of seconds while still scattering timestamps
    # across the second.
    micro = [f".{(k * 7813) % 1000000:06d}" for k in range(128)]
    return day_str, tod, micro


def iter_rows(vehicles, sim: Simulator, net: CameraNet, start_epoch: int, end_epoch: int,
              progress_every: int = 1_000_000):
    """
    Stream (14-tuple) rows for executemany. A generator, not a list: 12M tuples
    materialised would be ~6GB of Python objects.
    """
    day_str, tod, micro = build_time_tables(start_epoch, end_epoch)
    ids, lats, lons, dirs = net.ids, net.lat, net.lon, net.direction
    # Per-camera local track counters — a real single-camera tracker hands out
    # incrementing ids, and this is what local_track_id means.
    track = [0] * net.n
    # Confidence table instead of round(uniform(...), 2) per row.
    conf = [round(0.82 + 0.17 * (k / 255.0), 2) for k in range(256)]

    n = 0
    t0 = time.perf_counter()
    for vi, veh in enumerate(vehicles):
        plate = veh.plate
        gid = veh.global_id
        vtype = veh.vtype
        color = veh.color
        mic = micro[vi & 127]
        for cam, epoch, speed in sim.simulate(veh):
            n += 1
            track[cam] += 1
            yield (
                f"d{n:09d}",
                ids[cam],
                str(track[cam]),
                day_str[epoch // 86400] + tod[epoch % 86400] + mic,
                plate,
                conf[n & 255],
                lats[cam],
                lons[cam],
                dirs[cam],
                vtype,
                color,
                round(speed, 1),
                gid,
                None,  # created_at patched in by the loader (constant per run)
            )
            if n % progress_every == 0:
                el = time.perf_counter() - t0
                print(f"    generated {n:>12,} events  ({n / el:>9,.0f} ev/s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Bulk load
# ─────────────────────────────────────────────────────────────────────────────

LOAD_PRAGMAS = (
    "PRAGMA journal_mode=WAL",       # kept for the demo: feeder writes, API reads
    "PRAGMA synchronous=OFF",        # a crash mid-load just means re-running the generator
    "PRAGMA temp_store=MEMORY",
    "PRAGMA cache_size=-200000",     # ~200MB page cache
    "PRAGMA foreign_keys=OFF",       # camera FK is satisfied by construction
)

# Built after the load, not during: maintaining four B-trees while inserting 12M
# rows roughly triples load time and fragments every index.
POST_LOAD_INDEXES = (
    # Leftmost prefix serves the plain `WHERE plate = ?` lookup, the trailing
    # columns make the trajectory query's ORDER BY free and let the RoadUsage
    # window scan run straight off the index with no 12M-row sort. One index,
    # three jobs.
    ("ix_vehicle_events_plate_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_vehicle_events_plate_timestamp "
     "ON vehicle_events (plate, timestamp, camera_id)"),
    ("ix_vehicle_events_camera_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_vehicle_events_camera_timestamp "
     "ON vehicle_events (camera_id, timestamp)"),
    ("ix_vehicle_events_timestamp",
     "CREATE INDEX IF NOT EXISTS ix_vehicle_events_timestamp "
     "ON vehicle_events (timestamp)"),
    ("ix_vehicle_events_global_vehicle_id",
     "CREATE INDEX IF NOT EXISTS ix_vehicle_events_global_vehicle_id "
     "ON vehicle_events (global_vehicle_id)"),
)


def open_raw(engine):
    """The DBAPI connection under SQLAlchemy, in autocommit so BEGIN/COMMIT are explicit."""
    raw = engine.raw_connection()
    dbapi = raw.driver_connection
    dbapi.isolation_level = None
    return raw, dbapi


def load_events(engine, row_iter, batch_size: int, created_at: str) -> tuple[int, float]:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    for p in LOAD_PRAGMAS:
        cur.execute(p)

    # Indexes are dropped rather than kept: rebuilding them from scratch on the
    # finished table is far cheaper than 12M incremental B-tree inserts.
    for name, _ in POST_LOAD_INDEXES:
        cur.execute(f"DROP INDEX IF EXISTS {name}")

    total = 0
    t0 = time.perf_counter()
    batch: list[tuple] = []
    append = batch.append
    try:
        cur.execute("BEGIN")
        for row in row_iter:
            append(row[:13] + (created_at,))
            if len(batch) >= batch_size:
                cur.executemany(INSERT_EVENT_SQL, batch)
                total += len(batch)
                batch.clear()
                cur.execute("COMMIT")
                cur.execute("BEGIN")
        if batch:
            cur.executemany(INSERT_EVENT_SQL, batch)
            total += len(batch)
        cur.execute("COMMIT")
    finally:
        cur.close()
        raw.close()
    return total, time.perf_counter() - t0


def create_indexes(engine) -> list[tuple[str, float]]:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    for p in LOAD_PRAGMAS:
        cur.execute(p)
    # Index builds sort the whole table. temp_store=MEMORY would try to hold a
    # multi-GB sorter in RAM; spilling to disk is the safer trade at this scale.
    cur.execute("PRAGMA temp_store=FILE")
    timings = []
    for name, ddl in POST_LOAD_INDEXES:
        t0 = time.perf_counter()
        cur.execute(ddl)
        timings.append((name, time.perf_counter() - t0))
        print(f"    {name:<44} {timings[-1][1]:>7.1f}s", flush=True)
    cur.execute("ANALYZE")
    cur.close()
    raw.close()
    return timings


def upsert_cameras(engine, cameras: list[dict], deactivate_others: bool) -> None:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    rows = [
        (c["camera_id"], c["name"], c["location"], c["latitude"], c["longitude"],
         c.get("road"), c.get("direction"), c.get("camera_type", "ANPR"),
         c.get("deployment", DEPLOYMENT_TAG), float(c.get("speed_limit_kmh") or 60.0), 1, now)
        for c in cameras
    ]
    cur.execute("BEGIN")
    cur.executemany(
        "INSERT OR REPLACE INTO cameras (camera_id, name, location, latitude, longitude, "
        "road, direction, camera_type, deployment, speed_limit_kmh, is_active, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    if deactivate_others:
        # Leftover cameras from earlier synthetic runs otherwise show up on the
        # Delhi map as dead pins in Bengaluru with zero traffic.
        cur.execute("UPDATE cameras SET is_active = 0 WHERE deployment != ?", (DEPLOYMENT_TAG,))
    cur.execute("COMMIT")
    cur.close()
    raw.close()


def reset_tables(engine, reset_alerts: bool) -> None:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    for p in LOAD_PRAGMAS:
        cur.execute(p)
    cur.execute("BEGIN")
    # Bare DELETE hits SQLite's truncate optimisation, so this is fast even on
    # a 12M-row table.
    for table in ("vehicle_events", "road_usage", "camera_hourly",
                  "camera_totals", "dataset_kpi"):
        cur.execute(f"DELETE FROM {table}")
    if reset_alerts:
        cur.execute("DELETE FROM alerts")
    cur.execute("COMMIT")
    cur.close()
    raw.close()


def seed_blacklist(engine) -> None:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    cur.execute("BEGIN")
    cur.executemany(
        "INSERT OR REPLACE INTO blacklist (plate, reason, added_at) VALUES (?,?,?)",
        [(p, POI_REASONS[i % len(POI_REASONS)], now) for i, p in enumerate(POI_PLATES)],
    )
    cur.execute("COMMIT")
    cur.close()
    raw.close()


def backfill_poi_alerts(engine, per_plate_cap: int = 400) -> int:
    """
    Historical BLACKLIST alerts for the watchlist plates.

    The bulk path deliberately skips alert_service (it is a per-event DB round
    trip), but an alerts panel that is empty until the live feeder starts looks
    broken. These are derived from the loaded sightings via the plate index, so
    they are real hits on real rows rather than invented ones. Capped per plate
    because a high-mileage POI generates thousands of sightings a month and the
    demo only needs a populated, believable feed.
    """
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    inserted = 0
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    cur.execute("BEGIN")
    for i, plate in enumerate(POI_PLATES):
        reason = POI_REASONS[i % len(POI_REASONS)]
        rows = cur.execute(
            "SELECT event_id, camera_id, timestamp FROM vehicle_events "
            "WHERE plate = ? ORDER BY timestamp DESC LIMIT ?",
            (plate, per_plate_cap),
        ).fetchall()
        cur.executemany(
            "INSERT OR REPLACE INTO alerts (alert_id, vehicle_id, camera_id, alert_type, "
            "description, status, timestamp, created_at) VALUES (?,?,?,?,?,?,?,?)",
            [(f"bl_{plate}_{eid}", plate, cam, "BLACKLIST",
              f"Blacklisted plate {plate} detected. Reason: {reason}",
              "ACTIVE", ts, now) for eid, cam, ts in rows],
        )
        inserted += len(rows)
    cur.execute("COMMIT")
    cur.close()
    raw.close()
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# Aggregates
# ─────────────────────────────────────────────────────────────────────────────

def build_aggregates(engine, net: CameraNet) -> dict:
    """
    Populate road_usage / camera_hourly / camera_totals / dataset_kpi with SQL
    GROUP BYs over the loaded events.

    Deliberately not a Python loop over 12M rows: SQLite's aggregation runs in
    C over the table it already has cached, and the only data that crosses into
    Python is the few thousand already-reduced rows.
    """
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    for p in LOAD_PRAGMAS:
        cur.execute(p)
    cur.execute("PRAGMA temp_store=FILE")
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    timings: dict = {}

    # ── camera_hourly ────────────────────────────────────────────────────────
    # substr() on the stored text is the cheapest possible hour truncation —
    # strftime()/datetime() would re-parse every one of 12M timestamps. The
    # result string is already the canonical storage format, so SQLAlchemy
    # reads it straight back as a datetime.
    t0 = time.perf_counter()
    cur.execute("BEGIN")
    cur.execute(
        "INSERT INTO camera_hourly (camera_id, hour_bucket, vehicle_count, avg_speed, unique_vehicles) "
        "SELECT camera_id, substr(timestamp, 1, 13) || ':00:00', "
        "       COUNT(*), AVG(speed), COUNT(DISTINCT plate) "
        "FROM vehicle_events "
        "GROUP BY camera_id, substr(timestamp, 1, 13)"
    )
    cur.execute("COMMIT")
    timings["camera_hourly"] = time.perf_counter() - t0

    # ── camera_totals ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cur.execute("BEGIN")
    cur.execute(
        "INSERT INTO camera_totals (camera_id, vehicle_count, unique_vehicles, avg_speed, "
        "  first_seen, last_seen, road, latitude, longitude, peak_hour, peak_hour_count, computed_at) "
        "SELECT e.camera_id, COUNT(*), COUNT(DISTINCT e.plate), AVG(e.speed), "
        "       MIN(e.timestamp), MAX(e.timestamp), c.road, c.latitude, c.longitude, NULL, 0, ? "
        "FROM vehicle_events e JOIN cameras c ON c.camera_id = e.camera_id "
        "GROUP BY e.camera_id",
        (now,),
    )
    # Peak hour comes off camera_hourly (144k rows), not the raw events — same
    # answer, one thousandth of the work.
    cur.execute(
        "WITH hod AS ("
        "  SELECT camera_id, CAST(strftime('%H', hour_bucket) AS INTEGER) AS hod, "
        "         SUM(vehicle_count) AS cnt FROM camera_hourly GROUP BY camera_id, hod"
        "), best AS ("
        "  SELECT camera_id, hod, cnt, "
        "         ROW_NUMBER() OVER (PARTITION BY camera_id ORDER BY cnt DESC) AS rn FROM hod"
        ") "
        "UPDATE camera_totals SET peak_hour = best.hod, peak_hour_count = best.cnt "
        "FROM best WHERE best.camera_id = camera_totals.camera_id AND best.rn = 1"
    )
    cur.execute("COMMIT")
    timings["camera_totals"] = time.perf_counter() - t0

    # ── road_usage ───────────────────────────────────────────────────────────
    # LAG over (plate, timestamp) reconstructs every observed trip leg. With the
    # (plate, timestamp, camera_id) index this is a covering index scan, so the
    # window function needs no sort of the 12M rows.
    t0 = time.perf_counter()
    legs = cur.execute(
        "WITH legs AS ("
        "  SELECT LAG(camera_id) OVER w AS from_cam, camera_id AS to_cam, "
        "         (julianday(timestamp) - julianday(LAG(timestamp) OVER w)) * 1440.0 AS mins "
        "  FROM vehicle_events WHERE plate IS NOT NULL "
        "  WINDOW w AS (PARTITION BY plate ORDER BY timestamp)"
        ") "
        "SELECT from_cam, to_cam, COUNT(*), AVG(mins), MIN(mins), MAX(mins) FROM legs "
        f"WHERE from_cam IS NOT NULL AND from_cam <> to_cam AND mins > 0 AND mins <= {MAX_SEGMENT_GAP_MINUTES} "
        "GROUP BY from_cam, to_cam"
    ).fetchall()
    timings["road_usage_scan"] = time.perf_counter() - t0

    idx = {cid: i for i, cid in enumerate(net.ids)}
    lat, lon, road = net.lat, net.lon, net.road
    ru_rows = []
    max_implied = 0.0
    implied_speeds = []
    for from_cam, to_cam, cnt, avg_min, min_min, max_min in legs:
        a, b = idx.get(from_cam), idx.get(to_cam)
        if a is None or b is None:
            continue
        dist = haversine_km(lat[a], lon[a], lat[b], lon[b])
        avg_speed = dist / (avg_min / 60.0) if avg_min else None
        # Fastest observed traversal of this segment: the tightest physical
        # plausibility check available, and it costs one extra MIN() above.
        fastest = dist / (min_min / 60.0) if min_min else None
        if fastest:
            max_implied = max(max_implied, fastest)
            implied_speeds.append(fastest)
        ru_rows.append((
            from_cam, to_cam, cnt,
            round(avg_min, 3) if avg_min else None,
            round(avg_speed, 2) if avg_speed else None,
            round(fastest, 2) if fastest else None,
            round(dist, 4), road[a], road[b],
            f"{road[a]} → {road[b]}",
            round((lat[a] + lat[b]) / 2, 6), round((lon[a] + lon[b]) / 2, 6), now,
        ))

    t0 = time.perf_counter()
    cur.execute("BEGIN")
    cur.executemany(
        "INSERT OR REPLACE INTO road_usage (from_camera_id, to_camera_id, trip_count, "
        "  avg_travel_minutes, avg_speed_kmh, max_speed_kmh, distance_km, from_road, to_road, "
        "  road_label, mid_latitude, mid_longitude, computed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ru_rows,
    )
    cur.execute("COMMIT")
    timings["road_usage_write"] = time.perf_counter() - t0

    # ── dataset_kpi ──────────────────────────────────────────────────────────
    # total_events / avg_speed roll up from camera_totals (200 rows). The
    # distinct-plate count is served as an index-only scan of the plate index,
    # which is the one thing /analytics/summary cannot avoid on its own.
    t0 = time.perf_counter()
    total_events, weighted_speed, first_seen, last_seen = cur.execute(
        "SELECT SUM(vehicle_count), SUM(avg_speed * vehicle_count) / SUM(vehicle_count), "
        "       MIN(first_seen), MAX(last_seen) FROM camera_totals"
    ).fetchone()
    unique_vehicles = cur.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT plate FROM vehicle_events WHERE plate IS NOT NULL)"
    ).fetchone()[0]
    camera_count = cur.execute("SELECT COUNT(*) FROM cameras WHERE is_active = 1").fetchone()[0]
    cur.execute("BEGIN")
    cur.execute(
        "INSERT OR REPLACE INTO dataset_kpi (scope, total_events, unique_vehicles, camera_count, "
        "  avg_speed_kmh, segment_count, first_event_at, last_event_at, computed_at) "
        "VALUES ('global',?,?,?,?,?,?,?,?)",
        (total_events, unique_vehicles, camera_count,
         round(weighted_speed, 2) if weighted_speed else None,
         len(ru_rows), first_seen, last_seen, now),
    )
    cur.execute("COMMIT")
    timings["dataset_kpi"] = time.perf_counter() - t0

    cur.close()
    raw.close()

    implied_speeds.sort()
    return {
        "timings": timings,
        "segments": len(ru_rows),
        "total_events": total_events,
        "unique_vehicles": unique_vehicles,
        "max_implied_kmh": max_implied,
        "implied_speeds": implied_speeds,
        "legs_counted": sum(r[2] for r in ru_rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = min(len(sorted_vals) - 1, max(0, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def verify(engine, net: CameraNet, agg: dict, sample_plates: list[str]) -> None:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    cur.execute("PRAGMA cache_size=-200000")
    bar = lambda v, mx, w=52: "#" * max(0, int(round(v / mx * w))) if mx else ""

    print("\n" + "=" * 84)
    print("VERIFICATION")
    print("=" * 84)

    # (a) Physical plausibility of consecutive-sighting speeds.
    speeds = agg["implied_speeds"]
    print("\n[a] Implied camera-to-camera speed across consecutive sightings")
    print(f"    directed segments: {agg['segments']:,}   trip legs: {agg['legs_counted']:,}")
    print("    per-segment FASTEST observed traversal (the worst case that exists):")
    for p in (50, 75, 90, 99, 100):
        print(f"      p{p:<3} {percentile(speeds, p):>8.1f} km/h")
    print("    per-segment AVERAGE traversal speed:")
    avg_speeds = sorted(r[0] for r in cur.execute(
        "SELECT avg_speed_kmh FROM road_usage WHERE avg_speed_kmh IS NOT NULL"))
    for p in (5, 25, 50, 75, 95):
        print(f"      p{p:<3} {percentile(avg_speeds, p):>8.1f} km/h")
    worst = cur.execute(
        "SELECT road_label, max_speed_kmh, distance_km, trip_count FROM road_usage "
        "ORDER BY max_speed_kmh DESC LIMIT 3").fetchall()
    for label, mx, dist, cnt in worst:
        print(f"      fastest segment: {mx:>6.1f} km/h over {dist:.2f} km  ({cnt:,} legs)  {label[:46]}")
    verdict = "PLAUSIBLE" if agg["max_implied_kmh"] < 160 else "*** IMPLAUSIBLE ***"
    print(f"    global max implied speed: {agg['max_implied_kmh']:.1f} km/h  -> {verdict}")

    # An independent per-row check, not derived from the aggregates, over a
    # sample of vehicles — in case the rollup's own filters were hiding
    # something.
    total_pairs = 0
    over_150 = 0
    row_max = 0.0
    idx = {cid: i for i, cid in enumerate(net.ids)}
    for plate in sample_plates[:3000]:
        rows = cur.execute(
            "SELECT camera_id, timestamp FROM vehicle_events WHERE plate = ? ORDER BY timestamp",
            (plate,)).fetchall()
        prev = None
        for cam, ts in rows:
            t = datetime.fromisoformat(ts)
            if prev and prev[0] != cam:
                dt_h = (t - prev[1]).total_seconds() / 3600.0
                if 0 < dt_h <= MAX_SEGMENT_GAP_MINUTES / 60.0:
                    a, b = idx[prev[0]], idx[cam]
                    v = haversine_km(net.lat[a], net.lon[a], net.lat[b], net.lon[b]) / dt_h
                    total_pairs += 1
                    row_max = max(row_max, v)
                    if v > 150:
                        over_150 += 1
            prev = (cam, t)
    print(f"    independent per-row check over {len(sample_plates[:3000]):,} sampled vehicles: "
          f"{total_pairs:,} legs, max {row_max:.1f} km/h, "
          f"{over_150} above MAX_PLAUSIBLE_SPEED_KMH(150)")

    # (b) One vehicle's trajectory, chronological and geographically contiguous.
    probe = cur.execute(
        "SELECT plate FROM vehicle_events WHERE plate = ? LIMIT 1", (POI_PLATES[0],)).fetchone()
    probe_plate = probe[0] if probe else sample_plates[0]
    traj = cur.execute(
        "SELECT camera_id, timestamp, latitude, longitude, speed FROM vehicle_events "
        "WHERE plate = ? ORDER BY timestamp LIMIT 14", (probe_plate,)).fetchall()
    print(f"\n[b] Trajectory of {probe_plate} (first 14 sightings)")
    prev = None
    ordered = True
    for cam, ts, la, lo, sp in traj:
        t = datetime.fromisoformat(ts)
        if prev:
            if t < prev[1]:
                ordered = False
            gap_km = haversine_km(prev[2], prev[3], la, lo)
            gap_min = (t - prev[1]).total_seconds() / 60.0
            implied = gap_km / (gap_min / 60.0) if gap_min > 0 else 0.0
            print(f"    {ts[:19]}  {cam:<26} {sp:>5.1f} km/h  "
                  f"| +{gap_km:5.2f} km in {gap_min:7.2f} min -> {implied:6.1f} km/h implied")
        else:
            print(f"    {ts[:19]}  {cam:<26} {sp:>5.1f} km/h  | trip start")
        prev = (cam, t, la, lo)
    print(f"    chronologically ordered: {ordered}")

    # (c) Diurnal curve. Read off camera_hourly, which is the point of having it.
    print("\n[c] Events per hour-of-day (from camera_hourly)")
    hod = cur.execute(
        "SELECT CAST(strftime('%H', hour_bucket) AS INTEGER) h, SUM(vehicle_count) c "
        "FROM camera_hourly GROUP BY h ORDER BY h").fetchall()
    mx = max(c for _, c in hod) if hod else 1
    print("     UTC   IST    events")
    for h, c in hod:
        ist_h = (h * 60 + 330) // 60 % 24
        ist_m = (h * 60 + 330) % 60
        print(f"    {h:02d}:00 {ist_h:02d}:{ist_m:02d} {c:>10,}  {bar(c, mx)}")

    print("\n    Events per day-of-week (weekly rhythm)")
    dow = cur.execute(
        "SELECT CAST(strftime('%w', hour_bucket) AS INTEGER) d, SUM(vehicle_count) c "
        "FROM camera_hourly GROUP BY d ORDER BY d").fetchall()
    names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    mx = max(c for _, c in dow) if dow else 1
    for d, c in dow:
        print(f"    {names[d]} {c:>10,}  {bar(c, mx, 40)}")

    # (d) Reconciliation — an aggregate that disagrees with its source is worse
    #     than no aggregate at all.
    print("\n[d] Aggregate reconciliation")
    raw_count = cur.execute("SELECT COUNT(*) FROM vehicle_events").fetchone()[0]
    hourly_sum, hourly_rows = cur.execute(
        "SELECT SUM(vehicle_count), COUNT(*) FROM camera_hourly").fetchone()
    totals_sum, totals_rows = cur.execute(
        "SELECT SUM(vehicle_count), COUNT(*) FROM camera_totals").fetchone()
    kpi = cur.execute(
        "SELECT total_events, unique_vehicles, camera_count, avg_speed_kmh, segment_count "
        "FROM dataset_kpi WHERE scope='global'").fetchone()
    ru_rows, ru_trips = cur.execute(
        "SELECT COUNT(*), SUM(trip_count) FROM road_usage").fetchone()
    print(f"    vehicle_events COUNT(*)      {raw_count:>14,}")
    print(f"    camera_hourly  SUM(count)    {hourly_sum:>14,}   ({hourly_rows:,} buckets)"
          f"   match={hourly_sum == raw_count}")
    print(f"    camera_totals  SUM(count)    {totals_sum:>14,}   ({totals_rows:,} cameras)"
          f"   match={totals_sum == raw_count}")
    print(f"    dataset_kpi    total_events  {kpi[0]:>14,}   match={kpi[0] == raw_count}")
    print(f"    dataset_kpi    unique_vehicles {kpi[1]:>12,}   avg_speed={kpi[3]} km/h")
    print(f"    road_usage     {ru_rows:,} segments, {ru_trips:,} trip legs "
          f"({ru_trips / raw_count * 100:.1f}% of events are a leg end)")
    busiest = cur.execute(
        "SELECT road_label, trip_count, avg_travel_minutes, avg_speed_kmh FROM road_usage "
        "ORDER BY trip_count DESC LIMIT 5").fetchall()
    print("    top road segments by usage:")
    for label, cnt, mins, spd in busiest:
        print(f"      {cnt:>8,} legs  {mins:>6.1f} min  {spd:>5.1f} km/h  {label[:52]}")

    cur.close()
    raw.close()


DEMO_QUERIES = (
    ("plate trajectory lookup (indexed point query)",
     "SELECT event_id, camera_id, timestamp, latitude, longitude, speed "
     "FROM vehicle_events WHERE plate = :plate ORDER BY timestamp"),
    ("summary KPIs (from dataset_kpi)",
     "SELECT total_events, unique_vehicles, camera_count, avg_speed_kmh FROM dataset_kpi "
     "WHERE scope = 'global'"),
    ("per-camera density, lifetime (from camera_totals)",
     "SELECT camera_id, vehicle_count, unique_vehicles, avg_speed, peak_hour "
     "FROM camera_totals ORDER BY vehicle_count DESC"),
    ("per-camera density, last 24h (from camera_hourly)",
     "SELECT camera_id, SUM(vehicle_count), AVG(avg_speed) FROM camera_hourly "
     "WHERE hour_bucket >= :since GROUP BY camera_id"),
    ("road-usage top 20 (from road_usage)",
     "SELECT road_label, trip_count, avg_travel_minutes, avg_speed_kmh, "
     "mid_latitude, mid_longitude FROM road_usage ORDER BY trip_count DESC LIMIT 20"),
    ("city time series, hourly, last 7d (from camera_hourly)",
     "SELECT hour_bucket, SUM(vehicle_count) FROM camera_hourly "
     "WHERE hour_bucket >= :week GROUP BY hour_bucket ORDER BY hour_bucket"),
    ("recent alerts feed",
     "SELECT alert_id, vehicle_id, camera_id, alert_type, timestamp FROM alerts "
     "ORDER BY timestamp DESC LIMIT 50"),
    ("raw events at one camera, last hour (indexed range)",
     "SELECT COUNT(*), AVG(speed) FROM vehicle_events "
     "WHERE camera_id = :cam AND timestamp >= :hour"),
)


def time_demo_queries(engine, plate: str, camera_id: str) -> None:
    raw, conn = open_raw(engine)
    cur = conn.cursor()
    cur.execute("PRAGMA cache_size=-200000")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    params = {
        "plate": plate,
        "cam": camera_id,
        "since": (now - timedelta(hours=24)).isoformat(sep=" "),
        "week": (now - timedelta(days=7)).isoformat(sep=" "),
        "hour": (now - timedelta(hours=1)).isoformat(sep=" "),
    }

    print("\n" + "=" * 84)
    print("DEMO QUERY LATENCIES (cold-ish, single connection)")
    print("=" * 84)
    for label, sql in DEMO_QUERIES:
        bind = []
        stmt = sql
        for key in ("plate", "since", "week", "cam", "hour"):
            token = f":{key}"
            while token in stmt:
                stmt = stmt.replace(token, "?", 1)
                bind.append(params[key])
        best = math.inf
        rows = 0
        for _ in range(3):
            t0 = time.perf_counter()
            rows = len(cur.execute(stmt, bind).fetchall())
            best = min(best, time.perf_counter() - t0)
        flag = "  <-- SLOW" if best > 1.0 else ""
        print(f"    {best * 1000:>9.2f} ms  {rows:>8,} rows   {label}{flag}")

    print("\n  EXPLAIN QUERY PLAN — plate lookup must use the index, not scan 12M rows:")
    for row in cur.execute(
        "EXPLAIN QUERY PLAN SELECT event_id, camera_id, timestamp FROM vehicle_events "
        "WHERE plate = ? ORDER BY timestamp", ("DL01CP0001",)):
        print(f"    {row[-1]}")
    print("\n  EXPLAIN QUERY PLAN — road_usage top-N:")
    for row in cur.execute(
        "EXPLAIN QUERY PLAN SELECT road_label FROM road_usage ORDER BY trip_count DESC LIMIT 20"):
        print(f"    {row[-1]}")
    cur.close()
    raw.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def db_size_bytes(db_url: str) -> tuple[str, int]:
    if not db_url.startswith("sqlite"):
        return db_url, 0
    path = db_url.split("///", 1)[-1]
    p = Path(path) if Path(path).is_absolute() else (BASE_DIR / path)
    total = 0
    for suffix in ("", "-wal", "-shm"):
        f = Path(str(p) + suffix)
        if f.exists():
            total += f.stat().st_size
    return str(p), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicles", type=int, default=500_000,
                    help="unique vehicles / plates (default: 500,000)")
    ap.add_argument("--days", type=int, default=30, help="days of history ending now")
    ap.add_argument("--target-events", type=int, default=12_000_000,
                    help="event budget; trips are scaled to hit it (0 = no scaling)")
    ap.add_argument("--cameras", type=int, default=None,
                    help="use only the first N cameras from the manifest (debug)")
    ap.add_argument("--batch-size", type=int, default=50_000, help="executemany batch size")
    ap.add_argument("--db-url", default=None,
                    help="override DATABASE_URL, e.g. sqlite:///./scratch.db")
    ap.add_argument("--seed", type=int, default=1947)
    ap.add_argument("--no-reset", action="store_true",
                    help="append instead of clearing vehicle_events + aggregates first")
    ap.add_argument("--reset-alerts", action="store_true",
                    help="also clear the alerts table (stale alerts from earlier runs)")
    ap.add_argument("--keep-other-cameras", action="store_true",
                    help="leave non-delhi cameras active")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--ground-truth-out", default=str(BASE_DIR / "deployments" / "delhi" / "ground_truth.json"),
                    help="where to write the blacklist/speeder ground truth ('' to skip)")
    args = ap.parse_args()

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    from sqlalchemy import create_engine

    from backend.config import get_settings
    from backend.database import Base
    from backend.models import analytics_agg, alert, camera, trajectory, traffic_snapshot, vehicle_event  # noqa: F401

    get_settings.cache_clear()
    db_url = args.db_url or get_settings().DATABASE_URL
    engine = create_engine(db_url, connect_args={"check_same_thread": False}
                           if "sqlite" in db_url else {})

    wall0 = time.perf_counter()
    print("=" * 84)
    print("DELHI CITY-SCALE ANPR DATASET GENERATOR")
    print("=" * 84)
    print(f"  database      {db_url}")
    print(f"  vehicles      {args.vehicles:,}")
    print(f"  history       {args.days} days ending now")
    print(f"  event budget  {args.target_events:,}")

    rng = random.Random(args.seed)

    # ── 1. Network ───────────────────────────────────────────────────────────
    print("\n[1/7] Building camera network from real OSM junctions...")
    t0 = time.perf_counter()
    cams = load_cameras(args.cameras)
    net = CameraNet(cams)
    lat_span = (min(net.lat), max(net.lat))
    lon_span = (min(net.lon), max(net.lon))
    print(f"  {net.n} cameras, {net.edge_count()} road segments "
          f"(k={KNN_NEIGHBOURS} NN, <={KNN_MAX_KM}km), "
          f"{getattr(net, 'components_joined', 0)} component stitch(es)")
    print(f"  coverage lat {lat_span[0]:.3f}-{lat_span[1]:.3f}, "
          f"lon {lon_span[0]:.3f}-{lon_span[1]:.3f}")
    seg_dists = sorted(d for i in range(net.n) for _, d in net.adj[i])
    print(f"  segment length: median {percentile(seg_dists, 50):.2f} km, "
          f"p95 {percentile(seg_dists, 95):.2f} km, max {seg_dists[-1]:.2f} km")
    print(f"  built in {time.perf_counter() - t0:.2f}s")

    # ── 2. Schema + reset ────────────────────────────────────────────────────
    print("\n[2/7] Ensuring schema and clearing previous dataset...")
    Base.metadata.create_all(bind=engine)
    if not args.no_reset:
        reset_tables(engine, args.reset_alerts)
        print("  cleared vehicle_events, road_usage, camera_hourly, camera_totals, dataset_kpi"
              + (", alerts" if args.reset_alerts else ""))
    upsert_cameras(engine, cams, not args.keep_other_cameras)
    seed_blacklist(engine)
    print(f"  {net.n} cameras upserted, {len(POI_PLATES)} POI plates blacklisted")

    # ── 3. Population ────────────────────────────────────────────────────────
    print(f"\n[3/7] Assigning behaviour to {args.vehicles:,} vehicles...")
    t0 = time.perf_counter()
    plates = make_plates(args.vehicles, rng)
    plates[:len(POI_PLATES)] = POI_PLATES  # guarantee the watchlist is simulated
    vehicles, speeders = assign_vehicles(plates, net, rng)
    from collections import Counter
    prof_mix = Counter(v.profile for v in vehicles)
    type_mix = Counter(v.vtype for v in vehicles)
    print(f"  {len(vehicles):,} vehicles in {time.perf_counter() - t0:.1f}s")
    print("  profile mix: " + "  ".join(
        f"{k}={v:,}({v / len(vehicles) * 100:.1f}%)" for k, v in prof_mix.most_common()))
    print("  vehicle mix: " + "  ".join(
        f"{k}={v / len(vehicles) * 100:.1f}%" for k, v in type_mix.most_common()))
    print(f"  habitual speeders: {len(speeders):,} ({len(speeders) / len(vehicles) * 100:.1f}%)")

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end_epoch = int(now.replace(tzinfo=timezone.utc).timestamp())
    start_epoch = end_epoch - args.days * 86400

    # ── 4. Calibrate the trip budget ─────────────────────────────────────────
    # Rather than hand-tuning profile constants against a target, measure the
    # profile mix's real yield on a small sample and scale trips per day. The
    # sample uses the same simulator, so the estimate tracks any later change to
    # the behaviour model automatically.
    trip_scale = 1.0
    if args.target_events > 0:
        print("\n[4/7] Calibrating trip volume against the event budget...")
        t0 = time.perf_counter()
        sample_n = max(1000, min(4000, len(vehicles) // 100))
        probe_sim = Simulator(net, args.days, end_epoch, random.Random(args.seed + 1))
        step = max(1, len(vehicles) // sample_n)
        sample = vehicles[::step][:sample_n]
        sampled_events = sum(1 for v in sample for _ in probe_sim.simulate(v))
        projected = sampled_events / len(sample) * len(vehicles)
        trip_scale = min(4.0, max(0.05, args.target_events / projected)) if projected else 1.0
        print(f"  sampled {len(sample):,} vehicles -> {sampled_events:,} events "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"  unscaled projection {projected / 1e6:.2f}M events -> trip_scale {trip_scale:.3f}")
    else:
        print("\n[4/7] Trip budget scaling disabled (--target-events 0)")

    # ── 5. Generate + bulk load ──────────────────────────────────────────────
    print(f"\n[5/7] Generating and bulk-loading events (batch {args.batch_size:,})...")
    sim = Simulator(net, args.days, end_epoch, rng, trip_scale)
    created_at = now.isoformat(sep=" ")
    rows = iter_rows(vehicles, sim, net, start_epoch, end_epoch)
    total, load_secs = load_events(engine, rows, args.batch_size, created_at)
    print(f"  loaded {total:,} events in {load_secs:.1f}s "
          f"({total / load_secs:,.0f} rows/sec)")

    print("\n  Creating indexes on the finished table...")
    t0 = time.perf_counter()
    create_indexes(engine)
    index_secs = time.perf_counter() - t0
    print(f"  indexes + ANALYZE in {index_secs:.1f}s")

    poi_alerts = backfill_poi_alerts(engine)
    print(f"  backfilled {poi_alerts:,} historical BLACKLIST alerts for {len(POI_PLATES)} POI plates")

    # ── 6. Aggregates ────────────────────────────────────────────────────────
    print("\n[6/7] Building precomputed analytics aggregates (SQL GROUP BY)...")
    agg = build_aggregates(engine, net)
    for k, v in agg["timings"].items():
        print(f"    {k:<20} {v:>7.1f}s")
    print(f"  road_usage: {agg['segments']:,} directed segments  |  "
          f"camera_hourly + camera_totals + dataset_kpi populated")

    # WAL checkpoint so the reported file size is the real on-disk footprint.
    raw, conn = open_raw(engine)
    conn.cursor().execute("PRAGMA wal_checkpoint(TRUNCATE)")
    raw.close()
    db_path, size = db_size_bytes(db_url)

    # ── 7. Verify ────────────────────────────────────────────────────────────
    if not args.skip_verify:
        print("\n[7/7] Verifying...")
        sample_plates = [v.plate for v in vehicles[::max(1, len(vehicles) // 3000)]]
        verify(engine, net, agg, sample_plates)
        busiest_cam = net.ids[0]
        raw, conn = open_raw(engine)
        row = conn.cursor().execute(
            "SELECT camera_id FROM camera_totals ORDER BY vehicle_count DESC LIMIT 1").fetchone()
        raw.close()
        if row:
            busiest_cam = row[0]
        time_demo_queries(engine, POI_PLATES[0], busiest_cam)
    else:
        print("\n[7/7] Verification skipped (--skip-verify)")

    # ── Ground truth ─────────────────────────────────────────────────────────
    speeder_list = sorted(speeders)
    if args.ground_truth_out:
        Path(args.ground_truth_out).write_text(json.dumps({
            "generated_at": now.isoformat(),
            "seed": args.seed,
            "blacklist_poi": [{"plate": p, "reason": POI_REASONS[i % len(POI_REASONS)]}
                              for i, p in enumerate(POI_PLATES)],
            "habitual_speeder_count": len(speeder_list),
            "habitual_speeders": speeder_list,
        }, indent=2))

    wall = time.perf_counter() - wall0
    print("\n" + "=" * 84)
    print("COMPLETE")
    print("=" * 84)
    print(f"  cameras            {net.n} (real OSM junctions, deployment='{DEPLOYMENT_TAG}')")
    print(f"  unique vehicles    {len(vehicles):,}")
    print(f"  vehicle_events     {total:,}")
    print(f"  history            {args.days} days, "
          f"{datetime.fromtimestamp(start_epoch, timezone.utc):%Y-%m-%d %H:%M} to "
          f"{now:%Y-%m-%d %H:%M} UTC")
    print(f"  load throughput    {total / load_secs:,.0f} rows/sec "
          f"({load_secs:.1f}s load + {index_secs:.1f}s indexes)")
    print(f"  total wall time    {wall / 60:.1f} min")
    print(f"  database           {db_path}  {size / 1e9:.2f} GB")
    print(f"\n  GROUND TRUTH — blacklisted persons of interest (live alerts fire on these):")
    for i, p in enumerate(POI_PLATES):
        print(f"    {p}   {POI_REASONS[i % len(POI_REASONS)]}")
    print(f"\n  GROUND TRUTH — habitual speeders: {len(speeder_list):,} plates, e.g.")
    print("    " + "  ".join(speeder_list[:8]))
    if args.ground_truth_out:
        print(f"    full list: {args.ground_truth_out}")
    print("\n  Next: .venv/bin/python scripts/live_event_feeder.py --events-per-sec 40")


if __name__ == "__main__":
    main()

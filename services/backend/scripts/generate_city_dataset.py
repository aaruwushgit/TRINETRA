"""
City-Wide Synthetic ANPR Dataset Generator.

Builds a realistic camera *road network* (not just a random bag of cameras)
across a synthetic city, then simulates thousands of vehicles making real
multi-hop trips through it, and streams the resulting detections into the
live backend through its real public API (POST /cameras, POST /alerts/
blacklist, POST /events/bulk-ingest) — the same endpoints a real camera
worker would call.

This deliberately excludes any actual license-plate/vehicle *detection* —
there is no video, no YOLO, no OCR here. It only generates the structured
event records (plate, camera, timestamp, GPS, speed, confidence) that a
real ANPR pipeline would have produced, so the rest of the platform
(tracking, trajectory, alerts, analytics, ML prediction) can be exercised
and load-tested at city scale.

Why a graph instead of independent random events per camera:
  Earlier scale tests (scripts/test_analytics_big_data.py) assigned each
  event an independent random timestamp within a 12h window. Two sightings
  of the same plate at cameras 17km apart could land a few seconds apart,
  implying a ~15,000 km/h "speed" — which broke the speed-defaulters logic
  and hides the real question this project needs to answer (do consecutive
  camera sightings, timestamps and speeds compose into a physically
  plausible trajectory?). Here, timestamps are derived by walking a real
  camera adjacency graph edge by edge, each timestamp computed forward
  from the previous one using that edge's distance and the vehicle's
  simulated speed on it.

Usage:
  python scripts/generate_city_dataset.py                    # defaults: 64 cameras, 4000 vehicles
  python scripts/generate_city_dataset.py --cameras 80 --vehicles 6000
  python scripts/generate_city_dataset.py --backend-url http://localhost:8000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

CITY_CENTER_LAT = 12.9716
CITY_CENTER_LON = 77.5946  # Bengaluru, arbitrary synthetic-city anchor
DEPLOYMENT_TAG = "citywide_demo"

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck", "auto"]
VEHICLE_CLASS_WEIGHTS = [0.58, 0.22, 0.07, 0.07, 0.06]

ROAD_TYPES = {
    # name -> (speed_limit_kmh, weight)
    "residential": (40.0, 0.55),
    "arterial": (60.0, 0.30),
    "ring_road": (80.0, 0.15),
}


@dataclass
class CameraNode:
    camera_id: str
    row: int
    col: int
    latitude: float
    longitude: float
    road: str
    road_type: str
    speed_limit_kmh: float
    direction: str
    neighbors: list[str] = field(default_factory=list)


def build_city_graph(rows: int, cols: int) -> dict[str, CameraNode]:
    """
    Grid-of-junctions road network. Each node is a camera at an
    intersection; edges are the road segments between adjacent
    intersections (4-connected grid) plus sparse diagonal "ring road"
    shortcuts so trips aren't perfectly rectilinear.
    """
    step_deg = 0.008  # ~0.9 km per grid step
    nodes: dict[str, CameraNode] = {}

    rng = random.Random(42)  # deterministic city layout

    for r in range(rows):
        for c in range(cols):
            cam_id = f"CAM_{r:02d}_{c:02d}"
            is_perimeter = r in (0, rows - 1) or c in (0, cols - 1)
            road_type = "ring_road" if is_perimeter else rng.choices(
                list(ROAD_TYPES.keys()), weights=[w for _, w in ROAD_TYPES.values()]
            )[0]
            speed_limit = ROAD_TYPES[road_type][0]
            nodes[cam_id] = CameraNode(
                camera_id=cam_id,
                row=r,
                col=c,
                latitude=round(CITY_CENTER_LAT + (r - rows / 2) * step_deg, 6),
                longitude=round(CITY_CENTER_LON + (c - cols / 2) * step_deg, 6),
                road=f"{'Ring Road' if is_perimeter else 'Corridor'} {r}-{c}",
                road_type=road_type,
                speed_limit_kmh=speed_limit,
                direction=rng.choice(["NORTH", "SOUTH", "EAST", "WEST"]),
            )

    # 4-connected grid edges
    def link(a: str, b: str):
        nodes[a].neighbors.append(b)
        nodes[b].neighbors.append(a)

    for r in range(rows):
        for c in range(cols - 1):
            link(f"CAM_{r:02d}_{c:02d}", f"CAM_{r:02d}_{c+1:02d}")
    for r in range(rows - 1):
        for c in range(cols):
            link(f"CAM_{r:02d}_{c:02d}", f"CAM_{r+1:02d}_{c:02d}")

    # Sparse diagonal shortcuts to break up the pure grid (~1 per 6 nodes)
    all_ids = list(nodes.keys())
    for cam_id in all_ids:
        if rng.random() < 0.15:
            node = nodes[cam_id]
            dr, dc = rng.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
            tr, tc = node.row + dr, node.col + dc
            target = f"CAM_{tr:02d}_{tc:02d}"
            if target in nodes and target not in node.neighbors:
                link(cam_id, target)

    return nodes


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math
    r = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass
class SimEvent:
    camera_id: str
    timestamp: datetime
    plate: str
    plate_confidence: float
    latitude: float
    longitude: float
    direction: str
    vehicle_type: str
    speed: float


def simulate_trips(
    graph: dict[str, CameraNode],
    num_vehicles: int,
    days: int,
    defaulter_rate: float,
    rng: random.Random,
) -> tuple[list[SimEvent], set[str], set[str]]:
    """
    Walk `num_vehicles` random routes through the camera graph, producing
    one SimEvent per camera hop with a timestamp derived from the previous
    hop's distance/speed — never an independently-random timestamp.

    Returns (events, defaulter_plates, blacklist_plates).
    """
    node_ids = list(graph.keys())
    events: list[SimEvent] = []
    defaulter_plates: set[str] = set()

    # A handful of "hero" vehicles that are pre-registered on the blacklist
    # so the alert pipeline has something guaranteed to fire on.
    blacklist_plates = {f"KA51HERO{i:03d}" for i in range(1, 9)}

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window_start = now - timedelta(days=days)

    all_plates = [f"KA{rng.randint(1,99):02d}CD{i:04d}" for i in range(1, num_vehicles + 1)]
    all_plates[: len(blacklist_plates)] = list(blacklist_plates)  # guarantee they get simulated

    for plate in all_plates:
        is_defaulter = rng.random() < defaulter_rate
        if is_defaulter:
            defaulter_plates.add(plate)
        speed_factor = rng.uniform(1.25, 1.7) if is_defaulter else rng.uniform(0.6, 0.95)
        vehicle_type = rng.choices(VEHICLE_CLASSES, weights=VEHICLE_CLASS_WEIGHTS)[0]

        # Generate trip start times up front and simulate them in chronological
        # order, threading the vehicle's location from one trip to the next.
        # Otherwise each trip would start from an unrelated random node —
        # manufacturing "teleports" between trips that have nothing to do
        # with real speeding and would swamp the alert numbers with noise.
        num_trips = rng.randint(1, 4)
        trip_starts = sorted(
            window_start + timedelta(seconds=rng.randint(0, days * 86400))
            for _ in range(num_trips)
        )
        home_node = rng.choice(node_ids)
        current = home_node

        for trip_start in trip_starts:
            hops = rng.randint(3, 9)
            prev = None
            t = trip_start

            node = graph[current]
            events.append(SimEvent(
                camera_id=current,
                timestamp=t,
                plate=plate,
                plate_confidence=round(rng.uniform(0.85, 0.99), 2),
                latitude=node.latitude,
                longitude=node.longitude,
                direction=node.direction,
                vehicle_type=vehicle_type,
                speed=0.0,
            ))

            for _ in range(hops):
                node = graph[current]
                candidates = [n for n in node.neighbors if n != prev] or node.neighbors
                if not candidates:
                    break
                nxt = rng.choice(candidates)
                nxt_node = graph[nxt]

                dist_km = haversine_km(node.latitude, node.longitude, nxt_node.latitude, nxt_node.longitude)
                road_speed = min(node.speed_limit_kmh, nxt_node.speed_limit_kmh) * speed_factor
                road_speed = max(8.0, road_speed + rng.uniform(-5, 5))
                travel_hours = dist_km / road_speed
                dwell_seconds = rng.uniform(5, 45)  # intersection/signal delay
                t = t + timedelta(hours=travel_hours) + timedelta(seconds=dwell_seconds)

                events.append(SimEvent(
                    camera_id=nxt,
                    timestamp=t,
                    plate=plate,
                    plate_confidence=round(rng.uniform(0.85, 0.99), 2),
                    latitude=nxt_node.latitude,
                    longitude=nxt_node.longitude,
                    direction=nxt_node.direction,
                    vehicle_type=vehicle_type,
                    speed=round(road_speed, 1),
                ))

                prev, current = current, nxt

    events.sort(key=lambda e: e.timestamp)
    return events, defaulter_plates, blacklist_plates


def push_cameras(graph: dict[str, CameraNode], backend_url: str) -> float:
    t0 = time.perf_counter()
    session = requests.Session()
    for node in graph.values():
        payload = {
            "camera_id": node.camera_id,
            "name": f"{node.road} Junction Cam",
            "location": f"Grid ({node.row},{node.col})",
            "latitude": node.latitude,
            "longitude": node.longitude,
            "road": node.road,
            "direction": node.direction,
            "camera_type": "ANPR",
            "deployment": DEPLOYMENT_TAG,
            "speed_limit_kmh": node.speed_limit_kmh,
        }
        res = session.post(f"{backend_url}/cameras", json=payload, timeout=10)
        if res.status_code not in (201, 409):
            print(f"  ⚠️ Camera {node.camera_id} registration failed: {res.status_code} {res.text}")
    return time.perf_counter() - t0


def push_blacklist(plates: set[str], backend_url: str) -> None:
    session = requests.Session()
    for plate in plates:
        session.post(
            f"{backend_url}/alerts/blacklist",
            json={"plate": plate, "reason": "City dataset demo — POI watchlist"},
            timeout=10,
        )


def push_events(events: list[SimEvent], backend_url: str, batch_size: int = 500) -> tuple[float, int]:
    session = requests.Session()
    t0 = time.perf_counter()
    fired = 0
    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        payload = [
            {
                "camera_id": e.camera_id,
                "timestamp": e.timestamp.isoformat(),
                "plate": e.plate,
                "plate_confidence": e.plate_confidence,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "direction": e.direction,
                "vehicle_type": e.vehicle_type,
                "speed": e.speed or None,
            }
            for e in batch
        ]
        res = session.post(f"{backend_url}/events/bulk-ingest", json=payload, timeout=60)
        if res.status_code != 200:
            print(f"  ⚠️ Batch {i}-{i+len(batch)} failed: {res.status_code} {res.text[:200]}")
            continue
        fired += sum(1 for r in res.json() if r.get("alert_fired"))
        if (i // batch_size) % 10 == 0:
            print(f"    ...{i + len(batch)}/{len(events)} events ingested")
    return time.perf_counter() - t0, fired


def main():
    parser = argparse.ArgumentParser(description="Generate a city-wide synthetic ANPR dataset")
    parser.add_argument("--rows", type=int, default=8, help="Grid rows (rows*cols = camera count)")
    parser.add_argument("--cols", type=int, default=8, help="Grid cols (rows*cols = camera count)")
    parser.add_argument("--vehicles", type=int, default=4000, help="Unique vehicles to simulate")
    parser.add_argument("--days", type=int, default=3, help="Days of history to spread trips across")
    parser.add_argument("--defaulter-rate", type=float, default=0.03, help="Fraction of vehicles that are consistent speeders")
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("=" * 80)
    print(f"🏙️  CITY-WIDE SYNTHETIC ANPR DATASET GENERATOR")
    print("=" * 80)
    print(f"Grid: {args.rows}x{args.cols} = {args.rows * args.cols} cameras (road network, not random points)")

    graph = build_city_graph(args.rows, args.cols)
    print(f"✅ Built road network: {len(graph)} cameras, "
          f"{sum(len(n.neighbors) for n in graph.values()) // 2} road segments")

    print(f"\n[1] Registering {len(graph)} cameras with the backend ({args.backend_url})...")
    reg_time = push_cameras(graph, args.backend_url)
    print(f"  ✓ Done in {reg_time:.2f}s")

    print(f"\n[2] Simulating {args.vehicles} vehicles over {args.days} days "
          f"(defaulter rate: {args.defaulter_rate*100:.0f}%)...")
    t0 = time.perf_counter()
    events, defaulter_plates, blacklist_plates = simulate_trips(
        graph, args.vehicles, args.days, args.defaulter_rate, rng
    )
    print(f"  ✓ Generated {len(events)} physically-coherent detection events "
          f"in {time.perf_counter() - t0:.2f}s")
    print(f"  ✓ {len(defaulter_plates)} vehicles simulated as consistent speeders (ground truth)")
    print(f"  ✓ {len(blacklist_plates)} vehicles pre-registered on the blacklist (ground truth)")

    print(f"\n[3] Adding {len(blacklist_plates)} POI vehicles to the blacklist...")
    push_blacklist(blacklist_plates, args.backend_url)
    print("  ✓ Done")

    print(f"\n[4] Streaming {len(events)} events through POST /events/bulk-ingest "
          f"in batches of {args.batch_size} (chronological order, like real camera workers)...")
    ingest_time, alerts_fired = push_events(events, args.backend_url, args.batch_size)
    throughput = len(events) / ingest_time if ingest_time > 0 else 0
    print(f"  ✓ Ingested {len(events)} events in {ingest_time:.1f}s ({throughput:.0f} events/sec)")
    print(f"  ✓ {alerts_fired} alerts fired during ingestion (BLACKLIST + SPEED_VIOLATION + ROUTE_ANOMALY)")

    print("\n" + "=" * 80)
    print("✅ DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Cameras:      {len(graph)}")
    print(f"  Vehicles:     {args.vehicles}")
    print(f"  Events:       {len(events)}")
    print(f"  Ground-truth defaulters: {len(defaulter_plates)}")
    print(f"  Ground-truth blacklist:  {sorted(blacklist_plates)}")
    print("\nNext: run scripts/verify_city_dataset.py to benchmark analytics/defaulters/prediction on this data.")


if __name__ == "__main__":
    main()

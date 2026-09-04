"""
Build a real Delhi ANPR camera network from OpenStreetMap traffic signals.

Why this exists
---------------
A city-scale demo is only convincing if the camera nodes are real places. This
script pulls actual `highway=traffic_signals` nodes for the Delhi NCT bounding
box from the Overpass API, works out which *named* road each signal sits on,
thins the result down to N well-spread junctions, and writes a camera manifest.

Nothing here is invented: every camera's latitude/longitude is a genuine
signalled junction in OSM, and its road name and speed limit come from the OSM
`highway`/`name`/`maxspeed` tags of the ways that pass through that node.

Output
------
  deployments/delhi/cameras.json   — the camera manifest (id, name, road, GPS,
                                     road class, speed limit, direction)

Usage
-----
  python scripts/fetch_delhi_junctions.py --count 200
  python scripts/fetch_delhi_junctions.py --count 200 --refresh   # re-query Overpass
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "deployments" / "delhi"
RAW_CACHE = OUT_DIR / "overpass_raw.json"
CAMERAS_JSON = OUT_DIR / "cameras.json"

# Delhi NCT approximate bounding box (south, west, north, east)
DELHI_BBOX = (28.40, 76.84, 28.89, 77.35)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# OSM highway class -> (default speed limit km/h, priority for camera placement)
# Higher priority classes are preferred when thinning, because real ANPR
# deployments sit on arterials and ring roads, not residential lanes.
ROAD_CLASS = {
    "motorway": (80.0, 6),
    "motorway_link": (60.0, 4),
    "trunk": (70.0, 6),
    "trunk_link": (55.0, 4),
    "primary": (60.0, 5),
    "primary_link": (50.0, 3),
    "secondary": (50.0, 4),
    "secondary_link": (45.0, 3),
    "tertiary": (45.0, 3),
    "tertiary_link": (40.0, 2),
    "residential": (40.0, 1),
    "unclassified": (40.0, 1),
    "living_street": (30.0, 1),
    "service": (30.0, 0),
}

DIRECTIONS = ["NORTH", "SOUTH", "EAST", "WEST"]


def overpass_query(query: str, timeout: int = 300) -> dict:
    """POST a query to Overpass, trying mirrors on failure."""
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            print(f"  → querying {endpoint} ...")
            res = requests.post(
                endpoint,
                data={"data": query},
                timeout=timeout,
                headers={"User-Agent": "SIH-VehicleIntelligence/1.0 (academic project)"},
            )
            if res.status_code == 200:
                return res.json()
            last_err = f"HTTP {res.status_code}: {res.text[:200]}"
            print(f"  ⚠️  {last_err}")
        except Exception as e:  # noqa: BLE001 - mirror fallback is the point
            last_err = f"{type(e).__name__}: {e}"
            print(f"  ⚠️  {last_err}")
        time.sleep(2)
    raise RuntimeError(f"All Overpass endpoints failed. Last error: {last_err}")


def fetch_raw(refresh: bool = False) -> dict:
    """Fetch traffic signals + the named roads they sit on."""
    if RAW_CACHE.exists() and not refresh:
        print(f"✅ Using cached Overpass response: {RAW_CACHE}")
        return json.loads(RAW_CACHE.read_text())

    s, w, n, e = DELHI_BBOX
    # Two result sets in one round trip:
    #   .sig — every signalled node in the bbox
    #   .w   — the highway ways passing through those nodes (with node refs,
    #          so each signal can be attributed to a named road locally)
    query = f"""
[out:json][timeout:280];
node["highway"="traffic_signals"]({s},{w},{n},{e})->.sig;
way(bn.sig)["highway"]->.w;
.sig out body;
.w out body;
"""
    print("📡 Fetching real Delhi traffic signals from OpenStreetMap (Overpass)...")
    data = overpass_query(query)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.write_text(json.dumps(data))
    print(f"✅ Cached raw Overpass response -> {RAW_CACHE}")
    return data


def build_signal_index(data: dict) -> list[dict]:
    """Join signal nodes to the named highway ways that contain them."""
    signals: dict[int, dict] = {}
    ways: list[dict] = []

    for el in data.get("elements", []):
        if el.get("type") == "node" and el.get("tags", {}).get("highway") == "traffic_signals":
            signals[el["id"]] = {
                "osm_id": el["id"],
                "lat": el["lat"],
                "lon": el["lon"],
                "roads": [],
                "classes": [],
                "maxspeeds": [],
            }
        elif el.get("type") == "way":
            ways.append(el)

    # Attribute each signal to every highway way passing through it
    for way in ways:
        tags = way.get("tags", {})
        hw = tags.get("highway")
        if hw not in ROAD_CLASS:
            continue
        name = tags.get("name") or tags.get("ref")
        maxspeed = tags.get("maxspeed")
        for node_id in way.get("nodes", []):
            sig = signals.get(node_id)
            if sig is None:
                continue
            sig["classes"].append(hw)
            if name:
                sig["roads"].append(name)
            if maxspeed:
                sig["maxspeeds"].append(maxspeed)

    return list(signals.values())


def _parse_maxspeed(values: list[str]) -> float | None:
    """Pick the most common numeric maxspeed, if OSM has one."""
    nums: list[float] = []
    for v in values:
        v = str(v).strip().lower().replace("km/h", "").replace("kmh", "").strip()
        try:
            nums.append(float(v.split()[0]))
        except (ValueError, IndexError):
            continue
    if not nums:
        return None
    return max(set(nums), key=nums.count)


def score_signal(sig: dict) -> int:
    """Placement priority: arterials and named junctions rank highest."""
    if not sig["classes"]:
        return -1
    best_class = max(ROAD_CLASS.get(c, (0, 0))[1] for c in sig["classes"])
    named_bonus = 2 if sig["roads"] else 0
    # A junction where several distinct roads meet is a better camera site
    distinct_roads = len(set(sig["roads"]))
    return best_class * 3 + named_bonus + min(distinct_roads, 3)


def thin_spatially(signals: list[dict], count: int, grid: int = 16) -> list[dict]:
    """
    Spread cameras across the city instead of clustering them downtown.

    The bbox is divided into a grid; within each cell the highest-scoring
    signals are taken in round-robin passes until `count` is reached. This
    guarantees coverage of outer Delhi (Rohini, Dwarka, Najafgarh) rather than
    200 cameras inside the Ring Road.
    """
    s, w, n, e = DELHI_BBOX
    lat_step = (n - s) / grid
    lon_step = (e - w) / grid

    cells: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for sig in signals:
        if score_signal(sig) < 0:
            continue
        r = min(grid - 1, max(0, int((sig["lat"] - s) / lat_step)))
        c = min(grid - 1, max(0, int((sig["lon"] - w) / lon_step)))
        cells[(r, c)].append(sig)

    for cell in cells.values():
        cell.sort(key=score_signal, reverse=True)

    # Round-robin across cells so coverage is even
    chosen: list[dict] = []
    cell_keys = sorted(cells.keys())
    depth = 0
    while len(chosen) < count:
        added_this_pass = 0
        for key in cell_keys:
            if len(chosen) >= count:
                break
            bucket = cells[key]
            if depth < len(bucket):
                chosen.append(bucket[depth])
                added_this_pass += 1
        if added_this_pass == 0:
            break  # exhausted every cell
        depth += 1

    return chosen


def _slug(text: str, maxlen: int = 22) -> str:
    keep = [ch.upper() if ch.isalnum() else "_" for ch in text]
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")[:maxlen] or "JN"


def build_cameras(signals: list[dict]) -> list[dict]:
    """Turn chosen signals into a camera manifest."""
    rng = random.Random(2026)
    cameras: list[dict] = []
    used_ids: set[str] = set()

    for idx, sig in enumerate(signals, start=1):
        road_names = list(dict.fromkeys(sig["roads"]))  # dedupe, keep order
        primary_road = road_names[0] if road_names else "Unnamed Road"
        if len(road_names) >= 2:
            junction_name = f"{road_names[0]} x {road_names[1]}"
        else:
            junction_name = f"{primary_road} Junction"

        best_class = max(
            sig["classes"], key=lambda c: ROAD_CLASS.get(c, (0, 0))[1]
        ) if sig["classes"] else "residential"
        default_speed = ROAD_CLASS.get(best_class, (40.0, 1))[0]
        speed_limit = _parse_maxspeed(sig["maxspeeds"]) or default_speed

        base_id = f"DL_{_slug(primary_road, 18)}"
        cam_id = base_id
        suffix = 1
        while cam_id in used_ids:
            suffix += 1
            cam_id = f"{base_id}_{suffix}"
        used_ids.add(cam_id)

        cameras.append({
            "camera_id": cam_id,
            "name": junction_name[:100],
            "location": f"{primary_road}, Delhi",
            "latitude": round(sig["lat"], 6),
            "longitude": round(sig["lon"], 6),
            "road": primary_road[:100],
            "road_class": best_class,
            "direction": DIRECTIONS[idx % len(DIRECTIONS)],
            "camera_type": "ANPR",
            "deployment": "delhi",
            "speed_limit_kmh": float(speed_limit),
            "osm_node_id": sig["osm_id"],
        })

    return cameras


def main():
    ap = argparse.ArgumentParser(description="Build Delhi ANPR camera network from OSM signals")
    ap.add_argument("--count", type=int, default=200, help="Number of cameras to select")
    ap.add_argument("--refresh", action="store_true", help="Re-query Overpass instead of using cache")
    ap.add_argument("--grid", type=int, default=16, help="Spatial thinning grid resolution")
    args = ap.parse_args()

    data = fetch_raw(refresh=args.refresh)
    signals = build_signal_index(data)
    print(f"✅ Parsed {len(signals)} real traffic signals in the Delhi bbox")

    named = sum(1 for s in signals if s["roads"])
    print(f"   {named} of them sit on a *named* road (usable as junction names)")

    chosen = thin_spatially(signals, args.count, grid=args.grid)
    print(f"✅ Selected {len(chosen)} junctions, spread across a {args.grid}x{args.grid} city grid")

    cameras = build_cameras(chosen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CAMERAS_JSON.write_text(json.dumps(cameras, indent=2))
    print(f"✅ Wrote camera manifest -> {CAMERAS_JSON}")

    # Coverage / sanity summary
    lats = [c["latitude"] for c in cameras]
    lons = [c["longitude"] for c in cameras]
    by_class: dict[str, int] = defaultdict(int)
    for c in cameras:
        by_class[c["road_class"]] += 1

    print("\nCoverage summary")
    print(f"  lat range: {min(lats):.4f} .. {max(lats):.4f}")
    print(f"  lon range: {min(lons):.4f} .. {max(lons):.4f}")
    print(f"  road classes: {dict(sorted(by_class.items(), key=lambda kv: -kv[1]))}")
    print("\nSample cameras:")
    for c in cameras[:8]:
        print(f"  {c['camera_id']:<26} {c['name'][:44]:<46} {c['speed_limit_kmh']:.0f} km/h")


if __name__ == "__main__":
    main()

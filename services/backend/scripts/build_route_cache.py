"""
Offline precompute of the camera→camera road-route cache.

Why this script exists: the live demo must draw road-accurate trajectories with
NO network. The public OSRM demo server is rate-limited and venue Wi-Fi is not
trustworthy, so every camera pair the map is likely to draw gets fetched once,
here, ahead of time, and persisted into route_segments. After this has run,
/routing/trajectory/{plate} makes zero outbound calls.

Usage
─────
    # ~1440 directed pairs (200 cameras x 6 nearest neighbours, both ways)
    .venv/bin/python scripts/build_route_cache.py --pairs adjacent --k 6

    # only the corridors the demo dataset actually uses
    .venv/bin/python scripts/build_route_cache.py --pairs top-used --limit 400

    # every ordered pair — 200*199 = 39,800 calls, hours of polite requests.
    # Don't. Listed for completeness only.
    .venv/bin/python scripts/build_route_cache.py --pairs all --limit 2000

Flags:
    --pairs adjacent|top-used|all   which pairs to cache (default adjacent)
    --k N                           neighbours per camera for `adjacent` (default 6)
    --limit N                       cap the number of pairs processed
    --sleep S                       extra delay between OSRM calls (default 0.15)
    --refresh                       re-fetch pairs already cached
    --deployment NAME               restrict to one deployment (default delhi)
    --dry-run                       list what would be fetched, make no calls

Resumable and idempotent: already-cached pairs are skipped unless --refresh, so
re-running after an interruption (or after Ctrl-C) picks up where it stopped.
Ctrl-C is caught and the partial progress is already committed — nothing is
lost, and the summary still prints.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect, text

from backend.database import SessionLocal
from backend.models.route_segment import RouteSegment
from backend.services import routing_service as rs

# Set when SIGINT arrives. We finish the in-flight pair and stop cleanly rather
# than dying mid-commit — the whole point of a resumable cache builder.
_stop = False


def _handle_sigint(_signum, _frame):
    global _stop
    if _stop:
        print("\n  Second Ctrl-C — exiting immediately.")
        raise SystemExit(130)
    _stop = True
    print("\n⏸  Ctrl-C received — finishing current pair, then stopping. "
          "Progress so far is already saved; re-run to resume.")


def load_cameras(db, deployment: str | None) -> list[rs.CameraRef]:
    """Camera positions (deployment JSON first, topped up from the cameras table)."""
    rs.deployment_cameras(refresh=True)  # pick up edits to cameras.json between runs
    return rs.camera_universe(db, deployment)


def top_used_pairs(db, limit: int) -> list[tuple[str, str]]:
    """
    The camera pairs the demo will actually draw, ranked by real traffic.

    Prefers a `road_usage` aggregate table if a teammate's dataset generator has
    produced one (any (from,to,count)-shaped column naming is probed, because
    that table is not ours to define). Falls back to deriving consecutive
    sightings straight out of vehicle_events — correct but a full scan, so it is
    bounded to the most recent rows.
    """
    inspector = inspect(db.get_bind())
    tables = set(inspector.get_table_names())

    if "road_usage" in tables:
        cols = {c["name"] for c in inspector.get_columns("road_usage")}
        from_col = next((c for c in ("from_camera_id", "from_camera", "src_camera_id") if c in cols), None)
        to_col = next((c for c in ("to_camera_id", "to_camera", "dst_camera_id") if c in cols), None)
        # Ordered by preference. `trip_count` first because that is what the
        # Delhi dataset generator actually emits; the rest are defensive
        # synonyms so this keeps working if that table is ever renamed/reshaped
        # (road_usage is not ours to define). Without a match we fall back to
        # unordered table order, which with 6000+ rows and a --limit means
        # caching an arbitrary slice instead of the busiest corridors — so it is
        # worth keeping this list in sync.
        cnt_col = next((c for c in ("trip_count", "vehicle_count", "trips",
                                    "usage_count", "count", "total") if c in cols), None)
        if from_col and to_col:
            # Column names come from the inspector, not user input, so this
            # interpolation cannot carry injected SQL.
            order = f"ORDER BY {cnt_col} DESC" if cnt_col else ""
            rows = db.execute(text(
                f"SELECT {from_col}, {to_col} FROM road_usage {order} LIMIT :lim"
            ), {"lim": limit}).all()
            pairs = [(r[0], r[1]) for r in rows if r[0] and r[1] and r[0] != r[1]]
            if pairs:
                print(f"  Source: road_usage table ({len(pairs)} pairs, ranked by {cnt_col or 'table order'})")
                return pairs
        print("  road_usage exists but has no usable (from, to) rows — falling back to vehicle_events.")

    if "vehicle_events" not in tables:
        print("  No road_usage and no vehicle_events table — nothing to rank. "
              "Use --pairs adjacent instead.")
        return []

    # Derive from consecutive sightings. Bounded row count: we only need the
    # POPULAR corridors, and 200k recent events already reveal them.
    scan_limit = 200_000
    rows = db.execute(text(
        "SELECT plate, camera_id FROM vehicle_events "
        "WHERE plate IS NOT NULL "
        "ORDER BY plate, timestamp LIMIT :lim"
    ), {"lim": scan_limit}).all()

    counter: Counter[tuple[str, str]] = Counter()
    prev_plate, prev_cam = None, None
    for plate, cam in rows:
        if plate == prev_plate and prev_cam and cam and prev_cam != cam:
            counter[(prev_cam, cam)] += 1
        prev_plate, prev_cam = plate, cam
    pairs = [p for p, _ in counter.most_common(limit)]
    print(f"  Source: vehicle_events consecutive sightings "
          f"({len(rows)} rows scanned → {len(counter)} distinct pairs, keeping {len(pairs)})")
    return pairs


def all_pairs(cams: list[rs.CameraRef]) -> list[tuple[str, str]]:
    """Every ordered pair, nearest-first so a --limit still yields useful hops."""
    pairs: list[tuple[tuple[str, str], float]] = []
    for a in cams:
        for b in cams:
            if a.camera_id == b.camera_id:
                continue
            pairs.append(((a.camera_id, b.camera_id),
                          rs.haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)))
    pairs.sort(key=lambda t: t[1])
    return [p for p, _ in pairs]


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute the camera→camera road route cache.")
    ap.add_argument("--pairs", choices=["adjacent", "top-used", "all"], default="adjacent")
    ap.add_argument("--k", type=int, default=6, help="Nearest neighbours per camera (adjacent mode)")
    ap.add_argument("--limit", type=int, default=None, help="Max pairs to process")
    ap.add_argument("--sleep", type=float, default=0.15, help="Extra delay between OSRM calls (s)")
    ap.add_argument("--refresh", action="store_true", help="Re-fetch pairs already cached")
    ap.add_argument("--deployment", default="delhi", help="Restrict to this deployment ('' for all)")
    ap.add_argument("--dry-run", action="store_true", help="List pairs, make no OSRM calls")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Own table creation: init_db() does not know about RouteSegment yet.
    rs.ensure_route_cache_schema()

    db = SessionLocal()
    try:
        deployment = args.deployment or None
        cams = load_cameras(db, deployment)
        print(f"Cameras loaded: {len(cams)}"
              f"{f' (deployment={deployment})' if deployment else ''}")
        if len(cams) < 2:
            print("✗ Need at least 2 cameras with coordinates. Nothing to do.")
            return 1

        if args.pairs == "adjacent":
            pairs = rs.adjacent_pairs(cams, k=args.k)
            print(f"Mode: adjacent (k={args.k}) → {len(pairs)} directed pairs")
        elif args.pairs == "top-used":
            pairs = top_used_pairs(db, args.limit or 500)
            known = {c.camera_id for c in cams}
            before = len(pairs)
            pairs = [p for p in pairs if p[0] in known and p[1] in known]
            print(f"Mode: top-used → {len(pairs)} pairs "
                  f"({before - len(pairs)} dropped: camera not in this deployment)")
            if before and not pairs:
                print("  ⚠️  Every top-used pair references cameras outside "
                      f"deployment '{deployment}'. The events in this DB belong to a "
                      "different deployment — re-run with --deployment '' to cache them.")
        else:
            pairs = all_pairs(cams)
            print(f"Mode: all → {len(pairs)} directed pairs (this is a LOT of requests)")

        if args.limit:
            pairs = pairs[: args.limit]

        # Idempotency: skip what we already have unless --refresh.
        existing = {
            (r.from_camera_id, r.to_camera_id)
            for r in db.query(RouteSegment.from_camera_id, RouteSegment.to_camera_id).all()
        }
        todo = pairs if args.refresh else [p for p in pairs if p not in existing]
        print(f"Already cached: {len(existing)} segments. To fetch now: {len(todo)}"
              f"{' (--refresh: re-fetching everything)' if args.refresh else ''}")

        if args.dry_run:
            for p in todo[:20]:
                print(f"  would fetch {p[0]} → {p[1]}")
            if len(todo) > 20:
                print(f"  ... and {len(todo) - 20} more")
            return 0
        if not todo:
            print("✓ Nothing to do — cache already covers every requested pair.")
            _summary(db, deployment, args.k)
            return 0

        by_id = {c.camera_id: c for c in cams}
        ok = fallback = errors = 0
        ratios: list[float] = []
        started = time.monotonic()

        for i, (a_id, b_id) in enumerate(todo, start=1):
            if _stop:
                print(f"\nStopped after {i - 1}/{len(todo)} pairs (progress saved).")
                break
            a, b = by_id.get(a_id), by_id.get(b_id)
            if a is None or b is None:
                errors += 1
                continue
            try:
                route = rs.get_route(a, b, db, refresh=args.refresh)
            except Exception as err:  # noqa: BLE001 - one bad pair must not abort a 20-min run
                errors += 1
                print(f"\n  ! {a_id} → {b_id}: {type(err).__name__}: {err}")
                continue

            if route.source == "osrm":
                ok += 1
                if route.straight_line_m > 0:
                    ratios.append(route.detour_ratio)
            else:
                fallback += 1

            elapsed = time.monotonic() - started
            rate = i / elapsed if elapsed > 0 else 0.0
            eta_s = (len(todo) - i) / rate if rate > 0 else 0.0
            mean_ratio = (sum(ratios) / len(ratios)) if ratios else 0.0
            print(
                f"\r[{i}/{len(todo)}] {i * 100 // len(todo):3d}%  "
                f"ok={ok} fallback={fallback} err={errors}  "
                f"ratio~{mean_ratio:.2f}  {rate:.1f}/s  ETA {eta_s / 60:.1f}m   ",
                end="", flush=True,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        print()
        print(f"Fetched this run: {ok} via OSRM, {fallback} straight-line fallbacks, {errors} errors.")
        if fallback and not ok:
            print("⚠️  EVERY pair fell back to a straight line — the router is unreachable. "
                  "Check network / OSRM_BASE_URL. The cache is NOT road geometry yet.")
        _summary(db, deployment, args.k)
        return 0
    finally:
        db.close()


def _summary(db, deployment: str | None, k: int) -> None:
    """Cache-wide sanity report. The detour ratio is the honesty check."""
    stats = rs.cache_stats(db, k=k, deployment=deployment)
    sample = stats["detour_ratio_sample"]
    print("\n── Route cache summary ─────────────────────────────")
    print(f"  segments cached        : {stats['segments_cached']}")
    print(f"  real OSRM road routes  : {stats['osrm_segments']}")
    print(f"  straight-line fallbacks: {stats['fallback_segments']}")
    print(f"  points/segment         : mean {stats['mean_point_count']}, max {stats['max_point_count']}")
    print(f"  road/straight ratio    : median {sample['median']} "
          f"(mean {sample['mean']}, p10 {sample['p10']}, p90 {sample['p90']}) "
          f"over {sample['segments_in_sample']} hops >= "
          f"{rs.RATIO_SAMPLE_MIN_STRAIGHT_M:.0f} m apart")
    adj = stats["adjacency"]
    print(f"  adjacency coverage     : {adj['pairs_cached']}/{adj['pairs_wanted']} "
          f"({adj['coverage_pct']}%) at k={adj['k']}, deployment={adj['deployment']}")
    ratio = sample["median"]
    if ratio is None:
        print("  ✗ No OSRM segments long enough to judge — geometry unverified.")
    elif ratio < 1.05:
        print("  ✗ Median ratio ~1.0: geometry is suspiciously straight, expected 1.15–1.6 "
              "for real Delhi road routes. Investigate before demoing.")
    else:
        print("  ✓ Ratio is in the expected range for genuine road routing.")


if __name__ == "__main__":
    raise SystemExit(main())

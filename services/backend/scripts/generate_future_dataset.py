#!/usr/bin/env python3
"""
Stage the *next* month of sightings, for the simulation clock to release one by one.

The dashboard is demonstrated across a two-month window centred on now:

    [ -30 days ............ now ............ +30 days ]
      already in vehicle_events      staged in future_events
      -> history, aggregates,        -> released one at a time by the
         trained transition model       simulation clock: live feed, live
                                        heatmap, live trajectories, and
                                        predictions that can be scored
                                        against what actually happens

This script fills the right-hand half. It reuses `scripts/generate_delhi_dataset`
wholesale — the camera graph, the behaviour profiles, the congestion curve, the
speed model, the plate format — so the future obeys the same physics as the
past. That matters more than it sounds: the whole point of the exercise is that
a model fitted on the left half predicts the right half. If the two halves came
from different generators, every "prediction" would be measuring the difference
between two code paths.

Continuity with the past
------------------------
By default the plate population is seeded from the plates that actually appear
in the last week of history (`--reuse-plates`), so a vehicle with a month of
trips behind it keeps driving into the future rather than vanishing at midnight
and being replaced by a stranger. Without that, every live trajectory would be
one hop long and every next-hop prediction would have no history to work from.

Volume
------
The past month holds ~12M events because it has to look like a real city's
archive. The future does not need that: it is *replayed*, and at the default
60x clock a month of sim-time passes in twelve hours of wall-clock. 1.5M events
over 30 days is ~2,000 sim-events an hour, i.e. ~35 releases a second at 60x —
fast enough that the live feed never looks idle, slow enough that a single
SQLite writer keeps up.

Usage
-----
  .venv/bin/python scripts/generate_future_dataset.py
  .venv/bin/python scripts/generate_future_dataset.py --days 30 --target-events 1500000
  .venv/bin/python scripts/generate_future_dataset.py --batch demo2 --no-reset
  .venv/bin/python scripts/generate_future_dataset.py --db-url sqlite:///./scratch.db
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

GENERATOR_PATH = Path(__file__).resolve().parent / "generate_delhi_dataset.py"


def load_generator():
    """Import the Delhi generator by path.

    `scripts/` is not a package, so a plain `import generate_delhi_dataset`
    works only when the CWD happens to be scripts/. Loading by absolute path
    makes this script runnable from anywhere, which is what a cron entry or a
    demo runbook will actually do.
    """
    spec = importlib.util.spec_from_file_location("delhi_generator", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the dataset generator at {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["delhi_generator"] = module
    spec.loader.exec_module(module)
    return module


INSERT_FUTURE_SQL = (
    "INSERT INTO future_events "
    "(camera_id, local_track_id, timestamp, plate, plate_confidence, "
    " latitude, longitude, direction, vehicle_type, vehicle_color, speed, "
    " global_vehicle_id, released_at, batch) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?)"
)


def recent_plates(engine, limit: int, days: int = 7) -> list[tuple[str, str | None]]:
    """Plates seen in the last `days` of history, most recently first.

    An index range scan on `timestamp`, not a scan of the 12M-row table: the
    ORDER BY / LIMIT is pushed into SQLite, which walks the tail of the
    timestamp index and stops. Returns (plate, global_vehicle_id) so a vehicle
    keeps its identity across the boundary.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).replace(tzinfo=None)
    sql = (
        "SELECT plate, MAX(global_vehicle_id) FROM vehicle_events "
        "WHERE timestamp >= ? AND plate IS NOT NULL "
        "GROUP BY plate LIMIT ?"
    )
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(sql, (since.strftime("%Y-%m-%d %H:%M:%S.%f"), limit))
        rows = [(r[0], r[1]) for r in cur.fetchall()]
        cur.close()
    finally:
        raw.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--days", type=int, default=30, help="days of future to stage (default 30)")
    ap.add_argument("--vehicles", type=int, default=60_000,
                    help="unique plates in the future population (default 60,000)")
    ap.add_argument("--target-events", type=int, default=1_500_000,
                    help="event budget for the whole window (0 = no scaling)")
    ap.add_argument("--reuse-plates", dest="reuse_plates", action="store_true", default=True,
                    help="seed the population from plates already in history (default)")
    ap.add_argument("--fresh-plates", dest="reuse_plates", action="store_false",
                    help="use an entirely new plate population (breaks continuity)")
    ap.add_argument("--batch", default=None, help="batch tag (default: future_<UTC date>)")
    ap.add_argument("--no-reset", action="store_true",
                    help="append instead of clearing previously staged future events")
    ap.add_argument("--cameras", type=int, default=None, help="use only the first N cameras (debug)")
    ap.add_argument("--batch-size", type=int, default=50_000)
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    gen = load_generator()

    from sqlalchemy import create_engine, text

    from backend.config import get_settings
    from backend.database import Base
    from backend.models import future_event  # noqa: F401 — registers the table

    get_settings.cache_clear()
    db_url = args.db_url or get_settings().DATABASE_URL
    engine = create_engine(
        db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
    )

    batch = args.batch or f"future_{datetime.now(timezone.utc):%Y%m%d}"
    wall0 = time.perf_counter()

    print("=" * 84)
    print("FUTURE WINDOW GENERATOR — staging the next month for the simulation clock")
    print("=" * 84)
    print(f"  database      {db_url}")
    print(f"  window        now -> now + {args.days} days")
    print(f"  batch tag     {batch}")
    print(f"  event budget  {args.target_events:,}")

    rng = random.Random(args.seed)

    # ── 1. Network ───────────────────────────────────────────────────────────
    print("\n[1/5] Building the camera network (same graph as the history)...")
    cams = gen.load_cameras(args.cameras)
    net = gen.CameraNet(cams)
    print(f"  {net.n} cameras, {net.edge_count()} road segments")

    # ── 2. Schema ────────────────────────────────────────────────────────────
    print("\n[2/5] Ensuring the future_events table exists...")
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        if not args.no_reset:
            deleted = conn.execute(text("DELETE FROM future_events")).rowcount
            print(f"  cleared {max(deleted, 0):,} previously staged event(s)")
        else:
            print("  appending to whatever is already staged (--no-reset)")

    # ── 3. Population ────────────────────────────────────────────────────────
    print(f"\n[3/5] Assigning behaviour to {args.vehicles:,} vehicles...")
    t0 = time.perf_counter()

    carried = 0
    if args.reuse_plates:
        known = recent_plates(engine, args.vehicles)
        carried = len(known)
        plates = [p for p, _ in known]
        print(f"  carried {carried:,} plate(s) forward from the last week of history")
    else:
        plates = []

    if len(plates) < args.vehicles:
        # Top up with new plates — a city gains vehicles too, and a population
        # that is 100% carried-forward would have no cold-start cases to show.
        extra = gen.make_plates(args.vehicles - len(plates), rng)
        existing = set(plates)
        plates.extend(p for p in extra if p not in existing)

    # The watchlist has to keep driving, or the alert demo has nothing to fire on.
    for i, poi in enumerate(gen.POI_PLATES):
        if poi not in plates:
            plates[i] = poi

    vehicles, speeders = gen.assign_vehicles(plates, net, rng)
    print(f"  {len(vehicles):,} vehicles in {time.perf_counter() - t0:.1f}s "
          f"({len(speeders):,} habitual speeders)")

    # ── 4. Calibrate ─────────────────────────────────────────────────────────
    # The generator's Simulator walks BACKWARD from `end_epoch` over `days`.
    # Handing it an end_epoch one month in the FUTURE therefore produces a
    # window running from now to now+days — the same code, pointed the other
    # way, rather than a second simulator that would drift from the first.
    now = datetime.now(timezone.utc)
    end_epoch = int(now.timestamp()) + args.days * 86400
    start_epoch = end_epoch - args.days * 86400

    trip_scale = 1.0
    if args.target_events > 0:
        print("\n[4/5] Calibrating trip volume against the event budget...")
        t0 = time.perf_counter()
        sample_n = max(500, min(3000, len(vehicles) // 20))
        probe = gen.Simulator(net, args.days, end_epoch, random.Random(args.seed + 1))
        step = max(1, len(vehicles) // sample_n)
        sample = vehicles[::step][:sample_n]
        sampled = sum(1 for v in sample for _ in probe.simulate(v))
        projected = sampled / len(sample) * len(vehicles) if sample else 0
        trip_scale = min(4.0, max(0.05, args.target_events / projected)) if projected else 1.0
        print(f"  sampled {len(sample):,} vehicles -> {sampled:,} events "
              f"({time.perf_counter() - t0:.1f}s)")
        print(f"  unscaled projection {projected / 1e6:.2f}M -> trip_scale {trip_scale:.3f}")
    else:
        print("\n[4/5] Budget scaling disabled")

    # ── 5. Generate + stage ──────────────────────────────────────────────────
    print(f"\n[5/5] Generating and staging (batch {args.batch_size:,})...")
    sim = gen.Simulator(net, args.days, end_epoch, rng, trip_scale=trip_scale)
    rows = gen.iter_rows(vehicles, sim, net, start_epoch, end_epoch)

    raw, conn = gen.open_raw(engine)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=OFF")

    # Drop the indexes for the load and rebuild them after. Same reasoning as
    # the history loader: rebuilding two B-trees once beats 1.5M incremental
    # inserts into them.
    for name in ("ix_future_events_due", "ix_future_events_plate",
                 "ix_future_events_released_at", "ix_future_events_batch"):
        cur.execute(f"DROP INDEX IF EXISTS {name}")

    total = 0
    t0 = time.perf_counter()
    batch_rows: list[tuple] = []
    try:
        cur.execute("BEGIN")
        for row in rows:
            # iter_rows yields the 14-tuple the history loader wants:
            #   (event_id, camera_id, track, ts, plate, conf, lat, lon, dir,
            #    vtype, color, speed, gid, created_at)
            # future_events has no event_id (it gets one when promoted) and no
            # created_at, so drop the first and last and append the batch tag.
            batch_rows.append(row[1:13] + (batch,))
            if len(batch_rows) >= args.batch_size:
                cur.executemany(INSERT_FUTURE_SQL, batch_rows)
                total += len(batch_rows)
                batch_rows.clear()
                cur.execute("COMMIT")
                cur.execute("BEGIN")
        if batch_rows:
            cur.executemany(INSERT_FUTURE_SQL, batch_rows)
            total += len(batch_rows)
        cur.execute("COMMIT")

        print(f"  staged {total:,} events in {time.perf_counter() - t0:.1f}s "
              f"({total / max(time.perf_counter() - t0, 1e-9):,.0f} rows/s)")

        print("  rebuilding indexes...")
        ti = time.perf_counter()
        cur.execute("CREATE INDEX IF NOT EXISTS ix_future_events_due "
                    "ON future_events (released_at, timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_future_events_plate "
                    "ON future_events (plate)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_future_events_batch "
                    "ON future_events (batch)")
        print(f"  indexes built in {time.perf_counter() - ti:.1f}s")
    finally:
        cur.close()
        raw.close()

    # ── Verify ───────────────────────────────────────────────────────────────
    with engine.connect() as conn2:
        staged, first, last, plates_n, cams_n = conn2.execute(text(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp), "
            "COUNT(DISTINCT plate), COUNT(DISTINCT camera_id) FROM future_events"
        )).one()
        hist_last = conn2.execute(text(
            "SELECT MAX(timestamp) FROM vehicle_events"
        )).scalar()
        overlap = conn2.execute(text(
            "SELECT COUNT(*) FROM future_events WHERE timestamp <= :now"
        ), {"now": now.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S.%f")}).scalar()

    print()
    print("=" * 84)
    print(f"  staged events      {staged:,}")
    print(f"  unique plates      {plates_n:,}   cameras {cams_n}")
    print(f"  future window      {first}  ->  {last}")
    print(f"  history ends       {hist_last}")
    print(f"  already-due rows   {overlap:,}  (released on the first tick)")
    print(f"  carried-forward    {carried:,} plate(s) shared with history")
    print(f"  total wall time    {time.perf_counter() - wall0:.1f}s")
    print("=" * 84)
    print()
    print("Start the clock:  curl -X POST 'http://localhost:8000/simulation/start?speed=60'")
    print("Watch it live:    http://localhost:8000/app/live")


if __name__ == "__main__":
    main()

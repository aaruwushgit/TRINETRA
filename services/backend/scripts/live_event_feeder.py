"""
Live ANPR event feeder — "happening now" on top of the month of history.

A month of loaded history makes the dashboard look real; a stream of events
arriving while the audience watches makes it look *alive*. This is the second
half of the demo: it keeps generating plausible sightings at "now" across the
same 200 Delhi junctions, at a configurable rate, and commits often enough that
the dashboard sees them within a second.

It reuses scripts/generate_delhi_dataset.py wholesale — the camera graph, the
routing matrix, the congestion curve, the speed model, the plate format. The
history and the live tail therefore obey identical physics, so a vehicle whose
last historical sighting was 40 minutes ago and whose next sighting is live
still produces a plausible implied speed. Reimplementing the model here would
guarantee the two halves drifted apart.

Three things it does that a naive "insert random rows" loop would not
--------------------------------------------------------------------

1. Vehicles are *in flight*, not independent.
   State is a heap of live trips keyed by when their next hop is due. Popping a
   trip emits one sighting and reschedules it one hop later, using that hop's
   real distance and a congestion-adjusted speed. That is what gives trajectory
   reconstruction and next-hop prediction something to actually predict. It also
   means the process warms up: it pre-seeds trips with staggered due times so
   the stream is continuous from the first second rather than being all
   trip-starts for the first five minutes.

2. It mostly reuses plates that already exist in the database.
   A live stream of entirely new plates would make every trajectory one event
   long and every "vehicle seen at N cameras" panel read 1. The pool is seeded
   from the most recent historical sightings (an index scan, not a table scan),
   and only --new-plate-rate of trips introduce a fresh plate.

3. Watchlist vehicles go through the real ingestion path.
   Ordinary events are direct bulk inserts — at 40/s, running tracking +
   alerting per event is wasted work. But the POI vehicles are routed through
   kafka_consumer.process_event_payload(), which is the same function the Kafka
   worker and the REST endpoint use: it validates the camera, normalises the
   plate, assigns global_vehicle_id via tracking_service and fires
   alert_service. So the BLACKLIST alerts appearing on screen during the demo
   are produced by the production code path, not fabricated here.

Aggregate freshness
-------------------
The rollups in backend/models/analytics_agg.py are refreshed incrementally from
counters this process keeps as it inserts, then UPSERTed — not by re-running the
generator's GROUP BYs, which would rescan millions of rows every minute. The
one known approximation: unique_vehicles is advanced by *this process's*
newly-seen plates per bucket, so a plate seen live that was already in that
hour's history is double-counted until the next full rebuild. Volume, speed and
road-usage figures stay exact.

Usage
-----
  .venv/bin/python scripts/live_event_feeder.py                       # 40 ev/s
  .venv/bin/python scripts/live_event_feeder.py --events-per-sec 120
  .venv/bin/python scripts/live_event_feeder.py --diurnal-rate        # rate follows rush hour
  .venv/bin/python scripts/live_event_feeder.py --db-url sqlite:///./scratch.db
  Ctrl-C to stop (flushes and prints a summary).
"""
from __future__ import annotations

import argparse
import heapq
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import generate_delhi_dataset as gen  # noqa: E402  (same directory, path fixed above)

TICK_SECONDS = 0.1           # commit cadence; the dashboard should never be >1s stale
# Recent sightings scanned to seed the reuse pool. This has to comfortably
# exceed the in-flight trip count, or every pool pick collides with a plate
# already on the road and the feeder falls back to inventing new plates —
# which is exactly the trajectory-continuity problem the pool exists to solve.
POOL_QUERY_LIMIT = 250_000
AVG_HOP_SECONDS = 240        # rough steady-state hop time, used to size the trip pool


@dataclass(slots=True)
class Trip:
    plate: str
    global_id: str
    vtype: str
    color: str
    node: int          # camera index where the next sighting happens
    speed: float       # speed to report at `node`
    target: int
    hops_left: int
    bias: float
    cap: float
    due: float         # epoch seconds
    via_api: bool = False


@dataclass(slots=True)
class Bucket:
    """Per (camera, hour) counters accumulated between aggregate refreshes."""
    count: int = 0
    speed_sum: float = 0.0
    plates: set = field(default_factory=set)
    plates_reported: int = 0


class Feeder:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random()
        self.running = True

        if args.db_url:
            os.environ["DATABASE_URL"] = args.db_url

        from sqlalchemy import create_engine

        from backend.config import get_settings
        from backend.database import Base
        from backend.models import analytics_agg, alert, camera, trajectory, traffic_snapshot, vehicle_event  # noqa: F401

        get_settings.cache_clear()
        self.db_url = args.db_url or get_settings().DATABASE_URL
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {},
        )
        Base.metadata.create_all(bind=self.engine)

        self.net = self._load_network()
        self.pool = self._load_vehicle_pool()

        # Event ids must not collide with the generator's "d%09d" space or with a
        # previous feeder run. Process start time gives a per-run namespace.
        self.id_prefix = f"L{int(time.time()):x}_"
        self.gid_prefix = "VEH_L"
        self.seq = 0
        self.track = [0] * self.net.n
        self.new_plates = 0
        self.new_plates_reported = 0

        self.heap: list[tuple[float, int, Trip]] = []
        self.active_plates: set[str] = set()
        self.cam_index = {cid: i for i, cid in enumerate(self.net.ids)}
        self.tiebreak = 0
        # Generous: trips are ~200 bytes each, and an under-sized cap would stop
        # the feeder reaching its target rate. The budget loop finds the real
        # equilibrium on its own.
        self.max_inflight = int(args.events_per_sec * AVG_HOP_SECONDS * 3) + 500

        self.hour_buckets: dict[tuple[str, str], Bucket] = {}
        self.cam_totals: dict[str, list] = {}
        self.segments: dict[tuple[str, str], list] = {}

        self.inserted = 0
        self.poi_events = 0
        self.poi_alerts = 0
        self.refreshes = 0
        self.started = time.time()
        self.last_poi = 0.0
        self.last_agg = time.time()
        self.last_beat = 0.0
        self.beat_base = 0

        self.hour_picker = gen._weighted_picker(gen.HOUR_WEIGHTS)
        self.mean_hour_weight = sum(gen.HOUR_WEIGHTS) / 24.0

    # ── setup ────────────────────────────────────────────────────────────────

    def _load_network(self) -> gen.CameraNet:
        """
        Build the graph from the cameras actually in the database, not from the
        manifest file. If an operator deactivated a junction, the feeder should
        stop routing traffic through it — and it guarantees every camera_id the
        feeder emits satisfies the vehicle_events foreign key.
        """
        raw, conn = gen.open_raw(self.engine)
        rows = conn.cursor().execute(
            "SELECT camera_id, name, location, latitude, longitude, road, direction, "
            "       camera_type, deployment, speed_limit_kmh FROM cameras "
            "WHERE is_active = 1 AND deployment = ? ORDER BY camera_id",
            (gen.DEPLOYMENT_TAG,),
        ).fetchall()
        raw.close()
        if not rows:
            raise SystemExit(
                f"No active '{gen.DEPLOYMENT_TAG}' cameras in {self.db_url}. "
                "Run scripts/generate_delhi_dataset.py first."
            )
        cams = [
            {"camera_id": r[0], "name": r[1], "location": r[2], "latitude": r[3],
             "longitude": r[4], "road": r[5], "direction": r[6], "camera_type": r[7],
             "deployment": r[8], "speed_limit_kmh": r[9],
             # road_class isn't stored on Camera; the manifest has it, but the
             # feeder only needs it for truck homing, and freight_nodes falls
             # back to all cameras when it's absent.
             "road_class": None}
            for r in rows
        ]
        return gen.CameraNet(cams)

    def _load_vehicle_pool(self) -> list[tuple[str, str, str, str, int]]:
        """
        Seed the reuse pool from the newest historical sightings. ORDER BY
        timestamp DESC LIMIT N rides the timestamp index, so this is milliseconds
        even against 12M rows — a SELECT DISTINCT plate would be a full scan.
        """
        raw, conn = gen.open_raw(self.engine)
        cur = conn.cursor()
        cur.execute("PRAGMA cache_size=-100000")
        rows = cur.execute(
            "SELECT plate, global_vehicle_id, vehicle_type, vehicle_color, camera_id "
            "FROM vehicle_events WHERE plate IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT ?",
            (POOL_QUERY_LIMIT,),
        ).fetchall()
        raw.close()

        idx = {cid: i for i, cid in enumerate(self.net.ids)}
        seen: set[str] = set()
        pool = []
        for plate, gid, vtype, color, cam in rows:
            if plate in seen or plate in gen.POI_PLATES:
                continue
            seen.add(plate)
            pool.append((plate, gid or f"{gen.GLOBAL_ID_PREFIX}UNKNOWN", vtype or "car",
                         color or "white", idx.get(cam, 0)))
        return pool

    # ── trip lifecycle ───────────────────────────────────────────────────────

    def _ist_hour(self, epoch: float) -> int:
        return int((epoch + gen.IST_OFFSET) % 86400) // 3600

    def _plan(self, trip: Trip, epoch: float) -> bool:
        """
        Work out the hop the vehicle is about to make and when it lands.
        Returns False when the trip is over.

        Same three ingredients as the history: the stricter of the two cameras'
        posted limits, the congestion factor for the current Delhi hour, and a
        junction dwell that grows as congestion bites.
        """
        net = self.net
        if trip.hops_left <= 0:
            return False
        nxt = net.next_hop[trip.target][trip.node]
        if nxt < 0 or trip.node == trip.target:
            return False
        if self.rng.random() < 0.12:
            cand = net.adj[trip.node]
            nxt = cand[self.rng.randrange(len(cand))][0]

        dist = 0.0
        for v, w in net.adj[trip.node]:
            if v == nxt:
                dist = w
                break
        if dist <= 0.0:
            dist = gen.haversine_km(net.lat[trip.node], net.lon[trip.node],
                                    net.lat[nxt], net.lon[nxt])

        hour = self._ist_hour(epoch)
        cong = gen.CONGESTION[hour]
        limit = min(net.limit[trip.node], net.limit[nxt])
        speed = gen._hop_speed(limit, cong, trip.bias, trip.cap, self.rng.random)
        dwell = 3.0 + 22.0 * (1.0 - cong) + self.rng.random() * (9.0 + 93.0 * (1.0 - cong))

        # Bookkeeping for the road_usage rollup: this hop is a trip leg the
        # moment both of its sightings land.
        travel_min = (dist / speed * 3600.0 + dwell) / 60.0
        key = (net.ids[trip.node], net.ids[nxt])
        seg = self.segments.get(key)
        if seg is None:
            self.segments[key] = [1, travel_min, travel_min, dist]
        else:
            seg[0] += 1
            seg[1] += travel_min
            if travel_min < seg[2]:
                seg[2] = travel_min

        trip.node = nxt
        trip.speed = speed
        trip.hops_left -= 1
        trip.due = epoch + dist / speed * 3600.0 + dwell
        return True

    def _spawn(self, epoch: float, poi_plate: str | None = None) -> Trip | None:
        """
        Start a new trip and emit its first sighting.

        The active_plates guard is load-bearing. With ~12k trips in flight drawn
        from a 60k plate pool, collisions are frequent, and one plate on two
        concurrent trips means interleaved sightings at unrelated junctions —
        i.e. exactly the implausible implied speeds this whole model exists to
        avoid.
        """
        net = self.net
        rng = self.rng
        if poi_plate:
            if poi_plate in self.active_plates:
                return None
            plate = poi_plate
            gid = None  # tracking_service resolves it from history
            vtype, color = "car", "white"
            start = rng.randrange(net.n)
        else:
            plate = None
            if self.pool and rng.random() >= self.args.new_plate_rate:
                for _ in range(6):
                    cand, gid, vtype, color, start = self.pool[rng.randrange(len(self.pool))]
                    if cand not in self.active_plates:
                        plate = cand
                        break
            if plate is None:
                # Either a deliberate new plate, or the pool is saturated (small
                # dataset / very high rate). Inventing one keeps the target rate
                # rather than silently under-feeding the dashboard.
                plate = gen.make_plates(1, rng)[0]
                self.new_plates += 1
                gid = f"{self.gid_prefix}{self.new_plates:07d}"
                vtype = self._sample_type()
                color = gen.VEHICLE_COLORS[rng.randrange(len(gen.VEHICLE_COLORS))]
                start = rng.randrange(net.n)

        target = rng.randrange(net.n)
        if target == start:
            return None
        trip = Trip(
            plate=plate,
            global_id=gid or "",
            vtype=vtype if vtype in gen.TYPE_SPEED_CAP else "car",
            color=color,
            node=start,
            speed=0.0,
            target=target,
            hops_left=rng.randint(3, 9),
            bias=rng.uniform(*(gen.SPEEDER_BIAS if rng.random() < gen.SPEEDER_RATE
                               else gen.NORMAL_BIAS)) * gen.TYPE_SPEED_FACTOR.get(vtype, 1.0),
            cap=gen.TYPE_SPEED_CAP.get(vtype, 135.0),
            due=epoch,
            via_api=poi_plate is not None,
        )
        # _plan() advances trip.node to the next junction and sets trip.speed to
        # the speed on the leg being entered — which is also the right speed to
        # report at the departure junction. So plan first, emit at the original
        # node, then leave trip.node on the next hop.
        start_node = trip.node
        if not self._plan(trip, epoch):
            return None
        next_node = trip.node
        trip.node = start_node
        self._emit(trip, epoch)
        trip.node = next_node
        self.active_plates.add(plate)
        return trip

    def _sample_type(self) -> str:
        r = self.rng.random()
        acc = 0.0
        for t, w in zip(gen.VEHICLE_TYPES, gen.VEHICLE_TYPE_WEIGHTS):
            acc += w
            if r < acc:
                return t
        return "car"

    # ── emission ─────────────────────────────────────────────────────────────

    def _emit(self, trip: Trip, epoch: float) -> None:
        net = self.net
        cam = net.ids[trip.node]
        ts = datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None)
        spd = round(trip.speed, 1)
        payload = {
            "camera_id": cam,
            "timestamp": ts.isoformat(sep=" "),
            "plate": trip.plate,
            "plate_confidence": round(self.rng.uniform(0.82, 0.99), 2),
            "latitude": net.lat[trip.node],
            "longitude": net.lon[trip.node],
            "direction": net.direction[trip.node],
            "vehicle_type": trip.vtype,
            "vehicle_color": trip.color,
            "speed": spd,
        }
        if trip.via_api:
            self.api_queue.append(payload)
        else:
            self.seq += 1
            # local_track_id is a *per-camera* tracker id, matching what the
            # single-camera trackers actually emit — a global sequence here
            # would be a different thing wearing the same column name.
            self.track[trip.node] += 1
            self.batch.append((
                f"{self.id_prefix}{self.seq:08d}", cam, str(self.track[trip.node]),
                payload["timestamp"],
                trip.plate, payload["plate_confidence"], payload["latitude"],
                payload["longitude"], payload["direction"], trip.vtype, trip.color, spd,
                trip.global_id or f"{self.gid_prefix}{self.seq:07d}",
                payload["timestamp"],
            ))
        self._account(cam, payload["timestamp"], spd, trip.plate)

    def _account(self, cam: str, ts: str, speed: float, plate: str) -> None:
        hb = ts[:13] + ":00:00"
        key = (cam, hb)
        b = self.hour_buckets.get(key)
        if b is None:
            b = self.hour_buckets[key] = Bucket()
        b.count += 1
        b.speed_sum += speed
        b.plates.add(plate)

        t = self.cam_totals.get(cam)
        if t is None:
            self.cam_totals[cam] = [1, speed, ts]
        else:
            t[0] += 1
            t[1] += speed
            t[2] = ts

    # ── persistence ──────────────────────────────────────────────────────────

    def _flush_batch(self) -> None:
        if not self.batch:
            return
        self.cur.execute("BEGIN")
        self.cur.executemany(gen.INSERT_EVENT_SQL, self.batch)
        self.cur.execute("COMMIT")
        self.inserted += len(self.batch)
        self.batch.clear()

    def _flush_api(self) -> None:
        """
        Watchlist events through the production ingestion function, so tracking
        and alert_service actually run and the alert lands in the same table the
        dashboard is polling.
        """
        if not self.api_queue:
            return
        from backend.services.kafka_consumer import process_event_payload

        db = self.SessionLocal()
        try:
            for payload in self.api_queue:
                res = process_event_payload(payload, db)
                if "error" in res:
                    print(f"\n  ! POI ingest rejected: {res['error']}")
                    continue
                self.poi_events += 1
                # process_event_payload runs alert_service.check_and_fire, which
                # writes the row itself. Counting alerts would mean a COUNT(*)
                # on a growing table per event; the blacklist invariant is
                # cheaper and just as true — every POI sighting is a hit.
                self.poi_alerts += 1
            db.commit()
        finally:
            db.close()
            self.api_queue.clear()

    def _refresh_aggregates(self) -> None:
        """
        Merge this process's counters into the rollup tables with UPSERTs.

        In SQLite's DO UPDATE, a bare column reference is the *existing* row's
        value, so the weighted-average update below reads the old count and the
        old mean regardless of the order of the SET clauses.
        """
        now_s = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
        cur = self.cur
        cur.execute("BEGIN")

        hourly_rows = []
        for (cam, hb), b in self.hour_buckets.items():
            if b.count == 0:
                continue
            new_plates = len(b.plates) - b.plates_reported
            hourly_rows.append((cam, hb, b.count, b.speed_sum / b.count, max(0, new_plates)))
        if hourly_rows:
            cur.executemany(
                "INSERT INTO camera_hourly (camera_id, hour_bucket, vehicle_count, avg_speed, "
                "  unique_vehicles) VALUES (?,?,?,?,?) "
                "ON CONFLICT(camera_id, hour_bucket) DO UPDATE SET "
                "  avg_speed = (COALESCE(avg_speed,0) * vehicle_count + "
                "               excluded.avg_speed * excluded.vehicle_count) / "
                "              (vehicle_count + excluded.vehicle_count), "
                "  vehicle_count = vehicle_count + excluded.vehicle_count, "
                "  unique_vehicles = unique_vehicles + excluded.unique_vehicles",
                hourly_rows,
            )

        total_rows = []
        for cam, (cnt, spd_sum, last_ts) in self.cam_totals.items():
            i = self.cam_index[cam]
            # peak_hour_count is NOT NULL with only a Python-side default, so a
            # raw INSERT has to spell it out; peak hour is a full-rebuild concern
            # and is left alone on conflict.
            total_rows.append((cam, cnt, 0, spd_sum / cnt, last_ts, last_ts,
                               self.net.road[i], self.net.lat[i], self.net.lon[i], 0, now_s))
        if total_rows:
            cur.executemany(
                "INSERT INTO camera_totals (camera_id, vehicle_count, unique_vehicles, avg_speed, "
                "  first_seen, last_seen, road, latitude, longitude, peak_hour_count, computed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(camera_id) DO UPDATE SET "
                "  avg_speed = (COALESCE(avg_speed,0) * vehicle_count + "
                "               excluded.avg_speed * excluded.vehicle_count) / "
                "              (vehicle_count + excluded.vehicle_count), "
                "  vehicle_count = vehicle_count + excluded.vehicle_count, "
                "  last_seen = excluded.last_seen, computed_at = excluded.computed_at",
                total_rows,
            )

        seg_rows = []
        for (a, b), (cnt, mins_sum, min_min, dist) in self.segments.items():
            if a == b:
                continue
            ia, ib = self.cam_index[a], self.cam_index[b]
            avg_min = mins_sum / cnt
            seg_rows.append((
                a, b, cnt, avg_min, dist / (avg_min / 60.0) if avg_min else None,
                dist / (min_min / 60.0) if min_min else None, dist,
                self.net.road[ia], self.net.road[ib],
                f"{self.net.road[ia]} → {self.net.road[ib]}",
                round((self.net.lat[ia] + self.net.lat[ib]) / 2, 6),
                round((self.net.lon[ia] + self.net.lon[ib]) / 2, 6), now_s,
            ))
        if seg_rows:
            cur.executemany(
                "INSERT INTO road_usage (from_camera_id, to_camera_id, trip_count, "
                "  avg_travel_minutes, avg_speed_kmh, max_speed_kmh, distance_km, from_road, "
                "  to_road, road_label, mid_latitude, mid_longitude, computed_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(from_camera_id, to_camera_id) DO UPDATE SET "
                "  avg_travel_minutes = (COALESCE(avg_travel_minutes,0) * trip_count + "
                "                        excluded.avg_travel_minutes * excluded.trip_count) / "
                "                       (trip_count + excluded.trip_count), "
                "  trip_count = trip_count + excluded.trip_count, "
                "  avg_speed_kmh = excluded.distance_km / "
                "    (((COALESCE(avg_travel_minutes,0) * trip_count + "
                "       excluded.avg_travel_minutes * excluded.trip_count) / "
                "      (trip_count + excluded.trip_count)) / 60.0), "
                "  max_speed_kmh = MAX(COALESCE(max_speed_kmh,0), COALESCE(excluded.max_speed_kmh,0)), "
                "  computed_at = excluded.computed_at",
                seg_rows,
            )

        delta = sum(b.count for b in self.hour_buckets.values())
        speed_sum = sum(b.speed_sum for b in self.hour_buckets.values())
        # Only plates this process *invented* are new to the dataset. Summing
        # the per-bucket unique counts instead would count one plate once per
        # camera it drove past, on top of the history that already contains it —
        # a >50% over-count on the dashboard's headline number.
        new_uniques = self.new_plates - self.new_plates_reported
        self.new_plates_reported = self.new_plates
        cur.execute(
            "UPDATE dataset_kpi SET total_events = total_events + ?, "
            "  unique_vehicles = unique_vehicles + ?, "
            "  avg_speed_kmh = CASE WHEN ? > 0 THEN "
            "      (COALESCE(avg_speed_kmh, 0) * total_events + ?) / (total_events + ?) "
            "    ELSE avg_speed_kmh END, "
            "  last_event_at = ?, computed_at = ? WHERE scope = 'global'",
            (delta, new_uniques, delta, speed_sum, delta, now_s, now_s),
        )
        cur.execute("COMMIT")

        # Counters consumed. Keep the plate sets for the *current* hour so the
        # next refresh reports only genuinely new plates, and drop older hours
        # so a long-running demo doesn't accumulate them forever.
        current_hour = now_s[:13]
        for key in list(self.hour_buckets):
            b = self.hour_buckets[key]
            if key[1][:13] == current_hour:
                b.plates_reported = len(b.plates)
                b.count = 0
                b.speed_sum = 0.0
            else:
                del self.hour_buckets[key]
        self.cam_totals.clear()
        self.segments.clear()
        self.refreshes += 1

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        from backend.database import SessionLocal
        self.SessionLocal = SessionLocal

        raw, conn = gen.open_raw(self.engine)
        self.raw, self.cur = raw, conn.cursor()
        self.cur.execute("PRAGMA journal_mode=WAL")
        self.cur.execute("PRAGMA synchronous=NORMAL")   # a live feed can afford a fsync per commit
        self.cur.execute("PRAGMA busy_timeout=10000")   # the API may be reading concurrently
        self.batch: list[tuple] = []
        self.api_queue: list[dict] = []

        signal.signal(signal.SIGINT, self._stop)
        signal.signal(signal.SIGTERM, self._stop)

        print("=" * 78)
        print("LIVE ANPR EVENT FEEDER")
        print("=" * 78)
        print(f"  database        {self.db_url}")
        print(f"  cameras         {self.net.n} active '{gen.DEPLOYMENT_TAG}' junctions, "
              f"{self.net.edge_count()} segments")
        print(f"  reuse pool      {len(self.pool):,} existing plates "
              f"(new-plate rate {self.args.new_plate_rate:.0%})")
        print(f"  target rate     {self.args.events_per_sec} events/sec"
              + ("  (modulated by rush hour)" if self.args.diurnal_rate else ""))
        print(f"  POI routing     every {self.args.poi_interval}s via "
              f"kafka_consumer.process_event_payload -> live BLACKLIST alerts")
        print(f"  agg refresh     every {self.args.agg_refresh}s (incremental UPSERT)")
        print("  Ctrl-C to stop.\n")

        now = time.time()
        self._prewarm(now)

        budget = 0.0
        last = time.time()
        while self.running:
            now = time.time()
            rate = self.args.events_per_sec
            if self.args.diurnal_rate:
                rate *= gen.HOUR_WEIGHTS[self._ist_hour(now)] / self.mean_hour_weight
            budget = min(budget + (now - last) * rate, rate * 2.0)
            last = now

            while self.heap and self.heap[0][0] <= now:
                _, _, trip = heapq.heappop(self.heap)
                self._emit(trip, trip.due)
                budget -= 1.0
                if self._plan(trip, trip.due):
                    self.tiebreak += 1
                    heapq.heappush(self.heap, (trip.due, self.tiebreak, trip))
                else:
                    self.active_plates.discard(trip.plate)

            while budget >= 1.0 and len(self.heap) < self.max_inflight:
                trip = self._spawn(now)
                budget -= 1.0
                if trip:
                    self.tiebreak += 1
                    heapq.heappush(self.heap, (trip.due, self.tiebreak, trip))

            if now - self.last_poi >= self.args.poi_interval:
                self.last_poi = now
                plate = gen.POI_PLATES[self.rng.randrange(len(gen.POI_PLATES))]
                trip = self._spawn(now, poi_plate=plate)
                if trip:
                    self.tiebreak += 1
                    heapq.heappush(self.heap, (trip.due, self.tiebreak, trip))

            self._flush_batch()
            self._flush_api()

            if now - self.last_agg >= self.args.agg_refresh:
                self.last_agg = now
                self._refresh_aggregates()

            if now - self.last_beat >= 1.0:
                self._heartbeat(now)

            time.sleep(TICK_SECONDS)

        self._shutdown()

    def _prewarm(self, now: float) -> None:
        """
        Seed trips with due times spread across one hop interval.

        Without this, the first ~5 minutes of the stream are almost entirely
        first-sightings, because a freshly spawned trip's second hop is minutes
        away. The demo would open on a wall of one-event trajectories.
        """
        want = min(self.max_inflight, int(self.args.events_per_sec * AVG_HOP_SECONDS))
        made = 0
        for _ in range(want * 2):
            if made >= want:
                break
            # Backdate each trip's opening sighting into the last hop interval
            # so the warm-up doesn't land as one instantaneous spike of
            # thousands of events at the same second.
            t0 = now - self.rng.random() * AVG_HOP_SECONDS
            trip = self._spawn(t0)
            if not trip:
                continue
            # Push the next hop into the coming window so arrivals are smooth
            # from the first tick. max() never moves an arrival *earlier* than
            # the planned travel time, so the hop stays physically valid — the
            # vehicle just dawdles a little longer at the junction.
            trip.due = max(trip.due, now + self.rng.random() * AVG_HOP_SECONDS)
            self.tiebreak += 1
            heapq.heappush(self.heap, (trip.due, self.tiebreak, trip))
            made += 1
        self._flush_batch()
        print(f"  pre-warmed {made:,} in-flight trips "
              f"({self.inserted:,} back-dated warm-up sightings)\n")
        # Don't let the warm-up burst show up as the first heartbeat's rate.
        self.last_beat = time.time()
        self.beat_base = self.inserted

    def _heartbeat(self, now: float) -> None:
        elapsed = now - self.last_beat if self.last_beat else 1.0
        rate = (self.inserted - self.beat_base) / elapsed
        self.last_beat = now
        self.beat_base = self.inserted
        print(f"\r  [{datetime.now():%H:%M:%S}] {rate:6.1f} ev/s | "
              f"inserted {self.inserted:>9,} | in-flight {len(self.heap):>6,} | "
              f"POI {self.poi_events:>4} ({self.poi_alerts} alerts) | "
              f"agg x{self.refreshes}", end="", flush=True)

    def _stop(self, *_):
        self.running = False

    def _shutdown(self) -> None:
        print("\n\n  stopping — flushing...")
        self._flush_batch()
        self._flush_api()
        self._refresh_aggregates()
        self.cur.close()
        self.raw.close()
        wall = time.time() - self.started
        print("=" * 78)
        print(f"  ran for            {wall / 60:.1f} min")
        print(f"  events inserted    {self.inserted:,} ({self.inserted / wall:.1f}/sec avg)")
        print(f"  POI events via API {self.poi_events:,} -> {self.poi_alerts:,} BLACKLIST alerts")
        print(f"  new plates seen    {self.new_plates:,}")
        print(f"  aggregate refreshes {self.refreshes}")
        print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events-per-sec", type=float, default=40.0)
    ap.add_argument("--diurnal-rate", action="store_true",
                    help="scale the rate by the Delhi rush-hour curve instead of holding it flat")
    ap.add_argument("--new-plate-rate", type=float, default=0.03,
                    help="fraction of trips that introduce a never-seen plate")
    ap.add_argument("--poi-interval", type=float, default=45.0,
                    help="seconds between routing a blacklisted POI vehicle through the city")
    ap.add_argument("--agg-refresh", type=float, default=30.0,
                    help="seconds between incremental aggregate UPSERTs")
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop after N seconds (0 = run until Ctrl-C); useful for smoke tests")
    args = ap.parse_args()

    feeder = Feeder(args)
    if args.max_seconds > 0:
        deadline = time.time() + args.max_seconds
        original = feeder._heartbeat

        def bounded(now):
            original(now)
            if now >= deadline:
                feeder.running = False

        feeder._heartbeat = bounded
    feeder.run()


if __name__ == "__main__":
    main()

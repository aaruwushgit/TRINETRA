"""
The simulation clock — turns the staged future month into a live event stream.

The platform is demonstrated over a two-month window centred on now. The past
month is already in `vehicle_events`; the next month sits in `future_events`
(see backend/models/future_event.py). This service is the thing in the middle:
a clock that walks forward through the staged month and promotes each sighting
into the live table at the moment it comes due, **one by one**.

Everything else in the "live" half of the product is a consequence of that:

    future_events ──tick──▶ vehicle_events ──▶ live feed      (this window's stream)
                                          ├──▶ live heatmap   (rolling N-minute counts)
                                          ├──▶ live trajectory(a plate's trail as it forms)
                                          └──▶ next-hop prediction + confidence

Why a clock rather than "insert some rows every second"
-------------------------------------------------------
Because predictions have to be falsifiable. A prediction made at sim-time T
about where a vehicle will be at T+8min can be scored the moment the clock
reaches T+8min, against a row that was written before the prediction was made
and that the predictor never read. That is a real measurement. A feeder that
invents events as it goes could never be wrong.

The clock also makes the demo controllable: `speed` compresses sim-time against
wall-time (60x = one simulated hour per real minute), so a month of city traffic
is watchable in an afternoon, and the whole thing can be paused mid-sentence.

Threading
---------
One daemon thread owns the promotion loop and the only write session. The API
handlers read from in-memory buffers under a lock and never touch the future
table, so a burst of dashboard polling cannot slow ingestion down. WebSocket
subscribers get their own bounded queue; a subscriber that stops draining is
dropped rather than allowed to back-pressure the clock.

The rolling buffers are deliberately bounded and in-process. They are a *view*
of the last few minutes, not a store — the durable record is `vehicle_events`,
which is where the ordinary analytics endpoints read from. Restarting the API
loses the view and keeps the data, which is the right way round.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)

# ── tuning ───────────────────────────────────────────────────────────────────

# Wall-clock seconds between promotion passes. Half a second keeps the live
# feed visibly continuous without turning the clock into a busy loop.
TICK_SECONDS = 0.5

# Ceiling on rows promoted in a single tick. Without it, starting the clock
# with a month of already-due rows staged would try to insert 1.5M rows in one
# statement. With it, the backlog drains at a visible, controllable rate.
MAX_PROMOTE_PER_TICK = 4000

# The live view. 20k hits at ~35/s is roughly the last ten minutes of sim-time
# at 60x, which is what the heatmap and the feed actually display.
LIVE_BUFFER = 20_000

# Rolling window the live heatmap sums over, in *simulated* minutes.
DEFAULT_HEATMAP_MINUTES = 15

# How many recent historical events to fit the transition model on. One index
# range scan over the tail of `timestamp`; large enough that a busy camera has
# hundreds of observed departures, small enough to build in a couple of seconds.
TRANSITION_SAMPLE = 300_000

# Shrinkage constant for the confidence score. A camera with 5 observed
# departures should not report a confident prediction however lopsided those 5
# happen to be; n/(n+PRIOR) is the standard way to say so.
CONFIDENCE_PRIOR = 20.0

INSERT_EVENT_SQL = (
    "INSERT INTO vehicle_events "
    "(event_id, camera_id, local_track_id, timestamp, plate, plate_confidence, "
    " latitude, longitude, direction, vehicle_type, vehicle_color, speed, "
    " global_vehicle_id, created_at) "
    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)

SELECT_DUE_SQL = (
    "SELECT future_id, camera_id, local_track_id, timestamp, plate, plate_confidence, "
    "       latitude, longitude, direction, vehicle_type, vehicle_color, speed, "
    "       global_vehicle_id "
    "FROM future_events "
    "WHERE released_at IS NULL AND timestamp <= ? "
    "ORDER BY timestamp ASC LIMIT ?"
)


# ── state ────────────────────────────────────────────────────────────────────

@dataclass
class LiveHit:
    """One promoted sighting, as the live window renders it."""
    event_id: str
    camera_id: str
    camera_name: str | None
    plate: str | None
    plate_confidence: float | None
    latitude: float | None
    longitude: float | None
    direction: str | None
    vehicle_type: str | None
    vehicle_color: str | None
    speed: float | None
    global_vehicle_id: str | None
    sim_timestamp: datetime
    released_at: datetime
    watchlisted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "plate": self.plate,
            "plate_confidence": self.plate_confidence,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "direction": self.direction,
            "vehicle_type": self.vehicle_type,
            "vehicle_color": self.vehicle_color,
            "speed": self.speed,
            "global_vehicle_id": self.global_vehicle_id,
            "timestamp": self.sim_timestamp.isoformat(),
            "released_at": self.released_at.isoformat(),
            "watchlisted": self.watchlisted,
        }


@dataclass
class Counters:
    ticks: int = 0
    promoted: int = 0
    watchlist_hits: int = 0
    errors: int = 0
    last_error: str | None = None
    started_wall: datetime | None = None
    last_tick_wall: datetime | None = None
    last_tick_ms: float = 0.0
    max_tick_ms: float = 0.0
    backlog_at_last_tick: int = 0


@dataclass
class TransitionModel:
    """P(next camera | current camera), fitted on recent history.

    Counts only, deliberately. A heavier model (a sequence net over the last k
    hops) would need training infrastructure and a GPU to serve, and on this
    data would be fitting the same first-order structure — vehicles follow
    roads, and the road you are on determines where you can go next. What the
    counts *cannot* do is express uncertainty, so that is added explicitly
    rather than hidden: see `confidence`.
    """
    counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    totals: Counter = field(default_factory=Counter)
    fitted_at: datetime | None = None
    fitted_on_events: int = 0
    live_updates: int = 0

    def observe(self, from_cam: str, to_cam: str) -> None:
        if from_cam == to_cam:
            return
        self.counts[from_cam][to_cam] += 1
        self.totals[from_cam] += 1

    def distribution(self, camera_id: str) -> list[tuple[str, float, int]]:
        """[(camera_id, probability, observed_count)], most likely first."""
        total = self.totals.get(camera_id, 0)
        if not total:
            return []
        return [
            (cam, n / total, n)
            for cam, n in self.counts[camera_id].most_common()
        ]


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """95% lower bound on a proportion.

    3 of 4 and 750 of 1000 are both 0.75, and only one of them is worth acting
    on. The lower bound separates them (0.30 vs 0.72), which is exactly the
    distinction an operator deciding where to send a unit needs.
    """
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials)
    return max(0.0, (centre - margin) / denom)


def _normalised_entropy(probs: list[float]) -> float:
    """0 = one certain outcome, 1 = every outcome equally likely."""
    live = [p for p in probs if p > 0]
    if len(live) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in live)
    return h / math.log(len(live))


# ── the clock ────────────────────────────────────────────────────────────────

class SimulationClock:
    """Owns sim-time, the promotion loop, and the live view of the last minutes."""

    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    EXHAUSTED = "EXHAUSTED"   # clock still ticking, nothing left staged

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.state = self.STOPPED
        self.speed = 60.0
        self._sim_time: datetime = datetime.now(timezone.utc).replace(tzinfo=None)
        self._anchor_wall: float = time.monotonic()
        self._anchor_sim: datetime = self._sim_time

        self.counters = Counters()
        self.model = TransitionModel()
        # Set when start() had to jump the clock over a gap before the first
        # staged event. Reported, not hidden — the clock moved.
        self.skipped_to: datetime | None = None

        # The live view. One buffer serves the feed, the heatmap and the
        # trails: three deques would triple the memory and could disagree with
        # each other about what "the last ten minutes" means.
        self._hits: deque[LiveHit] = deque(maxlen=LIVE_BUFFER)
        self._trails: dict[str, deque[LiveHit]] = {}
        self._camera_names: dict[str, str] = {}
        self._camera_pos: dict[str, tuple[float, float]] = {}
        self._watchlist: set[str] = set()

        # Bounded queues, one per WebSocket subscriber.
        self._subscribers: list[deque[dict[str, Any]]] = []

    # ── sim-time ─────────────────────────────────────────────────────────

    @property
    def sim_time(self) -> datetime:
        """Current simulated instant, interpolated between ticks.

        Derived from a wall-clock anchor rather than accumulated per tick, so a
        slow tick (a long insert, a GC pause) does not make sim-time drift
        behind wall-time permanently.
        """
        with self._lock:
            if self.state != self.RUNNING:
                return self._sim_time
            elapsed = time.monotonic() - self._anchor_wall
            return self._anchor_sim + timedelta(seconds=elapsed * self.speed)

    def _reanchor(self, sim_time: datetime | None = None) -> None:
        self._sim_time = sim_time if sim_time is not None else self.sim_time
        self._anchor_sim = self._sim_time
        self._anchor_wall = time.monotonic()

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self, speed: float | None = None, start_at: datetime | None = None,
              skip_to_first: bool = True, reset_view: bool = False) -> dict[str, Any]:
        with self._lock:
            if speed is not None:
                self.speed = max(0.1, min(float(speed), 10_000.0))
            if start_at is not None:
                self._sim_time = _naive(start_at)
            elif self.state == self.STOPPED:
                self._sim_time = datetime.now(timezone.utc).replace(tzinfo=None)

            # Close the dead zone.
            #
            # The staged month begins at the first trip the generator happened
            # to produce, which can be hours after "now". Starting the clock at
            # now would then show nothing at all until sim-time crosses that
            # gap — six real minutes of an apparently broken feed at 60x. So
            # if nothing is due yet, jump to just before the first staged
            # event. `skipped_to` in the status records that this happened,
            # rather than silently moving the clock.
            self.skipped_to = None
            if start_at is None and skip_to_first:
                first = self._first_unreleased_timestamp()
                if first is not None and first > self._sim_time:
                    self.skipped_to = first
                    self._sim_time = first - timedelta(seconds=1)

            if reset_view:
                self._hits.clear()
                self._trails.clear()
            self._reanchor(self._sim_time)
            self.state = self.RUNNING
            self.counters.started_wall = datetime.now(timezone.utc)

            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(
                    target=self._run, name="simulation-clock", daemon=True
                )
                self._thread.start()
        return self.status()

    def _first_unreleased_timestamp(self) -> datetime | None:
        """Earliest staged sighting not yet promoted, or None if nothing is staged."""
        from backend.database import engine

        try:
            raw = engine.raw_connection()
            try:
                cur = raw.driver_connection.cursor()
                # MIN over the (released_at, timestamp) index — an index probe,
                # not a scan of the staged month.
                row = cur.execute(
                    "SELECT MIN(timestamp) FROM future_events WHERE released_at IS NULL"
                ).fetchone()
                cur.close()
            finally:
                raw.close()
        except Exception:
            return None
        return _parse_db_ts(row[0]) if row and row[0] else None

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self.state == self.RUNNING:
                self._reanchor()          # freeze sim-time where it stands
                self.state = self.PAUSED
        return self.status()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self.state in (self.PAUSED, self.EXHAUSTED):
                self._reanchor(self._sim_time)
                self.state = self.RUNNING
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self.state = self.STOPPED
            self._reanchor()
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=3.0)
        with self._lock:
            self._thread = None
        return self.status()

    def set_speed(self, speed: float) -> dict[str, Any]:
        with self._lock:
            # Re-anchor before changing the multiplier, or the elapsed real
            # seconds since the last anchor would be retroactively rescaled and
            # sim-time would jump.
            self._reanchor()
            self.speed = max(0.1, min(float(speed), 10_000.0))
            self._anchor_wall = time.monotonic()
        return self.status()

    def seek(self, to: datetime) -> dict[str, Any]:
        """Move the clock. Rewinding un-releases anything after the new instant."""
        target = _naive(to)
        with self._lock:
            rewinding = target < self._sim_time
            self._reanchor(target)
        if rewinding:
            self._unrelease_after(target)
            with self._lock:
                self._hits.clear()
                self._trails.clear()
        return self.status()

    def reset(self) -> dict[str, Any]:
        """Back to now, nothing released, buffers empty."""
        self.stop()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self._unrelease_after(now)
        with self._lock:
            self._sim_time = now
            self._reanchor(now)
            self._hits.clear()
            self._trails.clear()
            self.counters = Counters()
        return self.status()

    # ── the loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._load_context()
        except Exception as exc:  # noqa: BLE001
            log.exception("simulation: could not load camera/watchlist context")
            with self._lock:
                self.counters.last_error = f"context load failed: {exc}"

        try:
            self.fit_transition_model()
        except Exception as exc:  # noqa: BLE001
            log.exception("simulation: could not fit the transition model")
            with self._lock:
                self.counters.last_error = f"model fit failed: {exc}"

        while not self._stop.is_set():
            if self.state != self.RUNNING:
                self._stop.wait(TICK_SECONDS)
                continue
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001 — a bad tick must not kill the clock
                log.exception("simulation: tick failed")
                with self._lock:
                    self.counters.errors += 1
                    self.counters.last_error = f"{type(exc).__name__}: {exc}"
                # Back off after a failure so a persistent problem (a locked
                # database) does not spin at full rate writing log lines.
                self._stop.wait(2.0)
                continue
            ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.counters.ticks += 1
                self.counters.last_tick_ms = ms
                self.counters.max_tick_ms = max(self.counters.max_tick_ms, ms)
                self.counters.last_tick_wall = datetime.now(timezone.utc)
            self._stop.wait(TICK_SECONDS)

    def _tick(self) -> None:
        from backend.database import engine

        now_sim = self.sim_time
        cutoff = now_sim.strftime("%Y-%m-%d %H:%M:%S.%f")

        raw = engine.raw_connection()
        try:
            dbapi = raw.driver_connection
            dbapi.isolation_level = None
            cur = dbapi.cursor()
            cur.execute("PRAGMA busy_timeout=8000")
            cur.execute(SELECT_DUE_SQL, (cutoff, MAX_PROMOTE_PER_TICK))
            due = cur.fetchall()

            if not due:
                remaining = cur.execute(
                    "SELECT COUNT(*) FROM future_events WHERE released_at IS NULL"
                ).fetchone()[0]
                with self._lock:
                    self.counters.backlog_at_last_tick = 0
                    if remaining == 0 and self.state == self.RUNNING:
                        # Nothing staged ahead of us. Keep the clock running —
                        # sim-time still advances, so the dashboard's windows
                        # stay coherent — but say so, rather than looking like
                        # a stall.
                        self.state = self.EXHAUSTED
                cur.close()
                return

            released_at = datetime.now(timezone.utc).replace(tzinfo=None)
            released_str = released_at.strftime("%Y-%m-%d %H:%M:%S.%f")

            rows = []
            hits: list[LiveHit] = []
            for r in due:
                (_fid, camera_id, track, ts, plate, conf, lat, lon, direction,
                 vtype, color, speed, gid) = r
                event_id = uuid.uuid4().hex
                rows.append((event_id, camera_id, track, ts, plate, conf, lat, lon,
                             direction, vtype, color, speed, gid, released_str))
                hits.append(LiveHit(
                    event_id=event_id,
                    camera_id=camera_id,
                    camera_name=self._camera_names.get(camera_id),
                    plate=plate,
                    plate_confidence=conf,
                    latitude=lat,
                    longitude=lon,
                    direction=direction,
                    vehicle_type=vtype,
                    vehicle_color=color,
                    speed=speed,
                    global_vehicle_id=gid,
                    sim_timestamp=_parse_db_ts(ts),
                    released_at=released_at,
                    watchlisted=bool(plate) and plate in self._watchlist,
                ))

            ids = [r[0] for r in due]
            cur.execute("BEGIN")
            cur.executemany(INSERT_EVENT_SQL, rows)
            cur.executemany(
                "UPDATE future_events SET released_at = ? WHERE future_id = ?",
                [(released_str, fid) for fid in ids],
            )
            cur.execute("COMMIT")

            backlog = cur.execute(
                "SELECT COUNT(*) FROM future_events "
                "WHERE released_at IS NULL AND timestamp <= ?", (cutoff,)
            ).fetchone()[0]
            cur.close()
        finally:
            raw.close()

        self._record(hits, backlog)

    def _record(self, hits: list[LiveHit], backlog: int) -> None:
        """Fold a batch of promotions into the live view and the model."""
        watch_hits = []
        with self._lock:
            self.counters.promoted += len(hits)
            self.counters.backlog_at_last_tick = backlog
            for hit in hits:
                self._hits.append(hit)
                key = hit.plate or hit.global_vehicle_id
                if key:
                    trail = self._trails.get(key)
                    if trail is None:
                        # Long enough to draw a journey, short enough that
                        # 60k tracked vehicles do not become a memory leak.
                        trail = self._trails.setdefault(key, deque(maxlen=40))
                    if trail and trail[-1].camera_id != hit.camera_id:
                        # Learn the transition the moment it happens, so the
                        # model reflects tonight's traffic and not last week's.
                        self.model.observe(trail[-1].camera_id, hit.camera_id)
                        self.model.live_updates += 1
                    trail.append(hit)
                if hit.watchlisted:
                    self.counters.watchlist_hits += 1
                    watch_hits.append(hit)

            # Bound the trail table. Dropping the least recently touched
            # vehicles is right: a plate not seen in the live window is one the
            # live view has nothing to say about.
            if len(self._trails) > 80_000:
                stale = sorted(
                    self._trails.items(),
                    key=lambda kv: kv[1][-1].released_at,
                )[:20_000]
                for key, _ in stale:
                    self._trails.pop(key, None)

        self._publish({
            "type": "hits",
            "sim_time": self.sim_time.isoformat(),
            "count": len(hits),
            "backlog": backlog,
            # Cap the payload: a subscriber does not need 4,000 rows a tick to
            # render a scrolling feed, and a 2 MB frame every half second would
            # make the browser the bottleneck.
            "hits": [h.as_dict() for h in hits[-120:]],
        })
        for hit in watch_hits:
            self._publish({
                "type": "watchlist",
                "sim_time": self.sim_time.isoformat(),
                "hit": hit.as_dict(),
            })

    # ── context + model ──────────────────────────────────────────────────

    def _load_context(self) -> None:
        """Camera names/positions and the watchlist, read once per run."""
        from backend.database import SessionLocal
        from backend.models.alert import Blacklist
        from backend.models.camera import Camera

        session = SessionLocal()
        try:
            names, pos = {}, {}
            for cam in session.query(
                Camera.camera_id, Camera.name, Camera.latitude, Camera.longitude
            ).all():
                names[cam.camera_id] = cam.name
                if cam.latitude is not None and cam.longitude is not None:
                    pos[cam.camera_id] = (cam.latitude, cam.longitude)
            watch = {b.plate for b in session.query(Blacklist).all() if b.plate}
        finally:
            session.close()

        with self._lock:
            self._camera_names = names
            self._camera_pos = pos
            self._watchlist = watch

    def fit_transition_model(self) -> dict[str, Any]:
        """Fit P(next | current) on the tail of history.

        One index range scan over `timestamp` DESC with a LIMIT — not a scan of
        the 12M-row table. Recency is a feature, not a compromise: a transition
        model for "where does traffic go from here *now*" should be fitted on
        now, and a month-wide fit would average this evening's flows with a
        public holiday three weeks ago.
        """
        from backend.database import engine

        t0 = time.perf_counter()
        raw = engine.raw_connection()
        try:
            cur = raw.driver_connection.cursor()
            cur.execute(
                "SELECT global_vehicle_id, camera_id, timestamp FROM vehicle_events "
                "WHERE global_vehicle_id IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?", (TRANSITION_SAMPLE,)
            )
            rows = cur.fetchall()
            cur.close()
        finally:
            raw.close()

        # Rows arrive newest-first and interleaved across vehicles, so group by
        # vehicle in Python and order within each group. Asking SQLite to
        # ORDER BY global_vehicle_id, timestamp over the same window would cost
        # an external sort of the whole sample.
        by_vehicle: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for gid, cam, ts in rows:
            by_vehicle[gid].append((ts, cam))

        model = TransitionModel()
        for hops in by_vehicle.values():
            if len(hops) < 2:
                continue
            hops.sort()
            prev = hops[0][1]
            for _ts, cam in hops[1:]:
                model.observe(prev, cam)
                prev = cam

        model.fitted_at = datetime.now(timezone.utc)
        model.fitted_on_events = len(rows)

        with self._lock:
            self.model = model

        return {
            "fitted_on_events": len(rows),
            "vehicles": len(by_vehicle),
            "source_cameras": len(model.totals),
            "transitions": int(sum(model.totals.values())),
            "seconds": round(time.perf_counter() - t0, 2),
        }

    # ── reads ────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        with self._lock:
            c = self.counters
            sim = self.sim_time
            wall = datetime.now(timezone.utc).replace(tzinfo=None)
            return {
                "state": self.state,
                "speed": self.speed,
                "skipped_to_first_event": (
                    self.skipped_to.isoformat() if self.skipped_to else None
                ),
                "sim_time": sim.isoformat(),
                "wall_time": wall.isoformat(),
                # Positive = the simulation is ahead of real time, which is the
                # normal state at any speed above 1x.
                "sim_ahead_of_wall_seconds": round((sim - wall).total_seconds(), 1),
                "live_buffer": len(self._hits),
                "tracked_vehicles": len(self._trails),
                "counters": {
                    "ticks": c.ticks,
                    "events_promoted": c.promoted,
                    "watchlist_hits": c.watchlist_hits,
                    "errors": c.errors,
                    "last_error": c.last_error,
                    "backlog": c.backlog_at_last_tick,
                    "last_tick_ms": round(c.last_tick_ms, 1),
                    "max_tick_ms": round(c.max_tick_ms, 1),
                    "started": c.started_wall.isoformat() if c.started_wall else None,
                    "last_tick": c.last_tick_wall.isoformat() if c.last_tick_wall else None,
                },
                "model": {
                    "fitted_at": self.model.fitted_at.isoformat() if self.model.fitted_at else None,
                    "fitted_on_events": self.model.fitted_on_events,
                    "source_cameras": len(self.model.totals),
                    "live_updates": self.model.live_updates,
                },
            }

    def recent_hits(self, limit: int = 100, camera_id: str | None = None,
                    watchlist_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            for hit in reversed(self._hits):
                if camera_id and hit.camera_id != camera_id:
                    continue
                if watchlist_only and not hit.watchlisted:
                    continue
                out.append(hit.as_dict())
                if len(out) >= limit:
                    break
            return out

    def live_heatmap(self, minutes: int = DEFAULT_HEATMAP_MINUTES) -> dict[str, Any]:
        """Per-camera counts over the last `minutes` of *simulated* time.

        Summed from the in-memory buffer rather than queried: the equivalent
        SQL is a GROUP BY over the newest rows of a 12M-row table every few
        seconds, and the buffer already holds exactly those rows.
        """
        now = self.sim_time
        since = now - timedelta(minutes=minutes)
        counts: Counter = Counter()
        speeds: dict[str, list[float]] = defaultdict(list)
        plates: dict[str, set[str]] = defaultdict(set)
        oldest: datetime | None = None

        with self._lock:
            for hit in self._hits:
                if hit.sim_timestamp < since:
                    continue
                counts[hit.camera_id] += 1
                if hit.speed is not None:
                    speeds[hit.camera_id].append(hit.speed)
                if hit.plate:
                    plates[hit.camera_id].add(hit.plate)
                if oldest is None or hit.sim_timestamp < oldest:
                    oldest = hit.sim_timestamp
            pos = dict(self._camera_pos)
            names = dict(self._camera_names)
            truncated = len(self._hits) >= LIVE_BUFFER

        peak = max(counts.values()) if counts else 0
        points = []
        for cam, weight in counts.most_common():
            coords = pos.get(cam)
            if coords is None:
                continue
            avg_speed = (sum(speeds[cam]) / len(speeds[cam])) if speeds.get(cam) else None
            points.append({
                "camera_id": cam,
                "camera_name": names.get(cam),
                "latitude": coords[0],
                "longitude": coords[1],
                "weight": weight,
                # Normalised so a Leaflet heat layer does not need to know the
                # absolute scale, which changes as the window slides.
                "intensity": round(weight / peak, 4) if peak else 0.0,
                "unique_plates": len(plates.get(cam, ())),
                "avg_speed_kmh": round(avg_speed, 1) if avg_speed is not None else None,
                "congestion": _congestion_label(avg_speed),
            })

        return {
            "sim_time": now.isoformat(),
            "window_minutes": minutes,
            "window_start": since.isoformat(),
            "total_events": sum(counts.values()),
            "active_cameras": len(points),
            "peak_camera_events": peak,
            "points": points,
            # True when the ring buffer, not the requested window, decided how
            # far back this goes. Saying so beats quietly under-reporting.
            "window_truncated_by_buffer": truncated and oldest is not None and oldest > since,
            "buffer_oldest": oldest.isoformat() if oldest else None,
        }

    def live_trajectory(self, plate: str) -> dict[str, Any]:
        """The trail a vehicle has laid down since the clock started."""
        key = plate.upper().replace(" ", "").replace("-", "")
        with self._lock:
            trail = list(self._trails.get(key, ()))
            names = dict(self._camera_names)

        points = [{
            "camera_id": h.camera_id,
            "camera_name": names.get(h.camera_id),
            "latitude": h.latitude,
            "longitude": h.longitude,
            "timestamp": h.sim_timestamp.isoformat(),
            "speed": h.speed,
            "direction": h.direction,
            "plate_confidence": h.plate_confidence,
        } for h in trail]

        elapsed_s = 0.0
        distance_km = 0.0
        for a, b in zip(trail, trail[1:]):
            elapsed_s += (b.sim_timestamp - a.sim_timestamp).total_seconds()
            if None not in (a.latitude, a.longitude, b.latitude, b.longitude):
                distance_km += _haversine_km(a.latitude, a.longitude, b.latitude, b.longitude)

        return {
            "plate": key,
            "sim_time": self.sim_time.isoformat(),
            "points": points,
            "sightings": len(points),
            "straight_line_km": round(distance_km, 3),
            "elapsed_minutes": round(elapsed_s / 60.0, 1),
            "avg_speed_kmh": (
                round(distance_km / (elapsed_s / 3600.0), 1) if elapsed_s > 0 else None
            ),
            "last_camera": trail[-1].camera_id if trail else None,
            "note": (
                None if trail else
                "This vehicle has not been seen since the simulation clock started. "
                "Its historical trajectory is at GET /vehicles/{plate}/trajectory."
            ),
        }

    def predict(self, plate: str, top_n: int = 3) -> dict[str, Any]:
        """Next-hop prediction for a vehicle the live stream is tracking.

        The probability is the transition model's; the *confidence* is a
        separate, explicit number, because those are different questions.
        "Which exit is most likely?" and "should you act on that?" come apart
        exactly when the evidence is thin or the distribution is flat, which is
        when a single blended score would mislead.

        confidence = P(candidate) x support x decisiveness

          support      n/(n+20) on the observed departures from this camera —
                       4 observations cannot support a confident claim.
          decisiveness 1 - normalised entropy of the distribution. A junction
                       that splits evenly five ways is genuinely unpredictable
                       and should say so.

        `probability_lower_bound` is the 95% Wilson bound on the same count,
        for anyone who would rather reason about the interval directly.
        """
        key = plate.upper().replace(" ", "").replace("-", "")
        with self._lock:
            trail = list(self._trails.get(key, ()))
            model = self.model
            pos = dict(self._camera_pos)
            names = dict(self._camera_names)
            watch = key in self._watchlist

        if not trail:
            return {
                "plate": key,
                "error": "not_tracked_live",
                "detail": (
                    "No live sightings for this plate since the clock started. "
                    "Use GET /vehicles/{plate}/predict-next-location for a "
                    "prediction from history."
                ),
            }

        last = trail[-1]
        dist = model.distribution(last.camera_id)
        total = model.totals.get(last.camera_id, 0)

        if not dist:
            return {
                "plate": key,
                "sim_time": self.sim_time.isoformat(),
                "last_sighting": _sighting_dict(last, names),
                "predictions": [],
                "confidence": 0.0,
                "basis": "no_observed_departures",
                "detail": (
                    f"No vehicle has been observed leaving {last.camera_id} in the "
                    "fitted sample, so there is nothing to predict from."
                ),
            }

        decisiveness = 1.0 - _normalised_entropy([p for _c, p, _n in dist])
        support = total / (total + CONFIDENCE_PRIOR)

        # Recent speed governs the ETA. The vehicle's own last reading beats a
        # camera average: a truck and a bike leaving the same junction do not
        # arrive together.
        speed = last.speed or 30.0
        speed = max(8.0, min(speed, 120.0))

        # Bearing of the last hop, to drop candidates that would be a U-turn.
        heading = None
        if len(trail) >= 2:
            prev = trail[-2]
            if None not in (prev.latitude, prev.longitude, last.latitude, last.longitude):
                heading = _bearing(prev.latitude, prev.longitude, last.latitude, last.longitude)

        out = []
        for cam, prob, n in dist:
            coords = pos.get(cam)
            if coords is None:
                continue
            km = None
            eta = None
            if last.latitude is not None and last.longitude is not None:
                km = _haversine_km(last.latitude, last.longitude, coords[0], coords[1])
                eta = (km / speed) * 60.0
                if heading is not None:
                    to_bearing = _bearing(last.latitude, last.longitude, coords[0], coords[1])
                    if _bearing_similarity(heading, to_bearing) < -0.35:
                        # A hard reversal. Vehicles do turn around, but the
                        # transition counts already include the ones that do;
                        # what this removes is the mirror-image candidate that
                        # the count picked up from traffic going the other way.
                        continue

            conf = prob * support * decisiveness
            out.append({
                "camera_id": cam,
                "camera_name": names.get(cam),
                "latitude": coords[0],
                "longitude": coords[1],
                "probability": round(prob, 4),
                "probability_lower_bound": round(_wilson_lower_bound(n, total), 4),
                "observed_transitions": n,
                "confidence": round(conf, 4),
                "confidence_label": _confidence_label(conf),
                "distance_km": round(km, 2) if km is not None else None,
                "eta_minutes": round(eta, 1) if eta is not None else None,
                "eta_at": (
                    (last.sim_timestamp + timedelta(minutes=eta)).isoformat()
                    if eta is not None else None
                ),
            })
            if len(out) >= top_n:
                break

        top_conf = out[0]["confidence"] if out else 0.0
        return {
            "plate": key,
            "watchlisted": watch,
            "sim_time": self.sim_time.isoformat(),
            "last_sighting": _sighting_dict(last, names),
            "live_sightings": len(trail),
            "predictions": out,
            "confidence": round(top_conf, 4),
            "confidence_label": _confidence_label(top_conf),
            "basis": {
                "observed_departures_from_camera": total,
                "support": round(support, 3),
                "decisiveness": round(decisiveness, 3),
                "model_fitted_on_events": model.fitted_on_events,
                "live_transitions_learned": model.live_updates,
            },
            "suggested_interception": out[0]["camera_id"] if out else None,
        }

    def horizon(self, hours: int = 24) -> dict[str, Any]:
        """What is staged ahead of the clock — the 'future' side of the timeline."""
        from backend.database import engine

        now = self.sim_time
        raw = engine.raw_connection()
        try:
            cur = raw.driver_connection.cursor()
            total, first, last = cur.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM future_events "
                "WHERE released_at IS NULL"
            ).fetchone()
            released, rel_first, rel_last = cur.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM future_events "
                "WHERE released_at IS NOT NULL"
            ).fetchone()
            soon = cur.execute(
                "SELECT COUNT(*) FROM future_events "
                "WHERE released_at IS NULL AND timestamp <= ?",
                ((now + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S.%f"),),
            ).fetchone()[0]
            cur.close()
        finally:
            raw.close()

        return {
            "sim_time": now.isoformat(),
            "staged_remaining": total or 0,
            "staged_window": [first, last],
            "already_released": released or 0,
            "released_window": [rel_first, rel_last],
            f"due_within_{hours}h": soon or 0,
        }

    def _unrelease_after(self, instant: datetime) -> int:
        """Rewind: put promoted rows back on the shelf and delete their events.

        The deletion is by event timestamp within the future window, which is
        safe because the historical month ends before the window begins — no
        real history is in range.
        """
        from backend.database import engine

        stamp = instant.strftime("%Y-%m-%d %H:%M:%S.%f")
        raw = engine.raw_connection()
        try:
            dbapi = raw.driver_connection
            dbapi.isolation_level = None
            cur = dbapi.cursor()
            cur.execute("PRAGMA busy_timeout=15000")
            cur.execute("BEGIN")
            deleted = cur.execute(
                "DELETE FROM vehicle_events WHERE timestamp > ?", (stamp,)
            ).rowcount
            cur.execute(
                "UPDATE future_events SET released_at = NULL WHERE timestamp > ?", (stamp,)
            )
            cur.execute("COMMIT")
            cur.close()
        finally:
            raw.close()
        return max(deleted, 0)

    # ── pub/sub ──────────────────────────────────────────────────────────

    def subscribe(self) -> deque[dict[str, Any]]:
        """A bounded mailbox for one WebSocket client."""
        queue: deque[dict[str, Any]] = deque(maxlen=64)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: deque[dict[str, Any]]) -> None:
        with self._lock:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    def _publish(self, message: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            # A bounded deque drops the oldest frame when a client falls behind.
            # A slow browser tab must never be able to stall ingestion.
            queue.append(message)


# ── small helpers ────────────────────────────────────────────────────────────

def _naive(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def _parse_db_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _naive(value)
    text = str(value)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
         - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dl))
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _bearing_similarity(a: float, b: float) -> float:
    diff = abs(a - b)
    if diff > 180:
        diff = 360 - diff
    return 1.0 - diff / 90.0


def _confidence_label(value: float) -> str:
    if value >= 0.45:
        return "HIGH"
    if value >= 0.20:
        return "MEDIUM"
    if value >= 0.07:
        return "LOW"
    return "SPECULATIVE"


def _congestion_label(avg_speed: float | None) -> str | None:
    if avg_speed is None:
        return None
    if avg_speed < 15:
        return "SEVERE"
    if avg_speed < 25:
        return "HEAVY"
    if avg_speed < 40:
        return "MODERATE"
    return "FREE_FLOW"


def _sighting_dict(hit: LiveHit, names: dict[str, str]) -> dict[str, Any]:
    return {
        "camera_id": hit.camera_id,
        "camera_name": names.get(hit.camera_id),
        "timestamp": hit.sim_timestamp.isoformat(),
        "latitude": hit.latitude,
        "longitude": hit.longitude,
        "speed_kmh": hit.speed,
        "direction": hit.direction,
    }


# The single process-wide clock. There is one simulated city, and two clocks
# racing to promote the same rows would double-insert.
simulation_clock = SimulationClock()

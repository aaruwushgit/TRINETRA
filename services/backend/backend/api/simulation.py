"""
Simulation router — drive the clock, and read the live view it produces.

The clock (backend/services/simulation_service.py) walks forward through the
staged future month and promotes sightings into `vehicle_events` one by one.
This router is the control surface for that, plus the read endpoints the live
window and the dashboard's live heatmap poll.

Two distinct kinds of endpoint live here, and it is worth being clear which is
which:

* `/simulation/heatmap`, `/trajectory`, `/predict` read the **live view** — the
  in-memory buffer of the last few minutes of simulated time. They are cheap,
  they are about *now*, and they are empty before the clock has run.
* Everything under `/analytics` and `/vehicles` reads the **database** — the
  full two-month record, including everything this clock has promoted. They are
  the historical view.

They are separate paths on purpose. A dashboard panel titled "live" should not
silently show a month of history when the stream stalls; it should show that it
is empty.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from backend.services.simulation_service import (
    DEFAULT_HEATMAP_MINUTES,
    simulation_clock,
)

_rest = APIRouter(prefix="/simulation", tags=["Simulation"])
_ws = APIRouter(tags=["Simulation"])


# ── control ──────────────────────────────────────────────────────────────────

@_rest.get("/status")
def status():
    """
    Clock state, throughput, and how the transition model was fitted.

    Poll this to drive a status bar. `counters.backlog` is the honest health
    signal: a backlog that keeps growing means the clock is set faster than the
    database can promote, and the fix is a lower `speed`, not a bigger machine.
    """
    return simulation_clock.status()


@_rest.post("/start")
def start(
    speed: float = Query(
        default=60.0, ge=0.1, le=10_000,
        description="Simulated seconds per real second. 60 = one sim-hour per real minute.",
    ),
    start_at: datetime | None = Query(
        default=None,
        description="Begin at this instant instead of now (ISO 8601). Must lie inside "
                    "the staged window to release anything.",
    ),
    skip_to_first: bool = Query(
        default=True,
        description="If nothing is due yet, jump the clock to just before the first "
                    "staged event instead of ticking through an empty gap.",
    ),
    reset_view: bool = Query(
        default=False, description="Clear the live buffer and trails before starting."
    ),
):
    """
    Start (or restart) the clock.

    Idempotent: calling it while running re-anchors sim-time and applies the new
    speed rather than starting a second clock. There is one simulated city, and
    two clocks promoting the same rows would double-insert.

    The staged month begins at the first trip the generator produced, which can
    be hours after "now". By default the clock therefore jumps that gap rather
    than ticking through it — otherwise pressing start looks like nothing
    happening for several real minutes. The jump is reported back as
    `skipped_to_first_event`; pass `skip_to_first=false` to tick through.
    """
    return simulation_clock.start(speed=speed, start_at=start_at,
                                  skip_to_first=skip_to_first, reset_view=reset_view)


@_rest.post("/pause")
def pause():
    """Freeze sim-time. Nothing is promoted until resumed; nothing is lost."""
    return simulation_clock.pause()


@_rest.post("/resume")
def resume():
    """Continue from where the clock was paused."""
    return simulation_clock.resume()


@_rest.post("/stop")
def stop():
    """Stop the promotion thread. Released events stay in the database."""
    return simulation_clock.stop()


@_rest.post("/speed")
def set_speed(
    speed: float = Query(..., ge=0.1, le=10_000, description="Simulated seconds per real second"),
):
    """
    Change the clock rate without restarting.

    Sim-time is re-anchored first, so the elapsed real seconds since the last
    anchor are not retroactively rescaled — otherwise raising 1x to 600x would
    make the clock jump hours.
    """
    return simulation_clock.set_speed(speed)


@_rest.post("/seek")
def seek(
    to: datetime = Query(..., description="Move the clock to this instant (ISO 8601)"),
):
    """
    Jump the clock. **Rewinding is destructive by design**: every event promoted
    after the target instant is deleted from `vehicle_events` and its staged row
    is put back on the shelf, so replaying gives the same stream again rather
    than a doubled one.

    Only the future window is touched — the historical month ends before the
    staged window begins, so real history is never in range.
    """
    return simulation_clock.seek(to)


@_rest.post("/reset")
def reset():
    """
    Back to the starting line: clock at now, nothing released, buffers empty.

    Deletes every event this simulation has promoted. The month of history is
    untouched.
    """
    return simulation_clock.reset()


@_rest.post("/refit")
def refit():
    """
    Re-fit the next-hop transition model on the most recent history.

    Worth calling after a long run: by then the clock has promoted hours of new
    sightings, and the model fitted at start-up is describing traffic from
    before the simulation began.
    """
    return simulation_clock.fit_transition_model()


# ── the live view ────────────────────────────────────────────────────────────

@_rest.get("/hits")
def hits(
    limit: int = Query(default=100, ge=1, le=2000),
    camera_id: str | None = Query(default=None, description="Only this camera"),
    watchlist_only: bool = Query(default=False, description="Only watchlisted plates"),
):
    """
    The most recent promoted sightings, newest first.

    This is the polling fallback for the live feed; prefer the WebSocket at
    `/ws/simulation`, which pushes each batch as it is promoted instead of
    re-sending a window every second.
    """
    return {
        "sim_time": simulation_clock.sim_time.isoformat(),
        "state": simulation_clock.state,
        "hits": simulation_clock.recent_hits(
            limit=limit, camera_id=camera_id, watchlist_only=watchlist_only
        ),
    }


@_rest.get("/heatmap")
def heatmap(
    minutes: int = Query(
        default=DEFAULT_HEATMAP_MINUTES, ge=1, le=180,
        description="Rolling window, in SIMULATED minutes.",
    ),
):
    """
    Live traffic density: per-camera counts over the last `minutes` of sim-time.

    Summed from the in-memory buffer of promoted events, not queried — the
    equivalent SQL is a GROUP BY over the newest rows of a 12M-row table every
    few seconds, and the buffer already holds exactly those rows.

    `intensity` is normalised against the busiest camera in the window, so a
    client can feed it straight to a heat layer without tracking the absolute
    scale, which shifts as the window slides. `window_truncated_by_buffer` says
    when the ring buffer, rather than the requested window, decided how far back
    the answer goes.
    """
    return simulation_clock.live_heatmap(minutes=minutes)


@_rest.get("/trajectory/{plate}")
def live_trajectory(plate: str):
    """
    The trail a vehicle has laid down since the clock started — its live path.

    Distinct from `/vehicles/{plate}/trajectory`, which is the full historical
    record. This one answers "where has it been in the last few minutes", which
    is the question during a pursuit.
    """
    return simulation_clock.live_trajectory(plate)


@_rest.get("/predict/{plate}")
def predict(
    plate: str,
    top_n: int = Query(default=3, ge=1, le=10),
):
    """
    Where a live-tracked vehicle goes next, with an explicit confidence.

    Probability and confidence are reported separately because they answer
    different questions. The probability is the transition model's estimate of
    which exit is most likely. The confidence is whether that estimate is worth
    acting on — it folds in how many departures from this camera were actually
    observed (`support`) and how lopsided the distribution is
    (`decisiveness`). A junction seen four times, or one that splits evenly five
    ways, reports a low confidence even when one candidate leads.

    `probability_lower_bound` is the 95% Wilson bound on the same count, for
    reasoning about the interval directly: 3-of-4 and 750-of-1000 are both a
    probability of 0.75 and only one of them is worth a unit.
    """
    result = simulation_clock.predict(plate, top_n=top_n)
    if result.get("error") == "not_tracked_live":
        raise HTTPException(status_code=404, detail=result["detail"])
    return result


@_rest.get("/horizon")
def horizon(
    hours: int = Query(default=24, ge=1, le=24 * 60,
                       description="Count staged events due within this many hours."),
):
    """
    The shape of the timeline: how much is staged ahead, how much is released.

    This is what tells an operator the demo has fuel left in it.
    """
    return simulation_clock.horizon(hours=hours)


@_rest.get("/timeline")
def timeline(
    past_days: int = Query(default=30, ge=1, le=90),
    future_days: int = Query(default=30, ge=1, le=90),
    buckets: int = Query(default=60, ge=6, le=240,
                         description="Resolution of the returned histogram."),
):
    """
    One histogram spanning both halves of the window — the picture of the whole run.

    Past buckets count rows in `vehicle_events`; future buckets count rows still
    staged in `future_events`. The join between them is `now`, which is the
    marker a client should draw. Built with two grouped queries over indexed
    ranges rather than per-bucket counts, so the resolution is free.
    """
    from backend.database import engine

    now = simulation_clock.sim_time
    start = now - timedelta(days=past_days)
    end = now + timedelta(days=future_days)
    span_s = (end - start).total_seconds()
    bucket_s = span_s / buckets

    def fmt(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    # SQLite has no date_trunc, but the timestamps are stored as sortable text,
    # so bucketing is arithmetic on julianday — one pass, one index range.
    sql = (
        "SELECT CAST((julianday(timestamp) - julianday(:start)) * 86400.0 / :bucket AS INTEGER) AS b, "
        "       COUNT(*) "
        "FROM {table} WHERE timestamp >= :start AND timestamp < :end {extra} "
        "GROUP BY b ORDER BY b"
    )

    from sqlalchemy import text

    past = [0] * buckets
    future = [0] * buckets
    with engine.connect() as conn:
        params = {"start": fmt(start), "end": fmt(now), "bucket": bucket_s}
        for b, n in conn.execute(text(sql.format(table="vehicle_events", extra="")), params):
            if 0 <= b < buckets:
                past[b] = n
        params = {"start": fmt(start), "end": fmt(end), "bucket": bucket_s}
        try:
            rows = conn.execute(
                text(sql.format(table="future_events", extra="AND released_at IS NULL")),
                params,
            )
            for b, n in rows:
                if 0 <= b < buckets:
                    future[b] = n
        except Exception:
            # No staged future yet — an empty right-hand half is the correct
            # answer, not an error.
            pass

    edges = [(start + timedelta(seconds=bucket_s * i)).isoformat() for i in range(buckets + 1)]
    now_bucket = int((now - start).total_seconds() / bucket_s)
    return {
        "sim_time": now.isoformat(),
        "window": [start.isoformat(), end.isoformat()],
        "bucket_seconds": round(bucket_s, 1),
        "bucket_edges": edges,
        "now_bucket": now_bucket,
        "past_counts": past,
        "future_counts": future,
        "totals": {"past": sum(past), "future_staged": sum(future)},
    }


# ── stream ───────────────────────────────────────────────────────────────────

@_ws.websocket("/ws/simulation")
async def stream(websocket: WebSocket):
    """
    Push each batch of promoted sightings as it happens.

    Frames are `{"type": "hits", ...}` for the stream, `{"type": "watchlist",
    ...}` for a watchlisted plate (sent separately so a client can alert on it
    without filtering every frame), and a periodic `{"type": "status", ...}`
    heartbeat that also keeps the connection from idling out through a proxy.

    Each subscriber has a bounded 64-frame mailbox. A tab that stops draining
    loses its oldest frames rather than applying back-pressure — ingestion must
    never be slowed by a browser.
    """
    await websocket.accept()
    queue = simulation_clock.subscribe()
    try:
        await websocket.send_json({
            "type": "hello",
            "status": simulation_clock.status(),
            "recent": simulation_clock.recent_hits(limit=60),
        })
        last_status = 0.0
        while True:
            sent = 0
            while queue and sent < 20:
                await websocket.send_json(queue.popleft())
                sent += 1

            loop_now = asyncio.get_event_loop().time()
            if loop_now - last_status > 2.0:
                await websocket.send_json({
                    "type": "status", "status": simulation_clock.status()
                })
                last_status = loop_now

            if not sent:
                await asyncio.sleep(0.2)
            else:
                # Yield without a full sleep so a burst drains promptly.
                await asyncio.sleep(0)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return
    except Exception:
        return
    finally:
        simulation_clock.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass


router = APIRouter()
router.include_router(_rest)
router.include_router(_ws)

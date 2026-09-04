"""
Staged future sightings — the "not yet happened" half of the timeline.

The platform is demonstrated over a two-month window: a month of history that
is already in `vehicle_events`, and a month ahead that is *generated in advance*
and parked here. The simulation clock (backend/services/simulation_service.py)
walks forward through this table and promotes each row into `vehicle_events` at
the moment its timestamp comes due, one by one — which is what makes the live
feed, the live heatmap and the live trajectories real consequences of ingestion
rather than animations.

Why stage them instead of generating on the fly
-----------------------------------------------
Pre-generating buys three things that matter more than the disk it costs:

1. **Predictions are checkable.** A next-hop prediction made at sim-time T can
   be scored against what the vehicle actually does at T+1, because that future
   already exists in a row nobody has read yet. Generating on the fly would make
   every prediction unfalsifiable.
2. **The run is reproducible.** The same seed replays the same month, so a demo
   that worked in rehearsal works on stage.
3. **The clock can be moved.** Rewinding or jumping the simulation is a query
   against `released_at`, not a re-simulation.

`released_at` (not a boolean) is deliberate: it records *when* a row entered the
live table, so a paused-and-resumed run can tell an event that has been ingested
from one that merely came due while the clock was stopped.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class FutureEvent(Base):
    __tablename__ = "future_events"

    # Assigned by the generator (sequential, not a UUID) so the promotion pass
    # can page through by primary key without a sort.
    future_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    camera_id: Mapped[str] = mapped_column(String(50))
    local_track_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)

    plate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(20), nullable=True)

    vehicle_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vehicle_color: Mapped[str | None] = mapped_column(String(50), nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)

    global_vehicle_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # NULL until the simulation clock promotes this row into vehicle_events.
    released_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    # Which generator run produced this row. Lets a re-seed replace one batch
    # without truncating a batch someone else is mid-demo on.
    batch: Mapped[str] = mapped_column(String(40), default="default", index=True)


# The promotion query is exactly "un-released rows due by now, oldest first".
# A composite index on (released_at, timestamp) turns that into a range scan
# over the head of the table instead of a scan of the whole staged month —
# which at ~1.5M rows is the difference between a 5 ms tick and a 900 ms one.
Index("ix_future_events_due", FutureEvent.released_at, FutureEvent.timestamp)
Index("ix_future_events_plate", FutureEvent.plate)

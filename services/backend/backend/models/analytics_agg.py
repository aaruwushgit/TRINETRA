"""
Precomputed analytics aggregates.

Why these tables exist at all: the dashboard's headline panels (city heatmap,
per-camera density, "most used roads", KPI header) are all GROUP BY queries over
`vehicle_events`. At the demo's scale — ~12M rows for one month over 200
junctions — every one of those is a multi-second full table scan on SQLite, and
the /analytics/* endpoints re-run them on every page load. Rolling the answers
up once at load time turns each panel into a lookup over a few hundred to a few
thousand rows, which is what makes the map feel instant.

The tradeoff is honest and worth naming: these are *stale by design*. They are
recomputed in bulk by scripts/generate_delhi_dataset.py and then refreshed
incrementally by scripts/live_event_feeder.py. Anything that must be
to-the-second accurate (a specific plate's trajectory, a live alert feed) should
still hit `vehicle_events` directly — those are point lookups on an index, not
scans, so they're already fast.

Design notes:
  * Natural composite primary keys, not surrogate UUIDs. These rows are
    identified by what they aggregate, so a surrogate key would only add a
    second index to keep and would let duplicate buckets exist.
  * Denormalised road names / lat-lon / distances. The heatmap and the road
    ranking need to draw geometry, and joining 200 cameras back in per request
    is pure overhead when the values never change for a given camera.
"""
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class RoadUsage(Base):
    """
    One row per *directed* camera-pair segment actually travelled.

    A "segment" here is a consecutive pair of sightings of the same vehicle
    (A then B, no other sighting in between) — i.e. an observed trip leg, not a
    road from a map. That's what makes this a usage measure rather than a
    network description: a segment only appears if vehicles really drove it, and
    trip_count is how many times.

    Directed rather than undirected on purpose — Delhi's morning and evening
    peaks run in opposite directions on the same tarmac, and collapsing the two
    would hide exactly the asymmetry a traffic operator cares about.

    Legs are only counted when the gap between the two sightings is plausible
    for a single journey (see MAX_SEGMENT_GAP_MINUTES in the generator);
    otherwise "last sighting last night, first sighting this morning" would be
    recorded as a 9-hour trip on a 3km road and wreck avg_travel_minutes.
    """

    __tablename__ = "road_usage"

    from_camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id"), primary_key=True
    )
    to_camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id"), primary_key=True
    )

    trip_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_travel_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Derived as distance_km / travel_time, so it is the *implied* speed over the
    # whole leg (includes signal waits), not the average of the instantaneous
    # `speed` readings at either camera. It is therefore always the lower, and
    # more useful, of the two numbers for congestion ranking.
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Fastest single traversal ever observed on this segment. Kept because it is
    # the cheapest possible physical-plausibility tripwire: if this column ever
    # shows four digits, the timestamp simulation has broken.
    max_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)

    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    from_road: Mapped[str | None] = mapped_column(String(200), nullable=True)
    to_road: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Human label for the segment, e.g. "Ring Road → Vikas Marg". Precomputed so
    # map tooltips don't have to concatenate on the client.
    road_label: Mapped[str | None] = mapped_column(String(420), nullable=True)

    # Midpoint of the two cameras — where a heatmap should draw the segment's
    # weight. Straight-line, not route geometry; good enough at city zoom.
    mid_latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    mid_longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        # The road-usage panel is almost always "top N busiest segments", which
        # is a descending sort over the whole table. Cheap to index, and it keeps
        # that query off a sort of every segment.
        Index("ix_road_usage_trip_count", "trip_count"),
    )


class CameraHourly(Base):
    """
    Per-camera, per-hour rollup. Powers the time-series charts and the
    "density over the last N hours" panel without touching raw events.

    Hourly is a deliberate floor on resolution. 200 cameras x 720 hours is
    ~144k rows for the month — small enough to scan entirely — whereas
    15-minute buckets would be 576k rows for detail no dashboard chart at
    city scale can actually render.

    unique_vehicles is stored alongside vehicle_count because they answer
    different questions (throughput vs. distinct population) and COUNT(DISTINCT)
    is precisely the part you cannot afford to compute on demand. Note it is not
    additive across buckets: summing an hour's unique_vehicles over a day
    over-counts vehicles seen in two hours. Sum vehicle_count for volume; use
    CameraTotals.unique_vehicles for lifetime distinct counts.
    """

    __tablename__ = "camera_hourly"

    camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id"), primary_key=True
    )
    # Truncated to the hour, naive UTC — matches VehicleEvent.timestamp so the
    # two can be compared without timezone juggling.
    hour_bucket: Mapped[datetime] = mapped_column(DateTime, primary_key=True)

    vehicle_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_speed: Mapped[float | None] = mapped_column(Float, nullable=True)
    unique_vehicles: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        # City-wide time series slices by time across all cameras, which the
        # (camera_id, hour_bucket) primary key can't serve.
        Index("ix_camera_hourly_hour_bucket", "hour_bucket"),
    )


class CameraTotals(Base):
    """
    Lifetime per-camera totals — 200 rows, the entire map layer in one query.

    peak_hour is hour-of-day (0-23) rather than a specific timestamp: the
    dashboard wants "this junction's rush hour is 09:00", which is a property of
    the junction, not of one particular Tuesday.
    """

    __tablename__ = "camera_totals"

    camera_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("cameras.camera_id"), primary_key=True
    )

    vehicle_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_vehicles: Mapped[int] = mapped_column(Integer, default=0)
    avg_speed: Mapped[float | None] = mapped_column(Float, nullable=True)

    peak_hour: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0-23
    peak_hour_count: Mapped[int] = mapped_column(Integer, default=0)

    first_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Copied from Camera so the heatmap layer is a single-table read.
    road: Mapped[str | None] = mapped_column(String(200), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        # Heatmap weighting and "busiest junctions" both rank on volume.
        Index("ix_camera_totals_vehicle_count", "vehicle_count"),
    )


class DatasetKpi(Base):
    """
    Single-row snapshot of the whole-dataset headline numbers.

    This exists because /analytics/summary computes COUNT(*) and
    COUNT(DISTINCT plate) over `vehicle_events` on every call. Both are full
    scans; at 12M rows the DISTINCT one costs seconds and it sits in the
    dashboard *header*, so it delays first paint of every page. Keeping the
    answer in one row makes it a single-row read.

    Keyed by a fixed scope string ("global", or a deployment tag) so a
    multi-city deployment can hold one row per city without a schema change.
    """

    __tablename__ = "dataset_kpi"

    scope: Mapped[str] = mapped_column(String(50), primary_key=True, default="global")

    total_events: Mapped[int] = mapped_column(Integer, default=0)
    unique_vehicles: Mapped[int] = mapped_column(Integer, default=0)
    camera_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)

    first_event_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    computed_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

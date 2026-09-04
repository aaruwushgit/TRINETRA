"""
RouteSegment ORM model — persistent cache of REAL road geometry between two cameras.

Why this table exists
─────────────────────
A trajectory drawn as straight lines between cameras is a lie: it shows a
vehicle flying over buildings and the Yamuna. To draw the *actual* road the
vehicle must have taken we need the driving route between every camera pair,
which comes from a road-network router (OSRM over OpenStreetMap).

We cannot call OSRM at request time for every leg of every trajectory:
  * the public OSRM demo server is rate-limited and would throttle us mid-demo,
  * a 6-hop trajectory would add ~6 network round-trips (seconds) to one API call,
  * and venue Wi-Fi fails. The demo must render road-accurate paths OFFLINE.

So every camera→camera route is fetched once, decimated, and stored here.
After scripts/build_route_cache.py has run, the whole map works with zero
network access.

Design notes / tradeoffs
───────────────────────
* The primary key is (from_camera_id, to_camera_id) and is DIRECTIONAL on
  purpose. Delhi is full of one-ways, flyovers and central medians, so the
  A→B road path is genuinely not the reverse of B→A. Caching one direction and
  reversing it would silently draw vehicles down the wrong carriageway.
* Deliberately NO ForeignKey to cameras.camera_id. The route cache is built
  offline from deployments/delhi/cameras.json, which may be precomputed before
  (or independently of) the cameras being seeded into this database — and the
  same dev.db currently holds cameras from a different deployment. A hard FK
  would make the cache builder fail on exactly the workflow it exists for.
  We accept a nominally-orphanable row in exchange for a cache that can be
  built ahead of, and shipped independently of, the camera table.
* `straight_line_m` is stored alongside `road_distance_m` not for rendering but
  for honesty: it lets the UI (and /routing/cache/stats) show the detour ratio.
  A ratio of ~1.0 across the board is the tell-tale that geometry is fake /
  fell back to a straight line; real Delhi road routes land around 1.15–1.6x.
* `source` records provenance ("osrm" vs "fallback_straight"). We never hide a
  fallback: a degraded leg is drawn, but labelled, so nobody demos a straight
  line believing it is a road.
"""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class RouteSegment(Base):
    __tablename__ = "route_segments"

    # ── Identity: one row per ORDERED camera pair ─────────
    from_camera_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    to_camera_id: Mapped[str] = mapped_column(String(50), primary_key=True)

    # ── Routing result (from OSRM, metres / seconds) ──────
    road_distance_m: Mapped[float] = mapped_column(Float)
    road_duration_s: Mapped[float] = mapped_column(Float)
    # Haversine great-circle distance between the two cameras. Kept so we can
    # report the road/straight-line detour ratio without re-deriving camera
    # coordinates at read time.
    straight_line_m: Mapped[float] = mapped_column(Float)

    # ── Geometry ─────────────────────────────────────────
    # JSON list of [lat, lon] pairs — ALREADY FLIPPED from OSRM's GeoJSON
    # [lon, lat] order so the frontend can hand it straight to Leaflet's
    # L.polyline() with no transformation. Flipping once, here, is much safer
    # than trusting every consumer to remember which order it got.
    geometry: Mapped[list] = mapped_column(JSON, default=list)
    point_count: Mapped[int] = mapped_column(Integer, default=0)

    # "osrm" = genuine road geometry. "fallback_straight" = router unreachable,
    # 2-point straight line, visibly degraded and reported as such.
    source: Mapped[str] = mapped_column(String(30), default="osrm")

    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (
        # Trajectory rendering looks up legs by exact pair (covered by the PK),
        # but the cache builder and /routing/cache/stats scan "all segments
        # leaving camera X" and "all fallbacks", hence these two.
        Index("ix_route_segments_from", "from_camera_id"),
        Index("ix_route_segments_to", "to_camera_id"),
        Index("ix_route_segments_source", "source"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<RouteSegment {self.from_camera_id}->{self.to_camera_id} "
            f"{self.road_distance_m:.0f}m/{self.point_count}pts src={self.source}>"
        )

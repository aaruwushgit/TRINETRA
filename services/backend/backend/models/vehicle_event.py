"""
Vehicle detection event ORM model.

Every time a camera sees a vehicle and ANPR reads its plate, one VehicleEvent
row is created. Multi-camera association (MTMC) later assigns a global_vehicle_id.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    event_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    # ── Camera sighting ──────────────────────────────────
    camera_id: Mapped[str] = mapped_column(String(50), ForeignKey("cameras.camera_id"))
    local_track_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)

    # ── ANPR result ───────────────────────────────────────
    plate: Mapped[str | None] = mapped_column(String(20), nullable=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Location ─────────────────────────────────────────
    latitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float, nullable=True)
    direction: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # ── Vehicle attributes ───────────────────────────────
    vehicle_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vehicle_color: Mapped[str | None] = mapped_column(String(50), nullable=True)
    speed: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Attribute-only identity ──────────────────────────────────
    # A camera does not always get the plate. A vehicle crossing at 90 km/h,
    # a plate occluded by the car in front, a bike with a bent plate — the
    # detector still sees a silver hatchback heading north at 88 km/h, and
    # throwing that away is throwing away the only record that it passed.
    # These columns let a sighting be ingested on attributes alone: it shows
    # in counts, heatmap and speed analytics, and a partial plate plus make,
    # model and colour is often enough for an investigator to match it by hand
    # against the plate-identified sightings either side of it.
    #
    # `plate` stays NULL for these rows rather than being filled with a guess,
    # so no query can mistake an attribute match for a plate read.
    vehicle_make: Mapped[str | None] = mapped_column(String(50), nullable=True)
    vehicle_model: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Whatever characters were legible, e.g. "MH01??1234". Indexed because
    # "find every sighting whose partial is consistent with this plate" is the
    # query an investigator actually runs.
    plate_partial: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    # Unformatted OCR output before grammar correction — kept for audit: it is
    # the evidence behind a contested plate read.
    plate_raw: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Detector/classifier confidence in the *attributes*, separate from
    # plate_confidence, which is confidence in the plate text.
    attribute_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Multi-camera association (filled by MTMC service) ─
    # This is the bridge between single-camera tracks and global vehicle identity.
    # When MTMC is not yet running, this stays None and trajectory still works
    # for vehicles identified only by plate number.
    global_vehicle_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

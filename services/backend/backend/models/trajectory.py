"""Trajectory ORM model — reconstructed vehicle path across cameras."""
import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Trajectory(Base):
    """
    A trajectory is a time-ordered list of camera sightings for one vehicle.
    Built lazily from VehicleEvent rows — either keyed by plate (when MTMC is
    not running) or by global_vehicle_id (when MTMC is active).
    """

    __tablename__ = "trajectories"

    trajectory_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    vehicle_key: Mapped[str] = mapped_column(String(100), index=True)  # plate or global_vehicle_id
    key_type: Mapped[str] = mapped_column(String(20), default="plate")  # "plate" | "global_id"

    # Ordered list of event_ids that make up this trajectory
    event_ids: Mapped[list] = mapped_column(JSON, default=list)

    start_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

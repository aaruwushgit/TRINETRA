"""Camera ORM model."""
from datetime import datetime

from sqlalchemy import DateTime, Float, String, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    camera_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    location: Mapped[str] = mapped_column(String(200))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    road: Mapped[str | None] = mapped_column(String(200), nullable=True)
    direction: Mapped[str | None] = mapped_column(String(20), nullable=True)  # NORTH/SOUTH/EAST/WEST
    camera_type: Mapped[str] = mapped_column(String(50), default="ANPR")
    deployment: Mapped[str] = mapped_column(String(50), default="default")
    # Legal speed limit for the road segment this camera watches. Real cities
    # don't have one city-wide limit — a ring road and a residential lane
    # differ by 2x. Checkpoint-pair speed violations use the lower of the
    # two cameras' limits on a hop instead of one flat global default.
    speed_limit_kmh: Mapped[float] = mapped_column(Float, default=60.0)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

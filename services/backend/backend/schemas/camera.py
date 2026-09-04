"""Pydantic schemas for camera endpoints."""
from pydantic import BaseModel


class CameraCreate(BaseModel):
    camera_id: str
    name: str
    location: str
    latitude: float
    longitude: float
    road: str | None = None
    direction: str | None = None
    camera_type: str = "ANPR"
    deployment: str = "default"
    speed_limit_kmh: float = 60.0


class CameraOut(CameraCreate):
    is_active: bool
    model_config = {"from_attributes": True}

"""Pydantic schemas for traffic analytics, heatmaps, and snapshots."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HeatmapPoint(BaseModel):
    latitude: float
    longitude: float
    weight: int = Field(description="Detection intensity/count at this location")
    camera_id: str | None = None


class SpeedStat(BaseModel):
    camera_id: str
    avg_speed_kmh: float
    sample_count: int


class CongestionReport(BaseModel):
    camera_id: str
    congestion_level: str  # LOW, MEDIUM, HIGH
    vehicle_count: int
    avg_speed_kmh: float | None
    road: str | None = None


class ODMatrixEntry(BaseModel):
    origin_camera_id: str
    destination_camera_id: str
    trip_count: int
    avg_duration_minutes: float | None = None


class TrafficFlowPoint(BaseModel):
    timestamp_bucket: str
    camera_id: str
    count: int


class TrafficSnapshotCreate(BaseModel):
    camera_id: str
    window_start: datetime
    window_end: datetime
    vehicle_count: int
    avg_speed: float | None = None
    peak_density: int = 0
    class_counts: dict[str, Any] = Field(default_factory=dict)
    congestion_level: str = "LOW"


class TrafficSnapshotOut(TrafficSnapshotCreate):
    snapshot_id: str
    created_at: datetime

    model_config = {"from_attributes": True}

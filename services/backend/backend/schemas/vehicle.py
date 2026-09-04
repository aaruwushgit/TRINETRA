"""Pydantic schemas for vehicle events and ANPR ingestion."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestEventRequest(BaseModel):
    """
    What AI workers POST to /events/ingest.
    This is THE contract. Every module (ANPR, MTMC, trackers) must produce this.
    """
    camera_id: str
    timestamp: datetime
    local_track_id: str | None = None
    plate: str | None = None
    plate_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latitude: float | None = None
    longitude: float | None = None
    direction: str | None = None
    vehicle_type: str | None = None
    vehicle_color: str | None = None
    speed: float | None = None

    @field_validator("plate")
    @classmethod
    def normalize_plate(cls, v: str | None) -> str | None:
        """
        Normalize plates once, at the point of entry, so every downstream
        lookup (tracking, alerts, trajectory, defaulters) can rely on exact
        string matches instead of re-normalizing inconsistently everywhere.
        """
        if v is None:
            return v
        cleaned = v.upper().replace(" ", "").replace("-", "").strip()
        return cleaned or None


class IngestEventResponse(BaseModel):
    event_id: str
    global_vehicle_id: str | None
    alert_fired: bool


# ── Frame upload ──────────────────────────────────────────────────────────────

class FrameAnalysisResponse(BaseModel):
    """Response from POST /events/analyze-frame (for testing ANPR via HTTP)."""
    plate: str | None
    confidence: float


# ── Vehicle lookup ────────────────────────────────────────────────────────────

class VehicleEventOut(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    plate: str | None
    plate_confidence: float | None
    latitude: float | None
    longitude: float | None
    direction: str | None
    vehicle_type: str | None
    speed: float | None = None
    global_vehicle_id: str | None

    model_config = {"from_attributes": True}


class TrajectoryPoint(BaseModel):
    camera_id: str
    timestamp: datetime
    latitude: float | None
    longitude: float | None
    direction: str | None
    plate_confidence: float | None
    speed: float | None = None


class TrajectoryResponse(BaseModel):
    plate: str
    global_vehicle_id: str | None
    points: list[TrajectoryPoint]

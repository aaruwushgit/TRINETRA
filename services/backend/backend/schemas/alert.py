"""Pydantic schemas for alerts."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AlertOut(BaseModel):
    alert_id: str
    vehicle_id: str
    camera_id: str | None
    alert_type: str
    description: str | None
    status: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class BlacklistAddRequest(BaseModel):
    plate: str
    reason: str | None = None


class BlacklistOut(BaseModel):
    plate: str
    reason: str | None
    added_at: datetime

    model_config = {"from_attributes": True}

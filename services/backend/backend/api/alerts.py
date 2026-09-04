"""Alerts router — blacklist management and alert queries."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.alert import Alert, Blacklist
from backend.schemas.alert import AlertOut, BlacklistAddRequest, BlacklistOut
from backend.services.alert_service import alert_service

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("/", response_model=list[AlertOut])
def list_alerts(status: str = "ACTIVE", db: Session = Depends(get_db)):
    return db.query(Alert).filter(Alert.status == status).order_by(Alert.timestamp.desc()).all()


@router.post("/blacklist", response_model=BlacklistOut, status_code=201)
def add_to_blacklist(payload: BlacklistAddRequest, db: Session = Depends(get_db)):
    return alert_service.add_to_blacklist(payload.plate, payload.reason, db)


@router.delete("/blacklist/{plate}")
def remove_from_blacklist(plate: str, db: Session = Depends(get_db)):
    removed = alert_service.remove_from_blacklist(plate, db)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{plate} not in blacklist.")
    return {"message": f"{plate.upper()} removed from blacklist."}


@router.get("/blacklist", response_model=list[BlacklistOut])
def list_blacklist(db: Session = Depends(get_db)):
    return db.query(Blacklist).all()

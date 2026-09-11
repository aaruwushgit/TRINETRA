"""
Patterns router — what a specific vehicle normally does, and when it doesn't.

Registered as an optional router (see backend/main.py) because it is the only
feature that hard-depends on scikit-learn. If that import fails the rest of the
API is unaffected, which is the same contract the jobs/routing/benchmarks
routers get.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services import pattern_service

router = APIRouter(prefix="/patterns", tags=["Patterns"])


@router.get("/{plate}")
def vehicle_profile(
    plate: str,
    days: int = Query(365, ge=7, le=1095, description="History window to mine"),
    db: Session = Depends(get_db),
):
    """Inferred home/workplace, routine statistics and anomalous days.

    Returns `status: "insufficient"` rather than an error when a vehicle has too
    few sightings to have a pattern — that is a legitimate answer about the
    vehicle, not a failure of the request.
    """
    result = pattern_service.profile_vehicle(db, plate.strip().upper(), days=days)
    if result.get("status") == "insufficient" and result.get("sightings") == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No sightings of '{plate}' in the last {days} days.",
        )
    return result


@router.get("/{plate}/anomalies")
def vehicle_anomalies(
    plate: str,
    days: int = Query(365, ge=14, le=1095),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Just the anomalous days, for a client that does not need the full profile."""
    sightings = pattern_service._sightings(db, plate.strip().upper(), days)
    if not sightings:
        raise HTTPException(status_code=404, detail=f"No sightings of '{plate}'.")

    places = pattern_service.infer_places(sightings)
    found = pattern_service.find_anomalies(sightings, places["home"], limit=limit)
    return {
        "plate": plate.strip().upper(),
        "window_days": days,
        "days_analysed": len({s["day"] for s in sightings}),
        "minimum_days": pattern_service.MIN_DAYS_FOR_ANOMALY,
        "anomalies": [a.__dict__ for a in found],
    }


@router.get("")
def pattern_candidates(
    limit: int = Query(20, ge=1, le=100),
    min_sightings: int = Query(60, ge=12, le=5000),
    days: int = Query(365, ge=7, le=1095),
    db: Session = Depends(get_db),
):
    """Vehicles with enough history to profile, busiest first.

    Exists so a client has somewhere to start: mining a pattern needs a vehicle
    with a pattern, and picking a plate at random from 150k mostly-occasional
    vehicles usually lands on one with nine sightings.
    """
    rows = db.execute(
        text(
            "SELECT plate, COUNT(*) AS n, COUNT(DISTINCT camera_id) AS cams "
            "FROM vehicle_events "
            "WHERE plate IS NOT NULL AND timestamp >= NOW() - (:days || ' days')::interval "
            "GROUP BY plate HAVING COUNT(*) >= :min_sightings "
            "ORDER BY n DESC LIMIT :limit"
        )
        if db.bind.dialect.name.startswith("postgres")
        else text(
            "SELECT plate, COUNT(*) AS n, COUNT(DISTINCT camera_id) AS cams "
            "FROM vehicle_events "
            "WHERE plate IS NOT NULL AND timestamp >= datetime('now', :days_sqlite) "
            "GROUP BY plate HAVING COUNT(*) >= :min_sightings "
            "ORDER BY n DESC LIMIT :limit"
        ),
        {
            "days": days,
            "days_sqlite": f"-{days} days",
            "min_sightings": min_sightings,
            "limit": limit,
        },
    ).fetchall()

    return {
        "window_days": days,
        "candidates": [
            {"plate": p, "sightings": n, "distinct_cameras": c} for p, n, c in rows
        ],
    }

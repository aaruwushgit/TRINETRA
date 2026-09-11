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
    profile: str = Query(
        "commuter",
        pattern="^(commuter|roamer|any)$",
        description="commuter = concentrated corridor; roamer = spread wide; any = by volume",
    ),
    db: Session = Depends(get_db),
):
    """Vehicles worth profiling, ranked by how much of a routine they have.

    Exists because ranking by raw sighting count is actively misleading. The
    highest-volume vehicles in this dataset are fleet and taxi profiles that
    roam the whole network: they have thousands of sightings across a hundred
    cameras and, correctly, no inferable home or workplace. Demonstrating the
    pattern miner on one of those shows it working and concluding nothing.

    So candidates are ranked by *corridor concentration* — the share of a
    vehicle's sightings falling on its busiest few cameras. High concentration
    is the signature of a commuter, which is the population the inference is
    designed for. `profile=roamer` inverts it, which is the useful view for
    "who is behaving unlike a commuter", and `any` restores volume ordering.

    COST: this aggregates every sighting in the window (36M+ rows at a year),
    so it takes tens of seconds and is a hash aggregate, not an index scan. It
    is a one-off "find me a vehicle to look at" query and is deliberately not
    called by the frontend on load — PatternPanel takes a plate it already
    has. Do not put it behind a polling panel.
    """
    pg = db.bind.dialect.name.startswith("postgres")
    since_clause = (
        "timestamp >= NOW() - (:days || ' days')::interval"
        if pg
        else "timestamp >= datetime('now', :days_sqlite)"
    )

    # concentration = sightings on the busiest camera / total sightings.
    # A cheap proxy for the full top-5 share used by routine_summary, and it
    # computes in one pass with no window function over 36M rows.
    order = {
        "commuter": "concentration DESC, n DESC",
        "roamer": "concentration ASC, n DESC",
        "any": "n DESC",
    }[profile]

    rows = db.execute(
        text(
            "WITH per_cam AS ("
            "  SELECT plate, camera_id, COUNT(*) AS c FROM vehicle_events "
            f" WHERE plate IS NOT NULL AND {since_clause} "
            "  GROUP BY plate, camera_id"
            "), totals AS ("
            "  SELECT plate, SUM(c) AS n, MAX(c) AS top_cam_count, "
            "         COUNT(*) AS cams FROM per_cam GROUP BY plate "
            "  HAVING SUM(c) >= :min_sightings"
            ") "
            "SELECT plate, n, cams, "
            "       CAST(top_cam_count AS FLOAT) / n AS concentration "
            f"FROM totals ORDER BY {order} LIMIT :limit"
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
        "ranked_by": profile,
        "candidates": [
            {
                "plate": plate,
                "sightings": int(n),
                "distinct_cameras": int(cams),
                "corridor_concentration": round(float(conc), 4),
            }
            for plate, n, cams, conc in rows
        ],
    }

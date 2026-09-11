"""
Behavioural pattern mining over a vehicle's own history.

What this answers that nothing else here does
---------------------------------------------
`prediction_service` answers "where does traffic go from camera X" — a property
of the *network*, learned from everyone's transitions. This module answers
"what does THIS vehicle normally do", which is a property of the *individual*,
and it is only answerable because there is now a year of history to look at.

Two capabilities, in order of how much they depend on each other:

**Home and workplace inference.** A vehicle's sightings are not uniformly
distributed in space or time. Cluster them geographically, then ask which
cluster dominates at night, and which dominates the weekday *commute* window
while being far enough away to be somewhere else. The night cluster is where it
sleeps; the other is where it goes. Neither label is in the data — they are
inferred, which is the point.

Note the commute window rather than the working day: a fixed camera network
watches roads, so a vehicle parked at its workplace is invisible from 10:00 to
16:00. Clustering that window found nothing for precisely the regular
commuters it should have worked best on.

**Anomaly detection against the vehicle's own baseline.** Once a vehicle has a
routine, a departure from it is detectable: a night drive for a vehicle that
has never driven at night, a corridor it has never used, twice its usual
distance. This is deliberately *per vehicle*, not per fleet. A taxi doing 300
km a day is unremarkable; a hatchback that has done 20 km a day for eleven
months suddenly doing 300 is the interesting one, and a population-level model
would rank them identically.

Why DBSCAN and IsolationForest
------------------------------
DBSCAN because the number of places a vehicle frequents is unknown and is
exactly what we want discovered — k-means would need `k` supplied, which is the
answer. DBSCAN also labels sparse sightings as noise rather than forcing them
into a cluster, and "this vehicle has no stable base" is a real and useful
output.

IsolationForest because the training set is one vehicle's own history: tens to
a few thousand rows, unlabelled, and we care about the tail rather than the
density everywhere. It needs no distributional assumption and no anomaly
examples, which we do not have.

Honest limits
-------------
* Inference quality scales with sighting count. Under MIN_SIGHTINGS_FOR_BASE
  the result is reported as `insufficient`, not guessed — a confidently wrong
  "home" is worse than an admitted gap.
* A camera network sees roads, not driveways. "Home" is the junction a vehicle
  is repeatedly seen near at night, which is a proxy for a neighbourhood, not
  an address. The API says so in its response.
* Anomaly scores are relative to one vehicle. They are not comparable across
  vehicles, and the endpoints do not pretend they are.
* IsolationForest's `contamination` is a quota, not a test — it will label a
  fixed fraction of days regardless. The forest therefore only ranks; a day is
  reported only if it also deviates by ANOMALY_MIN_Z on some concrete feature,
  so a genuinely regular vehicle returns no anomalies rather than a
  manufactured 6%.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

# Below this, we decline rather than guess.
MIN_SIGHTINGS_FOR_BASE = 12
MIN_DAYS_FOR_ANOMALY = 14

# All hour windows below are in LOCAL clock terms, and the loader converts
# stored timestamps into local time before bucketing. This matters more than it
# looks: `vehicle_events.timestamp` is naive UTC, so bucketing it directly in a
# +05:30 deployment puts "night" (22:00-06:00) at 03:30-11:30 local — the
# morning rush. The inference still produced confident-looking output, because a
# commuter oscillates between two cameras and any two disjoint windows will
# separate them, but the home/work labels were arbitrary and could as easily
# have been swapped.
#
# NIGHT is when a vehicle is near where it sleeps. Its sightings then are on the
# roads immediately around home, which is what makes the night cluster a home
# proxy.
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4, 5}

# DAY is deliberately the *commute* window and not 10:00-16:00, which is what
# this originally used and which was wrong for this sensor.
#
# A fixed camera network watches roads. A vehicle parked at its workplace is
# invisible for the whole working day — there is nothing to cluster between
# 10:00 and 16:00 for a normal commuter, so inferring a workplace from that
# window found nothing for exactly the vehicles it should have worked best on.
# What IS observable is the end of the morning outbound trip and the start of
# the evening return, so those are the hours used.
DAY_HOURS = {7, 8, 9, 10, 11, 16, 17, 18, 19, 20}

# How far a candidate workplace cluster must be from the inferred home before
# it counts. Without this, the densest daytime cluster for most vehicles is the
# junction outside their own house, and "work" comes back as a synonym for
# "home".
MIN_WORK_SEPARATION_KM = 1.5

# DBSCAN neighbourhood, in kilometres. ~700 m groups sightings around one
# junction cluster without merging adjacent neighbourhoods.
CLUSTER_RADIUS_KM = 0.7
CLUSTER_MIN_SAMPLES = 3

EARTH_RADIUS_KM = 6371.0

# IsolationForest's `contamination` is a *quota*, not a test: at 0.06 it will
# label ~6% of days anomalous whether or not anything unusual happened. On a
# perfectly regular vehicle that produced 10 "anomalies" out of 40 identical
# days, which is worse than useless — it manufactures suspicion.
#
# So the forest is used only to RANK days, with `contamination="auto"`, and a
# day is reported only if it also shows a concrete univariate deviation of at
# least ANOMALY_MIN_Z on some feature. A regular vehicle then yields none,
# which is the correct answer.
ANOMALY_CONTAMINATION = "auto"
ANOMALY_MIN_Z = 2.0


@dataclass
class Place:
    """An inferred significant location."""

    label: str                      # "home" | "work"
    latitude: float
    longitude: float
    camera_ids: list[str] = field(default_factory=list)
    sightings: int = 0
    confidence: float = 0.0         # share of that window's sightings here
    dominant_camera: str | None = None


@dataclass
class AnomalyDay:
    """One day that departed from the vehicle's own routine."""

    day: str
    score: float                    # more negative = more anomalous
    sightings: int
    distinct_cameras: int
    first_hour: int
    last_hour: int
    max_speed_kmh: float | None
    km_from_home: float | None
    reasons: list[str] = field(default_factory=list)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── loading ──────────────────────────────────────────────────────────────────

def _sightings(db: Session, plate: str, days: int) -> list[dict[str, Any]]:
    """Every sighting of a plate in the window, oldest first.

    Raw SQL rather than the ORM: this is a read-only projection of five columns
    over what can be thousands of rows, and it runs on the
    (plate, timestamp, camera_id) index as a covering scan.
    """
    from backend.config import get_settings

    offset = timedelta(minutes=get_settings().LOCAL_UTC_OFFSET_MINUTES)
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.execute(
        text(
            "SELECT camera_id, timestamp, latitude, longitude, speed "
            "FROM vehicle_events "
            "WHERE plate = :plate AND timestamp >= :since "
            "  AND latitude IS NOT NULL AND longitude IS NOT NULL "
            "ORDER BY timestamp"
        ),
        {"plate": plate, "since": since},
    ).fetchall()

    out = []
    for camera_id, ts, lat, lon, speed in rows:
        when = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
        # Local clock, for every time-of-day decision below. `when` is kept as
        # stored (UTC) so first_seen/last_seen stay consistent with the rest of
        # the API, which reports UTC.
        local = when + offset
        out.append(
            {
                "camera_id": camera_id,
                "when": when,
                "hour": local.hour,
                "dow": local.weekday(),         # 0=Mon, local
                # Grouped on the LOCAL date, so one evening's activity is not
                # split across two "days" by UTC midnight.
                "day": local.date().isoformat(),
                "lat": float(lat),
                "lon": float(lon),
                "speed": float(speed) if speed is not None else None,
            }
        )
    return out


# ── home / work inference ────────────────────────────────────────────────────

def _cluster(points: list[tuple[float, float]]) -> list[int]:
    """DBSCAN labels for lat/lon points, or all-noise if sklearn is missing.

    Haversine metric on radians is the correct distance for lat/lon — Euclidean
    on degrees would stretch longitude by cos(latitude) and, at Delhi's 28°N,
    make east-west neighbours look ~12% closer than they are.
    """
    if not points:
        return []
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
    except ImportError:
        return [-1] * len(points)

    radians = np.radians(np.asarray(points, dtype=float))
    model = DBSCAN(
        eps=CLUSTER_RADIUS_KM / EARTH_RADIUS_KM,
        min_samples=CLUSTER_MIN_SAMPLES,
        metric="haversine",
        algorithm="ball_tree",
    )
    return list(model.fit_predict(radians))


def _place_from(rows: list[dict[str, Any]], label: str) -> Place | None:
    """Cluster `rows` and return the dominant cluster as a Place."""
    if len(rows) < CLUSTER_MIN_SAMPLES:
        return None

    labels = _cluster([(r["lat"], r["lon"]) for r in rows])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for lab, row in zip(labels, rows):
        if lab != -1:                      # -1 is DBSCAN noise
            grouped[lab].append(row)
    if not grouped:
        return None

    biggest = max(grouped.values(), key=len)
    cams = Counter(r["camera_id"] for r in biggest)
    return Place(
        label=label,
        latitude=sum(r["lat"] for r in biggest) / len(biggest),
        longitude=sum(r["lon"] for r in biggest) / len(biggest),
        camera_ids=[c for c, _ in cams.most_common(5)],
        sightings=len(biggest),
        confidence=round(len(biggest) / len(rows), 3),
        dominant_camera=cams.most_common(1)[0][0],
    )


def infer_places(sightings: list[dict[str, Any]]) -> dict[str, Place | None]:
    """Infer home and workplace from when-and-where a vehicle is seen.

    Home first, then workplace as the densest weekday-commute cluster that is
    far enough from home to be somewhere else. Ordering matters: without the
    home fix the "work" cluster collapses onto the home junction, because the
    roads outside your house are where you are seen most often at any hour.
    """
    night = [s for s in sightings if s["hour"] in NIGHT_HOURS]
    home = _place_from(night, "home")

    # Weekdays only: a Saturday drive is leisure and drags the cluster towards
    # wherever the vehicle shops.
    daytime = [s for s in sightings if s["hour"] in DAY_HOURS and s["dow"] < 5]

    if home is not None:
        away = [
            s
            for s in daytime
            if haversine_km(home.latitude, home.longitude, s["lat"], s["lon"])
            >= MIN_WORK_SEPARATION_KM
        ]
    else:
        away = daytime

    return {"home": home, "work": _place_from(away, "work")}


# ── routine + anomalies ──────────────────────────────────────────────────────

def _daily_features(sightings: list[dict[str, Any]], home: Place | None):
    """One feature row per active day. Returns (days, matrix)."""
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sightings:
        by_day[s["day"]].append(s)

    days, matrix = [], []
    for day in sorted(by_day):
        rows = by_day[day]
        hours = [r["hour"] for r in rows]
        speeds = [r["speed"] for r in rows if r["speed"] is not None]
        km_home = (
            max(haversine_km(home.latitude, home.longitude, r["lat"], r["lon"]) for r in rows)
            if home
            else 0.0
        )
        # Span of the day's activity, in km, as a crude trip-extent proxy.
        spread = 0.0
        if len(rows) > 1:
            spread = max(
                haversine_km(rows[i]["lat"], rows[i]["lon"], rows[j]["lat"], rows[j]["lon"])
                for i in range(0, len(rows), max(1, len(rows) // 8))
                for j in range(i + 1, len(rows), max(1, len(rows) // 8))
            )
        days.append(day)
        matrix.append(
            [
                len(rows),                                   # how much it drove
                len({r["camera_id"] for r in rows}),         # how widely
                min(hours),
                max(hours),
                sum(hours) / len(hours),
                max(speeds) if speeds else 0.0,
                km_home,
                spread,
                1.0 if rows[0]["dow"] >= 5 else 0.0,         # weekend
            ]
        )
    return days, matrix


def find_anomalies(
    sightings: list[dict[str, Any]], home: Place | None, limit: int = 10
) -> list[AnomalyDay]:
    """Days that deviate from this vehicle's own routine, worst first."""
    days, matrix = _daily_features(sightings, home)
    if len(days) < MIN_DAYS_FOR_ANOMALY:
        return []

    try:
        import numpy as np
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return []

    features = np.asarray(matrix, dtype=float)
    # Scaled because the features are on wildly different units — a count of 4
    # and a distance of 40 km must not be compared raw.
    scaled = StandardScaler().fit_transform(features)
    model = IsolationForest(
        n_estimators=120,
        contamination=ANOMALY_CONTAMINATION,
        random_state=0,
    )
    model.fit(scaled)
    scores = model.score_samples(scaled)

    # Column means, to explain *why* a day was flagged. Without this the output
    # is a number an operator cannot act on.
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    names = [
        ("sightings", "unusually {dir} activity"),
        ("cameras", "{dir} spread across the network"),
        ("first_hour", "started {dir} than usual"),
        ("last_hour", "ended {dir} than usual"),
        ("mean_hour", "active at an unusual time of day"),
        ("max_speed", "{dir} peak speed"),
        ("km_from_home", "{dir} from its usual base"),
        ("trip_spread", "{dir} daily range"),
        ("weekend", "unusual weekend/weekday pattern"),
    ]

    # Rank by the forest, then gate on a real deviation. The gate is what stops
    # `contamination` from inventing a fixed quota of anomalies.
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1])

    out: list[AnomalyDay] = []
    for i, score in ranked:
        if len(out) >= limit:
            break
        reasons = []
        strongest = 0.0
        for col, (_, template) in enumerate(names):
            if stds[col] <= 0:
                continue
            z = (features[i][col] - means[col]) / stds[col]
            strongest = max(strongest, abs(z))
            if abs(z) >= ANOMALY_MIN_Z:
                reasons.append(
                    template.format(dir="higher" if z > 0 else "lower")
                    if "{dir}" in template
                    else template
                )
        if strongest < ANOMALY_MIN_Z:
            # Ranked low by the forest but not actually unlike anything. Because
            # `ranked` is sorted worst-first, everything after this is tamer
            # still.
            break
        out.append(
            AnomalyDay(
                day=days[i],
                score=round(float(score), 4),
                sightings=int(features[i][0]),
                distinct_cameras=int(features[i][1]),
                first_hour=int(features[i][2]),
                last_hour=int(features[i][3]),
                max_speed_kmh=round(float(features[i][5]), 1) or None,
                km_from_home=round(float(features[i][6]), 2) if home else None,
                reasons=reasons[:3],
            )
        )
    return out


def routine_summary(sightings: list[dict[str, Any]]) -> dict[str, Any]:
    """How regular this vehicle is, without any ML — plain counting.

    Reported alongside the model output because it is directly checkable, and
    a regularity figure an operator can verify by eye builds more trust in the
    inferred parts than a score alone.
    """
    if not sightings:
        return {}
    by_day = defaultdict(list)
    for s in sightings:
        by_day[s["day"]].append(s)

    span_days = (sightings[-1]["when"].date() - sightings[0]["when"].date()).days + 1
    active = len(by_day)
    hour_hist = Counter(s["hour"] for s in sightings)
    cam_hist = Counter(s["camera_id"] for s in sightings)
    dow_hist = Counter(s["dow"] for s in sightings)

    top_cams = cam_hist.most_common(5)
    concentration = sum(c for _, c in top_cams) / len(sightings)

    return {
        "first_seen": sightings[0]["when"].isoformat(),
        "last_seen": sightings[-1]["when"].isoformat(),
        "window_days": span_days,
        "active_days": active,
        "activity_rate": round(active / span_days, 3) if span_days else None,
        "total_sightings": len(sightings),
        "sightings_per_active_day": round(len(sightings) / active, 2) if active else None,
        "distinct_cameras": len(cam_hist),
        # High concentration means a small fixed corridor — the signature of a
        # commuter, and what makes next-hop prediction work for this vehicle.
        "corridor_concentration": round(concentration, 3),
        "top_cameras": [{"camera_id": c, "sightings": n} for c, n in top_cams],
        "peak_hour": hour_hist.most_common(1)[0][0],
        "busiest_weekday": dow_hist.most_common(1)[0][0],
        "weekend_share": round(
            sum(n for d, n in dow_hist.items() if d >= 5) / len(sightings), 3
        ),
    }


# ── the one entry point the API uses ─────────────────────────────────────────

def profile_vehicle(db: Session, plate: str, days: int = 365) -> dict[str, Any]:
    """Everything this module can say about one vehicle."""
    sightings = _sightings(db, plate, days)

    if len(sightings) < MIN_SIGHTINGS_FOR_BASE:
        return {
            "plate": plate,
            "status": "insufficient",
            "sightings": len(sightings),
            "required": MIN_SIGHTINGS_FOR_BASE,
            "detail": (
                f"Only {len(sightings)} sightings in {days} days. Home/work inference "
                "needs a stable pattern to find; guessing from this few would be "
                "confidently wrong."
            ),
        }

    places = infer_places(sightings)
    home = places["home"]
    anomalies = find_anomalies(sightings, home)

    return {
        "plate": plate,
        "status": "ok",
        "window_days": days,
        "routine": routine_summary(sightings),
        "places": {k: (asdict(v) if v else None) for k, v in places.items()},
        "commute_km": (
            round(
                haversine_km(
                    home.latitude, home.longitude,
                    places["work"].latitude, places["work"].longitude,
                ),
                2,
            )
            if home and places["work"]
            else None
        ),
        "anomalies": [asdict(a) for a in anomalies],
        "caveats": [
            "Home and workplace are INFERRED from when and where the vehicle is "
            "seen, not from any registry. A camera network sees roads, so these "
            "identify a neighbourhood junction, not an address.",
            "Anomaly scores are relative to THIS vehicle's own history and are "
            "not comparable between vehicles.",
        ],
    }

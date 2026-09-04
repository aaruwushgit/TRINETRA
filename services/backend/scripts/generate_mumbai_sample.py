#!/usr/bin/env python3
"""
Mumbai one-week sample dataset — the file to hand someone in a demo.

Produces a dataset in exactly the format `POST /jobs/dataset` accepts (see
`backend/services/dataset_service.SCHEMA`), so the sandbox's CITY DATASET tab
can be demonstrated end to end without asking the audience to bring their own
city: upload this, watch it validate, ingest it into an isolated database, and
see cameras, trajectories, heatmap and alerts come up on Mumbai.

Why generate rather than ship a static blob
-------------------------------------------
The events must sit in a *recent* week or the dashboard's default 24h/7d
windows show an empty map, so the timestamps are anchored to "now" at build
time. Regenerating is one command; a checked-in file with last year's dates is
a demo that fails silently.

What makes it a believable week rather than random rows
-------------------------------------------------------
* **Real geography.** 40 junctions on Mumbai's actual arterial network — the
  Western and Eastern Express Highways, the Sea Link, SV Road, LBS Marg, the
  Sion–Panvel corridor — with their true coordinates and plausible posted
  limits per road class.
* **Corridors, not a bag of points.** Cameras are grouped into named corridors
  and a vehicle drives *along* one, hop by hop, so consecutive sightings are
  adjacent cameras and implied speeds are physical. This is the property that
  makes trajectory reconstruction and next-hop prediction show something real;
  independent random events per camera would make every implied speed nonsense.
* **A commuter week.** Weekday morning and evening peaks, a flatter Saturday,
  a quiet Sunday, and a congestion curve that drags speeds down inside the
  peaks. Regulars repeat the same corridor at roughly the same time each
  working day, which is exactly what the Markov next-hop predictor learns from.
* **Ground truth to find.** A handful of habitual speeders and three watchlist
  plates are planted deliberately and written to a sidecar file, so a
  demonstration can end with "the platform found these" against a list that
  was decided in advance.

Usage
-----
  .venv/bin/python scripts/generate_mumbai_sample.py
  .venv/bin/python scripts/generate_mumbai_sample.py --vehicles 1200 --days 7
  .venv/bin/python scripts/generate_mumbai_sample.py --format csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUT_DIR = BASE_DIR / "deployments" / "mumbai"

# IST. The dataset is written with an explicit +05:30 offset because a demo
# audience reads "08:40 in the morning peak", not "03:10Z".
IST = timezone(timedelta(hours=5, minutes=30))


# ─────────────────────────────────────────────────────────────────────────────
# The network: real Mumbai junctions, grouped into corridors
# ─────────────────────────────────────────────────────────────────────────────
# (camera_id, name, lat, lon, road, road_class). Order within a corridor is
# geographic, so consecutive entries are genuinely adjacent on the ground and a
# hop between them is a drive a car could make.

CORRIDORS: dict[str, list[tuple[str, str, float, float, str, str]]] = {
    "WEH_NB": [
        ("MH_WEH_BANDRA",     "Western Express Hwy x Bandra",        19.0607, 72.8494, "Western Express Highway", "trunk"),
        ("MH_WEH_KHAR",       "Western Express Hwy x Khar Subway",   19.0728, 72.8478, "Western Express Highway", "trunk"),
        ("MH_WEH_SANTACRUZ",  "Western Express Hwy x Santacruz",     19.0810, 72.8420, "Western Express Highway", "trunk"),
        ("MH_WEH_VILEPARLE",  "Western Express Hwy x Vile Parle",    19.0990, 72.8460, "Western Express Highway", "trunk"),
        ("MH_WEH_ANDHERI",    "Western Express Hwy x Andheri E",     19.1180, 72.8500, "Western Express Highway", "trunk"),
        ("MH_WEH_JVLR",       "Western Express Hwy x JVLR",          19.1290, 72.8570, "Western Express Highway", "trunk"),
        ("MH_WEH_GOREGAON",   "Western Express Hwy x Goregaon E",    19.1640, 72.8580, "Western Express Highway", "trunk"),
        ("MH_WEH_BORIVALI",   "Western Express Hwy x Borivali E",    19.2290, 72.8570, "Western Express Highway", "trunk"),
        ("MH_WEH_DAHISAR",    "Western Express Hwy x Dahisar Check", 19.2500, 72.8590, "Western Express Highway", "trunk"),
    ],
    "EEH_NB": [
        ("MH_EEH_SION",       "Eastern Express Hwy x Sion",          19.0390, 72.8620, "Eastern Express Highway", "trunk"),
        ("MH_EEH_CHEMBUR",    "Eastern Express Hwy x Chembur",       19.0620, 72.8990, "Eastern Express Highway", "trunk"),
        ("MH_EEH_GHATKOPAR",  "Eastern Express Hwy x Ghatkopar",     19.0860, 72.9080, "Eastern Express Highway", "trunk"),
        ("MH_EEH_VIKHROLI",   "Eastern Express Hwy x Vikhroli",      19.1100, 72.9250, "Eastern Express Highway", "trunk"),
        ("MH_EEH_KANJURMARG", "Eastern Express Hwy x Kanjurmarg",    19.1290, 72.9360, "Eastern Express Highway", "trunk"),
        ("MH_EEH_MULUND",     "Eastern Express Hwy x Mulund Check",  19.1720, 72.9560, "Eastern Express Highway", "trunk"),
        ("MH_EEH_AIROLI",     "Airoli Bridge Toll Plaza",            19.1580, 72.9980, "Airoli Bridge",           "trunk"),
    ],
    "SEALINK_SB": [
        ("MH_SL_WORLI",       "Bandra-Worli Sea Link (Worli end)",   19.0080, 72.8180, "Bandra-Worli Sea Link", "motorway"),
        ("MH_SL_MID",         "Bandra-Worli Sea Link (mid-span)",    19.0280, 72.8180, "Bandra-Worli Sea Link", "motorway"),
        ("MH_SL_BANDRA",      "Bandra-Worli Sea Link (Bandra end)",  19.0430, 72.8200, "Bandra-Worli Sea Link", "motorway"),
        ("MH_MAHIM_CSWY",     "Mahim Causeway",                      19.0400, 72.8400, "Mahim Causeway",        "primary"),
    ],
    "SVROAD_NB": [
        ("MH_SV_MAHIM",       "SV Road x Mahim",                     19.0410, 72.8420, "Swami Vivekanand Road", "primary"),
        ("MH_SV_BANDRA",      "SV Road x Bandra Talao",              19.0550, 72.8400, "Swami Vivekanand Road", "primary"),
        ("MH_SV_KHAR",        "SV Road x Khar Danda",                19.0710, 72.8330, "Swami Vivekanand Road", "primary"),
        ("MH_SV_SANTACRUZ",   "SV Road x Santacruz W",               19.0820, 72.8390, "Swami Vivekanand Road", "primary"),
        ("MH_SV_ANDHERI",     "SV Road x Andheri W",                 19.1190, 72.8420, "Swami Vivekanand Road", "primary"),
        ("MH_SV_JOGESHWARI",  "SV Road x Jogeshwari W",              19.1360, 72.8450, "Swami Vivekanand Road", "primary"),
        ("MH_SV_MALAD",       "SV Road x Malad W",                   19.1870, 72.8400, "Swami Vivekanand Road", "primary"),
    ],
    "LBS_NB": [
        ("MH_LBS_SION",       "LBS Marg x Sion Circle",              19.0400, 72.8650, "LBS Marg", "primary"),
        ("MH_LBS_KURLA",      "LBS Marg x Kurla",                    19.0700, 72.8790, "LBS Marg", "primary"),
        ("MH_LBS_GHATKOPAR",  "LBS Marg x Ghatkopar W",              19.0870, 72.9080, "LBS Marg", "primary"),
        ("MH_LBS_BHANDUP",    "LBS Marg x Bhandup",                  19.1440, 72.9370, "LBS Marg", "primary"),
        ("MH_LBS_MULUND",     "LBS Marg x Mulund W",                 19.1720, 72.9490, "LBS Marg", "primary"),
    ],
    "SOUTH_CBD": [
        ("MH_WORLI_NAKA",     "Worli Naka Junction",                 19.0000, 72.8180, "Annie Besant Road",     "primary"),
        ("MH_HAJIALI",        "Haji Ali Junction",                   18.9800, 72.8100, "Lala Lajpatrai Marg",   "primary"),
        ("MH_PEDDAR",         "Peddar Road x Kemps Corner",          18.9660, 72.8080, "Peddar Road",           "primary"),
        ("MH_MARINE",         "Marine Drive x Charni Road",          18.9500, 72.8200, "Netaji Subhash Marg",   "primary"),
        ("MH_CST",            "CST x DN Road",                       18.9400, 72.8350, "Dr DN Road",            "secondary"),
        ("MH_COLABA",         "Colaba Causeway x Regal",             18.9220, 72.8320, "Shahid Bhagat Singh Rd", "secondary"),
    ],
    "SION_PANVEL": [
        ("MH_SP_CHEMBUR",     "Sion-Panvel Hwy x Chembur",           19.0500, 72.9000, "Sion-Panvel Highway", "trunk"),
        ("MH_SP_MANKHURD",    "Sion-Panvel Hwy x Mankhurd",          19.0480, 72.9300, "Sion-Panvel Highway", "trunk"),
        ("MH_SP_VASHI",       "Vashi Bridge Toll Plaza",             19.0680, 72.9990, "Sion-Panvel Highway", "trunk"),
    ],
}

# Posted limit by OSM road class, in km/h. Mumbai's expressways are signed at
# 80; arterials at 50; the island-city roads lower still.
CLASS_LIMIT = {"motorway": 80.0, "trunk": 80.0, "primary": 50.0, "secondary": 40.0}

# Cardinal heading of each corridor as traversed forwards, used as the camera's
# `direction`. A camera watches one carriageway.
CORRIDOR_HEADING = {
    "WEH_NB": ("NORTH", "SOUTH"),
    "EEH_NB": ("NORTH", "SOUTH"),
    "SEALINK_SB": ("NORTH", "SOUTH"),
    "SVROAD_NB": ("NORTH", "SOUTH"),
    "LBS_NB": ("NORTH", "SOUTH"),
    "SOUTH_CBD": ("SOUTH", "NORTH"),
    "SION_PANVEL": ("EAST", "WEST"),
}

# Corridors that physically meet, so a trip can change corridor mid-journey at
# the shared junction rather than teleporting.
INTERCHANGES = [
    ("WEH_NB", "MH_WEH_BANDRA", "SEALINK_SB", "MH_SL_BANDRA"),
    ("WEH_NB", "MH_WEH_SANTACRUZ", "SVROAD_NB", "MH_SV_SANTACRUZ"),
    ("SVROAD_NB", "MH_SV_MAHIM", "SEALINK_SB", "MH_MAHIM_CSWY"),
    ("EEH_NB", "MH_EEH_SION", "LBS_NB", "MH_LBS_SION"),
    ("EEH_NB", "MH_EEH_CHEMBUR", "SION_PANVEL", "MH_SP_CHEMBUR"),
    ("SEALINK_SB", "MH_SL_WORLI", "SOUTH_CBD", "MH_WORLI_NAKA"),
    ("LBS_NB", "MH_LBS_GHATKOPAR", "EEH_NB", "MH_EEH_GHATKOPAR"),
]

VEHICLE_TYPES = ("car", "motorcycle", "auto", "taxi", "bus", "truck")
VEHICLE_TYPE_WEIGHTS = (0.44, 0.26, 0.10, 0.11, 0.04, 0.05)
VEHICLE_COLORS = ("white", "silver", "grey", "black", "red", "blue", "brown")
VEHICLE_COLOR_WEIGHTS = (0.33, 0.20, 0.14, 0.12, 0.08, 0.08, 0.05)

# Heavy vehicles are slower and, in Mumbai, barred from several arterials in
# the day — approximated by capping their speed rather than routing them apart.
TYPE_SPEED_CAP = {"car": 999.0, "taxi": 999.0, "motorcycle": 90.0,
                  "auto": 55.0, "bus": 65.0, "truck": 70.0}

# Relative traffic volume by hour of day, weekday. Mumbai's peaks are sharp and
# the evening one is both later and broader than the morning's.
HOUR_WEIGHTS = (
    0.012, 0.007, 0.005, 0.005, 0.009, 0.022, 0.041, 0.068,   # 00–07
    0.092, 0.081, 0.056, 0.044, 0.043, 0.045, 0.043, 0.048,   # 08–15
    0.058, 0.078, 0.088, 0.076, 0.055, 0.038, 0.026, 0.017,   # 16–23
)
# Fraction of the free-flow limit actually achieved, by hour. 0.35 in the peak
# is the well-documented crawl on the Express Highways.
CONGESTION = (
    0.95, 0.97, 0.98, 0.98, 0.96, 0.90, 0.80, 0.62,
    0.42, 0.45, 0.62, 0.70, 0.72, 0.71, 0.70, 0.64,
    0.52, 0.38, 0.35, 0.44, 0.62, 0.76, 0.86, 0.92,
)
# Volume multiplier by weekday (Mon=0). Saturday is a working day for much of
# the city; Sunday is not.
DAY_VOLUME = {0: 1.00, 1: 1.03, 2: 1.02, 3: 1.03, 4: 1.06, 5: 0.78, 6: 0.48}

# A camera does not see every vehicle that passes: occlusion by the vehicle in
# front, glare, and plates outside the OCR grammar.
DETECTION_RATE = 0.88

# Maharashtra plate format: MH <RTO 2 digits> <1-2 letters> <4 digits>.
MH_RTO = ("01", "02", "03", "04", "05", "06", "43", "46", "47", "48")
OTHER_STATES = (("GJ", 0.30), ("KA", 0.20), ("RJ", 0.15), ("MP", 0.15),
                ("UP", 0.10), ("DL", 0.10))
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # I and O omitted, as on real plates

# Planted ground truth, so a demo can end by comparing findings to a list
# written down beforehand.
WATCHLIST = [
    ("MH01AB0007", "Stolen vehicle — FIR 214/2026, Bandra PS"),
    ("MH02XY1990", "Suspect vehicle — traffic bureau lookout notice"),
    ("MH43KL4242", "Unpaid challans exceeding threshold"),
]
# Shortest gap between one trip ending and the next beginning. A vehicle that
# arrives and leaves inside this window is not a trip, it is a data error.
MIN_PARK = timedelta(minutes=12)

SPEEDER_RATE = 0.04
SPEEDER_BIAS = (1.30, 1.70)
NORMAL_BIAS = (0.85, 1.10)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build_cameras() -> tuple[list[dict], dict[str, dict]]:
    """Flatten the corridors into the `cameras[]` artefact."""
    cameras: list[dict] = []
    index: dict[str, dict] = {}
    for corridor, stops in CORRIDORS.items():
        forward, _back = CORRIDOR_HEADING[corridor]
        for cid, name, lat, lon, road, klass in stops:
            if cid in index:      # a junction shared by two corridors
                continue
            cam = {
                "camera_id": cid,
                "name": name,
                "location": f"{road}, Mumbai",
                "latitude": lat,
                "longitude": lon,
                "road": road,
                "direction": forward,
                "speed_limit_kmh": CLASS_LIMIT[klass],
                "camera_type": "ANPR",
            }
            cameras.append(cam)
            index[cid] = cam
    return cameras, index


def make_plates(count: int, rng: random.Random) -> list[str]:
    """Unique, format-valid plates; ~82% Maharashtra, the rest visiting states."""
    seen: set[str] = set()
    out: list[str] = []
    other_states = [s for s, _ in OTHER_STATES]
    other_weights = [w for _, w in OTHER_STATES]
    while len(out) < count:
        if rng.random() < 0.82:
            series = "".join(rng.choice(LETTERS) for _ in range(rng.choice((1, 2))))
            plate = f"MH{rng.choice(MH_RTO)}{series}{rng.randrange(10000):04d}"
        else:
            state = rng.choices(other_states, other_weights)[0]
            series = "".join(rng.choice(LETTERS) for _ in range(2))
            plate = f"{state}{rng.randrange(1, 40):02d}{series}{rng.randrange(10000):04d}"
        if plate in seen:
            continue
        seen.add(plate)
        out.append(plate)
    return out


def corridor_path(corridor: str, start: int, hops: int, forward: bool,
                  rng: random.Random) -> list[tuple[str, str]]:
    """Walk `hops` cameras along a corridor, optionally switching at an interchange.

    Returns [(corridor, camera_id), ...]. Switching corridors is what turns a
    week of straight-line commutes into a network with origin–destination
    structure the OD-matrix endpoint can actually report on.
    """
    stops = CORRIDORS[corridor]
    step = 1 if forward else -1
    idx = start
    path: list[tuple[str, str]] = []
    remaining = hops

    while remaining > 0:
        if not (0 <= idx < len(stops)):
            break
        cid = stops[idx][0]
        path.append((corridor, cid))
        remaining -= 1

        # At a shared junction, sometimes carry on down the other road.
        if remaining > 1 and rng.random() < 0.18:
            for a_corr, a_cam, b_corr, b_cam in INTERCHANGES:
                pair = None
                if a_corr == corridor and a_cam == cid:
                    pair = (b_corr, b_cam)
                elif b_corr == corridor and b_cam == cid:
                    pair = (a_corr, a_cam)
                if pair is None:
                    continue
                new_corr, new_cam = pair
                new_stops = [s[0] for s in CORRIDORS[new_corr]]
                corridor, stops = new_corr, CORRIDORS[new_corr]
                idx = new_stops.index(new_cam)
                step = rng.choice((1, -1))
                break
        idx += step
    return path


def hop_speed(limit: float, hour: int, bias: float, cap: float, rng: random.Random) -> float:
    """Speed on one hop: the posted limit, scaled by congestion and driver bias."""
    free = limit * CONGESTION[hour]
    speed = free * bias * rng.uniform(0.90, 1.12)
    return max(6.0, min(speed, cap, limit * 1.9))


def generate(vehicles: int, days: int, seed: int, end: datetime) -> dict:
    rng = random.Random(seed)
    cameras, cam_index = build_cameras()
    corridors = list(CORRIDORS)

    plates = make_plates(vehicles, rng)
    # Guarantee the watchlist is actually driving around; an alert demo against
    # plates that never appear proves nothing.
    for i, (plate, _reason) in enumerate(WATCHLIST):
        plates[i] = plate

    speeders = set(rng.sample(plates, max(1, int(len(plates) * SPEEDER_RATE))))
    # Watchlist vehicles get a fixed home corridor and a peak-hour habit so
    # their trajectories are long and their next hop is genuinely predictable.
    profile: dict[str, dict] = {}
    for plate in plates:
        home = rng.choice(corridors)
        profile[plate] = {
            "home": home,
            "type": rng.choices(VEHICLE_TYPES, VEHICLE_TYPE_WEIGHTS)[0],
            "color": rng.choices(VEHICLE_COLORS, VEHICLE_COLOR_WEIGHTS)[0],
            # Regulars leave within a few minutes of the same time each day.
            "regular": rng.random() < 0.55,
            "am_minute": rng.randrange(7 * 60 + 30, 10 * 60),
            "pm_minute": rng.randrange(17 * 60, 20 * 60 + 30),
            "bias": rng.uniform(*(SPEEDER_BIAS if plate in speeders else NORMAL_BIAS)),
            "trips_per_day": rng.choices((0, 1, 2, 3), (0.18, 0.24, 0.44, 0.14))[0],
        }

    start = (end - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
    events: list[dict] = []
    track_counter: dict[str, int] = {}

    hour_choices = list(range(24))

    for plate in plates:
        cfg = profile[plate]
        vtype, vcolor = cfg["type"], cfg["color"]
        cap = TYPE_SPEED_CAP[vtype]
        is_watch = any(plate == w for w, _ in WATCHLIST)

        # Carried across days: a trip that runs past midnight still has to
        # finish before the next morning's commute starts.
        busy_until = start

        for day in range(days):
            day_start = start + timedelta(days=day)
            weekday = day_start.weekday()
            volume = DAY_VOLUME[weekday]
            trips = cfg["trips_per_day"]
            if is_watch:
                trips = max(trips, 2)          # keep the watchlist visible daily
            trips = sum(1 for _ in range(trips) if rng.random() < volume)

            # Departure times for the whole day, decided up front and sorted.
            # A vehicle can only be in one place at a time: generating trips
            # independently and sorting the events afterwards lets two trips of
            # the same plate interleave, which reads downstream as one vehicle
            # crossing the city in seconds. `busy_until` below enforces it.
            departures: list[int] = []
            for trip in range(trips):
                if cfg["regular"] and trip < 2 and weekday < 5:
                    base = cfg["am_minute"] if trip == 0 else cfg["pm_minute"]
                    minute = int(rng.gauss(base, 14))
                else:
                    hour = rng.choices(hour_choices, HOUR_WEIGHTS)[0]
                    minute = hour * 60 + rng.randrange(60)
                departures.append(max(0, min(minute, 24 * 60 - 1)))
            departures.sort()

            for trip, minute in enumerate(departures):
                corridor = cfg["home"] if rng.random() < 0.72 else rng.choice(corridors)
                stops = CORRIDORS[corridor]
                forward = (trip % 2 == 0) if cfg["regular"] else rng.random() < 0.5
                hops = rng.randint(3, min(7, len(stops)))
                span = max(0, len(stops) - hops)
                begin = rng.randint(0, span) if forward else rng.randint(hops - 1, len(stops) - 1)
                path = corridor_path(corridor, begin, hops, forward, rng)
                if len(path) < 2:
                    continue

                clock = day_start + timedelta(minutes=minute, seconds=rng.randrange(60))
                # Park for at least 12 minutes between trips. The comparison is
                # against busy_until *plus* that minimum, not against
                # busy_until itself: a departure drawn a few seconds after the
                # previous arrival passes a bare `<` test and produces a
                # vehicle apparently teleporting 10 km down the corridor.
                if clock < busy_until + MIN_PARK:
                    shifted = busy_until + timedelta(minutes=rng.randint(12, 90))
                    if shifted - clock > timedelta(hours=4):
                        # Delaying this far would smear the commute peaks the
                        # whole model exists to produce. The vehicle simply did
                        # not make this trip.
                        continue
                    clock = shifted
                if clock >= end:
                    break

                prev_cam = None
                seen_cams: set[str] = set()
                for _corr, cid in path:
                    if cid in seen_cams:
                        # An interchange switch can double back onto a junction
                        # already passed; a real trip does not.
                        continue
                    seen_cams.add(cid)
                    cam = cam_index[cid]
                    if prev_cam is not None:
                        km = haversine_km(prev_cam["latitude"], prev_cam["longitude"],
                                          cam["latitude"], cam["longitude"])
                        hour = clock.hour
                        speed = hop_speed(
                            min(prev_cam["speed_limit_kmh"], cam["speed_limit_kmh"]),
                            hour, cfg["bias"], cap, rng,
                        )
                        # Travel time from the hop's real distance and that
                        # speed — this is what keeps implied speeds physical.
                        clock = clock + timedelta(hours=km / speed)
                        # A junction costs time even when moving.
                        clock += timedelta(seconds=rng.randrange(5, 70))
                    else:
                        speed = hop_speed(cam["speed_limit_kmh"], clock.hour,
                                          cfg["bias"], cap, rng)

                    if clock >= end:
                        break
                    prev_cam = cam
                    if rng.random() > DETECTION_RATE:
                        continue   # camera missed this one — a real miss, not a gap in the model

                    track_counter[cid] = track_counter.get(cid, 0) + 1
                    events.append({
                        "camera_id": cid,
                        "timestamp": clock.astimezone(IST).isoformat(timespec="seconds"),
                        "plate": plate,
                        "plate_confidence": round(rng.uniform(0.79, 0.99), 2),
                        "vehicle_type": vtype,
                        "vehicle_color": vcolor,
                        "speed": round(speed, 1),
                        "direction": cam["direction"],
                        "local_track_id": f"{cid}_T{track_counter[cid]}",
                    })

                busy_until = clock

    events.sort(key=lambda e: e["timestamp"])

    ground_truth = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "watchlist": [{"plate": p, "reason": r} for p, r in WATCHLIST],
        "habitual_speeders": sorted(speeders),
        "note": (
            "Habitual speeders drive at 1.30-1.70x the congestion-adjusted free-flow "
            "speed. Load this dataset, then check GET /vehicles/analytics/speed-defaulters "
            "and the blacklist alerts against these lists."
        ),
    }
    return {"cameras": cameras, "events": events, "_ground_truth": ground_truth}


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicles", type=int, default=900,
                    help="unique plates (default 900 — ~25k events over a week)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1960)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--format", choices=("json", "csv", "both"), default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    end = datetime.now(timezone.utc)
    print(f"Generating {args.days}-day Mumbai sample ending {end.astimezone(IST):%Y-%m-%d %H:%M %Z}...")

    data = generate(args.vehicles, args.days, args.seed, end)
    ground_truth = data.pop("_ground_truth")
    cameras, events = data["cameras"], data["events"]

    if args.format in ("json", "both"):
        target = out_dir / "mumbai_week.json"
        target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {target}  ({target.stat().st_size / 1e6:.2f} MB)")

    if args.format in ("csv", "both"):
        cam_csv = out_dir / "mumbai_cameras.csv"
        evt_csv = out_dir / "mumbai_events.csv"
        write_csv(cam_csv, cameras,
                  ["camera_id", "name", "location", "latitude", "longitude",
                   "road", "direction", "speed_limit_kmh", "camera_type"])
        write_csv(evt_csv, events,
                  ["camera_id", "timestamp", "plate", "plate_confidence",
                   "vehicle_type", "vehicle_color", "speed", "direction", "local_track_id"])
        print(f"  wrote {cam_csv}")
        print(f"  wrote {evt_csv}  ({evt_csv.stat().st_size / 1e6:.2f} MB)")

    gt = out_dir / "mumbai_ground_truth.json"
    gt.write_text(json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {gt}")

    # A quick self-check on the physics: if any implied speed here is absurd,
    # the dataset would teach the platform nonsense, and it is better to find
    # that at generation time than in a demo.
    _, cam_index = build_cameras()
    by_plate: dict[str, list[dict]] = {}
    for ev in events:
        by_plate.setdefault(ev["plate"], []).append(ev)
    implied: list[float] = []
    for rows in by_plate.values():
        for a, b in zip(rows, rows[1:]):
            if a["camera_id"] == b["camera_id"]:
                continue
            ca, cb = cam_index[a["camera_id"]], cam_index[b["camera_id"]]
            hours = (datetime.fromisoformat(b["timestamp"])
                     - datetime.fromisoformat(a["timestamp"])).total_seconds() / 3600
            if hours <= 0:
                continue
            km = haversine_km(ca["latitude"], ca["longitude"], cb["latitude"], cb["longitude"])
            if km > 0.1:
                implied.append(km / hours)
    implied.sort()

    def pct(p: float) -> float:
        return implied[min(len(implied) - 1, int(len(implied) * p))] if implied else 0.0

    stamps = [e["timestamp"] for e in events]
    print()
    print(f"  cameras           {len(cameras)}")
    print(f"  events            {len(events):,}")
    print(f"  unique plates     {len(by_plate):,}")
    print(f"  window            {stamps[0]}  ->  {stamps[-1]}")
    print(f"  implied speed     median {pct(0.50):.1f} km/h, p95 {pct(0.95):.1f}, "
          f"max {implied[-1] if implied else 0:.1f}")
    print(f"  watchlist planted {len(WATCHLIST)}, habitual speeders "
          f"{len(ground_truth['habitual_speeders'])}")
    print()
    print("Upload deployments/mumbai/mumbai_week.json in the sandbox's CITY DATASET tab,")
    print("or:  curl -F file=@deployments/mumbai/mumbai_week.json -F sandbox=true \\")
    print("          http://localhost:8000/jobs/dataset")


if __name__ == "__main__":
    main()

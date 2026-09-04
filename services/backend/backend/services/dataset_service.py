"""
City-dataset ingestion — "bring your own city".

The sandbox page (frontend/test.html, CITY DATASET tab) lets anyone upload a
description of *their* camera network plus the plate sightings recorded on it,
and have the whole platform — map, trajectories, heatmap, alerts, analytics —
come up on that city instead of Delhi. This module is the server half of that.

Why the server re-validates what the browser already validated
-------------------------------------------------------------
The page validates client-side so the user gets a per-row error list without a
round trip. That is a *usability* feature, not a security boundary: the endpoint
is reachable with curl. So the same contract is enforced here, and the error
shape is identical (`artefact`, `row`, `msg`) so the sandbox can render server
errors in the exact widget it already uses for its own.

Why the contract lives in one place
-----------------------------------
`SCHEMA` below is the single source of truth and is *served* (GET
/jobs/dataset/schema). The sandbox renders its field table and its downloadable
templates from that response rather than from a hand-copied table, so the
documented format cannot drift away from the format the parser accepts.

Isolation
---------
`sandbox=True` writes into a throwaway per-upload SQLite file with the
production schema (`video_job_service._SandboxDB`) — the same isolation the
video/photo jobs use. A stranger's dataset must never be able to overwrite the
live city's cameras, so sandbox is the default everywhere it is offered.
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# ─────────────────────────────────────────────────────────────────────────────
# The contract
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA: dict[str, Any] = {
    "version": "1.0",
    "description": (
        "Two artefacts. `cameras` describe where you are watching; `events` are "
        "plate sightings at those cameras. Upload one JSON object holding either "
        "or both keys, a bare JSON array of one artefact, or a CSV of one artefact."
    ),
    "accepted_media": [
        {
            "format": "json",
            "extensions": [".json"],
            "shapes": [
                '{"cameras": [...], "events": [...]}',
                '{"cameras": [...]}',
                '{"events": [...]}',
                "[...]  (bare array — artefact detected from the keys present)",
            ],
        },
        {
            "format": "csv",
            "extensions": [".csv", ".txt"],
            "shapes": [
                "header row + one artefact per file; artefact detected from the "
                "headers, or forced with the `kind` form field (cameras|events)"
            ],
            "notes": [
                "Comma-delimited, RFC 4180 quoting. Unknown columns are ignored "
                "and reported back in `unknown_headers`.",
            ],
        },
    ],
    "artefacts": {
        "cameras": {
            "primary_key": "camera_id",
            "fields": [
                {"name": "camera_id", "type": "string", "required": True,
                 "description": "Stable id. Events reference this."},
                {"name": "name", "type": "string", "required": False},
                {"name": "location", "type": "string", "required": False},
                {"name": "latitude", "type": "float", "required": True, "range": [-90, 90]},
                {"name": "longitude", "type": "float", "required": True, "range": [-180, 180]},
                {"name": "road", "type": "string", "required": False},
                {"name": "direction", "type": "string", "required": False,
                 "enum": ["NORTH", "SOUTH", "EAST", "WEST", "NE", "NW", "SE", "SW"],
                 "description": "Free text is accepted; uppercased on ingest."},
                {"name": "speed_limit_kmh", "type": "number", "required": False,
                 "range": [1, 200], "default": 60.0},
                {"name": "camera_type", "type": "string", "required": False, "default": "ANPR"},
            ],
        },
        "events": {
            "primary_key": "event_id (generated)",
            "identity_rule": (
                "A sighting needs camera_id, timestamp, and SOME identity. Identity is "
                "either a `plate`, or — when the camera could not read one — at least one "
                "attribute from `plate_partial`, `vehicle_type`, `vehicle_color`, "
                "`vehicle_make`, `vehicle_model`. Rows with a plate are tracked across "
                "cameras; attribute-only rows are recorded as UNIDENTIFIED and still count "
                "toward volume, speed and heatmap analytics."
            ),
            "fields": [
                {"name": "camera_id", "type": "string", "required": True,
                 "description": "Must match a camera in this upload or an existing camera."},
                {"name": "timestamp", "type": "string", "required": True,
                 "format": "ISO 8601, e.g. 2026-08-31T09:14:02+05:30",
                 "description": "Offset-aware is preferred; naive is read as UTC."},
                {"name": "plate", "type": "string", "required": False,
                 "requires_one_of": ["plate", "plate_partial", "vehicle_type",
                                     "vehicle_color", "vehicle_make", "vehicle_model"],
                 "description": "Normalised on ingest: uppercased, spaces and hyphens "
                                "removed. Omit when the plate was not read — do not "
                                "invent one."},
                {"name": "plate_confidence", "type": "float", "required": False, "range": [0, 1]},
                {"name": "plate_partial", "type": "string", "required": False,
                 "description": "Partially legible plate. Use ? or * for unread characters, "
                                "e.g. MH01??1234. Searchable; never treated as a plate read."},
                {"name": "plate_raw", "type": "string", "required": False,
                 "description": "Unformatted OCR output before grammar correction, kept "
                                "for audit."},
                {"name": "vehicle_type", "type": "string", "required": False,
                 "enum": ["car", "motorcycle", "auto", "bus", "truck", "taxi",
                          "van", "tractor", "bicycle", "other"]},
                {"name": "vehicle_color", "type": "string", "required": False,
                 "enum": ["white", "silver", "grey", "black", "red", "blue", "brown",
                          "green", "yellow", "orange", "maroon", "beige", "other"],
                 "description": "Free text is accepted; lowercased on ingest."},
                {"name": "vehicle_make", "type": "string", "required": False,
                 "description": "Manufacturer, e.g. Maruti, Hyundai, Tata."},
                {"name": "vehicle_model", "type": "string", "required": False,
                 "description": "Model, e.g. Swift, i20, Nexon."},
                {"name": "attribute_confidence", "type": "float", "required": False,
                 "range": [0, 1],
                 "description": "Confidence in the type/colour/make/model classification, "
                                "separate from plate_confidence."},
                {"name": "speed", "type": "number", "required": False, "range": [0, 300],
                 "description": "km/h at the camera. Omit if unmeasured — do not send 0."},
                {"name": "direction", "type": "string", "required": False},
                {"name": "local_track_id", "type": "string", "required": False},
                {"name": "latitude", "type": "float", "required": False,
                 "description": "Defaults to the camera's own coordinate."},
                {"name": "longitude", "type": "float", "required": False},
            ],
        },
    },
    "limits": {
        "max_upload_bytes": 64 * 1024 * 1024,
        "max_events": 500_000,
        "max_cameras": 20_000,
        "max_reported_errors": 200,
    },
    "example": {
        "cameras": [
            {
                "camera_id": "CAM-MG-ROAD-01",
                "name": "MG Road Junction North",
                "location": "MG Road x Brigade Road",
                "latitude": 12.9752,
                "longitude": 77.6068,
                "road": "MG Road",
                "direction": "NORTH",
                "speed_limit_kmh": 50,
            },
            {
                "camera_id": "CAM-ORR-14",
                "name": "Outer Ring Road KM14",
                "location": "ORR near Marathahalli",
                "latitude": 12.9569,
                "longitude": 77.7011,
                "road": "Outer Ring Road",
                "direction": "EAST",
                "speed_limit_kmh": 80,
            },
        ],
        "events": [
            {
                "camera_id": "CAM-MG-ROAD-01",
                "timestamp": "2026-08-31T09:14:02+05:30",
                "plate": "KA01AB1234",
                "plate_confidence": 0.94,
                "vehicle_type": "car",
                "speed": 46.2,
            },
            {
                "camera_id": "CAM-ORR-14",
                "timestamp": "2026-08-31T09:18:47+05:30",
                "plate": "KA01AB1234",
                "plate_confidence": 0.88,
                "vehicle_type": "car",
                "vehicle_color": "white",
                "speed": 71.5,
            },
            {
                "_comment": (
                    "No plate read — the vehicle was moving too fast for a clean "
                    "crop. Attributes are enough to record the sighting."
                ),
                "camera_id": "CAM-ORR-14",
                "timestamp": "2026-08-31T09:19:03+05:30",
                "plate_partial": "KA05??77??",
                "vehicle_type": "motorcycle",
                "vehicle_color": "red",
                "vehicle_make": "Bajaj",
                "vehicle_model": "Pulsar",
                "attribute_confidence": 0.81,
                "speed": 94.0,
            },
        ],
    },
}

MAX_UPLOAD_BYTES: int = SCHEMA["limits"]["max_upload_bytes"]
MAX_EVENTS: int = SCHEMA["limits"]["max_events"]
MAX_CAMERAS: int = SCHEMA["limits"]["max_cameras"]
MAX_ERRORS: int = SCHEMA["limits"]["max_reported_errors"]

CAMERA_COLUMNS = [f["name"] for f in SCHEMA["artefacts"]["cameras"]["fields"]]
EVENT_COLUMNS = [f["name"] for f in SCHEMA["artefacts"]["events"]["fields"]]

_REQUIRED = {
    art: {f["name"] for f in spec["fields"] if f["required"]}
    for art, spec in SCHEMA["artefacts"].items()
}

# Headers that unambiguously identify a CSV's artefact when `kind=auto`.
_CAMERA_MARKERS = {"latitude", "longitude", "speed_limit_kmh", "camera_type"}
_EVENT_MARKERS = {"timestamp", "plate", "plate_confidence", "local_track_id",
                  "plate_partial", "vehicle_make", "vehicle_model", "attribute_confidence"}

# At least one of these must be present for a sighting to mean anything. A row
# with a camera and a time and nothing else records that *something* passed,
# which no downstream query can use.
_IDENTITY_FIELDS = ("plate", "plate_partial", "vehicle_type", "vehicle_color",
                    "vehicle_make", "vehicle_model")

_PLATE_STRIP = re.compile(r"[\s\-_.]+")


# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RowError:
    artefact: str
    row: int          # 1-based within its artefact, matching the sandbox's display
    msg: str

    def as_dict(self) -> dict[str, Any]:
        return {"artefact": self.artefact, "row": self.row, "msg": self.msg}


@dataclass
class ParsedDataset:
    """Everything the parser learned, whether or not it is ingestable."""
    format: str = "unknown"                       # json | csv
    cameras: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[RowError] = field(default_factory=list)
    unknown_headers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def rows_parsed(self) -> int:
        return len(self.cameras) + len(self.events)

    def summary(self) -> dict[str, Any]:
        stamps = [e["timestamp"] for e in self.events if e.get("timestamp")]
        plates = {e["plate"] for e in self.events if e.get("plate")}
        cam_ids = {c["camera_id"] for c in self.cameras}
        cam_ids |= {e["camera_id"] for e in self.events if e.get("camera_id")}
        attr_only = [e for e in self.events if not e.get("plate")]
        return {
            "format": self.format,
            "cameras": len(self.cameras),
            "events": len(self.events),
            "unique_camera_ids": len(cam_ids),
            "unique_plates": len(plates),
            # Split out rather than folded into the event count: an operator
            # needs to know how much of an upload is trackable across cameras
            # and how much is volume-only.
            "plate_identified_events": len(self.events) - len(attr_only),
            "attribute_only_events": len(attr_only),
            "attribute_only_with_partial": sum(
                1 for e in attr_only if e.get("plate_partial")
            ),
            "event_time_range": (
                [min(stamps).isoformat(), max(stamps).isoformat()] if stamps else None
            ),
            "rows_parsed": self.rows_parsed,
            "error_count": len(self.errors),
            "errors": [e.as_dict() for e in self.errors[:MAX_ERRORS]],
            "errors_truncated": max(0, len(self.errors) - MAX_ERRORS),
            "unknown_headers": self.unknown_headers,
            "warnings": self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Coercion helpers
# ─────────────────────────────────────────────────────────────────────────────

def _as_float(value: Any) -> float | None:
    """None for blank/absent; raises ValueError for genuinely bad input.

    CSV gives every cell as a string and an empty cell means "not provided",
    not "zero" — collapsing those two is how a dataset ends up with a city of
    vehicles apparently stopped at 0 km/h.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected a number, got a boolean")
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            raise ValueError("not a finite number")
        return f
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none", "nan", "-"}:
        return None
    return float(text)  # ValueError propagates with a useful message


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_plate(value: Any) -> str | None:
    text = _as_text(value)
    if text is None:
        return None
    return _PLATE_STRIP.sub("", text).upper() or None


def parse_timestamp(value: Any) -> datetime:
    """ISO 8601 in, naive-UTC datetime out (what the ORM column stores).

    Accepts a trailing `Z`, a space separator, and epoch seconds/milliseconds —
    all three turn up in real exports. An offset-aware value is converted to
    UTC; a naive one is *read* as UTC rather than as local time, because the
    server's timezone is not a property of the uploader's data.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        text = _as_text(value)
        if text is None:
            raise ValueError("timestamp is required")
        if re.fullmatch(r"-?\d{9,14}(\.\d+)?", text):
            epoch = float(text)
            if epoch > 1e11:  # milliseconds
                epoch /= 1000.0
            dt = datetime.fromtimestamp(epoch, timezone.utc)
        else:
            cleaned = text.replace("Z", "+00:00").replace("z", "+00:00")
            if " " in cleaned and "T" not in cleaned:
                cleaned = cleaned.replace(" ", "T", 1)
            dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ─────────────────────────────────────────────────────────────────────────────
# Row validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_camera(raw: dict[str, Any], row: int, errors: list[RowError]) -> dict[str, Any] | None:
    cid = _as_text(raw.get("camera_id"))
    if not cid:
        errors.append(RowError("cameras", row, "camera_id is required"))
        return None

    try:
        lat = _as_float(raw.get("latitude"))
        lon = _as_float(raw.get("longitude"))
    except ValueError as exc:
        errors.append(RowError("cameras", row, f"latitude/longitude must be numbers ({exc})"))
        return None
    if lat is None or lon is None:
        errors.append(RowError("cameras", row, "latitude and longitude are required"))
        return None
    if not (-90.0 <= lat <= 90.0):
        errors.append(RowError("cameras", row, f"latitude {lat} is outside -90..90"))
        return None
    if not (-180.0 <= lon <= 180.0):
        errors.append(RowError("cameras", row, f"longitude {lon} is outside -180..180"))
        return None

    try:
        limit = _as_float(raw.get("speed_limit_kmh"))
    except ValueError as exc:
        errors.append(RowError("cameras", row, f"speed_limit_kmh must be a number ({exc})"))
        return None
    if limit is not None and not (0 < limit <= 200):
        errors.append(RowError("cameras", row, f"speed_limit_kmh {limit} is outside 1..200"))
        return None

    direction = _as_text(raw.get("direction"))
    return {
        "camera_id": cid,
        "name": _as_text(raw.get("name")) or cid,
        "location": _as_text(raw.get("location")) or "Uploaded dataset",
        "latitude": lat,
        "longitude": lon,
        "road": _as_text(raw.get("road")),
        "direction": direction.upper() if direction else None,
        "camera_type": _as_text(raw.get("camera_type")) or "ANPR",
        "speed_limit_kmh": limit if limit is not None else 60.0,
    }


def _validate_event(raw: dict[str, Any], row: int, errors: list[RowError]) -> dict[str, Any] | None:
    cid = _as_text(raw.get("camera_id"))
    if not cid:
        errors.append(RowError("events", row, "camera_id is required"))
        return None

    try:
        ts = parse_timestamp(raw.get("timestamp"))
    except (ValueError, TypeError) as exc:
        errors.append(RowError("events", row, f"timestamp is not ISO 8601: {exc}"))
        return None

    # ── Identity: a plate, or attributes ──
    # A camera that could not read the plate has still seen a vehicle, and
    # discarding that sighting loses the only record it passed. So `plate` is
    # optional, provided the row carries something else to describe what went
    # by. What is *not* accepted is a row with neither.
    plate = normalize_plate(raw.get("plate"))
    partial = _as_text(raw.get("plate_partial"))
    vtype = _as_text(raw.get("vehicle_type"))
    vcolor = _as_text(raw.get("vehicle_color"))
    vmake = _as_text(raw.get("vehicle_make"))
    vmodel = _as_text(raw.get("vehicle_model"))

    if not any((plate, partial, vtype, vcolor, vmake, vmodel)):
        errors.append(RowError(
            "events", row,
            "no identity: give a `plate`, or — if the plate was not read — at least "
            "one of " + ", ".join(f for f in _IDENTITY_FIELDS if f != "plate"),
        ))
        return None

    # A partial that is actually complete is a plate. Accepting it under the
    # partial column would hide the sighting from every plate query.
    if partial and not plate and "?" not in partial and "*" not in partial:
        maybe = normalize_plate(partial)
        if maybe and len(maybe) >= 8:
            plate, partial = maybe, None

    try:
        conf = _as_float(raw.get("plate_confidence"))
        attr_conf = _as_float(raw.get("attribute_confidence"))
        speed = _as_float(raw.get("speed"))
        lat = _as_float(raw.get("latitude"))
        lon = _as_float(raw.get("longitude"))
    except ValueError as exc:
        errors.append(RowError("events", row, f"numeric field is not a number ({exc})"))
        return None

    for label, value in (("plate_confidence", conf), ("attribute_confidence", attr_conf)):
        if value is not None and not (0.0 <= value <= 1.0):
            errors.append(RowError(
                "events", row,
                f"{label} {value} is outside 0..1 (it is a fraction, not a percentage)",
            ))
            return None
    if conf is not None and plate is None:
        errors.append(RowError(
            "events", row,
            "plate_confidence was given but no plate was read — use "
            "attribute_confidence for a plate-less sighting",
        ))
        return None
    if speed is not None and not (0.0 <= speed <= 300.0):
        errors.append(RowError("events", row, f"speed {speed} km/h is outside 0..300"))
        return None

    direction = _as_text(raw.get("direction"))
    return {
        "camera_id": cid,
        "timestamp": ts,
        "plate": plate,
        "plate_confidence": conf,
        "plate_partial": partial.upper() if partial else None,
        "plate_raw": _as_text(raw.get("plate_raw")),
        "vehicle_type": vtype.lower() if vtype else None,
        "vehicle_color": vcolor.lower() if vcolor else None,
        "vehicle_make": vmake.title() if vmake else None,
        "vehicle_model": vmodel.title() if vmodel else None,
        "attribute_confidence": attr_conf,
        "speed": speed,
        "direction": direction.upper() if direction else None,
        "local_track_id": _as_text(raw.get("local_track_id")),
        "latitude": lat,
        "longitude": lon,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _classify_records(records: Iterable[dict[str, Any]]) -> str:
    """Guess the artefact of a bare array from the keys actually present."""
    keys: set[str] = set()
    for i, rec in enumerate(records):
        if isinstance(rec, dict):
            keys |= set(rec.keys())
        if i >= 50:
            break
    if keys & _EVENT_MARKERS:
        return "events"
    if keys & _CAMERA_MARKERS:
        return "cameras"
    return "events"


def _parse_rows(rows: list[Any], artefact: str, out: ParsedDataset) -> None:
    validate = _validate_camera if artefact == "cameras" else _validate_event
    target = out.cameras if artefact == "cameras" else out.events
    cap = MAX_CAMERAS if artefact == "cameras" else MAX_EVENTS

    for i, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            out.errors.append(RowError(artefact, i, "row is not an object"))
            continue
        if len(target) >= cap:
            out.warnings.append(
                f"{artefact}: stopped at the {cap:,}-row limit; "
                f"{len(rows) - i + 1:,} row(s) were not read"
            )
            break
        clean = validate(raw, i, out.errors)
        if clean is not None:
            target.append(clean)


def parse_dataset(data: bytes, filename: str | None = None, kind: str = "auto") -> ParsedDataset:
    """Bytes from an upload -> a ParsedDataset. Never raises on bad *content*.

    Raises ValueError only when the bytes are not a dataset at all (unreadable
    encoding, malformed JSON, headerless CSV) — the cases where there is no row
    to attach an error to.
    """
    out = ParsedDataset()

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File is not UTF-8 text ({exc.reason} at byte {exc.start}). "
            "Export as UTF-8 JSON or CSV."
        ) from exc

    stripped = text.lstrip()
    looks_json = stripped.startswith("{") or stripped.startswith("[")
    ext = (filename or "").lower().rsplit(".", 1)[-1] if filename and "." in filename else ""

    if looks_json or ext == "json":
        out.format = "json"
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc

        if isinstance(doc, dict):
            if "cameras" not in doc and "events" not in doc:
                raise ValueError(
                    'JSON object has neither a "cameras" nor an "events" key. '
                    'Send {"cameras": [...], "events": [...]}, or a bare array.'
                )
            for artefact in ("cameras", "events"):
                rows = doc.get(artefact)
                if rows is None:
                    continue
                if not isinstance(rows, list):
                    raise ValueError(f'"{artefact}" must be an array, got {type(rows).__name__}')
                _parse_rows(rows, artefact, out)
        elif isinstance(doc, list):
            artefact = kind if kind in ("cameras", "events") else _classify_records(doc)
            _parse_rows(doc, artefact, out)
        else:
            raise ValueError(f"Top-level JSON must be an object or an array, got {type(doc).__name__}")
        return out

    # ── CSV ──
    out.format = "csv"
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # a single-column file sniffs as an error; comma is fine
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise ValueError("CSV has no header row. The first line must name the columns.")

    headers = [(h or "").strip() for h in reader.fieldnames]
    header_set = {h.lower() for h in headers}

    if kind in ("cameras", "events"):
        artefact = kind
    elif header_set & {m.lower() for m in _EVENT_MARKERS}:
        artefact = "events"
    elif header_set & {m.lower() for m in _CAMERA_MARKERS}:
        artefact = "cameras"
    else:
        raise ValueError(
            "Could not tell whether this CSV holds cameras or events from its headers "
            f"({', '.join(headers[:10])}). Set the artefact type explicitly."
        )

    known = {c.lower() for c in (CAMERA_COLUMNS if artefact == "cameras" else EVENT_COLUMNS)}
    out.unknown_headers = [h for h in headers if h and h.lower() not in known]

    rows: list[dict[str, Any]] = []
    for raw in reader:
        rows.append({(k or "").strip().lower(): v for k, v in raw.items() if k})
    _parse_rows(rows, artefact, out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cross-artefact checks
# ─────────────────────────────────────────────────────────────────────────────

def check_references(parsed: ParsedDataset, known_camera_ids: set[str]) -> list[str]:
    """Report events pointing at cameras nobody has defined.

    A warning, not an error: an upload of events alone against an already
    registered network is a legitimate and common case. What is *not*
    legitimate is silently dropping them, which is what the ingest path would
    otherwise do (`process_event_payload` refuses unknown cameras).
    """
    defined = {c["camera_id"] for c in parsed.cameras} | known_camera_ids
    orphans = sorted({e["camera_id"] for e in parsed.events if e["camera_id"] not in defined})
    if not orphans:
        return []
    shown = ", ".join(orphans[:8])
    more = f" (+{len(orphans) - 8} more)" if len(orphans) > 8 else ""
    return [
        f"{len(orphans)} camera_id(s) referenced by events are not defined in this "
        f"upload and are not registered: {shown}{more}. Those events will be rejected "
        f"at ingest — add them to `cameras`, or upload the camera file first."
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _set_busy_timeout(session, milliseconds: int = 15_000) -> None:
    """Wait, briefly, for a competing SQLite writer instead of failing instantly.

    SQLite allows one writer. The simulation clock takes a short write
    transaction every half second, so an ingest that lands on top of one would
    otherwise raise 'database is locked' immediately. `busy_timeout` makes it
    retry for up to 15s, which is far longer than any tick.

    The bound matters as much as the retry: with no timeout at all the ingest
    waits indefinitely, which is how a stuck upload turns into a hung request
    that never answers.
    """
    from sqlalchemy import text

    try:
        session.execute(text(f"PRAGMA busy_timeout={int(milliseconds)}"))
    except Exception:
        # Not SQLite (Postgres has no such pragma) — irrelevant there, since
        # concurrent writers are the normal case.
        pass


def _upsert_cameras(session, cameras: list[dict[str, Any]], deployment: str) -> tuple[int, int]:
    """Insert new cameras, update existing ones. Returns (inserted, updated)."""
    from backend.models.camera import Camera

    if not cameras:
        return 0, 0

    # Last definition of a camera_id wins. Deduplicating before touching the
    # session matters: two rows with the same id would otherwise be two
    # session.add()s of the same primary key and an IntegrityError on flush,
    # losing the whole upload over a copy-pasted line.
    deduped: dict[str, dict[str, Any]] = {}
    for spec in cameras:
        deduped[spec["camera_id"]] = spec

    ids = list(deduped)
    existing: dict[str, Any] = {}
    # Chunked so a 20k-camera upload does not build one enormous IN (...).
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        for cam in session.query(Camera).filter(Camera.camera_id.in_(chunk)).all():
            existing[cam.camera_id] = cam

    inserted = updated = 0
    for cid, spec in deduped.items():
        cam = existing.get(cid)
        if cam is None:
            session.add(Camera(**spec, deployment=deployment, is_active=True))
            inserted += 1
        else:
            for key, value in spec.items():
                setattr(cam, key, value)
            cam.is_active = True
            updated += 1
    session.commit()
    return inserted, updated


def _insert_events(session, events: list[dict[str, Any]], camera_index: dict[str, Any],
                   batch_size: int = 2000) -> tuple[int, int, list[str]]:
    """Bulk-insert events. Returns (inserted, skipped, reasons).

    Bulk rather than `process_event_payload` per row: a city dataset is tens of
    thousands of rows and the per-row path runs tracking + alert evaluation,
    which at that volume turns a two-second ingest into a two-minute one. The
    trade-off is explicit — `global_vehicle_id` is derived here from the plate
    (which is exactly what tracking_service does for plate-identified vehicles)
    and blacklist alerts are evaluated in one pass afterwards instead of per
    row.
    """
    from backend.models.vehicle_event import VehicleEvent

    inserted = skipped = 0
    reasons: list[str] = []
    seen_reasons: set[str] = set()
    batch: list[dict[str, Any]] = []

    def note(msg: str) -> None:
        if msg not in seen_reasons and len(reasons) < MAX_ERRORS:
            seen_reasons.add(msg)
            reasons.append(msg)

    for ev in events:
        cam = camera_index.get(ev["camera_id"])
        if cam is None:
            skipped += 1
            note(f"unknown camera_id {ev['camera_id']} — not defined in this upload or registered")
            continue

        batch.append({
            "event_id": str(uuid.uuid4()),
            "camera_id": ev["camera_id"],
            "local_track_id": ev.get("local_track_id"),
            "timestamp": ev["timestamp"],
            "plate": ev["plate"],
            "plate_confidence": ev.get("plate_confidence"),
            "latitude": ev.get("latitude") if ev.get("latitude") is not None else cam["latitude"],
            "longitude": ev.get("longitude") if ev.get("longitude") is not None else cam["longitude"],
            "direction": ev.get("direction") or cam.get("direction"),
            "vehicle_type": ev.get("vehicle_type") or "car",
            "vehicle_color": ev.get("vehicle_color"),
            "vehicle_make": ev.get("vehicle_make"),
            "vehicle_model": ev.get("vehicle_model"),
            "plate_partial": ev.get("plate_partial"),
            "plate_raw": ev.get("plate_raw"),
            "attribute_confidence": ev.get("attribute_confidence"),
            "speed": ev.get("speed"),
            # Plate-identified vehicles get their global id from the plate —
            # the same rule tracking_service applies.
            #
            # Attribute-only sightings get NO global id. The tempting shortcut
            # is to hash the attributes into one, but "silver hatchback" is
            # tens of thousands of vehicles a day: that id would merge them all
            # into a single entity whose trajectory crosses the city at
            # impossible speeds and whose next-hop prediction is noise. NULL is
            # the honest answer — the sighting counts toward volume, speed and
            # heatmap, and a ReID pass can claim it later.
            "global_vehicle_id": (f"VEH_{ev['plate']}" if ev.get("plate") else None),
            "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        })

        if len(batch) >= batch_size:
            session.bulk_insert_mappings(VehicleEvent, batch)
            session.commit()
            inserted += len(batch)
            batch.clear()

    if batch:
        session.bulk_insert_mappings(VehicleEvent, batch)
        session.commit()
        inserted += len(batch)

    return inserted, skipped, reasons


def _evaluate_blacklist(session, plates: set[str]) -> int:
    """One pass over the uploaded plates against the blacklist.

    Cheaper than per-event alert evaluation and produces the same alerts an
    operator cares about: "a watchlisted vehicle appears in this dataset".
    """
    try:
        from backend.models.alert import Alert, Blacklist
    except ImportError:
        return 0

    try:
        watched = {
            b.plate for b in session.query(Blacklist).all()
            if getattr(b, "plate", None)
        }
    except Exception:
        return 0

    hits = plates & watched
    if not hits:
        return 0

    from backend.models.vehicle_event import VehicleEvent

    fired = 0
    for plate in sorted(hits):
        latest = (
            session.query(VehicleEvent)
            .filter(VehicleEvent.plate == plate)
            .order_by(VehicleEvent.timestamp.desc())
            .first()
        )
        if latest is None:
            continue
        try:
            session.add(Alert(
                vehicle_id=plate,
                camera_id=latest.camera_id,
                alert_type="BLACKLIST",
                description=(
                    f"Watchlisted plate {plate} present in an uploaded city dataset "
                    f"(most recent sighting at {latest.camera_id})"
                ),
                timestamp=latest.timestamp,
            ))
            fired += 1
        except Exception:
            session.rollback()
            break
    if fired:
        session.commit()
    return fired


def ingest(parsed: ParsedDataset, sandbox: bool = True,
           deployment: str | None = None) -> dict[str, Any]:
    """Write a validated dataset into the sandbox (default) or the live city.

    Returns the shape the sandbox page renders: camera and event counts, the
    skip reasons, and — for a sandbox run — the path of the isolated database
    file, which is the page's proof that the live city was not touched.
    """
    if parsed.errors:
        raise ValueError(
            f"Refusing to ingest a dataset with {len(parsed.errors)} row error(s). "
            "Fix the rows listed in `errors` and resubmit."
        )

    upload_id = uuid.uuid4().hex[:12]
    tag = deployment or (f"upload_{upload_id}" if sandbox else "uploaded")

    sandbox_db = None
    if sandbox:
        from backend.services.video_job_service import _SandboxDB

        sandbox_db = _SandboxDB(f"dataset_{upload_id}")
        session_factory = sandbox_db.SessionLocal
        known_ids: set[str] = set()
    else:
        from backend.database import SessionLocal

        session_factory = SessionLocal
        known_ids = set()

    session = session_factory()
    try:
        _set_busy_timeout(session)
        from backend.models.camera import Camera

        # Cameras already registered in the target database — events may
        # legitimately reference these without redefining them.
        referenced = sorted({e["camera_id"] for e in parsed.events})
        camera_index: dict[str, dict[str, Any]] = {}
        for start in range(0, len(referenced), 500):
            chunk = referenced[start:start + 500]
            for cam in session.query(Camera).filter(Camera.camera_id.in_(chunk)).all():
                known_ids.add(cam.camera_id)
                camera_index[cam.camera_id] = {
                    "latitude": cam.latitude,
                    "longitude": cam.longitude,
                    "direction": cam.direction,
                }

        cams_inserted, cams_updated = _upsert_cameras(session, parsed.cameras, tag)
        for spec in parsed.cameras:
            camera_index[spec["camera_id"]] = {
                "latitude": spec["latitude"],
                "longitude": spec["longitude"],
                "direction": spec.get("direction"),
            }

        events_inserted, events_skipped, reasons = _insert_events(
            session, parsed.events, camera_index
        )
        alerts_fired = _evaluate_blacklist(
            session, {e["plate"] for e in parsed.events if e.get("plate")}
        )

        return {
            "ok": True,
            "upload_id": upload_id,
            "sandbox": sandbox,
            "deployment": tag,
            "database": (str(sandbox_db.path) if sandbox_db else "live"),
            "cameras_inserted": cams_inserted,
            "cameras_updated": cams_updated,
            "events_inserted": events_inserted,
            "events_skipped": events_skipped,
            "alerts_fired": alerts_fired,
            "skip_reasons": reasons,
            "summary": parsed.summary(),
        }
    finally:
        session.close()
        if sandbox_db is not None:
            sandbox_db.dispose()


def sandbox_dataset_events(upload_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Read back what a sandboxed dataset upload wrote, from its own file."""
    from backend.services.video_job_service import sandbox_events

    return sandbox_events(f"dataset_{upload_id}", limit=limit)

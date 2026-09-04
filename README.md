# Godseye — Citywide Vehicle Intelligence Platform

Godseye turns a network of roadside cameras into a single, queryable picture
of every vehicle moving through a city: who it is, where it has been, where
it is likely to go next, where the city is congested, and who is breaking
the law.

A YOLO-based plate detector plus OCR reads plates off camera frames, a
multi-camera tracker stitches sightings of the same vehicle across different
cameras (MTMC — Multi-Target Multi-Camera), and a FastAPI backend turns each
confirmed sighting into a `VehicleEvent` row. From that single table the
platform reconstructs trajectories snapped to real roads, predicts next-hop
movement with explicit statistical confidence, aggregates city-wide
congestion analytics, and raises real-time enforcement alerts.

---

## Table of contents

- [Architecture](#architecture)
- [Repository structure](#repository-structure)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Running everything with one command](#running-everything-with-one-command)
- [Services in detail](#services-in-detail)
- [Data model](#data-model)
- [Documentation](#documentation)
- [Local data (not tracked in git)](#local-data-not-tracked-in-git)
- [Credits & third-party code](#credits--third-party-code)
- [Roadmap / status](#roadmap--status)

---

## Architecture

```
 ┌── EDGE ────────────────┐    ┌── INGEST ──────────┐    ┌── CORE ─────────────────┐
 │ Camera / video file    │    │ POST /events/ingest│    │ tracking_service        │
 │   ↓ YOLOv8 plate det.  │───▶│ POST /events/bulk- │───▶│  → global_vehicle_id    │
 │   ↓ ByteTrack tracks   │    │      ingest        │    │ alert_service           │
 │   ↓ PaddleOCR + vote   │    │ Kafka consumer      │    │  → BLACKLIST / SPEED /  │
 │   ↓ plate grammar      │    │ POST /jobs/video    │    │    ROUTE_ANOMALY        │
 └────────────────────────┘    │ POST /jobs/dataset  │    └───────────┬─────────────┘
                                └────────────────────┘                │
                                                                       ▼
                                              ┌──────────── vehicle_events ────────────┐
                                              │  the single source of truth            │
                                              └───┬───────────┬───────────┬────────────┘
                                                  │           │           │
                      ┌───────────────────────────┘           │           └──────────────┐
                      ▼                                       ▼                          ▼
          routing_service (OSRM)              analytics_agg rollups          prediction / simulation
          road-snapped trajectories           heatmap, OD, congestion        Markov + confidence
                      │                                       │                          │
                      └───────────────┬───────────────────────┴──────────────────────────┘
                                      ▼
                      FastAPI  →  /app (dashboard) · /app/live (separate window)
                                  /app/test (sandbox) · /app/benchmarks · /docs
```

**Pipeline in one paragraph:** cameras produce frames → `services/alpr`
detects and reads plates (YOLOv8 + PaddleOCR + cross-frame voting) →
`services/vehicle-mtmc` and `services/traffic-detection-yolo` handle
multi-camera re-identification and general vehicle/traffic detection → every
confirmed sighting is ingested by `services/backend` as a `VehicleEvent` →
the backend derives trajectories (OSRM road-snapped), next-hop predictions
(Markov model with confidence bounds), congestion analytics, and alerts, and
serves it all through a FastAPI + vanilla-JS/Leaflet dashboard.

---

## Repository structure

```
SIH/
├── services/
│   ├── alpr/                     ANPR pipeline — YOLOv8 plate detection + PaddleOCR
│   ├── vehicle-mtmc/             Multi-camera vehicle re-identification & tracking
│   ├── traffic-detection-yolo/   General YOLO-based traffic/vehicle detection
│   └── backend/                  FastAPI backend — ingestion, trajectories,
│                                  prediction, analytics, alerts, dashboard/frontend
├── docs/                         Architecture notes, problem statement, workflow, integration guides
├── docker-compose.yml            One command to run the whole stack (see below)
├── .dockerignore
├── .gitignore
└── README.md                     You are here
```

Everything under `services/` is vendored directly into this repository —
there are no git submodules and no external checkouts required. `alpr`,
`vehicle-mtmc`, and `traffic-detection-yolo` originate from third-party
open-source projects (see [Credits & third-party
code](#credits--third-party-code)); `backend` is this project's own code,
written specifically to connect them into one pipeline. Vendoring them
in-tree keeps `git clone` + `docker compose up` sufficient to deploy the
whole platform, with no submodule init step and no dependency on those
upstream repos staying available.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI** + Uvicorn | Async WebSockets for the live stream; OpenAPI docs come free. |
| ORM / DB | **SQLAlchemy 2.0** + SQLite (WAL) for dev, **PostgreSQL + PostGIS** for prod | Swappable via `DATABASE_URL`; Alembic migrations in `services/backend/alembic/`. |
| Plate detection | **YOLOv8s** (Ultralytics) | `mAP@50 ≥ 0.85` gate; `s` variant chosen over `n` because plates are small objects. |
| OCR | **PaddleOCR** (via the `alpr` package) | Outperforms EasyOCR on Indian plates in-house testing. |
| Tracking | **ByteTrack** + cross-frame plate voting | A single frame's OCR is unreliable; voting across a track is what makes reads usable. |
| Road routing | **OSRM** over OpenStreetMap | Vehicles drive on roads, not straight lines — detour ratio 1.15–1.6× observed in Delhi. |
| Geo / camera graph | **Overpass API** (real OSM junctions) | Real junction graphs, not synthetic grids. |
| Streaming | **Kafka** (optional) + **Redis** pub/sub + WebSockets | All optional — the API degrades gracefully rather than failing without them. |
| Prediction | First-order **Markov transition matrix** + Wilson confidence bounds | Zero external ML deps, serves in microseconds, expresses its own uncertainty. |
| Frontend | Vanilla JS + **Leaflet** + `leaflet.heat` | No build step — static HTML served directly by the API. |

---

## Getting started

### Clone

```bash
git clone <this-repo-url>
cd TRINETRA
```

That's it — no submodule init step. Everything the app needs is already in
the tree.

### Prerequisites

- **Docker + Docker Compose** — the one-command path, recommended
- Or, for a native dev loop: Python 3.11+ (3.13 tested), `pip`, `venv`

---

## Running everything with one command

`docker-compose.yml` lives at the **repo root** and is the single entrypoint
for the whole platform: Postgres/PostGIS, Redis, Kafka + Zookeeper, the Kafka
consumer, and the backend. The backend container bundles `services/alpr` as
a real installed dependency (not a stub) and serves the frontend itself as
static pages — there is no separate frontend container to run.

```bash
# from the repo root
docker compose up --build
```

First build downloads/installs torch, ultralytics and paddleocr, so expect it
to take a while. Once it's up:

| URL | What |
|---|---|
| `http://localhost:8000/docs` | Interactive OpenAPI docs |
| `http://localhost:8000/app` | Main dashboard |
| `http://localhost:8000/app/live` | Live ingestion window (open as a second window) |
| `http://localhost:8000/app/test` | Sandbox — upload a video/photo/dataset and watch the pipeline run |
| `http://localhost:8000/app/benchmarks` | Measured throughput & scalability projections |

Seed demo cameras on first run (in a separate terminal, once the stack is up):

```bash
docker compose exec backend python scripts/seed_cameras.py
```

Stop everything with `docker compose down` (add `-v` to also drop the
Postgres volume).

### Running the backend natively instead (no Docker)

```bash
cd services/backend

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Install the ANPR service (with the OCR extra) as an editable local package
pip install -e "../alpr[ocr]"

# 4. Copy and edit the .env
cp .env.example .env
# ANPR_WEIGHTS_PATH already defaults to ../alpr/best.pt — override only if needed

# 5. Start the API (uses SQLite by default, no Postgres/Redis/Kafka needed)
uvicorn backend.main:app --reload

# 6. (first run only) seed demo cameras
python scripts/seed_cameras.py
```

Same URLs as above apply.

---

## Services in detail

### [`services/alpr`](services/alpr) — Automatic License Plate Recognition
YOLOv8 plate detector + PaddleOCR reader + ByteTrack-based per-track plate
voting + Indian plate grammar normalization. Ships `best.pt` weights,
CLI (`alpr` package), and its own test suite. See its own
[README](services/alpr/README.md) and [ROADMAP](services/alpr/ROADMAP.md).

### [`services/vehicle-mtmc`](services/vehicle-mtmc) — Multi-Target Multi-Camera tracking
Re-identifies the same vehicle across non-overlapping camera views using
appearance embeddings + spatio-temporal constraints, so a single vehicle's
sightings from different cameras can be merged into one trajectory.

### [`services/traffic-detection-yolo`](services/traffic-detection-yolo) — Traffic/vehicle detection
General-purpose YOLO-based vehicle/traffic object detection, used as an
additional detection source and for benchmarking against the ALPR pipeline's
own detector.

### [`services/backend`](services/backend) — FastAPI backend
The system of record. Ingests `VehicleEvent` rows from the detection
pipelines and exposes:
- `POST /events/ingest`, `/events/bulk-ingest` — production ingestion
- `POST /jobs/video`, `/jobs/dataset` — sandboxed video/dataset processing
- `GET /vehicles/{plate}/trajectory`, `GET /routing/trajectory/{plate}` — raw and road-snapped trajectories
- Prediction, analytics, alerts, benchmarks APIs
- The dashboard, live view, and sandbox frontends

See its own [README](services/backend/README.md) for the full API reference.

---

## Data model

`vehicle_events` is the single contract every producer targets:

| Field | Notes |
|---|---|
| `event_id`, `camera_id`, `local_track_id`, `timestamp` | Sighting identity |
| `plate`, `plate_confidence` | Normalized at ingest: uppercase, no spaces/hyphens |
| `latitude`, `longitude`, `direction` | Defaults to the camera's own position |
| `vehicle_type`, `vehicle_color`, `speed` | Attributes |
| `vehicle_make`, `vehicle_model`, `plate_partial`, `plate_raw`, `attribute_confidence` | Attribute-only identity (see below) |
| `global_vehicle_id` | MTMC identity: `VEH_<plate>` for plate reads, `NULL` for attribute-only |

**Attribute-only sightings.** A camera doesn't always get the plate (speed,
occlusion, angle). Rather than discard those, a sighting is accepted with a
plate *or* at least one attribute (`plate_partial` / `vehicle_type` /
`vehicle_color` / `vehicle_make` / `vehicle_model`). These count toward
volume/speed/heatmap analytics and are searchable by partial plate, but get
no `global_vehicle_id` — hashing attributes into an identity would wrongly
merge thousands of visually-similar vehicles into one impossible entity.

Other tables: `cameras`, `alerts`, `blacklist`, `trajectories`,
`route_segments` (OSRM cache), `future_events` (staged for simulation), and
rollups `road_usage` / `camera_hourly` / `camera_totals` / `dataset_kpi`.

---

## Documentation

| Doc | What's in it |
|---|---|
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, priority map, how every capability actually works |
| [docs/WorkFlow.md](docs/WorkFlow.md) | Team workflow / process |
| [docs/ALPR_Integration_Guide.md](docs/ALPR_Integration_Guide.md) | How the ANPR service plugs into the backend |
| [docs/Problem Statement Details.md](docs/Problem%20Statement%20Details.md) | Original problem statement |
| [docs/Open Source Projects and how to build this bitch.md](docs/Open%20Source%20Projects%20and%20how%20to%20build%20this%20bitch.md) | Survey of prior art / reference projects |

---

## Local data (not tracked in git)

These stay on disk but are `.gitignore`d — regenerate or re-download per the
relevant service's README as needed:

- `vehicle_models/`, `vehicle_models.zip` — pretrained ReID/classification model weights
- `archive/` — sample/demo video clips used for benchmarking and sandbox testing
- `services/backend/dev.db*`, `sandbox_dbs/`, `uploads/`, `job_outputs/` — local runtime data

---

## Credits & third-party code

`services/backend` is original code written for this project. The other
three services under `services/` are vendored copies of third-party
open-source projects, included directly in this repository (rather than as
git submodules) so the whole platform clones and deploys as one self-contained
unit. All credit for the original work in these directories goes to their
authors:

| Directory | Original project | Author | License |
|---|---|---|---|
| [`services/alpr`](services/alpr) | [Automatic-License-Plate-Recognition](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition) | Fayaz Hussain Syed | [MIT](services/alpr/LICENSE) |
| [`services/vehicle-mtmc`](services/vehicle-mtmc) | [vehicle_mtmc](https://github.com/regob/vehicle_mtmc) | Regő Borsodi | [MIT](services/vehicle-mtmc/LICENSE.md) |
| [`services/traffic-detection-yolo`](services/traffic-detection-yolo) | [traffic-detection-yolo](https://github.com/abrarCSE29/traffic-detection-yolo) | abrarCSE29 | See upstream repo |

Each directory retains its own original `README.md`/license file — see them
for the full license text and any additional attribution the original
authors require. If you redistribute this repository, keep those license
files intact alongside the code they cover.

---

## Roadmap / status

See [services/alpr/ROADMAP.md](services/alpr/ROADMAP.md) for the ANPR
service roadmap and [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) §2
"Priority map" for what's working (P0/P1) vs. planning-stage (P2/P3) across
the whole platform.

# Vehicle Intelligence Platform — Project Context

City-scale ANPR (Automatic Number Plate Recognition) and MTMC (Multi-Target
Multi-Camera) vehicle intelligence. Cameras see vehicles; the platform turns
those sightings into identity, trajectory, prediction and enforcement.

> **Read this first if you are new to the repo.** Everything below is what the
> code actually does today, not a roadmap. Where something is a limitation it
> says so.

---

## 1. The one-paragraph version

A network of roadside cameras produces frames. A YOLO plate detector plus
PaddleOCR reads plates off them; a tracker groups a camera's frames into
per-vehicle tracks and votes on the plate text across the track. Each confirmed
read becomes a `VehicleEvent` row — plate, camera, timestamp, GPS, speed,
attributes. From that single table the platform reconstructs where a vehicle
went (**trajectory**, snapped to real roads via OSRM), where it is likely to go
next (**Markov transition model with explicit confidence**), where the city is
congested (**heatmap, OD matrix, road usage**), and who is breaking the law
(**checkpoint-pair speed violations, watchlist alerts**). A simulation clock
replays a pre-generated future month one sighting at a time, so the live half of
the product — live feed, live heatmap, live trajectories, scoreable predictions
— runs against real ingestion rather than an animation.

---

## 2. Priority map — what matters most

| Rank | Capability | Why it ranks here | Where |
|---|---|---|---|
| **P0** | Plate → trajectory across cameras | The core claim. Nothing else works without correct multi-camera association. | `api/vehicles.py`, `api/routing.py`, `services/routing_service.py` |
| **P0** | Ingestion contract (`VehicleEvent`) | One schema every producer targets. Change it and everything downstream drifts. | `models/vehicle_event.py`, `schemas/vehicle.py` |
| **P1** | Live ingestion + simulation clock | Turns a static archive into a system you can watch working. | `services/simulation_service.py`, `api/simulation.py`, `frontend/live.html` |
| **P1** | Next-hop prediction **with confidence** | Interception decisions. Confidence is what makes it actionable rather than a guess. | `services/simulation_service.py`, `services/prediction_service.py` |
| **P1** | Alerts (watchlist, route anomaly, speeding) | The enforcement deliverable. | `services/alert_service.py`, `api/alerts.py` |
| **P2** | Analytics (heatmap, OD, congestion, road usage) | Planning value, not operational. | `api/analytics.py`, `models/analytics_agg.py` |
| **P2** | Bring-your-own-city dataset ingestion | Adoption path: any city, no code change. | `services/dataset_service.py`, `api/jobs.py` |
| **P2** | Sandbox (video/photo/dataset) | Public proof the pipeline is real, safely isolated. | `api/jobs.py`, `services/video_job_service.py`, `frontend/test.html` |
| **P3** | Benchmarks & scalability projections | Answers "will this run on the city's hardware". | `api/benchmarks.py`, `services/benchmark_service.py` |

---

## 3. Architecture

```
 ┌── EDGE ────────────────┐    ┌── INGEST ──────────┐    ┌── CORE ─────────────────┐
 │ Camera / video file    │    │ POST /events/ingest│    │ tracking_service        │
 │   ↓ YOLOv8 plate det.  │───▶│ POST /events/bulk- │───▶│  → global_vehicle_id    │
 │   ↓ ByteTrack tracks   │    │      ingest        │    │ alert_service           │
 │   ↓ PaddleOCR + vote   │    │ Kafka consumer     │    │  → BLACKLIST / SPEED /  │
 │   ↓ plate grammar      │    │ POST /jobs/video   │    │    ROUTE_ANOMALY        │
 └────────────────────────┘    │ POST /jobs/dataset │    └───────────┬─────────────┘
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

### The timeline model (what makes the demo live)

```
  [ ──────── 30 days past ──────── | now | ──────── 30 days staged ──────── ]
     vehicle_events (12.5M rows)      ▲      future_events (1.56M rows)
     history · aggregates ·           │      released ONE AT A TIME by the
     transition model fitted here     │      simulation clock as each comes due
                                      │
                            live feed · live heatmap · live trajectory
                            · predictions that can be SCORED, because the
                              answer already exists in a row the predictor
                              has not read
```

That last point is the reason the future is pre-generated rather than invented
on the fly: a prediction made at sim-time *T* about *T+8min* is checkable at
*T+8min*. A feeder that made events up as it went could never be wrong.

---

## 4. Tech stack

| Layer | Choice | Why this one |
|---|---|---|
| API | **FastAPI** + Uvicorn | Async WebSockets for the live stream; OpenAPI docs come free and are the integration contract. |
| ORM / DB | **SQLAlchemy 2.0** + **SQLite** (WAL) | SQLite holds 12.5M events in 6.5 GB and answers indexed trajectory queries in ms. Postgres/TimescaleDB is a URL change (`DATABASE_URL`); Alembic migrations are in `alembic/`. |
| Plate detection | **YOLOv8s** (Ultralytics), `best.pt` | mAP@50 gate ≥0.85. `yolov8s` over `n` because plates are small objects. |
| OCR | **PaddleOCR** via the `alpr` package | Measured better than EasyOCR on Indian plates; already normalises crops internally (see §9). |
| Tracking | ByteTrack (in `alpr.track`) + cross-frame **plate voting** | A single frame's OCR is unreliable; voting across a track is what makes reads usable. |
| Road routing | **OSRM** over OpenStreetMap, cached in `route_segments` | Vehicles drive on roads, not chords. Detour ratio 1.15–1.6× in Delhi — straight-line speeds systematically understate travel. |
| Geo / camera graph | **Overpass API** (real OSM junctions), k-NN adjacency | 200 real Delhi junctions; 41 real Mumbai arterial junctions. |
| Streaming | **Kafka** (optional) + **Redis** pub/sub + WebSockets | Kafka for volume ingest, Redis for cache + fan-out, WS for the browser. All optional — the API degrades rather than fails. |
| Prediction | First-order **Markov transition matrix** + Wilson bounds | Zero external ML deps, serves in µs, and — critically — can express its own uncertainty. |
| Frontend | Vanilla JS + **Leaflet** + `leaflet.heat` | No build step. The whole UI is four static HTML files served by the API. |
| Compute telemetry | `psutil`, per-stage timers | Real measured throughput, not claimed. |

**No framework the demo depends on requires a network connection at run time**
(route geometry is cached; tiles are the only external fetch).

---

## 5. Data model

**`vehicle_events`** — the contract. Everything else is derived.

| Field | Notes |
|---|---|
| `event_id`, `camera_id`, `local_track_id`, `timestamp` | Sighting identity |
| `plate`, `plate_confidence` | Normalised at ingest: uppercase, no spaces/hyphens |
| `latitude`, `longitude`, `direction` | Defaults to the camera's own position |
| `vehicle_type`, `vehicle_color`, `speed` | Attributes |
| `vehicle_make`, `vehicle_model`, `plate_partial`, `plate_raw`, `attribute_confidence` | **Attribute-only identity** — see below |
| `global_vehicle_id` | MTMC identity. `VEH_<plate>` for plate reads, **NULL** for attribute-only |

**Attribute-only sightings.** A camera does not always get the plate — a vehicle
at 90 km/h, a plate occluded by the car in front, a bent bike plate. Discarding
those loses the only record it passed. So a sighting is accepted with a plate
*or* with at least one of `plate_partial` / `vehicle_type` / `vehicle_color` /
`vehicle_make` / `vehicle_model`. Those rows count toward volume, speed and
heatmap analytics and are searchable by partial plate, but they get **no**
`global_vehicle_id`. The tempting shortcut — hashing attributes into an id — is
wrong: "silver hatchback" is tens of thousands of vehicles a day, and that id
would merge them into one entity crossing the city at impossible speeds.

Other tables: `cameras`, `alerts`, `blacklist`, `trajectories`,
`route_segments` (OSRM cache), `future_events` (staged), and the rollups
`road_usage` / `camera_hourly` / `camera_totals` / `dataset_kpi`.

---

## 6. How each capability actually works

### Multi-camera trajectory (P0)
`GET /vehicles/{plate}/trajectory` returns ordered sightings.
`GET /routing/trajectory/{plate}` returns the same journey **snapped to roads**:
per-leg OSRM geometry, direction arrows with bearings, road vs straight-line
distance, and both implied speeds. Legs that fell back to a chord say so
(`is_real_road: false`) rather than pretending.

The dashboard groups a month of sightings into **trips** — a gap over 45 minutes
between consecutive sightings ends one (the p99 camera-to-camera hop is well
under 30 min even in peak). Each trip gets its own colour; hovering any leg or
stop shows the weekday, date, both clock times and the implied speed; a day-strip
and two dropdowns filter by day or by individual trip. Without this, a month of
history is a solid scribble and "where was it on the 14th" is unanswerable.

### Next-hop prediction with confidence (P1)
Probability and confidence are reported **separately**, because they answer
different questions.

```
confidence = P(candidate) × support × decisiveness

  support      n/(n+20) on departures observed from this camera
               — 4 observations cannot support a confident claim
  decisiveness 1 − normalised entropy of the distribution
               — a junction that splits evenly five ways IS unpredictable
```

Also returned: `probability_lower_bound`, the 95% Wilson bound. 3-of-4 and
750-of-1000 are both p=0.75; the bounds are 0.30 and 0.72, and only one is worth
sending a unit to. The model is fitted on the most recent 300k events (one index
range scan) and **updated live** as the clock promotes transitions.

### Alerts (P1)
- **BLACKLIST** — watchlisted plate seen. Fires through the production ingest path.
- **SPEED_VIOLATION** — checkpoint-pair: distance between two cameras ÷ elapsed
  time, against the *stricter of the two cameras' own* posted limits. A ring
  road and a residential lane do not share one number.
- **ROUTE_ANOMALY** — implied speed above `MAX_PLAUSIBLE_SPEED_KMH`. This is a
  *data-quality* alert (bad match or clock skew), and those hops are excluded
  from the speed-defaulter report rather than counted as extreme speeders.

### Live ingestion (P1) — its own window
`/app/live` is a separate page, opened as a real second window from the
dashboard's `[L] LIVE ↗` button. During a demo it belongs on a second screen
beside the dashboard so both are visible at once. It shows the promoted-sighting
feed, a rolling live heatmap, the two-month timeline histogram, and
follow-a-vehicle with live trail + ranked predictions.

The dashboard's own `LIVE HEAT` toggle is deliberately **separate** from
`HEATMAP`: the latter sums 24h out of the database, the former sums only what
has been ingested in the last 15 simulated minutes. Merging them would make a
stalled stream look like healthy traffic.

### Bring-your-own-city (P2)
`GET /jobs/dataset/schema` serves the machine-readable contract — and the
sandbox renders its field table and templates *from that response*, so documented
format cannot drift from enforced format. `POST /jobs/dataset/validate` returns
per-row errors (`artefact`, `row`, `msg`); `POST /jobs/dataset` ingests. A
dataset with any row error is refused **whole** — half a city is worse than none,
because every downstream trajectory would then be silently missing hops.

Accepts JSON (`{cameras, events}`, either key, or a bare array) and CSV (artefact
auto-detected from headers). `sandbox=true` by default: rows land in a throwaway
SQLite file with the production schema, so a stranger's upload exercises the real
pipeline without touching the live city.

### Sandbox (P2)
Upload a video, a photo or a dataset and watch the real pipeline run: live
progress over WebSocket, per-stage compute telemetry, plate table with
provisional vs confirmed reads, and a full stats report (grammar-rejected,
too-few-reads, duplicates suppressed, dropped frames). Isolation is **physical**
— a separate database file per job — and the page shows the file path as proof.

---

## 7. Scalability

**Measured, on this machine (Apple Silicon, MPS):**

| Quantity | Value |
|---|---|
| Events in the archive | **12.54M** across 200 cameras, 30 days |
| Database size | 6.5 GB SQLite |
| Bulk load rate | **~250,000 events/sec** (executemany, indexes dropped and rebuilt) |
| Future staging | 1.56M events in **6.3 s** + 2.0 s index rebuild |
| Simulation promotion | 4,000 rows/tick ceiling, typical tick **~23 ms**, zero backlog at 600× |
| Trajectory query (495 sightings, month window) | milliseconds — indexed range scan |
| Dataset ingest | 31,144 events + 41 cameras in **0.58 s** |
| Transition model fit | 300k events → 200 source cameras in ~0.5 s |

**Design decisions that make it scale, and their reasoning:**

1. **Every hot query is bounded.** Speed-defaulters defaults to 24h, trajectory
   to 60 sightings, heatmap to a window. There is no endpoint whose cost grows
   with the age of the deployment.
2. **Rollups are incremental, not recomputed.** `analytics_agg` is UPSERTed from
   counters the writer keeps, not by re-running GROUP BYs over millions of rows.
   (Known approximation: `unique_vehicles` double-counts a live plate already in
   that hour's history until a full rebuild. Volume, speed and road usage stay
   exact.)
3. **Route geometry is cached, not fetched.** `route_segments` holds the OSRM
   answer per camera pair. The demo works with the network off;
   `GET /routing/cache/stats` reports coverage and a `median_detour_ratio`
   lie-detector (a value near 1.0 means chords are being drawn, not roads).
4. **Models load once per process.** YOLO + PaddleOCR construction is seconds;
   inference is ~25 ms. They are process-wide singletons behind a lock.
5. **Slow consumers cannot back-pressure ingestion.** Every WebSocket subscriber
   has a bounded 64-frame mailbox that drops its oldest frame. A stalled browser
   tab must never slow the writer.
6. **The live view is in-memory and bounded.** The heatmap sums a ring buffer,
   not a GROUP BY over the newest rows of a 12M-row table every four seconds.

**Where it would need work at true city scale (10k+ cameras):**
SQLite is one writer — swap `DATABASE_URL` to Postgres/TimescaleDB with
time-partitioned `vehicle_events`. Camera workers already produce to Kafka, so
horizontal ingest is a consumer-group change, not a rewrite. Edge inference
(TensorRT export exists at `scripts/export_tensorrt.py`) keeps video off the
network entirely — only ~200 bytes of JSON per sighting crosses it.

---

## 8. Feasibility

**What is proven in this repo, today:**
- Real OSM geography (200 Delhi junctions, 41 Mumbai arterial junctions), not invented coordinates.
- 12.5M-event archive that is *physically consistent* — timestamps derived by
  walking a real camera adjacency graph, so implied speeds are plausible. The
  Mumbai sample's implied speeds: median 32.4 km/h, p95 63.6, **max 125.1** —
  the max is the number that matters, because an earlier version produced
  18,771 km/h from two overlapping trips of one vehicle.
- Detection → OCR → voting → database on real footage through the sandbox.
- Ground truth written *before* the run (`deployments/*/ground_truth.json`), so
  a demo ends by comparing findings to a list decided in advance.

**Honest limitations:**
- The 12.5M archive is **synthetic** (generated from a physical model), not real
  captured traffic. Deliberate: real ANPR footage of Indian plates is personal
  data and cannot be redistributed.
- Attribute-only sightings are not re-identified across cameras. Correct, not
  lazy — see §5.
- Prediction is first-order Markov. It captures road structure well and knows
  when it does not know; it does not model destination intent.
- Fast-moving vehicles are the weakest link in OCR. See §9.

**Deployment cost is dominated by cameras, not compute.** `/app/benchmarks`
serves measured per-frame cost and projections to N cameras.

---

## 9. Known weakness: fast-moving vehicles

Motion blur at 80–120 km/h degrades plate reads. Two things worth knowing:

**Preprocessing does *not* fix it.** The obvious move — upscale + CLAHE + denoise
the crop — was ablated on 124 hand-labelled real crops and made accuracy
**worse** (CER 0.2291 raw vs 0.2410 enhanced). PaddleOCR already resizes and
normalises internally; enhancing first resamples twice and destroys detail the
model would have used. The code is kept in `anpr_service.preprocess_plate_crop`
with the measurements in its docstring, *specifically* so nobody re-adds it on
the same wrong intuition.

**What does help** is more evidence per vehicle (cross-frame voting over the
track, higher OCR cadence on fast tracks) and a detector fine-tuned on
motion-blurred data.

---

## 10. Use cases

| # | Use case | Endpoints / surface |
|---|---|---|
| 1 | **Stolen-vehicle interception** — watchlist hit, live trail, ranked next hops with ETA and confidence | `/alerts`, `/simulation/trajectory/{plate}`, `/simulation/predict/{plate}` |
| 2 | **Suspect movement reconstruction** — a month of trips, filterable by day, snapped to roads | `/routing/trajectory/{plate}`, dashboard trajectory tab |
| 3 | **Automated speed enforcement** — checkpoint-pair violations against per-segment limits, with fine bands | `/vehicles/analytics/speed-defaulters` |
| 4 | **Congestion monitoring** — live and historical heatmap, per-camera avg speed, congestion labels | `/analytics/heatmap`, `/simulation/heatmap`, `/analytics/congestion` |
| 5 | **Traffic planning** — origin-destination matrix, busiest road corridors along real geometry | `/analytics/od-matrix`, `/analytics/road-usage` |
| 6 | **Toll / checkpoint audit** — every sighting with confidence and raw OCR retained for dispute | `/vehicles/{plate}`, `plate_raw` |
| 7 | **New-city onboarding** — upload cameras + events, platform comes up on that city | `/jobs/dataset`, sandbox CITY DATASET tab |
| 8 | **Capability proof to a non-technical audience** — upload your own video, watch plates land in a database | `/app/test` |
| 9 | **Capacity planning** — measured compute cost projected to N cameras | `/app/benchmarks` |
| 10 | **Attribute-only BOLO** — "red Pulsar, partial MH05??77??" when no plate was read | `plate_partial` + attribute columns |

---

## 11. Repository map

```
vehicle-intelligence-backend/
├── backend/
│   ├── main.py                  app wiring; optional routers degrade, never crash
│   ├── api/                     auth cameras events vehicles analytics alerts
│   │                            websocket jobs routing benchmarks simulation
│   ├── models/                  camera vehicle_event alert trajectory
│   │                            traffic_snapshot analytics_agg route_segment
│   │                            future_event
│   ├── services/                anpr tracking alert prediction routing
│   │                            video_job dataset simulation kafka redis
│   │                            compute_monitor benchmark
│   └── schemas/                 pydantic contracts
├── frontend/                    index.html (dashboard) · live.html (live window)
│                                test.html (sandbox) · benchmarks.html
├── scripts/                     generate_delhi_dataset · generate_future_dataset
│                                generate_mumbai_sample · live_event_feeder
│                                camera_worker · run_benchmarks · export_tensorrt
│                                migrate_add_attribute_columns · fetch_delhi_junctions
├── deployments/                 delhi/ mumbai/ hamburg/ benchmarks/
└── sandbox_dbs/                 one throwaway SQLite per sandbox job
```

Sibling repos: `Automatic-License-Plate-Recognition/` (the `alpr` package —
detector, OCR, tracker, voting, plate grammar), `traffic-detection-yolo/`,
`vehicle_mtmc/`, `vehicle_models/`.

---

## 12. Running it

```bash
cd vehicle-intelligence-backend

# API + all four pages
.venv/bin/python -m uvicorn backend.main:app --port 8000
#   /app  /app/live  /app/test  /app/benchmarks  /docs

# Stage the next month, then start the clock
.venv/bin/python scripts/generate_future_dataset.py
curl -X POST 'http://localhost:8000/simulation/start?speed=60'

# Regenerate the Mumbai demo dataset (timestamps anchor to "now")
.venv/bin/python scripts/generate_mumbai_sample.py

# Rebuild the whole Delhi archive from scratch
.venv/bin/python scripts/generate_delhi_dataset.py
```

`GET /` reports which optional feature routers loaded — a broken module disables
one feature instead of taking the API down, which matters when the dashboard and
the ingestion pipeline still need to serve during a live demo.

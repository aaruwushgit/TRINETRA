# Godseye — Architecture

Complete reference for the system: what every component is, how data moves end
to end, what the data model is, and where the seams and known weak points are.

Written from the code as it stands at commit `d9401ca`. Where the code and the
older docs (`PROJECT_CONTEXT.md`, `WorkFlow.md`) disagree, this document
follows the code and says so.

**Contents**

1. [One-paragraph summary](#1-one-paragraph-summary)
2. [Component map](#2-component-map)
3. [What is actually wired up](#3-what-is-actually-wired-up)
4. [Layer 1 — Edge: the ALPR pipeline](#4-layer-1--edge-the-alpr-pipeline)
5. [Layer 2 — Ingestion](#5-layer-2--ingestion)
6. [Layer 3 — Identity and alerting](#6-layer-3--identity-and-alerting)
7. [Layer 4 — Derived intelligence](#7-layer-4--derived-intelligence)
8. [Layer 5 — API and frontend](#8-layer-5--api-and-frontend)
9. [Data model](#9-data-model)
10. [End-to-end dataflows](#10-end-to-end-dataflows), traced call by call
11. [Configuration and deployment](#11-configuration-and-deployment)
12. [Concurrency, threading and state](#12-concurrency-threading-and-state)
13. [Known weak points](#13-known-weak-points)

---

## 1. One-paragraph summary

Cameras produce frames. A YOLOv8 detector finds number plates, ByteTrack keeps
each plate as one track across frames, PaddleOCR reads every crop, and a
per-character weighted vote plus a country plate grammar turns many noisy reads
into one confirmed plate. Each confirmed plate becomes one `VehicleEvent` row
in the backend — **the single source of truth for the entire platform**. From
that one table, and nothing else, the backend derives global vehicle identity
across cameras, real-time alerts, road-snapped trajectories via OSRM, next-hop
predictions with statistical confidence bounds, and city-scale congestion
analytics, and serves all of it through FastAPI plus a Next.js frontend with a
3D MapLibre map. A year of that history then feeds a per-vehicle pattern miner
that infers where each vehicle lives and works and flags days it broke its own
routine.

The architectural bet: **one wide, denormalised event table, and everything
else derived from it.** Section 13 covers what that costs.

---

## 2. Component map

```
repo root
├── services/
│   ├── alpr/                    ★ ACTIVE — the ANPR library (7.4k LOC Python)
│   ├── backend/                 ★ ACTIVE — FastAPI platform (11.9k LOC + 7.5k LOC frontend)
│   ├── vehicle-mtmc/            ⚠ VENDORED, NOT WIRED IN (see §3)
│   └── traffic-detection-yolo/  ⚠ VENDORED, NOT WIRED IN (see §3)
├── archive/                     8 traffic clips (437 MB, untracked) — demo + training input
├── vehicle_models/              ⚠ UNUSED (381 MB) — duplicate of vehicle-mtmc/models/
├── vehicle_models.zip           ⚠ UNUSED (353 MB) — zip of the directory above
├── docs/
└── docker-compose.yml           postgres + redis + zookeeper + kafka + backend
                                 + live_feeder + kafka_consumer
```

### services/alpr — the ANPR library

Installable package (`hatchling`, `src/alpr`, console script `alpr`). 26 test
modules. Modules, by role:

| Module | Role |
|---|---|
| `detect.py` | YOLOv8 plate detection → `Detection` boxes |
| `track.py` | ByteTrack — one track id per plate across frames |
| `ocr.py` | `PlateReader` (PaddleOCR), `crop_plate`, `Preprocess` (all steps default **off**) |
| `vote.py` | Per-character confidence-weighted vote across a track's reads |
| `plates/` | Country grammars: `india.py`, `germany.py`, `poland.py`, `correct.py` |
| `dedup.py`, `dupes.py` | Duplicate-sighting cooldown |
| `pipeline.py` | `Pipeline.run()` — orchestrates all of the above, emits + logs |
| `sources.py` | Frame sources: file, directory, camera, RTSP |
| `excel.py` | Writes the confirmed-plate workbook (**the record**, see §4) |
| `data/` | Dataset build: `ingest`, `schema`, `split`, `export`, `manifest`, `stats` |
| `train.py`, `baseline.py` | Config-driven detector training + provenance |
| `evaluate.py`, `cer.py`, `endtoend.py` | Detection metrics, CER/exact-match ablations, outcome buckets |
| `label.py` | Crop extraction + browser labelling page (used to build OCR eval sets) |
| `cli.py` | `alpr env / train / run / label / fetch` |
| `viewer.py`, `env.py` | Live annotated window; interpreter/GPU/Colab report |
| `agent/`, `mcp_servers/`, `ui/` | **Empty stubs** — one `__init__.py`, no code |

### services/backend — the platform

```
backend/
├── main.py            FastAPI app, router registration, static frontend mounts
├── config.py          pydantic-settings — every knob, env-overridable
├── database.py        engine, Session, Base, init_db()
├── models/            9 SQLAlchemy 2.0 ORM models (§9)
├── schemas/           Pydantic request/response contracts
├── api/               12 routers (§8)
├── services/          14 service modules — all the logic
└── core/security.py   JWT + password hashing
frontend/              4 self-contained HTML pages, no build step
scripts/               24 operational + data-generation scripts
alembic/               one migration: a660cfb06462_initial_schema
training/fast_motion/  the high-speed detector fine-tune (dataset + run + weights)
deployments/           delhi / mumbai / benchmarks fixture data
```

---

## 3. What is actually wired up

This matters more than any other section, because the README's architecture
diagram and `main.py`'s own feature list both overstate it.

**Wired in and running:**

- `services/alpr` — a real dependency of the backend, imported by
  `anpr_service.py`. Not a stub.
- The full FastAPI backend, all 12 routers, all 4 frontend pages.
- Redis and Kafka — real integrations, both **off by default**
  (`USE_REDIS=False`, `USE_KAFKA=False`), both enabled in `docker-compose.yml`.

**Not wired in:**

- **`services/vehicle-mtmc` (1.8 GB) is dead code.** Nothing in
  `backend/` or `scripts/` imports it. The only reference anywhere is a demo
  video path in `scripts/cameras_config.json`. The multi-camera association that
  the product actually performs is
  `backend/services/tracking_service.py` — 122 lines of plate matching plus a
  haversine spatio-temporal gate. `main.py`'s description claiming
  "✅ MTMC (Multi-Camera Tracking & Spatio-Temporal ReID)" refers to *that*
  file, not to the vendored ReID models.
- **`services/traffic-detection-yolo` (20 MB) is dead code.** It is a
  self-contained FastAPI app of its own (`main.py`, `routes.py`,
  `detector.py`, its own templates). Nothing in the platform calls it. Its
  `demo_video/demo.mp4` is used as a camera source fixture.
- **Postgres is now the primary store** (as of the year-of-history work).
  `DATABASE_URL` is `postgresql+psycopg2://...` for backend, live_feeder and
  kafka_consumer. What forced the move: SQLite reached 6.5 GB at 13.3M rows and
  corrupted its own `vehicle_events` indexes under three concurrent writers on
  a macOS bind mount, and a year of history is ~40M rows.
  **PostGIS itself is still unused** — `geoalchemy2` is imported zero times and
  all geometry remains plain `Float` lat/lon with hand-written haversine. The
  image provides PostGIS; the schema does not use it.
- **Alembic.** One migration exists, and `init_db()` in `main.py`'s lifespan
  creates tables with `create_all()` instead. `alembic` is imported zero times
  from application code.

See `DEPENDENCY_AUDIT.md` for the full accounting.

---

## 4. Layer 1 — Edge: the ALPR pipeline

`Pipeline.run()` in `services/alpr/src/alpr/pipeline.py`, per frame:

```
frame
  │
  ├─▶ PlateDetector.detect()          YOLOv8s, 1 class, conf ≥ ANPR_CONFIDENCE (0.25)
  │      └─▶ [Detection(box, conf), ...]
  │
  ├─▶ Tracker.update()                ByteTrack: box → stable track_id
  │
  ├─▶ for each detection:
  │      crop_plate(frame, det, padding=0.08)
  │      PlateReader.read(crop)       PaddleOCR → OcrResult(text, confidence)
  │      voter.add(track_id, Read(text, confidence, frame))
  │
  └─▶ on track exit / cooldown:
         voter.result(track_id)       per-CHARACTER weighted vote, min 2 reads
         plates.parse(text, region)   grammar validate + confusable correction
         dedup                        suppress repeat sightings within cooldown
         excel.append(row)            ← the confirmed record
         on_frame(...)                ← the live decoration stream
```

Four design decisions here carry real weight:

**Per-character voting, not per-string** (`vote.py`). Four reads of the same
plate may each differ from every other, so no string has a majority and string
voting picks one by luck. Per-character voting is right at every position
because each error appears in one frame and is outvoted. **The output is
frequently a plate that no single frame read correctly.** This is the largest
accuracy mechanism in the pipeline and it costs nothing in model quality.

**Preprocessing is off by default.** Every `Preprocess` field defaults to
`False`. Measured on 124 labelled crops, each step made CER *worse* (0.2291 raw
vs 0.2410 enhanced): PaddleOCR normalises internally, so enhancing first
resamples twice. Padding stays on because it is a crop decision, not an image
transform.

**The Excel workbook is the source of truth for plates, the frame stream is
not.** `on_frame` sees the vote *in progress* — `KA02HN182` can become
`KA02HN1826` as more reads land. That is perfect for a live UI and wrong for a
database. Only rows that survived voting, the grammar and the cooldown reach the
workbook, and the workbook is what gets ingested
(`video_job_service.py` module docstring makes this explicit).

**Throughput knobs, in order of effect.** Detection is ~85% of per-frame cost,
so the levers are about doing less of it:

| Knob | Effect | Cost |
|---|---|---|
| `VideoFileSource(stride=N)` | ~N x — processes 1 frame in N, skipping with `grab()` so skipped frames pay no colour conversion or buffer copy | cheap here, because a vehicle spans many frames and `vote.py` aggregates per character across a track, so thinning removes redundancy before information |
| `PipelineConfig.batch` | 1.84x measured at 8 | **none** — same weights, resolution and frames, fewer dispatches. Tracking still sees frames one at a time in order |
| `ANPR_IMGSZ` 640 -> 512 | 1.33x | small recall loss, concentrated in the smallest plates |
| ~~ONNX runtime~~ | **rejected** | measured ~26% *slower* than torch on arm64 (24.3 vs 32.7 fps at 512). Kept as a switch for x86_64, default off |

Measured end to end on one clip: 68.7s -> 19.9s at matched resolution, ~6x
against the original 640/stride-1/batch-1 baseline. Frame indices stay the
**true** frame numbers under stride, because timestamps and logged frame
references derive from them.

**Live frame preview.** `Job.note_preview` encodes an annotated JPEG — boxes
with confidence, plus the in-progress vote — throttled to 6 fps at 720px, served
from `GET /jobs/{id}/preview.jpg`. Pulled as an image rather than pushed down
the progress WebSocket, so display rate is decoupled from inference rate and a
slow client cannot back up the pipeline. A preview failure can never fail a job.

**The OCR engine is switched by architecture.** `_paddle_static_engine_is_unreliable()`
(`ocr.py:190`) forces the `onnxruntime` engine on arm64, because paddle's native
`paddle_static` engine segfaults loading PIR-format models there. x86_64 keeps
the native engine.

---

## 5. Layer 2 — Ingestion

Four entry paths, all converging on one row-writing routine.

### 5.1 Direct HTTP — `backend/api/events.py`

- `POST /events/ingest` → one `IngestEventResponse`
- `POST /events/bulk-ingest` → a list of them
- `POST /events/analyze-frame` → runs ANPR on an uploaded frame synchronously
- `POST /events/traffic-snapshot` → aggregate counts, no plate
- `GET /events/recent`, `GET /events/{plate}`

Every ingest does the same three things, in order: write the `VehicleEvent`,
call `TrackingService.associate_event()` for `global_vehicle_id`, then call
`AlertService.check_and_fire()` and `check_checkpoint_speed()`.

### 5.2 Kafka — `services/kafka_consumer.py`, `scripts/run_kafka_consumer.py`

Topic `raw_vehicle_events` → the same ingest path. `USE_KAFKA=False` locally,
`true` in compose, where it runs as its own container. `kafka-python-ng`;
`kafka_service.py` is the producer side.

### 5.3 Media jobs — `api/jobs.py` + `services/video_job_service.py`

Upload a video or image, or pass a `source_path` (allowlisted directories only
— an unrestricted server-side path parameter is a file-read primitive). Returns
a job id immediately; a WebSocket streams progress.

Three properties of this runner are deliberate and worth knowing:

- **Jobs run on threads, not in the event loop.** A 30 s clip is ~30 s of solid
  inference; in a request handler that blocks every other request on the worker,
  including the progress WebSocket.
- **One job at a time — a single-worker executor.** The detector and reader are
  process-wide singletons (loading PaddleOCR twice costs seconds and hundreds of
  MB) and neither Ultralytics' nor PaddleOCR's `predict` promises thread safety
  on a shared model. `QUEUED` is a real state clients see.
- **`_SandboxDB`** — public uploads write into a throwaway per-job SQLite file
  with the production schema, so a stranger's video cannot touch the real table.

### 5.4 City datasets — `services/dataset_service.py` (946 LOC)

"Bring your own city": upload cameras + sightings, and the whole platform comes
up on that city instead of Delhi. `SCHEMA` is the single source of truth and is
*served* at `GET /jobs/dataset/schema`, so the sandbox renders its field table
and templates from the response rather than a hand-copied table — the documented
format cannot drift from the parsed format. Server-side validation duplicates
the browser's on purpose: the client-side pass is a usability feature, not a
security boundary.

### 5.5 Synthetic feeders — `scripts/`

`live_event_feeder.py` (runs as the `live_feeder` container, `--diurnal-rate`)
keeps "now" populated; without it the dashboard shows a month of history that
stops days ago and every last-hour panel reads zero.
`generate_delhi_dataset.py`, `generate_city_dataset.py`,
`generate_future_dataset.py`, `generate_mumbai_sample.py`, `hamburg_adapter.py`
build the fixtures.

---

## 6. Layer 3 — Identity and alerting

### TrackingService — `services/tracking_service.py` (122 LOC)

Assigns `global_vehicle_id`, hybrid strategy:

1. **Plate-based** (`_plate_based_associate`) — a plate is a global identifier;
   if the same plate exists, reuse its `global_vehicle_id`. Cheap, exact,
   and it covers most traffic.
2. **Spatio-temporal ReID** (`_spatio_temporal_reid`) — for plate-less
   sightings. Gated on haversine distance and elapsed time between candidate
   sightings: a vehicle cannot have covered the distance faster than physics
   allows.

This is the platform's entire multi-camera association. The vendored
`vehicle-mtmc` appearance-ReID models are *not* used (§3).

### AlertService — `services/alert_service.py` (212 LOC)

Three alert types, all fired inline on ingest:

| Type | Trigger |
|---|---|
| `BLACKLIST` | plate is in the `blacklist` table |
| `SPEED_VIOLATION` | checkpoint-pair speed (distance/time over two consecutive sightings) > the segment limit, default `DEFAULT_SPEED_LIMIT_KMH=60` |
| `ROUTE_ANOMALY` | speed > `MAX_PLAUSIBLE_SPEED_KMH=150` — physically implausible, so treated as clock skew, a duplicate plate or a bad match rather than as speeding |

That last distinction is the interesting one: the same computation produces
either an enforcement action or a data-quality flag depending on which side of
150 km/h it lands, so an impossible reading never becomes a ticket.

Alerts publish to Redis `alerts:live`; `api/websocket.py` bridges that to
`/ws/alerts`. With `USE_REDIS=False` the publish degrades to a no-op —
`redis_service.py` fails soft throughout.

---

## 7. Layer 4 — Derived intelligence

### RoutingService — `services/routing_service.py` (1171 LOC)

Turns sightings into **real road paths** by querying OSRM over OpenStreetMap for
every consecutive camera pair.

Why it exists: a straight line between two cameras draws the vehicle through
buildings and across the Yamuna, and — the part that actually matters — it
**systematically understates speed**. Straight-line distance between two Delhi
junctions is typically 60–85% of the driving distance, so a car that covered
26.5 km of road in 20 minutes (79 km/h, speeding) looks like 21.7 km
(65 km/h, barely over). **Every leg therefore reports both speeds**, so the
difference is explicit rather than quietly wrong.

Caching is not an optimisation here, it is a requirement: the public OSRM demo
server is rate-limited, a 6-hop trajectory would add six network round-trips to
one API call, and venue Wi-Fi fails. So `scripts/build_route_cache.py` fetches
every pair once, decimates the polyline, and stores it in `route_segments`.
After that the map works with **zero network access**. Falls back to
`_straight_line_route` with `is_real_road=False` when OSRM is unavailable, so
the map degrades visibly rather than lying.

The primary key `(from_camera_id, to_camera_id)` is **directional on purpose** —
Delhi is full of one-ways, flyovers and medians, and reversing a cached
direction would draw vehicles down the wrong carriageway.

### PredictionService — `services/prediction_service.py` (279 LOC)

Next-hop prediction, **zero ML dependencies — pure math on existing DB rows**:

1. Build a transition count matrix from history: `transitions[cam_a][cam_b]++`
2. Normalise per source camera → `P(next_cam | current_cam)`
3. Filter candidates by directional cosine/bearing similarity — removes U-turns
4. ETA = haversine distance / live average speed on that segment
5. Rank by probability, return top-N

Served at `GET /vehicles/{plate}/predict-next-location`.

### PatternService — `services/pattern_service.py`

Per-vehicle behavioural mining. The distinction from `PredictionService`
matters: that one answers "where does traffic go from camera X" — a property of
the *network*, learned from everyone. This answers "what does THIS vehicle
normally do", a property of the individual, and is only answerable because
there is a year of history.

**Home and workplace inference.** DBSCAN (haversine metric on radians) over a
vehicle's sightings, then ask which cluster dominates at night and which
dominates the weekday commute window. DBSCAN because the number of places a
vehicle frequents is unknown and is exactly the thing being discovered —
k-means needs `k`, which is the answer. It also labels sparse sightings as
noise rather than forcing them into a cluster, so "this vehicle has no stable
base" is a real output.

The window is the **commute** (7-11, 16-20), not the working day. This was
wrong on the first pass and worth recording: a fixed camera network watches
roads, so a vehicle parked at its workplace is invisible from 10:00 to 16:00.
Clustering that window found nothing for precisely the regular commuters the
feature should work best on. A candidate workplace must also sit >=1.5 km from
the inferred home, or the densest daytime cluster is the junction outside the
vehicle's own house and "work" becomes a synonym for "home".

**Anomaly detection against the vehicle's own baseline.** IsolationForest over
per-day features (sightings, distinct cameras, first/last/mean hour, max speed,
distance from home, daily range, weekend flag). Per vehicle on purpose: a taxi
doing 300 km a day is unremarkable, a hatchback that has done 20 km a day for
eleven months suddenly doing 300 is the interesting one, and a population-level
model ranks them identically.

`contamination` is a **quota, not a test** — at 0.06 it labels ~6% of days
anomalous whether or not anything happened, which on 40 identical days produced
10 "anomalies" and manufactured suspicion about a vehicle that did nothing. The
forest therefore only *ranks*; a day is reported only if it also deviates by
>=2 sigma on some concrete feature, and the z-scores double as the
human-readable reason. A regular vehicle now returns none.

Declines rather than guesses below 12 sightings or 14 active days. Registered
as an **optional** router — it is the only feature needing scikit-learn, so a
missing wheel degrades this alone.

### SimulationService — `services/simulation_service.py` (1116 LOC)

A **clock**, not a feeder. The demo spans two months centred on now: the past
month is in `vehicle_events`, the next month is staged in `future_events`. The
clock walks sim-time forward and promotes each staged sighting into the live
table when it comes due, one by one.

```
future_events ──tick──▶ vehicle_events ──▶ live feed
                                     ├──▶ live heatmap (rolling N-min counts)
                                     ├──▶ live trajectory (a trail as it forms)
                                     └──▶ next-hop prediction + confidence
```

The reason it is a clock: **predictions have to be falsifiable.** A prediction
made at sim-time T about T+8min can be scored when the clock reaches T+8min,
against a row written *before* the prediction was made and which the predictor
never read. A feeder that invents events as it goes could never be wrong.
`speed` compresses sim-time against wall-time (60× = one simulated hour per real
minute), and it can be paused mid-sentence.

Confidence is reported honestly: `_wilson_lower_bound` for a proportion from few
trials, `_normalised_entropy` for how spread the distribution is.

### Analytics — `api/analytics.py` (517 LOC) + `models/analytics_agg.py`

Endpoints: `/summary`, `/heatmap`, `/density`, `/flow`, `/congestion`,
`/od-matrix`, `/speed`, `/road-usage`, `/snapshots`,
`/analytics/speed-defaulters`.

These are all `GROUP BY` over `vehicle_events`. At ~12M rows over 200 junctions
each is a multi-second full table scan on SQLite, re-run on every page load. So
they are **precomputed into rollup tables** (`RoadUsage` and siblings): each
panel becomes a lookup over hundreds to thousands of rows, which is what makes
the map feel instant.

The tradeoff is stated in the code and is worth repeating: **the rollups are
stale by design.** Anything that must be to-the-second accurate — one plate's
trajectory, the live alert feed — hits `vehicle_events` directly, where it is a
point lookup on an index, not a scan. The rollups use natural composite primary
keys (the rows are identified by what they aggregate) and denormalise road
names, lat/lon and distances, because joining 200 cameras back in per request is
pure overhead for values that never change.

### Benchmarks — `services/benchmark_service.py` (2439 LOC), `services/compute_monitor.py`

The largest single module. Measures real inference cost on this hardware and
projects it to city scale. `GET /benchmarks/hardware`, `/projection`,
`/projection/scenarios`, `/assumptions`, `/runs`, `/run/{id}`, `POST /run`.
`/assumptions` exists so the projections can be argued with rather than
believed.

---

## 8. Layer 5 — API and frontend

`main.py` registers routers in two tiers.

**Core (hard failure if broken):** `auth`, `cameras`, `events`, `vehicles`,
`analytics`, `alerts`, `websocket`.

**Optional (imported defensively, degraded individually):** `jobs`, `routing`,
`benchmarks`, `simulation`. Each is `__import__`-ed in a try/except; a failure
records `FEATURE_STATUS[name] = "unavailable: ..."` and prints a warning
instead of taking the API down. This matters during a live demo: a broken
optional module must not stop the dashboard and the ingestion pipeline from
serving. `GET /` reports `features` so you can see what came up.

**Frontend** — a Next.js 15 / React 19 app (`services/frontend`, its own
container on host port 3001). Four runtime dependencies: next, react,
react-dom, maplibre-gl. No Tailwind and no react-leaflet — the terminal theme
is ported verbatim from the old `index.html` as plain CSS variables, so this is
a rebuild rather than a redesign, and `output: "standalone"` ships ~60 MB
instead of a full node_modules.

Two destinations, down from four. The old split meant no single screen looked
like a working product: a map of cameras that never changed on one page, an
empty trajectory panel on another, a static log on a third.

| Route | Contents |
|---|---|
| `/` | **Surveillance** — one operational screen: 3D map (camera network, congestion, optional TomTom traffic flow), detection log, trajectory tracker, pattern analysis, enforcement alerts, and the simulation-clock ingestion controls |
| `/sandbox` | Video ANPR (with live frame preview), Photo ANPR, City Dataset, and Benchmarks folded in as a tab |

The map is **MapLibre GL, not Leaflet** — Leaflet is a 2D raster compositor and
cannot tilt, so "3D" is not a setting it has. Vector tiles come from
OpenFreeMap (free, no key); its `dark` style carries per-building height for
the fill-extrusion layer. Three themes are selectable.

Two visual conventions carry meaning and are worth not breaking:
*blinking* is reserved for next-hop predictions (things about to happen, with
blink period scaled by probability), while inferred home/workplace are
**static** squares because they are conclusions drawn from history. And a
trajectory leg that fell back to a straight line is drawn dashed and amber,
because rendering a guess identically to a measurement is the map lying.

The old static pages are still served by FastAPI during the transition:

| Route | File | Purpose |
|---|---|---|
| `/app` | `index.html` (1692 lines) | Operator dashboard — Leaflet map, trajectories, alerts, analytics |
| `/app/test` | `test.html` (2747 lines) | Public sandbox — upload video / photo / city dataset |
| `/app/live` | `live.html` (906 lines) | Live ingestion monitor, **deliberately its own window** so it can sit on a second screen beside the dashboard during a demo |
| `/app/benchmarks` | `benchmarks.html` (2152 lines) | Measured compute cost and city-scale projections |

`CORSMiddleware` is `allow_origins=["*"]` — flagged in the code itself as
"Restrict in production".

---

## 9. Data model

Nine ORM models. `vehicle_events` is the hub; everything else is a dimension
table, a rollup, or a cache.

### `vehicle_events` — the single source of truth

| Group | Columns |
|---|---|
| identity | `event_id` (uuid PK) |
| sighting | `camera_id` (FK), `local_track_id`, `timestamp` |
| ANPR | `plate`, `plate_confidence` |
| location | `latitude`, `longitude`, `direction` |
| attributes | `vehicle_type`, `vehicle_color`, `speed` |
| attribute-only identity | `vehicle_make`, `vehicle_model`, `plate_partial` (indexed), `plate_raw`, `attribute_confidence` |
| MTMC | `global_vehicle_id` (indexed, nullable) |
| audit | `created_at` |

Two column groups here encode real decisions:

**Attribute-only sightings.** A camera does not always get the plate — a vehicle
crossing at 90 km/h, a plate occluded by the car ahead, a bent bike plate. But
the detector still saw a silver hatchback heading north at 88 km/h, and
discarding that discards the only record it passed. These rows are ingested on
attributes alone: they appear in counts, heatmap and speed analytics, and a
partial plate plus make/model/colour is often enough for an investigator to
match by hand against the plate-identified sightings either side.
**`plate` stays NULL rather than being filled with a guess**, so no query can
mistake an attribute match for a plate read. `plate_partial` is indexed because
"find every sighting consistent with this plate" is the query investigators
actually run.

**`plate_raw`** keeps the unformatted OCR output *before* grammar correction. It
is the evidence behind a contested read.

**`global_vehicle_id` nullable** means trajectory still works for
plate-identified vehicles when association has not run.

### The rest

| Model | Role |
|---|---|
| `Camera` | camera_id, lat/lon, road name, direction — the deployment |
| `Alert`, `Blacklist` | fired alerts; watchlist plates |
| `Trajectory` | reconstructed paths |
| `RouteSegment` | OSRM polyline cache, directional PK `(from, to)` |
| `TrafficSnapshot` | aggregate counts with no plate |
| `FutureEvent` | staged next month, drained by the simulation clock |
| `analytics_agg` (`RoadUsage` et al.) | precomputed rollups, stale by design |

**Storage.** PostgreSQL 15.4, in the `pgdata` named volume. Tuned for bulk
loading in `docker-compose.yml` (`shared_buffers=1GB`, `work_mem=64MB`,
`maintenance_work_mem=512MB`, `max_wal_size=4GB`) because the image defaults
killed the backend mid-load: a 40M-row `COPY` in one transaction emitted ~900 MB
of WAL against a 1 GB `max_wal_size` with nothing able to recycle it. The
loaders now `COPY` in 2M-row chunks so checkpoints can recycle between them.

Two operational notes that cost real time to learn:
* Bulk loads must run with the **frontend stopped**. It polls `/events/recent`
  every 2s; those `SELECT`s hold ACCESS SHARE, the generator's `TRUNCATE` needs
  ACCESS EXCLUSIVE, and the truncate then blocks every later read behind it.
* Do not `docker compose restart backend` while a load is running — the
  generator runs *inside* that container. The image bakes the source, so a
  restart cannot pick up host edits anyway; rebuild before starting a load.

---

## 10. End-to-end dataflows

### 10.1 Video upload → plate on the map

```
POST /jobs/video  (multipart file, or allowlisted source_path)
  └─ api/jobs.py                     validates extension, allocates job_id
     └─ video_job_service.submit_video_job()   → single-worker ThreadPoolExecutor
        │                                        returns job_id immediately
        │  ── client opens WS /jobs/{id}/ws ──▶ progress: frames, fps, live texts
        │
        └─ alpr Pipeline.run(source)
             per frame: YOLOv8 detect → ByteTrack → crop → PaddleOCR → voter.add
             on track exit: per-char vote → grammar parse → dedup → excel row
                            └─ on_frame(...) → WS (decoration only)
             │
             └─ workbook written (THE record)
                └─ rows read back and ingested as VehicleEvents
                   ├─ TrackingService.associate_event()  → global_vehicle_id
                   └─ AlertService.check_and_fire() + check_checkpoint_speed()
                        └─ redis publish 'alerts:live' → /ws/alerts → dashboard toast
                   └─ INSERT vehicle_events   (or _SandboxDB for public uploads)
                        └─ visible to /events/recent, /analytics/*, /vehicles/{plate}/*
```

### 10.2 Live camera / Kafka → alert

```
camera_worker.py / external producer
  └─ Kafka topic 'raw_vehicle_events'
     └─ scripts/run_kafka_consumer.py → services/kafka_consumer.py
        └─ same ingest path as 10.1's tail:
           VehicleEvent → TrackingService → AlertService → Redis → WebSocket
```

### 10.3 Trajectory request → road-snapped polyline

```
GET /routing/trajectory/{plate}
  └─ query vehicle_events for that plate, ordered by timestamp
     └─ for each consecutive pair (A,B):
          resolve_camera() → CameraRef (lat/lon)
          _memo_get((A,B))              in-process memo
            ↓ miss
          route_segments table          persistent OSRM cache
            ↓ miss
          fetch_osrm_route(A,B)         throttled HTTP to OSRM
            ↓ unavailable
          _straight_line_route()        is_real_road=False — degrade visibly
     └─ decimate_polyline, polyline_length_m, detour_ratio
     └─ each leg reports BOTH road-distance speed and haversine speed
  └─ Leaflet draws the polyline on /app
```

### 10.4 Simulation clock → prediction, then scored

```
POST /simulation/start {speed: 60}
  └─ SimulationClock.start() → background thread
     every tick:
       _first_unreleased_timestamp() → due FutureEvents
       promote each into vehicle_events (one by one)
       TransitionModel.observe(from_cam, to_cam)
       _record(hits, backlog) → live feed / heatmap / trajectory buffers
                                └─ redis 'stats:live' → /ws/stats

GET /vehicles/{plate}/predict-next-location   at sim-time T
  └─ PredictionService.predict()
       transition matrix → P(next|current)
       bearing filter (drop U-turns)
       ETA = haversine / live segment speed
       _wilson_lower_bound + _normalised_entropy → honest confidence
  └─ when the clock reaches T+ETA, the prediction is scored against a row
     written before it was made. Falsifiable.
```

### 10.5 Analytics page load

```
GET /app  →  index.html  →  parallel fetches:
  /analytics/summary      ─┐
  /analytics/heatmap       │ each: redis_service.get(key)  → hit? return
  /analytics/density       │       miss → SELECT from analytics_agg rollup
  /analytics/congestion    │              (NOT a scan of vehicle_events)
  /analytics/od-matrix     │       → redis_service.set(key, ttl)
  /analytics/road-usage   ─┘
  /alerts/                 → live table, direct index lookup
  WS /ws/stats             → rolling counters from redis pub/sub
```

---

## 11. Configuration and deployment

Every knob is in `backend/config.py`, env- or `.env`-overridable.

| Setting | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./dev.db` | compose: `postgresql+psycopg2://postgres:***@postgres:5432/vehicle_intelligence` |
| `ANPR_IMGSZ` | `512` | detector resolution; cost scales with the square |
| `ANPR_STRIDE` | `3` | process 1 frame in N — the dominant throughput knob |
| `ANPR_BATCH` | `8` | frames per detector call; accuracy-neutral |
| `ANPR_PREFER_ONNX` | `False` | measured *slower* than torch on arm64 |
| `ANPR_WEIGHTS_PATH` | `../alpr/best.pt` | set this to deploy retrained weights |
| `ANPR_REPO_PATH` | `../alpr` | sys.path fallback when `import alpr` fails |
| `ANPR_DEVICE` | `None` (auto: MPS → CUDA → CPU) | compose forces `cpu` |
| `ANPR_CONFIDENCE` | `0.25` | detector floor |
| `ANPR_REGION` | `IN` | selects the plate grammar |
| `DEFAULT_SPEED_LIMIT_KMH` | `60.0` | `SPEED_VIOLATION` threshold |
| `MAX_PLAUSIBLE_SPEED_KMH` | `150.0` | above this → `ROUTE_ANOMALY` |
| `USE_REDIS` / `REDIS_URL` | `False` / localhost | compose: true |
| `USE_KAFKA` / `KAFKA_BROKER` | `False` / localhost:9092 | compose: true, `kafka:29092` |
| `DEBUG` | `True` | |

**The path-resolution comment in `config.py` is a genuine scar and worth
reading:** an editable install writes an *absolute* path into a `.pth` inside
the venv, so *moving the project directory* — this one was created under
`~/Desktop/SIH` and now lives under `~/Documents/SIH` — silently breaks
`import alpr`, and the only symptom is every video job failing with
`No module named 'alpr'`. `_ensure_alpr_importable()` in `anpr_service.py:42`
resolves the repo relative to `__file__` to survive that.

`docker compose up --build` from the repo root brings up postgres, redis,
zookeeper, kafka, backend, live_feeder and kafka_consumer. No frontend
container — FastAPI serves the pages. Then open `http://localhost:8000/app`.

**Two hardcoded absolute paths are load-bearing in the compose file** and are
the deployment's most brittle point: `archive/` is mounted at
`/Users/aarushsharma/Documents/SIH/archive` *inside the container* because that
literal path is baked into `frontend/test.html` (`ARCHIVE_DIR`) and
`backend/api/jobs.py` (`ALLOWED_SOURCE_ROOTS`). This works only for this user on
this machine.

---

## 12. Concurrency, threading and state

| Concern | Mechanism |
|---|---|
| ANPR jobs | single-worker `ThreadPoolExecutor` — model singletons are not thread-safe |
| Model loading | process-wide singleton via `anpr_service`, guarded by a `threading` lock |
| Simulation clock | its own background thread, re-anchorable sim-time |
| Live push | Redis pub/sub (`alerts:live`, `stats:live`) bridged to WebSockets |
| Caching | Redis for analytics responses; in-process memo + DB table for OSRM routes |
| Public isolation | `_SandboxDB` — throwaway per-upload SQLite, production schema |
| DB contention | `lock_timeout` on Postgres sessions; `_set_busy_timeout(15_000)` on the SQLite path |
| Failure isolation | optional routers degrade individually; `redis_service` fails soft |

---

## 13. Known weak points

Ordered by how much they would cost to hit in production.

1. ~~SQLite at 6.1 GB~~ — **resolved**: Postgres is the primary store. The
   rollup tables remain, but they are now an optimisation rather than a
   workaround for an engine that could not serve the query pattern.
2. **`create_all()` instead of Alembic — now worse, not better.** The single
   migration `a660cfb06462_initial_schema` is an **empty stub** (`upgrade()` is
   `pass`), so the entire schema exists only via `create_all()`. On Postgres,
   with 40M rows, any schema change is now a manual operation against real
   data. This is the most significant remaining debt.
3. **Two hardcoded absolute host paths** in `docker-compose.yml`,
   `frontend/test.html` and `api/jobs.py` (§11). The deployment is
   machine-specific.
4. **`allow_origins=["*"]`** with JWT auth in front of an enforcement system.
5. **Alerts fire synchronously inside the ingest request.** At real camera-fleet
   throughput, ingest latency is bounded by alert evaluation, which itself runs
   DB queries per event. This is what Kafka is there to decouple — but the
   consumer calls the same synchronous path.
6. **`TrackingService` is 122 lines of plate matching.** It works well *because*
   plates are near-unique global identifiers, but plate-less association is a
   distance/time gate only — no appearance features. The vendored ReID models
   that would fix this are present and unused (§3).
7. **Rollup staleness is not surfaced to the UI.** The code is honest that the
   aggregates lag; the dashboard does not show *how much*.
8. **~1.9 GB of vendored dead code and ~734 MB of duplicated model weights**
   in the tree, plus ~619 MB of regenerable training JPEGs in git history. See
   `DEPENDENCY_AUDIT.md`.

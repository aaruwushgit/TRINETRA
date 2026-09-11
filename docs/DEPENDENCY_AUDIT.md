# Bloat and dependency audit

Findings from a full pass over the tree at commit `d9401ca`. Everything below
was verified against the code, the installed environments and git history — no
item is here on suspicion alone.

**Headline numbers**

| | Size |
|---|---|
| Working tree | **14.5 GB** |
| — of which three checked-out virtualenvs | **5.1 GB** |
| — of which vendored code that nothing imports | **1.9 GB** |
| — of which duplicated model weights | **734 MB** |
| — of which `dev.db` + WAL | **6.4 GB** |
| `.git` | **703 MB**, of which **619 MB is regenerable training JPEGs** |
| Backend venv site-packages | 114 packages, **~1.9 GB** for a plate reader |

Roughly **8.5 GB of the 14.5 GB is removable or regenerable** without changing
a single line of application behaviour, and the git repository can go from
703 MB to under 15 MB.

Findings are ordered by payoff. Each says what to do and what it costs.

---

## Tier 1 — Large, provably unused, safe to delete

### 1.1 `services/vehicle-mtmc/` — 1.8 GB of dead code · **delete**

Nothing in `services/backend/backend/` or `services/backend/scripts/` imports
it. The only textual reference anywhere in the platform is a demo video path in
`scripts/cameras_config.json`, and that points at `traffic-detection-yolo`, not
at this. Verified:

```bash
grep -rn "mtmc" services/backend/backend services/backend/scripts    # → 0 hits
```

The multi-camera association the product actually performs is
`backend/services/tracking_service.py` — 122 lines of plate matching plus a
haversine distance/time gate. `main.py`'s feature list saying
"✅ MTMC (Multi-Camera Tracking & Spatio-Temporal ReID)" describes *that file*,
not these weights.

Breakdown: `models/` 381 MB · `datasets/` 221 MB · `venv/` 1.2 GB ·
`assets/` 18 MB (including a 16 MB GIF) · `detection/yolov5/` a whole second
vendored YOLO fork with its own Dockerfiles · `reid/vehicle_reid/` with CUDA C++
extensions (`GPU-Re-Ranking/extension/*/setup.py`) that were never built.

**Replacement:** none needed today. If appearance-based ReID becomes a
requirement, the honest move is a single ONNX ReID embedder called from
`tracking_service.py` (~50 MB), not a vendored research repo. Keep the
upstream URL in a doc; the repo is public.

### 1.2 `vehicle_models/` (381 MB) + `vehicle_models.zip` (353 MB) · **delete both**

`vehicle_models/` contains exactly `resnet50_mixstyle`, `color_svm.pkl`,
`type_svm.pkl` — **byte-identical to `services/vehicle-mtmc/models/`**. And
`vehicle_models.zip` is a zip of that same directory. So this is the same
381 MB of weights stored **three times**, for a component that is not wired in
(§1.1). Zero code references; the only mentions are two lines of prose in
`README.md:275` and `docs/PROJECT_CONTEXT.md:356`.

Already gitignored, so this is disk only — but it is 734 MB of disk.

### 1.3 `services/traffic-detection-yolo/` — 20 MB · **delete, keep one file**

A complete, separate FastAPI application (`main.py`, `routes.py`,
`detector.py`, `config.py`, its own `templates/`, its own `requirements.txt`
pinning a *conflicting* `torch==2.11.0` / `numpy==2.2.6`). The platform never
calls it. Its stated model is trained on **Bangladeshi** urban traffic with 9
classes including CNG and rickshaw — a different problem from Indian plate
reading.

The only thing the platform uses from it is `demo_video/demo.mp4` (5.9 MB),
referenced three times by `scripts/cameras_config.json`. Also inside: a 5.0 MB
`docs/demo.gif`, a 3.3 MB notebook, a 6.2 MB `yolov8n.pt`, a **0-byte**
`test_filter.py`, an empty `cookies/` directory, and a committed `__pycache__/`.

**Action:** move `demo_video/demo.mp4` to `archive/`, update the three paths in
`cameras_config.json`, delete the rest.

### 1.4 Three checked-out virtualenvs — 5.1 GB · **rebuild, don't keep**

`services/alpr/.venv` 1.9 GB · `services/backend/.venv` 2.0 GB ·
`services/vehicle-mtmc/venv` 1.2 GB.

They are gitignored, but they are 5.1 GB of duplicated wheels: **torch is
installed three times** (529 + 583 + 529 MB), **paddle twice** (427 MB each),
`cv2` three times, `pandas` three times.

The mtmc venv goes with §1.1. The remaining two overlap almost entirely — alpr
is a dependency of backend, and the Dockerfile already installs them into **one**
environment. Do the same locally: one venv at the repo root with
`pip install -e services/alpr[ocr]` plus the backend requirements. Saves ~2 GB
and removes the class of bug where the two environments have different
`paddleocr` versions and only one of them works.

---

## Tier 2 — Git history: 703 MB → <15 MB

### 2.1 619 MB of regenerable training JPEGs are committed · **rewrite history**

```
531.3 MB  services/backend/training/fast_motion/dataset/images/train
 87.7 MB  services/backend/training/fast_motion/dataset/images/val
 42.9 MB  services/backend/training/fast_motion/runs/fast_motion/weights
```

That is **94% of the entire repository**. The images are 819 pseudo-labelled
frames written by `scripts/finetune_fast_motion.py --from-videos`: a script in
this repo, deterministic given `--seed`, regenerating from `archive/*.mp4`.
Committing them is committing the output of a build step.

Worse, they are not even *usable* as committed: `dataset/data.yaml` has an
absolute `path:` into `vehicle-intelligence-backend/`, a directory this repo no
longer has, and `labels/*.cache` are keyed to those dead paths.

**Action:**
1. Add to `.gitignore`:
   ```
   services/backend/training/**/dataset/
   services/backend/training/**/runs/*/weights/*.pt
   ```
2. Keep, and keep committing, the small artefacts that make a run
   *reproducible and auditable*: `provenance.json`, `results.csv`, `args.yaml`,
   and the PNG curves. Those are ~4.8 MB and they are the actual scientific
   record — the images are not.
3. Rewrite history with `git filter-repo --path services/backend/training --invert-paths`,
   then `git gc --prune=now --aggressive`.

**Cost, stated plainly:** history rewrite changes every commit SHA after the
first touch. Everyone with a clone must re-clone. Do it once, deliberately, and
announce it. Note that `provenance.json` records `git_sha: 4f998ea`, which the
rewrite will invalidate — copy the old SHA into a note in the file first.

### 2.2 12.2 MB `deployments/mumbai/` fixture data · **regenerate instead**

`mumbai_week.json` (9.3 MB) and `mumbai_events.csv` (2.9 MB) are the output of
`scripts/generate_mumbai_sample.py`, also in this repo. Same argument, one
thousandth the size — worth doing in the same history rewrite, not on its own.

---

## Tier 3 — Python dependencies

### 3.1 Three conflicting OpenCV builds are installed simultaneously · **fix now**

In `services/backend/.venv`:

```
opencv_python-5.0.0.93            (GUI build)
opencv_python_headless-5.0.0.93   (headless build)
opencv_contrib_python-4.10.0.84   (contrib, and a DIFFERENT major version)
```

All three install into the same `cv2` namespace. 221 MB, and whichever landed
last wins — which is not a decision anyone made. `alpr/pyproject.toml` even
documents the hazard ("Needs the GUI build of OpenCV, which conflicts with the
headless wheel above") and then the environment ends up with both anyway,
because **`ultralytics` hard-depends on `opencv-python`** (the GUI build) and
drags it in behind the headless pin.

**Fix:** pin exactly one — `opencv-python-headless` for the server, and install
`ultralytics` with `--no-deps` plus explicit deps, or accept the GUI build
everywhere and drop the headless pin. Uninstall `opencv-contrib-python`
entirely; nothing in this repo uses a contrib module.

### 3.2 `paddleocr` drags in `paddlex`, a whole model-zoo framework · **~450 MB**

`paddleocr → paddlex → modelscope, polars, pandas, aistudio-sdk, ruamel.yaml,
prettytable, ujson, huggingface-hub…`. Present in the backend venv as a direct
consequence:

| Package | Size | Why it is there |
|---|---|---|
| `paddle` | 427 MB | the inference runtime — genuinely needed |
| `_polars_runtime_32` + `polars` | 198 MB | pulled by `ultralytics`; a dataframe engine, for a plate reader |
| `sympy` | 72 MB | torch's symbolic shapes |
| `modelscope` | 60 MB | paddlex's model hub client |
| `pandas` | 70 MB | paddlex + ultralytics |
| `matplotlib` + `fontTools` | 51 MB | ultralytics plotting |
| `pypdfium2_raw` | 7.1 MB | paddlex's PDF support |
| `Crypto`, `cryptography` | 25 MB | transitive |
| `shapely`, `networkx` | 24 MB | transitive |

**The right replacement, and it is a real one:** the pipeline only ever runs
recognition on a plate crop of a few thousand pixels. `ocr.py:243` *already*
switches to the `onnxruntime` engine on arm64. Export the PP-OCR recogniser to
ONNX once and depend on `onnxruntime` (~40 MB) instead of
`paddlepaddle` + `paddleocr` + `paddlex` (~450 MB plus the transitive tail).
That drops the image by roughly a third and removes `polars`, `modelscope`,
`pypdfium2` and `aistudio-sdk` in one move. `PlateReader` is a thin,
well-isolated wrapper — this is a contained change behind one class, and
`cer.py` already exists to prove the swap did not cost accuracy.

### 3.3 `onnxruntime` is declared but **not installed** in the backend venv · **bug**

`alpr[ocr]` declares `onnxruntime>=1.18`, and `_paddle_static_engine_is_unreliable()`
switches to it on arm64 because paddle's native engine *segfaults* there. But:

```bash
ls services/backend/.venv/lib/python3.13/site-packages | grep -i onnx    # → nothing
```

So on this Apple Silicon machine the backend venv is one code path away from a
segfault, and the guard that was written to prevent it cannot fire. Install
`alpr[ocr]` into the backend environment (§1.4's single-venv fix does this), or
add `onnxruntime` to `requirements.txt`.

### 3.4 Declared-but-never-imported backend dependencies

Import counts across `backend/`, `scripts/` and `tests/`:

| Declared in `requirements.txt` | Imports | Verdict |
|---|---|---|
| `geoalchemy2>=0.14.0` | **0** | **Remove.** All geometry is plain `Float` lat/lon + hand-written haversine. |
| `psycopg2-binary>=2.9.9` | **0** | Remove *for now* — 9 MB, plus `libpq-dev` in the image, for a Postgres nothing connects to. Re-add with the real Postgres migration. |
| `alembic>=1.13.0` | **0** | Keep the dependency, but see §3.5 — it is not the dependency that is wrong. |
| `passlib[bcrypt]>=1.7.4` | 0 (`bcrypt` used directly) | Remove `passlib`; `core/security.py` uses `bcrypt` directly. |
| `pyjwt>=2.8.0` | 1 (as `jwt`) | Keep. |
| `ultralytics`, `redis`, `kafka-python-ng`, `requests`, `opencv-python`, `numpy`, `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic*`, `python-multipart`, `bcrypt` | used | Keep. |

`requirements.txt` also carries ~15 lines of commented-out instructions for a
`../Automatic-License-Plate-Recognition` path that no longer exists (the repo
was de-submoduled). It is stale advice that will actively mislead the next
person; delete it and point at `services/alpr`.

### 3.5 Postgres/PostGIS is provisioned and never connected · **decide**

`docker-compose.yml` starts `postgis/postgis:15-3.3`, healthchecks it, and makes
`backend` `depends_on` it — while every service sets
`DATABASE_URL: sqlite:////data/dev.db`. Meanwhile `dev.db` is **6.1 GB with a
302 MB WAL**, and `models/analytics_agg.py` exists specifically because SQLite
cannot serve the dashboard's `GROUP BY` queries at that size.

This is not really a dependency problem, it is a deferred decision, and both
resolutions are cheap:

- **Use it:** set `DATABASE_URL` to the Postgres service, run the Alembic
  migration instead of `init_db()`'s `create_all()`, and the rollup tables
  become an optimisation rather than a workaround.
- **Or drop it:** remove the postgres service, `psycopg2-binary`, `geoalchemy2`
  and `libpq-dev`. Saves ~600 MB of image pull and stops implying a capability
  the system does not have.

Leaving it as-is costs a 600 MB image pull and a healthcheck wait on every
`docker compose up`, for nothing.

### 3.6 `services/vehicle-mtmc/requirements.txt` pins conflict with everything

`torch==1.13.0`, `torchvision==0.14.0`, `numpy<1.23.0`, `pandas<2.0`,
`opencv-python==4.5.5.64`, `Pillow==9.1.0` — versus the backend's
`torch 2.14`, `numpy 2.x`, `pandas 2.x`. Three mutually unsatisfiable pin sets
in one repo (mtmc, traffic-detection-yolo, backend/alpr) is why there are three
venvs. Resolved by §1.1 and §1.3: delete two of the three.

---

## Tier 4 — Docker image

### 4.1 `build-essential` and `git` ship in the final image · **~400 MB**

`services/backend/Dockerfile` is single-stage, so the compiler toolchain
installed to build wheels stays in the runtime layer.

**Fix:** two-stage build — install into a venv in a `builder` stage, `COPY
--from=builder` the venv into a clean `python:3.11-slim`. Keep only the runtime
`.so` deps (`libgl1`, `libglib2.0-0`, `libgomp1`). Drop `libpq-dev` with §3.4,
and `git` unless a dependency is installed from a git URL — none is.

### 4.2 Three images are built from the same Dockerfile

`backend`, `live_feeder` and `kafka_consumer` all use
`dockerfile: services/backend/Dockerfile` with different `command:`s. That is
the right pattern, but confirm they share a layer cache rather than building
three times — pin an `image:` name on the first and reference it from the other
two.

### 4.3 The build context is the repo root, and `.dockerignore` has two gaps

`.dockerignore` correctly excludes `archive/`, `vehicle_models*`, the venvs,
`dev.db*`, `uploads/` and `sandbox_dbs/`. It does **not** exclude:

- `services/vehicle-mtmc/` — ~620 MB of models and datasets (excluding its
  venv, which *is* ignored), for code that is not wired in
- `services/backend/training/` — 619 MB of pseudo-labelled JPEGs

So every `docker compose up --build` ships ~1.2 GB of context to the daemon
before a single layer is built. §1.1 and §2.1 remove both at source; until then,
add them to `.dockerignore` — it is a two-line change and it is the cheapest
item in this document.

---

## Tier 5 — Small, cheap, do while you are in there

| Item | Action |
|---|---|
| `services/alpr/agent/`, `mcp_servers/`, `ui/` | Three directories containing one empty `__init__.py` each and no code. Delete, or put a one-line README saying what is planned. |
| `services/traffic-detection-yolo/test_filter.py` | 0 bytes. |
| `services/traffic-detection-yolo/cookies/` | Empty directory. |
| `services/traffic-detection-yolo/__pycache__/`, `services/alpr/src/alpr/__pycache__/` | Committed bytecode. `.gitignore` has `__pycache__/` — these predate it; `git rm -r --cached`. |
| `services/backend/uploads/` 410 MB, `sandbox_dbs/` 77 MB, `job_outputs/` 264 KB | Gitignored job scratch. Add a retention sweep — sandbox DBs are throwaway by design and nothing deletes them. |
| `services/backend/dev.db-wal` 302 MB | Checkpoint it: `sqlite3 dev.db "PRAGMA wal_checkpoint(TRUNCATE);"` |
| `services/alpr/test_car.jpeg`, `test_car2.jpg`, `test_image2.JPG`, `plates.xlsx`, `plates.xlsx.jsonl` | Ad-hoc fixtures in the package root. Move under `tests/fixtures/`. |
| `services/vehicle-mtmc/assets/highway_tracked.gif` | 16 MB GIF in git history — goes with §1.1 and the §2.1 rewrite. |
| `docs/Open Source Projects and how to build this bitch.md` | 949 lines of scaffolding notes, with a filename that will end up in a screenshot. Rename and fold the still-true parts into `ARCHITECTURE.md`. |
| Two hardcoded absolute host paths | `/Users/aarushsharma/Documents/SIH/archive` is baked into `docker-compose.yml`, `frontend/test.html` (`ARCHIVE_DIR`) and `backend/api/jobs.py` (`ALLOWED_SOURCE_ROOTS`). Not size bloat, but it makes the deployment work on exactly one machine. Make it a setting. |

---

## Suggested order of execution

Grouped so each step is independently verifiable and nothing is deleted before
it is proven unused.

**Step 1 — reclaim disk, no code change, no history change (~7.5 GB)**
```bash
cd /Users/aarushsharma/Documents/SIH
mv services/traffic-detection-yolo/demo_video/demo.mp4 archive/demo.mp4
# then update the 3 source paths in services/backend/scripts/cameras_config.json
rm -rf services/vehicle-mtmc services/traffic-detection-yolo
rm -rf vehicle_models vehicle_models.zip
rm -rf services/alpr/.venv services/backend/.venv
sqlite3 services/backend/dev.db "PRAGMA wal_checkpoint(TRUNCATE);"
```
Then rebuild **one** venv at the repo root and run both test suites
(`services/alpr`: 26 modules; `services/backend/tests`) plus
`docker compose up --build` and a click through `/app`, `/app/test`,
`/app/live`, `/app/benchmarks`. If anything imports what you deleted, you will
know here.

**Step 2 — dependencies.** Fix the OpenCV triple-install (§3.1), add
`onnxruntime` (§3.3), remove `geoalchemy2` / `passlib` / `psycopg2-binary`
(§3.4), delete the stale comment block. Re-run the suites.

**Step 3 — Docker.** Two-stage build (§4.1), verify `.dockerignore` (§4.3),
measure the image before and after.

**Step 4 — the ONNX OCR swap (§3.2).** The largest single win (~450 MB) and the
only item here that touches inference behaviour. Do it last, behind
`cer.py`'s measurement, so "accuracy unchanged" is a number rather than a hope.

**Step 5 — history rewrite (§2.1, §2.2).** Announce it, then
`git filter-repo`. 703 MB → <15 MB.

**Do not** batch steps 1 and 5 together. Step 1 is reversible from a backup;
step 5 is not, and you want a known-good tree before you rewrite history.

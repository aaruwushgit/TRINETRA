# Vehicle Intelligence Backend

City-wide vehicle tracking backend for SIH. Connects ANPR, exposes APIs for the dashboard, and has stubs for teammates to wire in MTMC.

## Architecture

```
Camera Feeds
     ↓
AI Workers  ←── POST /events/ingest
     ↓
FastAPI Backend
  ├── ANPR Service       ✅ Working (wraps Automatic-License-Plate-Recognition)
  ├── Tracking Service   🔧 Stub — plate-based for now, ReID slot ready
  ├── Alert Service      ✅ Working (blacklist + real-time alerts)
  └── Analytics          ✅ Working (density, summary stats)
     ↓
SQLite (dev) / PostgreSQL + PostGIS (prod)
     ↓
React Dashboard  ←── GET /vehicles/{plate}/trajectory, /alerts, /analytics
```

## Quickstart

```bash
# 1. Clone this repo
git clone <this-repo>
cd vehicle-intelligence-backend

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the ANPR repo (must be cloned separately)
pip install -e ../Automatic-License-Plate-Recognition

# 5. Copy and edit the .env
cp .env.example .env
# Edit ANPR_WEIGHTS_PATH to point at best.pt

# 6. Start the server
uvicorn backend.main:app --reload

# 7. Seed cameras (first run only)
python scripts/seed_cameras.py
```

Open http://localhost:8000/docs for interactive API docs.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/events/ingest` | AI workers push vehicle detection events |
| `POST` | `/events/analyze-frame` | Upload image → get plate text (test ANPR) |
| `GET`  | `/vehicles/{plate}` | All sightings of a plate |
| `GET`  | `/vehicles/{plate}/trajectory` | Full camera trajectory |
| `GET`  | `/cameras` | List all registered cameras |
| `POST` | `/cameras` | Register a new camera |
| `GET`  | `/analytics/density` | Vehicle count per camera |
| `GET`  | `/analytics/summary` | Dashboard summary stats |
| `GET`  | `/alerts` | Active alerts |
| `POST` | `/alerts/blacklist` | Add plate to blacklist |
| `GET`  | `/alerts/blacklist` | List blacklisted plates |

## For Teammates: Connecting MTMC

The MTMC integration slot is in [`backend/services/tracking_service.py`](backend/services/tracking_service.py).

Look for the `_reid_associate()` method — it has a `NotImplementedError` and a comment explaining exactly what to implement. The interface contract is fixed:

```python
def _reid_associate(self, event: VehicleEvent, db: Session) -> str | None:
    # Your code here: call vehicle_mtmc, return global_vehicle_id or None
```

Everything else in the backend already handles the rest.

## Tech Stack

- **FastAPI** — API framework
- **SQLAlchemy** — ORM
- **SQLite** (dev) / **PostgreSQL + PostGIS** (prod)
- **ANPR** — [`Automatic-License-Plate-Recognition`](https://github.com/fayazhussain2821/Automatic-License-Plate-Recognition)
- **MTMC** — [`vehicle_mtmc`](https://github.com/regob/vehicle_mtmc) (to be connected by teammates)

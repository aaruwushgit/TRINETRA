# Hamburg Live Camera Deployment (`deployment-hamburg`)

Real-time public traffic camera integration consuming live data from the city of Hamburg, Germany (**Verkehrskameras Hamburg**, BVM).

## Architecture Overview

```
REAL HAMBURG CAMERA (OGC API)
          │
          ▼
   HamburgCameraAdapter (scripts/hamburg_adapter.py)
          │
          ▼
   EXISTING CameraWorker (scripts/camera_worker.py)
          │
          ▼
   EXISTING AI PIPELINE (YOLOv8 + ByteTrack + ANPR)
          │
          ▼
   EXISTING VEHICLE EVENT SCHEMA
          │
          ▼
   FastAPI Backend (POST /events/bulk-ingest)
          │
          ▼
   PostgreSQL / SQLite Database
          │
          ▼
   React / Interactive Glassmorphic Frontend (OpenStreetMap / Leaflet)
```

## Source Data & Licensing

- **Dataset**: [Verkehrskameras Hamburg](https://suche.transparenz.hamburg.de/dataset/verkehrskameras-hamburg35)
- **API Endpoint**: `https://api.hamburg.de/datasets/v1/verkehrskameras`
- **License**: Datenlizenz Deutschland – Namensnennung – Version 2.0 (BVM Hamburg)
- **Cameras**: 18 active traffic junctions across Hamburg (A1, A7, Köhlbrandbrücke, Willy-Brandt-Straße, etc.)

## Quick Start

1. **Start Backend**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

2. **Run Hamburg Deployment**:
   ```bash
   python scripts/run_hamburg.py --cameras 18 --fetch-interval 3.0
   ```

3. **Run Test Suite**:
   ```bash
   python scripts/test_hamburg.py
   ```

4. **Launch Frontend**:
   Open `frontend/index.html` in your browser or run a simple local HTTP server.

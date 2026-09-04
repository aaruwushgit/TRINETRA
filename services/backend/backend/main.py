"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db
from backend.api import alerts, analytics, auth, cameras, events, vehicles, websocket

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks."""
    init_db()  # Create tables
    print(f"✅ {settings.APP_NAME} started. Docs: http://localhost:8000/docs")
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "City-wide vehicle intelligence backend.\n\n"
        "**Modules connected:**\n"
        "- ✅ Enterprise JWT Auth & RBAC\n"
        "- ✅ ANPR (Automatic License Plate Recognition with CLAHE)\n"
        "- ✅ MTMC (Multi-Camera Tracking & Spatio-Temporal ReID)\n"
        "- ✅ Alerts & Blacklist (Blacklist + Haversine Route Anomaly)\n"
        "- ✅ Traffic Analytics & GIS Heatmap\n"
        "- ✅ Trajectory Reconstruction\n"
        "- ✅ WebSockets & Redis Pub/Sub\n"
        "- ✅ Kafka Streaming Ingestion\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React frontend (and Swagger) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(events.router)
app.include_router(vehicles.router)
app.include_router(analytics.router)
app.include_router(alerts.router)
app.include_router(websocket.router)


# ── Optional feature routers ─────────────────────────────────────────────────
# These cover media jobs (video/photo ANPR), road-network routing and the
# benchmark suite. They are imported defensively and reported on startup:
# a missing or broken optional module degrades that one feature instead of
# taking the whole API down, which matters when the dashboard and the
# ingestion pipeline still need to serve during a live demo.
OPTIONAL_ROUTERS = [
    ("backend.api.jobs", "Media ANPR jobs (video/photo -> plates -> DB)"),
    ("backend.api.routing", "Road-network snapped trajectories"),
    ("backend.api.benchmarks", "Compute benchmarks & scalability projections"),
    ("backend.api.simulation", "Timeline clock: staged future -> live ingestion"),
]

FEATURE_STATUS: dict[str, str] = {}

for module_path, description in OPTIONAL_ROUTERS:
    short_name = module_path.rsplit(".", 1)[-1]
    try:
        module = __import__(module_path, fromlist=["router"])
        app.include_router(module.router)
        FEATURE_STATUS[short_name] = "ok"
        print(f"✅ {short_name:<11} enabled — {description}")
    except Exception as err:  # noqa: BLE001 - one bad feature must not kill the API
        FEATURE_STATUS[short_name] = f"unavailable: {type(err).__name__}: {err}"
        print(f"⚠️  {short_name:<11} DISABLED — {type(err).__name__}: {err}")


from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Static Frontend mounting
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    def _page(filename: str):
        """Serve a frontend page, falling back to the dashboard if absent."""
        target = FRONTEND_DIR / filename
        if not target.exists():
            target = FRONTEND_DIR / "index.html"
        return FileResponse(target)

    @app.get("/app", tags=["Frontend"])
    def serve_dashboard():
        """Operator dashboard: live GIS map, trajectories, alerts, analytics."""
        return _page("index.html")

    @app.get("/app/test", tags=["Frontend"])
    def serve_sandbox():
        """Public sandbox: upload a video / photo / dataset and watch it process."""
        return _page("test.html")

    @app.get("/app/benchmarks", tags=["Frontend"])
    def serve_benchmarks():
        """Measured compute cost and city-scale projections."""
        return _page("benchmarks.html")

    @app.get("/app/live", tags=["Frontend"])
    def serve_live():
        """
        Live ingestion monitor — its own window.

        Deliberately a separate page rather than a dashboard tab: during a
        demonstration this belongs on a second screen, running beside the
        dashboard, so the stream of sightings and the map reacting to it are
        visible at the same time.
        """
        return _page("live.html")


@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": app.version,
        "pages": {
            "dashboard": "/app",
            "sandbox": "/app/test",
            "benchmarks": "/app/benchmarks",
            "api_docs": "/docs",
        },
        "features": FEATURE_STATUS,
    }

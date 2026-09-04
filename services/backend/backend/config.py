"""
Application configuration.
All values come from environment variables (or .env file).
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./dev.db"  # swap for postgres in prod

    # ── ANPR ──────────────────────────────────────────────
    # Absolute path to the best.pt weights from the Automatic-License-Plate-Recognition repo
    ANPR_WEIGHTS_PATH: str = str(
        Path(__file__).parent.parent.parent
        / "alpr"
        / "best.pt"
    )
    # Root of the Automatic-License-Plate-Recognition checkout. Used only as a
    # fallback: if `import alpr` fails, `<repo>/src` is added to sys.path so a
    # sibling checkout works without a pip install.
    #
    # This is not belt-and-braces. An editable install writes an absolute path
    # into a .pth file inside the venv, so *moving the project directory* —
    # this one was created under ~/Desktop/SIH and now lives under
    # ~/Documents/SIH — silently breaks the import, and the only symptom is
    # every video job failing with "No module named 'alpr'". Resolving the
    # repo relative to this file survives that.
    ANPR_REPO_PATH: str = str(
        Path(__file__).parent.parent.parent / "alpr"
    )
    ANPR_DEVICE: str | None = None  # None = auto (MPS on Apple Silicon, CUDA, then CPU)
    ANPR_CONFIDENCE: float = 0.25
    ANPR_REGION: str | None = "IN"  # Indian plates

    # ── Speed enforcement ─────────────────────────────────
    # Default city speed limit used by the real-time SPEED_VIOLATION alert
    # (checkpoint-pair speed: distance/time between two consecutive camera
    # sightings of the same vehicle). Override per-deployment via .env.
    DEFAULT_SPEED_LIMIT_KMH: float = 60.0
    # Above this, the reading is treated as physically implausible for a
    # camera-to-camera hop (clock skew, duplicate plate, bad match) rather
    # than a real speeding violation — it's flagged as ROUTE_ANOMALY instead.
    MAX_PLAUSIBLE_SPEED_KMH: float = 150.0

    # ── Redis & Caching ───────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    USE_REDIS: bool = False

    # ── Kafka Message Broker ──────────────────────────────
    KAFKA_BROKER: str = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "raw_vehicle_events"
    KAFKA_TOPIC_SNAPSHOTS: str = "traffic_snapshots"
    USE_KAFKA: bool = False

    # ── Auth / misc ───────────────────────────────────────
    APP_NAME: str = "Vehicle Intelligence Backend"
    DEBUG: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()

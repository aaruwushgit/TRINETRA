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

    # ── ANPR throughput ───────────────────────────────────
    # Detector input resolution. Inference cost scales with the square of this,
    # so 512 is ~1.6x faster than the 640 the model was trained at, for a small
    # recall loss concentrated in the smallest plates. Must be a multiple of 32.
    # Raise back to 640 if plate recall matters more than framerate.
    ANPR_IMGSZ: int = 512
    # Prefer a pre-exported best.onnx over best.pt when running on CPU.
    #
    # OFF by default because it was measured and it LOST. On arm64 (Apple
    # Silicon, and the linux/arm64 container that runs on it) onnxruntime's CPU
    # provider is ~26% slower than torch for this model — 24.3 vs 32.7 fps at
    # imgsz 512, 15.5 vs 21.2 at 640, medians of 3x60 real 1080p frames. Torch
    # on arm64 has the better NEON kernels here.
    #
    # Kept as a switch rather than deleted: onnxruntime's x86_64 provider is a
    # different and much stronger implementation, so this may well win on an
    # Intel/AMD deployment. Measure before enabling — do not assume.
    ANPR_PREFER_ONNX: bool = False
    # Default frames-per-processed-frame for video jobs. Requests may override
    # per job; this is only the fallback when they do not.
    ANPR_STRIDE: int = 3
    # Frames handed to the detector per call. The only throughput knob here
    # with no accuracy cost — same weights, same resolution, same frames, just
    # fewer dispatches. Measured 1.84x at 8 vs 1 on linux/arm64 CPU. Costs
    # memory (8 decoded frames at once) and makes cancellation batch-granular,
    # so live camera workers should override this to 1 to avoid adding lag.
    ANPR_BATCH: int = 8

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

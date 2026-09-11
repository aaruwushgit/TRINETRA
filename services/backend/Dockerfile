# Built with the repo root as build context (see ../../docker-compose.yml),
# so this can COPY the sibling services/alpr checkout in alongside the
# backend and install it as a real dependency — not just API stubs.
#
# Multi-stage: everything compiled/downloaded lives in `builder` and only the
# resulting virtualenv is copied into the runtime image, so the compiler
# toolchain (build-essential, git) never ships.
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch, installed before anything that depends on it. Plain
# `pip install torch` resolves the CUDA-enabled wheel — including the full
# nvidia-*/triton runtime (~4 GB) — regardless of host architecture, because
# torch's own dependency markers gate on `platform_system == "Linux"`, not on
# GPU availability. This container runs ANPR_DEVICE=cpu and there is never a
# GPU to target, so pulling in that runtime is pure waste.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Backend Python requirements
COPY services/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ANPR service, installed as an editable-equivalent package (with OCR extra)
# so `import alpr` inside the backend resolves to real detection/OCR code.
COPY services/alpr /app/alpr
RUN pip install --no-cache-dir "/app/alpr[ocr]"

# No ONNX export here on purpose. It was the obvious move and it lost: on
# arm64 onnxruntime's CPU provider runs this model ~26% slower than torch
# (see ANPR_PREFER_ONNX in backend/config.py for the numbers). Exporting would
# add build time and 42 MB to the image for a pessimisation. `alpr.detect
# .export_onnx` is still there for an x86_64 deployment, where ORT may win —
# but enable it from a measurement, not from this comment.

# ── Runtime image ────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Only the shared-library runtime deps OpenCV/PaddleOCR need to execute —
# no compiler, no git, no libpq (nothing connects to Postgres; see
# requirements.txt).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Backend source (placed at /app so /app/backend/backend/config.py's
# `parent.parent.parent / "alpr"` resolves to /app/alpr, exactly like the
# sibling-checkout layout on a dev machine)
COPY services/alpr /app/alpr
COPY services/backend/. /app/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

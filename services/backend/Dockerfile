# Built with the repo root as build context (see ../../docker-compose.yml),
# so this can COPY the sibling services/alpr checkout in alongside the
# backend and install it as a real dependency — not just API stubs.
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV, PaddleOCR and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Backend Python requirements
COPY services/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ANPR service, installed as an editable-equivalent package (with OCR extra)
# so `import alpr` inside the backend resolves to real detection/OCR code.
COPY services/alpr /app/alpr
RUN pip install --no-cache-dir "/app/alpr[ocr]"

# Backend source (placed at /app so /app/backend/backend/config.py's
# `parent.parent.parent / "alpr"` resolves to /app/alpr, exactly like the
# sibling-checkout layout on a dev machine)
COPY services/backend/. /app/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

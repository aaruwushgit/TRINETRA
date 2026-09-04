"""
ANPR Service — wraps the Automatic-License-Plate-Recognition repo.

This is the abstraction layer. The rest of the backend never imports
from the ANPR repo directly — it only calls process_frame() and gets back
the standard PlateResult. This means you can swap the underlying OCR
engine without touching any other code.

SETUP:
  Preferred — install the ANPR repo into the same Python environment:
    pip install -e /path/to/Automatic-License-Plate-Recognition

  Fallback — point ANPR_REPO_PATH at the repo root and `<repo>/src` is added
  to sys.path on first use. This defaults to the sibling checkout, so a fresh
  clone works with no setup, and it is what keeps the service running when an
  editable install goes stale (that install pins an absolute path inside the
  venv, so moving the project directory breaks it silently).

MODEL LOADING IS THE EXPENSIVE PART. PaddleOCR's recognition model takes
several seconds to construct and YOLO weights another second; per-frame
inference is ~25 ms. So the detector and reader are process-wide singletons
built on first use and never rebuilt — a per-request construction would make
every API call look 100x slower than the model actually is. Construction is
guarded by a lock because FastAPI serves sync endpoints on a thread pool and
two concurrent first-requests would otherwise each load their own copy.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.config import get_settings

settings = get_settings()


def _ensure_alpr_importable() -> None:
    """Put the ALPR repo's `src` on sys.path if `alpr` is not already importable.

    Preferred route is still a proper `pip install -e` — this does not replace
    it. What it does is survive the failure mode that install has: an editable
    install writes an *absolute* path into a .pth inside the venv, so moving the
    project on disk breaks it with no warning. Every video job then fails with
    "No module named 'alpr'" and nothing points at the real cause.

    Idempotent, and a no-op when the import already works, so a correctly
    installed environment is unaffected.
    """
    import importlib.util

    if importlib.util.find_spec("alpr") is not None:
        return

    src = Path(settings.ANPR_REPO_PATH) / "src"
    if not src.is_dir():
        return  # nothing to add; _initialize() raises with a useful message

    path = str(src)
    if path not in sys.path:
        sys.path.insert(0, path)
        # find_spec caches negative results per finder; invalidate so the
        # freshly added directory is actually searched.
        importlib.invalidate_caches()


@dataclass
class PlateResult:
    """Standard output from the ANPR service — the contract."""
    plate: str | None
    confidence: float
    raw_text: str | None = None  # unformatted OCR output, useful for debugging


class ANPRService:
    """
    Singleton-style ANPR adapter.

    Loads the YOLO plate detector and OCR reader once, on first use rather than
    at import, so importing this module (which `backend.main` does at startup)
    never blocks on model files that may not be present yet.
    """

    def __init__(self) -> None:
        self._detector = None
        self._reader = None
        self._pipeline_config = None
        self._region = None
        self._initialized = False
        # Guards construction only. Once built, the detector and reader are
        # read-only and safe to call from several threads.
        self._init_lock = threading.Lock()

    # ── model lifecycle ──────────────────────────────────────────────────

    def _initialize(self) -> None:
        """Lazy init — loads models on first use, not on import."""
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:  # another thread won the race
                return

            weights = Path(settings.ANPR_WEIGHTS_PATH)
            if not weights.exists():
                raise FileNotFoundError(
                    f"ANPR weights not found at {weights}.\n"
                    "Fix by either:\n"
                    "  1. setting ANPR_WEIGHTS_PATH in .env to the absolute path of best.pt, or\n"
                    "  2. placing best.pt at the path above (it ships with the "
                    "Automatic-License-Plate-Recognition repo).\n"
                    "Nothing else in the backend needs to change — models load lazily."
                )

            _ensure_alpr_importable()

            try:
                from alpr.data.schema import Region
                from alpr.detect import PlateDetector
                from alpr.ocr import PlateReader
                from alpr.pipeline import PipelineConfig
            except ImportError as e:
                src = Path(settings.ANPR_REPO_PATH) / "src"
                raise ImportError(
                    "Could not import the ALPR library.\n"
                    f"Looked for it on sys.path and at {src} "
                    f"(exists: {src.exists()}).\n"
                    "Fix by either:\n"
                    "  1. setting ANPR_REPO_PATH to the repo root, or\n"
                    "  2. installing it into THIS venv:\n"
                    "     .venv/bin/python -m pip install -e "
                    "/path/to/Automatic-License-Plate-Recognition\n"
                    f"Original error: {e}"
                ) from e

            self._region = Region(settings.ANPR_REGION) if settings.ANPR_REGION else None

            self._detector = PlateDetector(
                weights=str(weights),
                device=settings.ANPR_DEVICE,  # None => auto: CUDA, then MPS, then CPU
                confidence=settings.ANPR_CONFIDENCE,
            )

            # PlateReader() with no arguments uses alpr's default Preprocess(),
            # which applies padding only. That is deliberate — see
            # preprocess_plate_crop() below for the measurement.
            self._reader = PlateReader()

            # A single-frame API call has no track history, so every threshold
            # that exists to accumulate evidence across frames drops to 1.
            self._pipeline_config = PipelineConfig(
                region=self._region,
                ocr_every=1,
                min_reads=1,
                min_hits=1,
                confidence=settings.ANPR_CONFIDENCE,
            )
            self._initialized = True

    @property
    def detector(self):
        """The shared PlateDetector, loading models if needed."""
        self._initialize()
        return self._detector

    @property
    def reader(self):
        """The shared PlateReader, loading models if needed."""
        self._initialize()
        return self._reader

    @property
    def region(self):
        """The configured plate-grammar region (alpr Region enum), or None."""
        self._initialize()
        return self._region

    def base_config(self):
        """A copy of the single-frame PipelineConfig, for callers to adjust."""
        self._initialize()
        from dataclasses import replace

        return replace(self._pipeline_config)

    def build_pipeline(self, config: Any | None = None):
        """A Pipeline bound to the *shared* detector and reader.

        Callers that process video build one of these per job. Constructing a
        Pipeline is free — it only holds references — so each job gets its own
        tracker and voter state while the multi-second model load is paid once
        for the whole process.
        """
        self._initialize()
        from alpr.pipeline import Pipeline

        return Pipeline(self._detector, self._reader, config or self._pipeline_config)

    def warmup(self) -> dict[str, Any]:
        """Force model loading and run one tiny inference.

        Worth calling before a demo: it moves the several-second model load off
        the first user-visible request, and it turns a broken install into an
        error at a time you can still fix it.
        """
        self._initialize()
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self._detector.detect(blank)
        # Touch the OCR model too — it is the slow one, and detecting on a blank
        # frame would otherwise never reach it.
        _ = self._reader.model
        return {
            "ready": True,
            "weights": str(settings.ANPR_WEIGHTS_PATH),
            "device": self._detector.device,
            "region": str(self._region) if self._region else None,
        }

    # ── preprocessing (opt-in; measured to hurt) ─────────────────────────

    @staticmethod
    def preprocess_plate_crop(crop: np.ndarray) -> np.ndarray:
        """
        Upscale + CLAHE + bilateral denoise on a plate crop.
        **NOT used by default, because measurement says it makes accuracy worse.**

        This looked like free accuracy — plate crops are small and low-contrast,
        and the recognition model wants ~48px-tall text. The upstream repo's
        ablation over 124 hand-labelled real crops says otherwise (lower CER is
        better):

            raw (control)                CER 0.2291   <- best
            upscale only                     0.2301
            gray + contrast                  0.2380
            upscale + gray + contrast        0.2410
            + sharpen                        0.2500

        The reason is that PaddleOCR already resizes and normalizes every crop
        to its own input spec. Enhancing first means resampling twice, and the
        second pass destroys detail the model would have used. See the module
        docstring of `alpr.ocr` for the full write-up.

        Kept — not deleted — because a future OCR backend that does *not*
        normalize internally would need exactly this, and because deleting the
        evidence would invite someone to re-add it on the same wrong intuition.
        Nothing in this service calls it.
        """
        import cv2

        if crop is None or crop.size == 0:
            return crop

        try:
            h, w = crop.shape[:2]
            if h < 40 or w < 100:
                scale = max(40.0 / h, 100.0 / w)
                crop = cv2.resize(
                    crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
                )

            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_lab = cv2.merge((clahe.apply(l), a, b))
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=50, sigmaSpace=50)
        except Exception:
            # A preprocessing failure must never lose the crop — the raw one
            # reads better than nothing, and (per the table above) better than
            # the enhanced one anyway.
            return crop

    # ── inference ────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> PlateResult:
        """
        Run ANPR on a single video frame (numpy BGR array from OpenCV).

        Returns the best plate detection, or PlateResult(plate=None, confidence=0.0)
        if nothing is found.

        `confidence` is min(detector, OCR) when text was read: localizing a
        plate and reading it are separate pieces of evidence, and the weaker one
        has to govern — a crisp box around unreadable characters is not a
        confident plate. This matches how alpr.pipeline scores its own rows.
        """
        self._initialize()

        if frame is None or getattr(frame, "size", 0) == 0:
            return PlateResult(plate=None, confidence=0.0)

        detections = self._detector.detect(frame)
        if not detections:
            return PlateResult(plate=None, confidence=0.0)

        # detect() returns highest-confidence first.
        best = detections[0]

        # PlateReader.read() takes the WHOLE FRAME as a PIL RGB image plus the
        # normalized detection, and crops internally (with padding, which a
        # hand-made crop would lose). Passing a pre-cropped BGR array — as this
        # method used to — raises inside the reader, and the old bare
        # `except Exception` swallowed it, so *every* plate came back None.
        # Reading a full frame per call is not wasteful: crop_plate is a cheap
        # PIL view compared with the recognition pass.
        from PIL import Image

        image = Image.fromarray(frame[:, :, ::-1])  # BGR (OpenCV) -> RGB (PIL)

        try:
            read = self._reader.read(image, best)
        except Exception as exc:
            # Narrowly reported rather than silently swallowed: an OCR failure
            # here is an integration bug, and hiding it is what caused BUG #1.
            return PlateResult(
                plate=None,
                confidence=float(best.confidence),
                raw_text=f"__ocr_error__: {type(exc).__name__}: {exc}",
            )

        raw_text = read.text or None
        if not raw_text:
            return PlateResult(plate=None, confidence=float(best.confidence), raw_text=None)

        confidence = min(float(best.confidence), float(read.confidence))

        # Run the country grammar so a single-frame read gets the same
        # normalization and OCR-confusion repair (O/0, I/1) that video rows get.
        plate = self._validate(raw_text)

        return PlateResult(plate=plate, confidence=confidence, raw_text=raw_text)

    def _validate(self, raw_text: str) -> str:
        """Grammar-normalize a raw OCR string, falling back to a clean version.

        A grammar rejection is informative for video — the pipeline drops those
        rows as false positives, and it can afford to because another frame is
        coming. A single uploaded photo has no other frame, so returning None
        would turn "we read it but the format is unusual" into "we found
        nothing". The raw reading is returned normalized instead, and the caller
        still sees the unmodified OCR output in `raw_text`.
        """
        try:
            from alpr.plates import parse

            match = parse(raw_text, region=self._region)
            if match is not None:
                return match.text
        except Exception:
            pass

        # Same normalization the /events/ingest schema applies, so a plate read
        # here matches one ingested there byte for byte.
        return "".join(ch for ch in raw_text.upper() if ch.isalnum())

    def process_image_path(self, image_path: str | Path) -> PlateResult:
        """
        Convenience: process a file path instead of a numpy array.
        Useful for testing and for the HTTP upload endpoint.
        """
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(
                f"Could not read image: {image_path} "
                "(missing file, or a format OpenCV cannot decode)"
            )
        return self.process_frame(frame)

    def process_frame_all(self, frame: np.ndarray) -> list[PlateResult]:
        """Every plate in a frame, not just the best one.

        A single-plate return is right for "test my ANPR" but wrong for a
        traffic frame with four cars in it. One PIL conversion is shared across
        all detections.
        """
        self._initialize()

        if frame is None or getattr(frame, "size", 0) == 0:
            return []
        detections = self._detector.detect(frame)
        if not detections:
            return []

        from PIL import Image

        image = Image.fromarray(frame[:, :, ::-1])
        out: list[PlateResult] = []
        for detection in detections:
            try:
                read = self._reader.read(image, detection)
            except Exception:
                continue
            if not read.text:
                continue
            out.append(
                PlateResult(
                    plate=self._validate(read.text),
                    confidence=min(float(detection.confidence), float(read.confidence)),
                    raw_text=read.text,
                )
            )
        return out


# Single shared instance — import this everywhere
anpr_service = ANPRService()

"""Plate detection inference.

A thin, device-aware wrapper over the trained Ultralytics model. It exists so
the rest of the pipeline never imports ultralytics directly: Phase 6 tracks
`Detection` objects, Phase 9 runs the same class on Apple Silicon, and neither
should care which library produced the boxes.

Coordinates are normalized here, matching `alpr.data.schema.PlateBox`, so a
detection and a ground-truth annotation are directly comparable without
carrying image dimensions around.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45


class DetectionError(RuntimeError):
    """Raised when the detector cannot be built or run."""


@dataclass(frozen=True)
class Detection:
    """One detected plate, in normalized corner coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def pixel_box(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        return (
            int(self.x1 * image_width),
            int(self.y1 * image_height),
            int(self.x2 * image_width),
            int(self.y2 * image_height),
        )

    def pixel_width(self, image_width: int) -> float:
        return self.width * image_width


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two normalized xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def select_device(preference: str | None = None) -> str:
    """Pick the fastest available device.

    Order is CUDA, then Apple's Metal backend, then CPU. MPS matters: Phase 9
    runs this same detector locally on an M4, where CPU inference would not
    hold a live camera framerate.
    """
    if preference:
        return preference

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PlateDetector:
    """Runs the trained detector over images."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU,
        imgsz: int = 640,
    ) -> None:
        self.weights = Path(weights)
        if not self.weights.exists():
            raise DetectionError(
                f"weights not found: {self.weights}. Train with Phase 2, or point at "
                "a downloaded best.pt."
            )
        if not 0.0 < confidence <= 1.0:
            raise DetectionError(f"confidence must be in (0, 1], got {confidence}")

        self.device = select_device(device)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self._model: Any | None = None

    @property
    def model(self):
        """The underlying model, loaded on first use.

        Deferred so constructing a detector stays cheap — building one to read
        its configuration should not pay for loading 22 MB of weights.
        """
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    def detect(self, image: Any, **kwargs) -> list[Detection]:
        """Detect plates in a single image (path, array, or PIL image)."""
        return self.detect_batch([image], **kwargs)[0]

    def detect_batch(self, images: Sequence[Any], **kwargs) -> list[list[Detection]]:
        """Detect over several images, returning one list per input."""
        if not images:
            return []

        results = self.model.predict(
            source=list(images),
            conf=kwargs.pop("confidence", self.confidence),
            iou=kwargs.pop("iou", self.iou_threshold),
            imgsz=kwargs.pop("imgsz", self.imgsz),
            device=self.device,
            verbose=False,
            **kwargs,
        )

        out: list[list[Detection]] = []
        for result in results:
            height, width = result.orig_shape
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        x1=x1 / width,
                        y1=y1 / height,
                        x2=x2 / width,
                        y2=y2 / height,
                        confidence=float(box.conf[0]),
                    )
                )
            # Highest confidence first, so a caller taking the top detection
            # gets the best one without having to sort.
            detections.sort(key=lambda d: d.confidence, reverse=True)
            out.append(detections)
        return out

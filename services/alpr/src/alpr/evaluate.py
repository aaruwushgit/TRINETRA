"""Detector evaluation on the held-out test split.

Ultralytics reports aggregate mAP, and that number goes in the README. But an
aggregate hides the thing that decides whether the *pipeline* works: a plate
missed by the detector can never be read by OCR, and misses are not spread
evenly — they concentrate in small plates.

So this module does two things Ultralytics does not:

**Size slices.** Precision and recall bucketed by ground-truth plate width in
pixels. A model at 0.98 overall can still be at 0.6 on plates under 32 px, and
that is exactly the population Phase 4 will fail on.

**A failure gallery.** The worst images rendered with predictions and ground
truth drawn on them. Ranked metrics tell you *that* something is wrong; only
looking at the images tells you what.

Matching is greedy by descending confidence, which is what the COCO and VOC
protocols do: each prediction claims the best unclaimed ground-truth box above
the IoU threshold, and anything left over is a false positive or a miss.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from alpr.data.schema import ImageRecord
from alpr.detect import Detection, iou

DEFAULT_IOU_THRESHOLD = 0.5

# Buckets by ground-truth plate width in pixels. The 32 px boundary is the
# same one `alpr.data.stats` calls unreadable: below it, OCR has no strokes to
# work with even when detection succeeds.
SIZE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("tiny (<32px)", 0.0, 32.0),
    ("small (32-64px)", 32.0, 64.0),
    ("medium (64-128px)", 64.0, 128.0),
    ("large (>=128px)", 128.0, float("inf")),
)


@dataclass
class Counts:
    """Match counts for one population of boxes."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def ground_truth(self) -> int:
        return self.true_positives + self.false_negatives


@dataclass
class ImageResult:
    """How the detector did on one image."""

    image_id: str
    counts: Counts
    matched: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_detections: list[int] = field(default_factory=list)
    unmatched_truth: list[int] = field(default_factory=list)

    @property
    def errors(self) -> int:
        """Total mistakes — the ranking key for the failure gallery."""
        return self.counts.false_positives + self.counts.false_negatives


def match(
    truth: Sequence[Sequence[float]],
    predictions: Sequence[Detection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> ImageResult:
    """Greedily match predictions to ground truth, highest confidence first.

    Args:
        truth: ground-truth boxes as normalized xyxy.
        predictions: detections, in any order.
        iou_threshold: overlap required to count as the same plate.

    Returns:
        Which predictions matched which truth, and what was left over.
    """
    ordered = sorted(range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True)

    claimed: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    unmatched_detections: list[int] = []

    for prediction_index in ordered:
        prediction = predictions[prediction_index]
        box = (prediction.x1, prediction.y1, prediction.x2, prediction.y2)

        best_index, best_iou = -1, iou_threshold
        for truth_index, truth_box in enumerate(truth):
            if truth_index in claimed:
                continue
            overlap = iou(box, truth_box)
            if overlap >= best_iou:
                best_index, best_iou = truth_index, overlap

        if best_index >= 0:
            claimed.add(best_index)
            matched.append((prediction_index, best_index, best_iou))
        else:
            unmatched_detections.append(prediction_index)

    unmatched_truth = [i for i in range(len(truth)) if i not in claimed]

    return ImageResult(
        image_id="",
        counts=Counts(
            true_positives=len(matched),
            false_positives=len(unmatched_detections),
            false_negatives=len(unmatched_truth),
        ),
        matched=matched,
        unmatched_detections=unmatched_detections,
        unmatched_truth=unmatched_truth,
    )


def bucket_for(width_px: float) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= width_px < high:
            return name
    return SIZE_BUCKETS[-1][0]


@dataclass
class EvaluationReport:
    """Aggregate and sliced results over a split."""

    overall: Counts
    by_size: dict[str, Counts]
    per_image: list[ImageResult]
    iou_threshold: float = DEFAULT_IOU_THRESHOLD

    def worst(self, n: int = 20) -> list[ImageResult]:
        """The images with the most errors — what the gallery renders."""
        return sorted(self.per_image, key=lambda r: (-r.errors, r.image_id))[:n]

    def report(self) -> str:
        lines = [
            f"IoU threshold      {self.iou_threshold}",
            f"images             {len(self.per_image)}",
            f"ground-truth       {self.overall.ground_truth}",
            f"true positives     {self.overall.true_positives}",
            f"false positives    {self.overall.false_positives}",
            f"missed             {self.overall.false_negatives}",
            f"precision          {self.overall.precision:.4f}",
            f"recall             {self.overall.recall:.4f}",
            f"f1                 {self.overall.f1:.4f}",
            "",
            f"{'by plate width':<20} {'n':>6} {'precision':>10} {'recall':>8}",
        ]
        for name, _, _ in SIZE_BUCKETS:
            counts = self.by_size.get(name)
            if not counts or not counts.ground_truth:
                continue
            lines.append(
                f"{name:<20} {counts.ground_truth:>6} "
                f"{counts.precision:>10.4f} {counts.recall:>8.4f}"
            )
        return "\n".join(lines)


def evaluate_records(
    records: Sequence[ImageRecord],
    predictions: Sequence[Sequence[Detection]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> EvaluationReport:
    """Score predictions against annotated records.

    Args:
        records: ground truth, one per image.
        predictions: detections, aligned with `records`.
        iou_threshold: overlap required to count as a hit.

    Raises:
        ValueError: when the two sequences are different lengths — a silent
            misalignment would produce plausible but meaningless numbers.
    """
    if len(records) != len(predictions):
        raise ValueError(
            f"{len(records)} record(s) but {len(predictions)} prediction list(s); "
            "they must be aligned"
        )

    overall = Counts()
    by_size: dict[str, Counts] = {name: Counts() for name, _, _ in SIZE_BUCKETS}
    per_image: list[ImageResult] = []

    for record, detections in zip(records, predictions, strict=True):
        truth_boxes = [box.xyxy for box in record.boxes]
        result = match(truth_boxes, detections, iou_threshold)
        result.image_id = record.image_id
        per_image.append(result)

        overall.true_positives += result.counts.true_positives
        overall.false_positives += result.counts.false_positives
        overall.false_negatives += result.counts.false_negatives

        # Slice by ground-truth size. False positives have no ground-truth box
        # to size, so they are attributed to the image's largest plate — or
        # skipped entirely on an image with no plates at all.
        for _, truth_index, _ in result.matched:
            width_px = record.boxes[truth_index].pixel_size(record.width, record.height)[0]
            by_size[bucket_for(width_px)].true_positives += 1
        for truth_index in result.unmatched_truth:
            width_px = record.boxes[truth_index].pixel_size(record.width, record.height)[0]
            by_size[bucket_for(width_px)].false_negatives += 1
        if result.unmatched_detections and record.boxes:
            widest = max(b.pixel_size(record.width, record.height)[0] for b in record.boxes)
            by_size[bucket_for(widest)].false_positives += len(result.unmatched_detections)

    return EvaluationReport(
        overall=overall,
        by_size=by_size,
        per_image=per_image,
        iou_threshold=iou_threshold,
    )


def render_failure(
    record: ImageRecord,
    detections: Sequence[Detection],
    result: ImageResult,
    image_root: str | Path,
    out_path: str | Path,
) -> Path | None:
    """Draw ground truth and predictions on one image.

    Green is a correct detection, red a miss, orange a false positive. Colour
    is the point: a page of these makes the failure mode obvious in a way a
    precision number never does.
    """
    from PIL import Image, ImageDraw

    image_root = Path(image_root)
    source = image_root / record.file_name if record.file_name else None
    if source is None or not source.exists():
        return None

    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def rect(box, colour, label):
        x1, y1, x2, y2 = box
        draw.rectangle([x1 * width, y1 * height, x2 * width, y2 * height], outline=colour, width=3)
        draw.text((x1 * width + 4, max(0, y1 * height - 12)), label, fill=colour)

    matched_truth = {truth_index for _, truth_index, _ in result.matched}
    for index, box in enumerate(record.boxes):
        if index in matched_truth:
            rect(box.xyxy, (60, 200, 60), "gt")
        else:
            rect(box.xyxy, (230, 60, 60), "MISSED")

    for index in result.unmatched_detections:
        d = detections[index]
        rect((d.x1, d.y1, d.x2, d.y2), (250, 150, 40), f"FP {d.confidence:.2f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def render_failure_gallery(
    report: EvaluationReport,
    records: Sequence[ImageRecord],
    predictions: Sequence[Sequence[Detection]],
    image_root: str | Path,
    out_dir: str | Path,
    *,
    limit: int = 20,
) -> list[Path]:
    """Render the worst images to disk. Returns the files written."""
    by_id = {record.image_id: (index, record) for index, record in enumerate(records)}
    out_dir = Path(out_dir)
    written: list[Path] = []

    for rank, result in enumerate(report.worst(limit), start=1):
        if result.errors == 0:
            break  # nothing left worth looking at
        entry = by_id.get(result.image_id)
        if entry is None:
            continue
        index, record = entry
        path = render_failure(
            record,
            predictions[index],
            result,
            image_root,
            out_dir / f"{rank:02d}_{result.errors}err_{record.image_id}.jpg",
        )
        if path:
            written.append(path)
    return written

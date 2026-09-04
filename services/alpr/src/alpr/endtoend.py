"""End-to-end evaluation: what fraction of plates the whole system gets right.

Phase 4 measured OCR on ground-truth crops, which isolates reading from
finding. This measures the pipeline a deployer actually runs — detect, crop,
read, validate — where a failure at any stage is a failure overall.

**The headline number is less useful than the breakdown.** "62% end-to-end" does
not say where to spend the next week. So every plate is attributed to the stage
that lost it:

    CORRECT             the pipeline produced exactly the right string
    DETECTION_MISS      no box overlapped the plate; nothing downstream could help
    OCR_ERROR           found, but read wrong, and the grammar could not repair it
    GRAMMAR_REJECTED    read something the grammar refused, so nothing was logged
    GRAMMAR_CORRUPTED   OCR was right and the grammar made it wrong

That last category is the one worth building the taxonomy for. A correction
step that damages correct reads is worse than no correction step, and it is
invisible in an aggregate score — the losses hide inside the same number as
the gains.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from alpr.data.schema import ImageRecord, Region
from alpr.detect import Detection, iou

DEFAULT_IOU_THRESHOLD = 0.5


class Outcome(StrEnum):
    CORRECT = "correct"
    DETECTION_MISS = "detection_miss"
    OCR_ERROR = "ocr_error"
    GRAMMAR_REJECTED = "grammar_rejected"
    GRAMMAR_CORRUPTED = "grammar_corrupted"


@dataclass(frozen=True)
class PlateOutcome:
    """What the pipeline did with one labelled plate."""

    crop_id: str
    truth: str
    outcome: Outcome
    raw_text: str = ""
    final_text: str = ""
    detection_confidence: float = 0.0

    @property
    def correct(self) -> bool:
        return self.outcome is Outcome.CORRECT


@dataclass
class EndToEndReport:
    outcomes: list[PlateOutcome] = field(default_factory=list)
    iou_threshold: float = DEFAULT_IOU_THRESHOLD

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(o.correct for o in self.outcomes)

    @property
    def accuracy(self) -> float:
        """Plates the whole pipeline got exactly right."""
        return self.correct / self.total if self.total else 0.0

    @property
    def reported(self) -> int:
        """Plates the pipeline produced a string for.

        A rejected read is not reported, which is the point of rejecting it —
        so it counts against recall but not against precision.
        """
        return sum(
            1
            for o in self.outcomes
            if o.outcome is not Outcome.GRAMMAR_REJECTED and o.outcome is not Outcome.DETECTION_MISS
        )

    @property
    def precision(self) -> float:
        """Of the plates logged, how many were right."""
        return self.correct / self.reported if self.reported else 0.0

    @property
    def recall(self) -> float:
        """Of the plates present, how many were logged correctly."""
        return self.accuracy

    def counts(self) -> dict[str, int]:
        return dict(Counter(o.outcome.value for o in self.outcomes))

    def report(self) -> str:
        counts = self.counts()
        lines = [
            f"plates              {self.total}",
            f"correct             {self.correct}",
            "",
            f"end-to-end accuracy {self.accuracy:.4f}",
            f"precision           {self.precision:.4f}   (of {self.reported} logged)",
            f"recall              {self.recall:.4f}",
            "",
            "where the losses happen:",
        ]
        for outcome in Outcome:
            n = counts.get(outcome.value, 0)
            if not n:
                continue
            share = n / self.total if self.total else 0
            lines.append(f"  {outcome.value:<20} {n:>4}  {share:>6.1%}")

        corrupted = counts.get(Outcome.GRAMMAR_CORRUPTED.value, 0)
        if corrupted:
            lines += [
                "",
                f"WARNING: the grammar broke {corrupted} read(s) that OCR got right. "
                "A correction step that damages correct reads is worse than none.",
            ]
        return "\n".join(lines)

    def failures(self, outcome: Outcome, limit: int = 10) -> list[PlateOutcome]:
        return [o for o in self.outcomes if o.outcome is outcome][:limit]


def classify(
    truth: str,
    raw_text: str,
    final_text: str | None,
) -> Outcome:
    """Attribute one plate's result to the stage that lost it.

    Args:
        truth: the hand-labelled plate.
        raw_text: what OCR returned before any grammar.
        final_text: what the grammar produced, or None if it rejected the read.
    """
    if final_text is None:
        return Outcome.GRAMMAR_REJECTED
    if final_text == truth:
        return Outcome.CORRECT
    # OCR had it and the grammar changed it into something wrong. Distinct
    # from an ordinary OCR error because the fix is to loosen the grammar,
    # not to improve the reader.
    if raw_text == truth:
        return Outcome.GRAMMAR_CORRUPTED
    return Outcome.OCR_ERROR


def best_detection_for(
    box_xyxy: Sequence[float],
    detections: Sequence[Detection],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Detection | None:
    """The detection that best covers a ground-truth box, if any does."""
    best, best_iou = None, iou_threshold
    for detection in detections:
        overlap = iou(box_xyxy, (detection.x1, detection.y1, detection.x2, detection.y2))
        if overlap >= best_iou:
            best, best_iou = detection, overlap
    return best


def evaluate_end_to_end(
    records: Sequence[ImageRecord],
    labels: dict[str, str],
    detector,
    reader,
    *,
    image_root: str | Path,
    region: Region | None = None,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    parse=None,
) -> EndToEndReport:
    """Run detect → crop → read → validate over labelled plates.

    Args:
        records: the dataset records the labels refer to.
        labels: `{crop_id: plate text}` from `alpr.label`.
        detector: anything with `.detect(image)` returning `Detection`s.
        reader: anything with `.read(image, detection)` returning an `OcrResult`.
        image_root: where the frames live.
        region: restrict the grammar to one country.
        parse: injected for testing; defaults to `alpr.plates.parse`.
    """
    from PIL import Image

    if parse is None:
        from alpr.plates import parse as parse

    image_root = Path(image_root)
    by_id = {r.image_id: r for r in records}
    report = EndToEndReport(iou_threshold=iou_threshold)

    for crop_id, truth in labels.items():
        image_id, _, index_text = crop_id.rpartition("#")
        record = by_id.get(image_id)
        if record is None or not record.file_name:
            continue
        box_index = int(index_text)
        if box_index >= len(record.boxes):
            continue

        path = image_root / record.file_name
        if not path.exists():
            continue

        with Image.open(path) as image:
            image = image.convert("RGB")
            detections = detector.detect(image)
            detection = best_detection_for(record.boxes[box_index].xyxy, detections, iou_threshold)

            if detection is None:
                # Nothing downstream can recover a plate that was never found.
                report.outcomes.append(
                    PlateOutcome(crop_id=crop_id, truth=truth, outcome=Outcome.DETECTION_MISS)
                )
                continue

            raw = reader.read(image, detection).text

        match = parse(raw, region=region) if raw else None
        final = match.text if match else None

        report.outcomes.append(
            PlateOutcome(
                crop_id=crop_id,
                truth=truth,
                outcome=classify(truth, raw, final),
                raw_text=raw,
                final_text=final or "",
                detection_confidence=detection.confidence,
            )
        )

    return report

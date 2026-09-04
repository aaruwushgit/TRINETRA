"""Dataset statistics.

The Phase 1 exit criteria are numbers, not vibes: how many plates, whether both
regions are represented, and how many plates are too small for OCR to have a
chance. This module produces them.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from alpr.data.schema import ImageRecord, Region, Split
from alpr.data.split import SplitAssignment

# Below roughly this width, a plate is a smear: PaddleOCR has no strokes to
# work with and Phase 4 will fail on it no matter how good the detector is.
# Used to report the share of the dataset that is unreadable in principle.
MIN_READABLE_PLATE_PX = 32


@dataclass(frozen=True)
class DatasetStats:
    images: int
    plates: int
    images_without_plates: int
    by_region: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_split: dict[str, int] = field(default_factory=dict)
    plates_by_split: dict[str, int] = field(default_factory=dict)
    groups: int = 0
    with_text: int = 0
    tiny_plates: int = 0
    median_plate_px: tuple[float, float] = (0.0, 0.0)
    plates_per_image: float = 0.0

    def report(self) -> str:
        """Human-readable summary, for the notebook and the PR description."""
        lines = [
            f"images            {self.images}",
            f"plates            {self.plates}  ({self.plates_per_image:.2f} per image)",
            f"groups            {self.groups}",
            f"background images {self.images_without_plates}",
            f"with OCR text     {self.with_text}",
            f"median plate      {self.median_plate_px[0]:.0f} x {self.median_plate_px[1]:.0f} px",
            f"tiny (<{MIN_READABLE_PLATE_PX}px wide)  {self.tiny_plates}"
            + (f"  [{self.tiny_plates / self.plates:.1%} of plates]" if self.plates else ""),
        ]
        if self.by_region:
            lines.append("region            " + _fmt_counter(self.by_region))
        if self.by_source:
            lines.append("source            " + _fmt_counter(self.by_source))
        if self.by_split:
            lines.append("split (images)    " + _fmt_counter(self.by_split))
        if self.plates_by_split:
            lines.append("split (plates)    " + _fmt_counter(self.plates_by_split))
        return "\n".join(lines)


def _fmt_counter(counts: dict[str, int]) -> str:
    return "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def compute_stats(
    records: Sequence[ImageRecord],
    assignment: SplitAssignment | None = None,
) -> DatasetStats:
    """Summarize a dataset, optionally broken down by split."""
    regions: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    plates_by_split: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []

    plates = 0
    with_text = 0
    tiny = 0
    empty_images = 0
    groups: set[str] = set()

    for record in records:
        groups.add(record.group_key)
        if not record.boxes:
            empty_images += 1
        if record.source:
            sources[record.source] += 1

        split_name = assignment.of(record).value if assignment else None
        if split_name:
            by_split[split_name] += 1
            plates_by_split[split_name] += len(record.boxes)

        for box in record.boxes:
            plates += 1
            regions[box.region.value] += 1
            if box.text:
                with_text += 1
            px_w, px_h = box.pixel_size(record.width, record.height)
            widths.append(px_w)
            heights.append(px_h)
            if px_w < MIN_READABLE_PLATE_PX:
                tiny += 1

    median = (statistics.median(widths), statistics.median(heights)) if widths else (0.0, 0.0)

    return DatasetStats(
        images=len(records),
        plates=plates,
        images_without_plates=empty_images,
        by_region=dict(regions),
        by_source=dict(sources),
        by_split=dict(by_split),
        plates_by_split=dict(plates_by_split),
        groups=len(groups),
        with_text=with_text,
        tiny_plates=tiny,
        median_plate_px=median,
        plates_per_image=plates / len(records) if records else 0.0,
    )


def check_exit_criteria(
    stats: DatasetStats,
    *,
    min_plates: int = 1000,
    required_regions: Sequence[Region] = (Region.INDIA, Region.EUROPE),
) -> list[str]:
    """Return the Phase 1 exit criteria that are not met yet.

    Returns a list of failures rather than raising, so the notebook can print
    the whole picture instead of stopping at the first shortfall.

    `required_regions` asks for India and Europe, not India and Germany: the
    openly-licensed European set is pan-European, and demanding a `DE` tag
    would fail on data that is perfectly good for training a detector. German
    plate handling is Phase 5's grammar, which needs no training data.
    """
    failures = []
    if stats.plates < min_plates:
        failures.append(f"only {stats.plates} plates, need >= {min_plates}")
    for region in required_regions:
        if stats.by_region.get(region.value, 0) == 0:
            failures.append(f"no plates for region {region.value}")
    for split in Split:
        if stats.by_split and stats.by_split.get(split.value, 0) == 0:
            failures.append(f"split {split.value} is empty")
    return failures

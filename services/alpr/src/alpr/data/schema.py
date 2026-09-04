"""Dataset record types.

The manifest is the source of truth; the YOLO directory tree is a generated
artifact (see `alpr.data.export`). The reason is `PlateBox.text`: YOLO label
files hold only `class cx cy w h`, so a YOLO tree cannot carry the plate string
that Phase 4 needs to score OCR against. Keeping the manifest canonical means
detection labels and recognition labels stay in one file that cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# Normalized coordinates are floats, and a box written as 1.0 can come back as
# 1.0000000000000002 after a round-trip through JSON or a resize. Validation
# uses this tolerance so honest boxes are not rejected for float noise.
_EPS = 1e-6


class Region(StrEnum):
    """Plate-format region.

    `GERMANY` is the Phase 5 grammar target. `EUROPE` is deliberately separate
    and is what the pan-European training data is tagged with: that set holds
    French, Spanish, Italian and other EU plates, and labelling it `GERMANY`
    would be a false claim about the data. Detection does not care — a plate
    detector transfers across countries — and the DE-specific work is the
    grammar in Phase 5, which needs no training data at all.
    """

    INDIA = "IN"
    GERMANY = "DE"
    POLAND = "PL"
    EUROPE = "EU"
    UNKNOWN = "XX"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetError(ValueError):
    """Raised when a record or dataset violates an invariant."""


@dataclass(frozen=True)
class PlateBox:
    """One license plate, in YOLO normalized center form.

    Coordinates are fractions of image width/height, so a box survives resizing
    unchanged — which is why they are stored this way rather than in pixels.

    Attributes:
        cx, cy: box center, in [0, 1].
        w, h: box size, in (0, 1].
        text: ground-truth plate string, when known. Detection-only images
            leave this None; Phase 4 scores OCR only on boxes that have it.
        region: plate format, used to stratify splits and pick a validator.
    """

    cx: float
    cy: float
    w: float
    h: float
    text: str | None = None
    region: Region = Region.UNKNOWN

    def __post_init__(self) -> None:
        for name in ("cx", "cy", "w", "h"):
            value = getattr(self, name)
            if not isinstance(value, int | float):
                raise DatasetError(f"{name} must be a number, got {value!r}")
        if not (0.0 < self.w <= 1.0 + _EPS) or not (0.0 < self.h <= 1.0 + _EPS):
            raise DatasetError(f"degenerate box size w={self.w} h={self.h}")
        if not (-_EPS <= self.cx <= 1.0 + _EPS) or not (-_EPS <= self.cy <= 1.0 + _EPS):
            raise DatasetError(f"box center outside image: cx={self.cx} cy={self.cy}")

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Corner form (x1, y1, x2, y2), still normalized."""
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )

    @property
    def area(self) -> float:
        return self.w * self.h

    def pixel_size(self, image_width: int, image_height: int) -> tuple[float, float]:
        """Box size in pixels, for the small-object slice in Phase 3."""
        return self.w * image_width, self.h * image_height

    def clipped(self) -> PlateBox:
        """Return a copy with the box clamped inside the image.

        A plate cut off by the frame edge is legitimately annotated past the
        border; training wants it clipped, but the annotation itself is not
        wrong, so this is opt-in rather than enforced at construction.
        """
        x1, y1, x2, y2 = self.xyxy
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        return PlateBox(
            cx=(x1 + x2) / 2,
            cy=(y1 + y2) / 2,
            w=max(x2 - x1, _EPS),
            h=max(y2 - y1, _EPS),
            text=self.text,
            region=self.region,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h}
        if self.text is not None:
            d["text"] = self.text
        if self.region is not Region.UNKNOWN:
            d["region"] = self.region.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlateBox:
        return cls(
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            w=float(d["w"]),
            h=float(d["h"]),
            text=d.get("text"),
            region=Region(d.get("region", Region.UNKNOWN.value)),
        )


# --- Clip identity -----------------------------------------------------------
#
# Frames sampled from one video are near-duplicates, so they must never straddle
# a split. Recognising them is entirely a filename question, and the filenames
# arrive in more than one shape.
#
# **The export suffix has to come off first.** Roboflow renames every exported
# file to `<original>_<ext>.rf.<32-hex-md5>`. Every one of the 188 surviving
# real identifiers in this project carries it. The frame patterns below are
# anchored at the end of the string, so with that suffix still attached they
# match nothing at all — which is exactly why the frame rule never once fired on
# the real dataset, and why frames 1062 and 1063 of one clip sat on opposite
# sides of a train/test split while the rule that was supposed to stop it looked
# correct in its own unit tests.
_EXPORT_SUFFIX = re.compile(
    r"_(?:jpe?g|png|bmp|webp|tiff?)\.rf\.[0-9a-f]{32}$",
    re.IGNORECASE,
)


def strip_export_suffix(name: str) -> str:
    """Remove a Roboflow export suffix (`_jpg.rf.<32-hex>`), if present.

    Matched strictly — a specific image extension, the literal `.rf.`, and
    exactly 32 hex characters — so it cannot bite a filename that merely
    contains a dot or a long token.
    """
    return _EXPORT_SUFFIX.sub("", name)


# Every pattern must anchor on an explicit marker and capture the clip identity
# in a `clip` group. A bare trailing-number rule is deliberately absent: it would
# collapse `license_plate_205` and `license_plate_242` into one group and hand
# most of the dataset to a single split. Perceptual grouping in `alpr.dupes` is
# the general safety net; these only catch what a filename can prove.
_CLIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Roboflow's video export: `dayride_type1_001-mp4-t-1062`. A video container
    # extension followed by a `t` timecode.
    re.compile(
        r"^(?P<clip>.+?)[-_](?:mp4|avi|mov|mkv|webm)[-_]t[-_]\d+$",
        re.IGNORECASE,
    ),
    # The Indian source's convention: `video3_2190`, `video11_1870` — the
    # literal word `video`, its clip number, then an underscore and the frame
    # number. Scoped tightly on purpose:
    #   matches     video3_2190, video11_1870, in-train-video8_870
    #   no match    license_plate_242  (no `video`)
    #   no match    video_2190         (no clip number after `video`)
    #   no match    myvideo3_2190      (`\b` — `video` must start a word)
    #   no match    video3             (no frame number)
    #   no match    video3_2190_640    (frame number must end the string)
    re.compile(r"^(?P<clip>.*\bvideo\d+)_\d+$", re.IGNORECASE),
    # The classic convention: `clip_042_frame_0137`.
    re.compile(r"^(?P<clip>.+?)[-_]?frame[-_]?\d+$", re.IGNORECASE),
)


def clip_id(name: str) -> str | None:
    """The clip a frame filename belongs to, or None if it is not a frame.

    The export suffix is stripped first, so this works on raw exported names.
    Returns None for an ordinary still, which is what keeps unrelated
    photographs in separate split groups.
    """
    base = strip_export_suffix(name)
    for pattern in _CLIP_PATTERNS:
        found = pattern.match(base)
        if found:
            clip = found.group("clip")
            if clip:
                return clip
    return None


@dataclass(frozen=True)
class ImageRecord:
    """One annotated image.

    Attributes:
        image_id: unique, stable identifier; also the label file stem.
        width, height: pixel dimensions, needed to un-normalize boxes.
        boxes: every plate in the image. May be empty — background images with
            no plates are legitimate training data and suppress false positives.
        file_name: image path relative to the dataset image root. Kept separate
            from `image_id` because sources collide on basenames — two datasets
            both containing `001.jpg` need distinct ids but keep their own
            paths.
        group: split-grouping key. Frames sampled from one video are
            near-duplicates; if some land in train and others in val, val
            accuracy measures memorization. Everything sharing a group is
            forced into the same split.
        source: provenance (dataset name, capture session). Not cosmetic —
            the sources carry different licences (ODbL, CC BY, CC BY-NC-ND),
            and this is what lets us exclude the non-redistributable ones
            before publishing a derived dataset.
    """

    image_id: str
    width: int
    height: int
    boxes: tuple[PlateBox, ...] = ()
    file_name: str | None = None
    group: str | None = None
    source: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id:
            raise DatasetError("image_id must be non-empty")
        if self.width <= 0 or self.height <= 0:
            raise DatasetError(f"bad image size {self.width}x{self.height} for {self.image_id}")
        if not isinstance(self.boxes, tuple):
            object.__setattr__(self, "boxes", tuple(self.boxes))

    @property
    def group_key(self) -> str:
        """The key used for grouped splitting.

        An explicit `group` always wins — that is the channel ingest uses to
        state a clip identity that a bare image id cannot express, and the
        channel `alpr.dupes` uses to merge perceptual duplicates.

        Without one, the id is read as a filename: a recognised frame yields its
        clip, and anything else is its own group. That makes an ungrouped
        dataset behave like a per-image split rather than silently lumping every
        record into one bucket.
        """
        if self.group is not None:
            return self.group
        base = strip_export_suffix(self.image_id)
        return clip_id(base) or base or self.image_id

    @property
    def regions(self) -> set[Region]:
        return {b.region for b in self.boxes}

    @property
    def primary_region(self) -> Region:
        """Region used to stratify the split.

        An image with plates from several regions is rare but possible; the
        most common one wins, with UNKNOWN only if nothing else is present.
        """
        known = [b.region for b in self.boxes if b.region is not Region.UNKNOWN]
        if not known:
            return Region.UNKNOWN
        return max(set(known), key=known.count)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "image_id": self.image_id,
            "width": self.width,
            "height": self.height,
            "boxes": [b.to_dict() for b in self.boxes],
        }
        for name in ("file_name", "group", "source"):
            value = getattr(self, name)
            if value is not None:
                d[name] = value
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageRecord:
        return cls(
            image_id=d["image_id"],
            width=int(d["width"]),
            height=int(d["height"]),
            boxes=tuple(PlateBox.from_dict(b) for b in d.get("boxes", ())),
            file_name=d.get("file_name"),
            group=d.get("group"),
            source=d.get("source"),
            meta=d.get("meta", {}),
        )

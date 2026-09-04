"""Adapters turning third-party datasets into manifest records.

Every source we use exports, or can be converted to, a YOLO directory —
Roboflow exports natively, UC3M-LP ships `labels2yolo.py`. So one adapter
covers the sources, and each gets its own `source` tag so licence provenance
survives into the manifest.

Ingest is deliberately lossy in one direction: source class ids are collapsed
to a single `license_plate` class. Sources that also label vehicles would
otherwise spend detector capacity on a class Phase 6 never uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from alpr.data.schema import DatasetError, ImageRecord, PlateBox, Region, clip_id

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class IngestReport:
    """What happened during an ingest, so problems are visible not silent."""

    records: list[ImageRecord]
    skipped_no_label: list[str]
    skipped_bad_label: list[tuple[str, str]]
    dropped_boxes: int

    def summary(self) -> str:
        lines = [
            f"ingested       {len(self.records)} image(s)",
            f"plates         {sum(len(r.boxes) for r in self.records)}",
        ]
        if self.skipped_no_label:
            lines.append(f"no label file  {len(self.skipped_no_label)}")
        if self.skipped_bad_label:
            example = self.skipped_bad_label[0]
            lines.append(
                f"bad label      {len(self.skipped_bad_label)} (e.g. {example[0]}: {example[1]})"
            )
        if self.dropped_boxes:
            lines.append(f"dropped boxes  {self.dropped_boxes} (non-plate class)")
        return "\n".join(lines)


def image_size(path: Path) -> tuple[int, int]:
    """Read (width, height) from the header without decoding pixels.

    Decoding a full 4K JPEG to learn its dimensions costs ~50 ms; across
    20k images that is 15 minutes of a Colab session for two integers.
    """
    with Image.open(path) as img:
        return img.width, img.height


def _iter_images(images_dir: Path) -> Iterator[Path]:
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            yield path


def parse_yolo_label(
    text: str, *, keep_classes: frozenset[int] | None
) -> tuple[list[PlateBox], int]:
    """Parse a YOLO label file body into boxes.

    Returns:
        The boxes kept, and how many were dropped for having a class id that
        is not a plate.

    Raises:
        DatasetError: on a malformed line — a silently skipped bad label is a
            silently smaller training set.
    """
    boxes: list[PlateBox] = []
    dropped = 0

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise DatasetError(f"line {lineno}: expected 5 fields, got {len(parts)}")
        try:
            class_id = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError as exc:
            raise DatasetError(f"line {lineno}: {exc}") from exc

        if keep_classes is not None and class_id not in keep_classes:
            dropped += 1
            continue

        try:
            boxes.append(PlateBox(cx=cx, cy=cy, w=w, h=h))
        except DatasetError as exc:
            raise DatasetError(f"line {lineno}: {exc}") from exc

    return boxes, dropped


def from_yolo_dir(
    images_dir: str | Path,
    labels_dir: str | Path | None = None,
    *,
    source: str,
    region: Region = Region.UNKNOWN,
    keep_classes: frozenset[int] | None = None,
    id_prefix: str | None = None,
    group_by_stem: bool = False,
    path_root: str | Path | None = None,
    strict: bool = True,
) -> IngestReport:
    """Ingest a YOLO-format directory.

    Args:
        images_dir: directory of images, searched recursively.
        labels_dir: matching `.txt` labels. Defaults to a sibling `labels/`.
        source: provenance tag — also the licence audit trail. Required.
        region: plate region for these images, when the source is single-country.
        keep_classes: source class ids that mean "plate". None keeps everything,
            which is right for single-class plate datasets.
        id_prefix: prefixed to every `image_id`. Two datasets both containing
            `001.jpg` would otherwise collide into one id.
        group_by_stem: derive the split group from the filename's frame suffix.
            Leave False for datasets of unrelated stills, where every image is
            already its own group.
        path_root: root that each record's `file_name` is stored relative to.
            Defaults to `images_dir`. Set it to the shared parent when merging
            several sources into one manifest — otherwise two sources both
            yield `img001.jpg` and the export cannot tell which file is meant.
        strict: raise on a malformed label instead of recording and skipping it.

    Returns:
        An `IngestReport`; the records still need splitting and exporting.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise DatasetError(f"images_dir does not exist: {images_dir}")

    labels_path = Path(labels_dir) if labels_dir else images_dir.parent / "labels"
    if not labels_path.is_dir():
        raise DatasetError(f"labels_dir does not exist: {labels_path}")

    root = Path(path_root) if path_root is not None else images_dir
    try:
        images_dir.resolve().relative_to(root.resolve())
    except ValueError:
        raise DatasetError(f"path_root {root} is not a parent of images_dir {images_dir}") from None

    records: list[ImageRecord] = []
    no_label: list[str] = []
    bad_label: list[tuple[str, str]] = []
    dropped_total = 0

    for image_path in _iter_images(images_dir):
        stem = image_path.stem
        label_file = labels_path / f"{stem}.txt"

        if not label_file.exists():
            # Not necessarily an error: some datasets omit labels for
            # background images. Recorded so the count is visible.
            no_label.append(stem)
            continue

        try:
            boxes, dropped = parse_yolo_label(
                label_file.read_text(encoding="utf-8"), keep_classes=keep_classes
            )
        except DatasetError as exc:
            if strict:
                raise DatasetError(f"{label_file}: {exc}") from exc
            bad_label.append((stem, str(exc)))
            continue

        dropped_total += dropped
        width, height = image_size(image_path)
        image_id = f"{id_prefix}{stem}" if id_prefix else stem

        records.append(
            ImageRecord(
                image_id=image_id,
                width=width,
                height=height,
                boxes=tuple(
                    PlateBox(b.cx, b.cy, b.w, b.h, text=b.text, region=region) for b in boxes
                ),
                file_name=str(image_path.resolve().relative_to(root.resolve())),
                group=None if group_by_stem else image_id,
                source=source,
            )
        )

    return IngestReport(
        records=records,
        skipped_no_label=no_label,
        skipped_bad_label=bad_label,
        dropped_boxes=dropped_total,
    )


# Roboflow's YOLO export writes one directory per split, each with its own
# images/ and labels/. It spells validation "valid"; Ultralytics says "val".
_ROBOFLOW_SPLIT_DIRS = ("train", "valid", "val", "test")


def _group_for(
    image_id: str,
    split_prefix: str,
    id_prefix: str | None,
    group_by_clip: bool,
) -> str:
    """The split group for one exported record.

    A recognised video frame gets `<source prefix><clip>` — no upstream
    directory, so every frame of that clip lands in one group whichever
    directory Roboflow exported it into. Anything else keeps a per-image group,
    which is the conservative choice: see `from_roboflow_export`.
    """
    if not group_by_clip:
        return image_id

    stem = image_id[len(split_prefix) :] if image_id.startswith(split_prefix) else image_id
    clip = clip_id(stem)
    return f"{id_prefix or ''}{clip}" if clip else image_id


def from_roboflow_export(
    location: str | Path,
    *,
    source: str,
    region: Region = Region.UNKNOWN,
    keep_classes: frozenset[int] | None = None,
    id_prefix: str | None = None,
    path_root: str | Path | None = None,
    group_by_clip: bool = True,
    strict: bool = False,
) -> IngestReport:
    """Ingest a Roboflow YOLO export, pooling all of its splits.

    **Roboflow's own train/valid/test split is discarded on purpose.** Their
    split is random per image, so frames from one scene can sit on both sides
    of it, and it knows nothing about our region stratification. We pool every
    image and re-split with `alpr.data.split`, which is grouped, stratified and
    deterministic. Trusting the upstream split would quietly reintroduce the
    leakage Phase 1 exists to prevent.

    Args:
        location: the export root — the directory holding `train/`, `valid/`,
            `test/` and `data.yaml`.
        source: provenance tag, also the licence audit trail.
        region: region to tag every plate with.
        keep_classes: source class ids meaning "plate"; None keeps all, which
            is correct for these single-class datasets.
        id_prefix: prefix for image ids, to keep sources from colliding.
        path_root: root the stored paths are relative to. Defaults to
            `location`'s parent so several exports compose under one root.
        group_by_clip: give frames of one video a shared split group derived
            from the filename, **independently of which upstream directory they
            were exported into**. See the section below; leave it on unless you
            are deliberately reproducing the old, leak-prone grouping.
        strict: raise on a malformed label rather than recording it.

    Raises:
        DatasetError: when the export contains none of the expected split
            directories.

    ### Why the group cannot simply be the image id

    `image_id` has to carry the upstream directory — stems are not unique across
    `train/`, `valid/` and `test/`, and a collision would abort `write_manifest`
    only after the whole multi-gigabyte download. But that makes the id useless
    as a *clip* identity: Roboflow scatters frames of one video across its own
    directories, so `dayride_type1_001` arrives as both `eu-train-…` and
    `eu-test-…`, and grouping on the id splits the clip in two.

    Measured on the 188 surviving real identifiers: 9 of 12 clip groups span
    more than one upstream directory, `dayride_type1_001` spanning all three.

    So the group is built from the *stem* — source prefix plus clip identity,
    with the directory dropped. Two frames of one clip therefore land in one
    group no matter where they were exported.

    ### Why only frames get this treatment

    Dropping the directory is safe exactly when the stem identifies something
    real. For a recognised frame it does. For an ordinary still it would mean
    trusting that two files named `001.jpg` in `train/` and `test/` are the same
    photograph — which nothing here establishes. Stills therefore keep a
    per-image group, and the false-merge surface stays limited to filenames that
    explicitly announce themselves as video frames. Perceptual hashing in
    `alpr.dupes` is what relates stills, and it looks at pixels rather than
    names.
    """
    location = Path(location)
    if not location.is_dir():
        raise DatasetError(f"export directory does not exist: {location}")

    root = Path(path_root) if path_root is not None else location.parent

    records: list[ImageRecord] = []
    no_label: list[str] = []
    bad_label: list[tuple[str, str]] = []
    dropped = 0
    found_any = False

    for split_dir in _ROBOFLOW_SPLIT_DIRS:
        images_dir = location / split_dir / "images"
        labels_dir = location / split_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue

        found_any = True
        # Namespace ids by the upstream split. Nothing guarantees a stem is
        # unique across train/valid/test, and a collision would only surface
        # later as a "duplicate image_id" abort from write_manifest — after
        # the multi-gigabyte download. The upstream split is kept in `meta`
        # for provenance; it is not used for splitting.
        split_prefix = f"{id_prefix or ''}{split_dir}-"
        report = from_yolo_dir(
            images_dir,
            labels_dir,
            source=source,
            region=region,
            keep_classes=keep_classes,
            id_prefix=split_prefix,
            path_root=root,
            strict=strict,
        )
        records.extend(
            replace(
                record,
                meta={**record.meta, "roboflow_split": split_dir},
                group=_group_for(record.image_id, split_prefix, id_prefix, group_by_clip),
            )
            for record in report.records
        )
        no_label.extend(report.skipped_no_label)
        bad_label.extend(report.skipped_bad_label)
        dropped += report.dropped_boxes

    if not found_any:
        raise DatasetError(
            f"no split directories found under {location}. Expected one of "
            f"{list(_ROBOFLOW_SPLIT_DIRS)}, each with images/ and labels/."
        )

    return IngestReport(
        records=records,
        skipped_no_label=no_label,
        skipped_bad_label=bad_label,
        dropped_boxes=dropped,
    )

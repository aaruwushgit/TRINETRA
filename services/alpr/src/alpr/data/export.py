"""Materialize a manifest + split into the directory layout Ultralytics wants.

    root/
      images/{train,val,test}/<file>
      labels/{train,val,test}/<stem>.txt
      data.yaml

Images are symlinked by default. A plate dataset is mostly JPEG bytes, and
copying tens of thousands of them on a Colab runtime wastes minutes of the
session and gigabytes of the disk quota for files that already exist.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from alpr.data.schema import DatasetError, ImageRecord, Split
from alpr.data.split import SplitAssignment

# Single class: everything is a plate. Vehicle boxes, where a source provides
# them, are dropped at ingest — Phase 6 tracks plates directly, so a vehicle
# class would cost training capacity and buy nothing.
CLASS_NAMES = {0: "license_plate"}

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class ExportResult:
    root: Path
    data_yaml: Path
    counts: dict[str, int]
    missing_images: list[str]

    def summary(self) -> str:
        counts = "  ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
        line = f"exported to {self.root}\n  {counts}"
        if self.missing_images:
            line += f"\n  WARNING {len(self.missing_images)} image(s) not found on disk"
        return line


def format_label_file(record: ImageRecord, *, clip: bool = True) -> str:
    """Render one record's boxes as YOLO label lines.

    Args:
        record: the annotated image.
        clip: clamp boxes to the image. Plates cut off by the frame edge are
            annotated past the border legitimately, but Ultralytics treats
            out-of-range coordinates as corrupt labels and drops the image.
    """
    lines = []
    for box in record.boxes:
        b = box.clipped() if clip else box
        lines.append(f"0 {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def _resolve_image(record: ImageRecord, image_root: Path) -> Path | None:
    """Find a record's image on disk, tolerating a missing extension."""
    if record.file_name:
        candidate = image_root / record.file_name
        return candidate if candidate.exists() else None

    for ext in _IMAGE_EXTENSIONS:
        candidate = image_root / f"{record.image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _link_or_copy(src: Path, dst: Path, *, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        # Relative target so the exported tree survives being moved or zipped
        # to the Hub and unpacked somewhere else.
        os.symlink(os.path.relpath(src, dst.parent), dst)
    else:
        shutil.copy2(src, dst)


def export_yolo(
    records: Sequence[ImageRecord],
    assignment: SplitAssignment,
    image_root: str | Path,
    out_root: str | Path,
    *,
    symlink: bool = True,
    allow_missing: bool = False,
) -> ExportResult:
    """Write the Ultralytics tree for `records`.

    Args:
        records: dataset to export.
        assignment: split assignment covering every record.
        image_root: directory the records' `file_name`s are relative to.
        out_root: destination; created if absent.
        symlink: link images instead of copying.
        allow_missing: keep going when an image is absent from disk. The label
            is still written, so the count mismatch shows up in the result
            rather than silently shrinking the training set.

    Raises:
        DatasetError: when an image is missing and `allow_missing` is False.
    """
    image_root = Path(image_root)
    out_root = Path(out_root)

    for split in Split:
        (out_root / "images" / split.value).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split.value).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {s.value: 0 for s in Split}
    missing: list[str] = []

    for record in records:
        split = assignment.of(record)
        source = _resolve_image(record, image_root)

        if source is None:
            missing.append(record.image_id)
            if not allow_missing:
                raise DatasetError(
                    f"image for {record.image_id!r} not found under {image_root} "
                    f"(file_name={record.file_name!r})"
                )
        else:
            suffix = source.suffix
            _link_or_copy(
                source,
                out_root / "images" / split.value / f"{record.image_id}{suffix}",
                symlink=symlink,
            )

        label_path = out_root / "labels" / split.value / f"{record.image_id}.txt"
        label_path.write_text(format_label_file(record), encoding="utf-8")
        counts[split.value] += 1

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(_render_data_yaml(out_root), encoding="utf-8")

    return ExportResult(root=out_root, data_yaml=data_yaml, counts=counts, missing_images=missing)


def _render_data_yaml(out_root: Path) -> str:
    names = "\n".join(f"  {idx}: {name}" for idx, name in sorted(CLASS_NAMES.items()))
    return (
        "# Generated by alpr.data.export — do not edit by hand.\n"
        f"path: {out_root.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{names}\n"
    )

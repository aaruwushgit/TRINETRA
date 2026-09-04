"""Running a detector baseline against a *frozen* dataset, and recording what ran.

Phase 2. This exists because the canonical dataset is an immutable artifact that
lives outside the repository, and a training run against it has to prove three
things before it is worth anything:

**That it trained on the dataset it claims to.** The canonical export is
identified by two SHA-256 hashes, not by a path. `verify_dataset` recomputes
them and refuses to continue on a mismatch — a run that quietly trained on a
rebuilt or partially-copied dataset is worse than no run at all, because it
looks like evidence.

**That it did not have to modify that dataset to run.** The canonical
`data.yaml` carries an absolute path to the machine that built it, so it cannot
be used anywhere else. `write_data_yaml` emits a *sibling* file for the new
location and never touches the original.

**That the settings are recoverable afterwards.** The historical run left an
`args.yaml` and nothing else, so its Ultralytics version, its resolved
optimizer and the provenance of its published metrics all had to be
reconstructed later, and one of them could not be. `baseline_provenance`
records those fields at the time they are known.

Nothing here trains, evaluates or downloads. It is the bookkeeping around a run,
kept in the package so it is testable rather than living in a notebook cell.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The export fingerprint deliberately excludes `data.yaml`: that file embeds an
# absolute `path:` for the machine it was generated on, so including it would
# make the hash location-dependent and the dataset unverifiable anywhere else.
# Splits and label bytes are what actually define the experiment.
SPLITS = ("train", "val", "test")


class BaselineError(RuntimeError):
    """Raised when a baseline run must not proceed."""


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks so a checkpoint does not sit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_fingerprint(export_root: str | Path) -> str:
    """Deterministic, location-independent hash of a YOLO export.

    Covers every label file's split, name and bytes, in sorted order — so it
    pins the split assignment and the annotations together. Two exports of the
    same dataset on different machines hash identically, which is what makes it
    usable as an identity check after a transfer.
    """
    export_root = Path(export_root)
    digest = hashlib.sha256()
    for split in SPLITS:
        for label in sorted((export_root / "labels" / split).glob("*.txt")):
            digest.update(f"{split}/{label.name}".encode())
            digest.update(label.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DatasetFacts:
    """What a YOLO export on disk actually contains."""

    images: dict[str, int]
    labels: dict[str, int]
    annotations: int
    export_sha256: str
    manifest_sha256: str | None = None
    manifest_records: int | None = None

    @property
    def total_images(self) -> int:
        return sum(self.images.values())

    def summary(self) -> str:
        counts = "  ".join(f"{s}={self.labels.get(s, 0)}" for s in SPLITS)
        lines = [
            f"images/labels    {self.total_images} / {sum(self.labels.values())}",
            f"split            {counts}",
            f"annotations      {self.annotations}",
            f"export sha256    {self.export_sha256}",
        ]
        if self.manifest_sha256:
            lines.append(f"manifest sha256  {self.manifest_sha256}")
            lines.append(f"manifest records {self.manifest_records}")
        return "\n".join(lines)


def inspect_dataset(export_root: str | Path, manifest: str | Path | None = None) -> DatasetFacts:
    """Measure a YOLO export without changing anything in it."""
    export_root = Path(export_root)
    images: dict[str, int] = {}
    labels: dict[str, int] = {}
    annotations = 0

    for split in SPLITS:
        image_dir = export_root / "images" / split
        label_dir = export_root / "labels" / split
        if not label_dir.is_dir():
            raise BaselineError(f"missing label directory: {label_dir}")
        # Counts symlinks and regular files alike; a dereferenced copy and the
        # original must measure the same.
        images[split] = (
            sum(1 for p in image_dir.iterdir() if not p.name.startswith("."))
            if image_dir.is_dir()
            else 0
        )
        label_files = sorted(label_dir.glob("*.txt"))
        labels[split] = len(label_files)
        for label in label_files:
            annotations += sum(1 for line in label.read_text().splitlines() if line.strip())

    manifest_sha = records = None
    if manifest is not None:
        manifest = Path(manifest)
        if not manifest.exists():
            raise BaselineError(f"manifest not found: {manifest}")
        manifest_sha = sha256_file(manifest)
        records = sum(1 for line in manifest.read_text().splitlines() if line.strip())

    return DatasetFacts(
        images=images,
        labels=labels,
        annotations=annotations,
        export_sha256=export_fingerprint(export_root),
        manifest_sha256=manifest_sha,
        manifest_records=records,
    )


def verify_dataset(
    facts: DatasetFacts,
    *,
    export_sha256: str,
    manifest_sha256: str | None = None,
    counts: dict[str, int] | None = None,
    annotations: int | None = None,
    manifest_records: int | None = None,
) -> None:
    """Raise unless the export is exactly the dataset that was expected.

    Every mismatch is collected before raising, so a broken transfer reports all
    of its symptoms at once instead of one per attempt.
    """
    problems: list[str] = []

    if facts.export_sha256 != export_sha256:
        problems.append(f"export sha256 {facts.export_sha256} != expected {export_sha256}")
    if manifest_sha256 is not None and facts.manifest_sha256 != manifest_sha256:
        problems.append(f"manifest sha256 {facts.manifest_sha256} != expected {manifest_sha256}")
    if counts is not None:
        for split, expected in counts.items():
            if facts.labels.get(split) != expected:
                problems.append(f"{split}: {facts.labels.get(split)} labels, expected {expected}")
            if facts.images.get(split) != expected:
                problems.append(f"{split}: {facts.images.get(split)} images, expected {expected}")
    if annotations is not None and facts.annotations != annotations:
        problems.append(f"annotations {facts.annotations}, expected {annotations}")
    if manifest_records is not None and facts.manifest_records != manifest_records:
        problems.append(f"manifest records {facts.manifest_records}, expected {manifest_records}")

    if problems:
        raise BaselineError(
            "the dataset is not the one this run expects — refusing to train:\n  "
            + "\n  ".join(problems)
        )


def write_data_yaml(
    export_root: str | Path,
    path: str | Path,
    *,
    class_names: dict[int, str] | None = None,
) -> Path:
    """Write a `data.yaml` for an export at its *current* location.

    The canonical export ships one already, but it carries the absolute path of
    the machine that produced it and is frozen read-only. Rather than edit a
    frozen artifact, this writes a new file beside it. Pass a `path` that is not
    `data.yaml` so the original cannot be overwritten by accident.

    Raises:
        BaselineError: if `path` would clobber the export's own `data.yaml`.
    """
    from alpr.data.export import CLASS_NAMES

    export_root = Path(export_root).resolve()
    path = Path(path)
    if path.resolve() == (export_root / "data.yaml").resolve():
        raise BaselineError(
            f"refusing to overwrite the export's own data.yaml at {path} — "
            "write a differently named file (e.g. data_colab.yaml)"
        )

    names = class_names or CLASS_NAMES
    rendered = "\n".join(f"  {index}: {name}" for index, name in sorted(names.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated by alpr.baseline.write_data_yaml for a relocated export.\n"
        "# The export's own data.yaml is left untouched.\n"
        f"path: {export_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"{rendered}\n",
        encoding="utf-8",
    )
    return path


@dataclass
class BaselineRecord:
    """Everything needed to identify a baseline run after the fact."""

    payload: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.payload, indent=2, sort_keys=False)
        path.write_text(payload + "\n", encoding="utf-8")
        return path


def baseline_provenance(
    *,
    dataset: DatasetFacts,
    dataset_paths: dict[str, str],
    train_config: dict[str, Any],
    resolved_optimizer: dict[str, Any] | None,
    environment: dict[str, Any],
    code: dict[str, Any],
    val_metrics: dict[str, float] | None = None,
    test_metrics: dict[str, float] | None = None,
    region_metrics: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
    timestamps: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> BaselineRecord:
    """Assemble the provenance record for one baseline run.

    Deliberately takes measured values rather than discovering them: every field
    is supplied by the caller that actually observed it, so nothing here can
    invent a metric. Sections that were not measured stay `None` and are written
    as null, which is the honest representation of "this did not happen".
    """
    return BaselineRecord(
        {
            "experiment": "canonical detector baseline",
            "intentional_change_vs_historical": "historical dataset -> canonical dataset",
            "timestamps": timestamps or {},
            "dataset": {
                "paths": dataset_paths,
                "split": dataset.labels,
                "images": dataset.images,
                "annotations": dataset.annotations,
                "export_sha256": dataset.export_sha256,
                "manifest_sha256": dataset.manifest_sha256,
                "manifest_records": dataset.manifest_records,
            },
            "training": train_config,
            "resolved_optimizer": resolved_optimizer,
            "checkpoint": checkpoint,
            "metrics": {
                "validation": val_metrics,
                "test": test_metrics,
                "test_by_region": region_metrics,
            },
            "evaluation": evaluation,
            "environment": environment,
            "code": code,
            "notes": notes or [],
        }
    )


def region_slice_counts(records: Any) -> dict[str, int]:
    """Images per region, for reporting a region-sliced evaluation."""
    return dict(Counter(record.primary_region.value for record in records))

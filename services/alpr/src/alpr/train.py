"""Detector training.

Config-driven so a run is reproducible from a committed file rather than from
whatever was typed into a Colab cell. The resolved config is written into the
run directory, so a result can always be traced back to the settings that
produced it — including the Ultralytics version, which changes defaults
between releases.

Augmentation is where a general-purpose recipe goes wrong for plates. The
defaults here are chosen for one specific object: a small, rigid, text-bearing
rectangle photographed from a vehicle's height.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from alpr import __version__


class TrainingError(RuntimeError):
    """Raised when a run cannot start or its configuration is invalid."""


@dataclass(frozen=True)
class TrainConfig:
    """Everything that determines a training run.

    Defaults are tuned for license plates rather than for COCO:

    `flipud=0.0` — a plate is never upside down. Vertical flipping spends
    capacity on an input that cannot occur.

    `degrees=10` and `perspective=0.0005` — plates *are* photographed rotated
    and off-axis, from a camera above bumper height. Ultralytics defaults both
    to zero, which is wrong for this object.

    `mosaic=1.0` with `close_mosaic=10` — mosaic tiles four images together,
    shrinking objects, which is exactly the small-object regime plates live
    in. Turning it off for the final epochs lets the model settle on
    undistorted images before it stops.

    `fliplr` is left at the Ultralytics default of 0.5 rather than disabled.
    Mirrored plates do not exist in the world, so there is a real argument for
    turning it off — but detection only has to find a bright rectangle, and
    halving the effective dataset to avoid an impossible input is the larger
    risk. It is exposed here precisely so Phase 3 can ablate it rather than
    leaving the question to taste.
    """

    # What to train
    model: str = "yolov8s.pt"
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    patience: int = 25
    seed: int = 0

    # Where results go
    project: str = "runs/detect"
    name: str = "plates"

    # Plate-specific augmentation
    degrees: float = 10.0
    translate: float = 0.10
    scale: float = 0.50
    shear: float = 2.0
    perspective: float = 0.0005
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    close_mosaic: int = 10
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # Runtime
    device: str = "0"
    workers: int = 8
    cache: bool = False
    resume: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise TrainingError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch < 1:
            raise TrainingError(f"batch must be at least 1, got {self.batch}")
        if self.imgsz < 32 or self.imgsz % 32:
            # Ultralytics strides by 32; anything else is silently rounded,
            # which makes a run's actual resolution differ from its config.
            raise TrainingError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        for name in ("flipud", "fliplr", "mosaic"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise TrainingError(f"{name} must be in [0, 1], got {value}")
        if self.close_mosaic > self.epochs:
            raise TrainingError(
                f"close_mosaic ({self.close_mosaic}) exceeds epochs ({self.epochs}): "
                "mosaic would never be enabled"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        unknown = set(payload) - known - {"extra"}
        if unknown:
            # A typo'd key would otherwise be silently ignored, and the run
            # would quietly use a default nobody intended.
            raise TrainingError(
                f"unknown config key(s) in {path}: {', '.join(sorted(unknown))}. "
                f"Put deliberate Ultralytics passthroughs under 'extra:'."
            )
        return cls(**payload)

    def to_ultralytics(self, data_yaml: str | Path) -> dict[str, Any]:
        """Render the arguments for `YOLO.train`."""
        args = {
            "data": str(data_yaml),
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "patience": self.patience,
            "seed": self.seed,
            "project": self.project,
            "name": self.name,
            "device": self.device,
            "workers": self.workers,
            "cache": self.cache,
            "resume": self.resume,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "close_mosaic": self.close_mosaic,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "exist_ok": True,
            "plots": True,
        }
        args.update(self.extra)
        return args


def provenance(config: TrainConfig, data_yaml: str | Path) -> dict[str, Any]:
    """Everything needed to explain a result later.

    Ultralytics changes augmentation defaults between minor versions, so a
    config alone does not pin a run — the library version has to travel with
    it, or a number becomes unreproducible six months on.
    """
    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except Exception:  # pragma: no cover - only when the import is broken
        ultralytics_version = "unknown"

    return {
        "alpr_version": __version__,
        "ultralytics_version": ultralytics_version,
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(),
        "data": str(data_yaml),
        "config": asdict(config),
    }


def validate_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Check the dataset before spending GPU time on it.

    Raises:
        TrainingError: when the file is missing, malformed, or points at
            directories that do not exist. Ultralytics would fail on these too,
            but often minutes in and with a less direct message.
    """
    path = Path(data_yaml)
    if not path.exists():
        raise TrainingError(f"dataset config not found: {path}. Run Phase 1 first.")

    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("train", "val", "names"):
        if key not in spec:
            raise TrainingError(f"{path} is missing the '{key}' key")

    root = Path(spec.get("path", path.parent))
    for split in ("train", "val"):
        directory = root / spec[split]
        if not directory.is_dir():
            raise TrainingError(f"{split} directory does not exist: {directory}")
        if not any(directory.iterdir()):
            raise TrainingError(f"{split} directory is empty: {directory}")

    return spec


def train(
    config: TrainConfig,
    data_yaml: str | Path,
    *,
    require_gpu: bool = True,
):
    """Run training.

    Args:
        config: the run's settings.
        data_yaml: the Ultralytics dataset config written by Phase 1.
        require_gpu: fail fast without CUDA. Training on CPU is not a slower
            path here — it is one that will not finish.

    Returns:
        The Ultralytics results object.
    """
    validate_dataset(data_yaml)

    if require_gpu:
        from alpr.env import require_gpu as _require_gpu

        _require_gpu()

    # Imported here rather than at module scope: ultralytics pulls in torch,
    # which costs seconds, and `alpr env` should stay instant.
    from ultralytics import YOLO

    record = provenance(config, data_yaml)

    model = YOLO(config.model)
    results = model.train(**config.to_ultralytics(data_yaml))

    # Ultralytics decides the real output directory itself, and it does not
    # always equal `project/name`: with a relative `project` it can resolve
    # against its own configured runs_dir, producing e.g.
    # runs/detect/runs/detect/plates. Writing provenance to a reconstructed
    # path put it in an empty directory while the weights landed elsewhere,
    # which defeats the point of recording it. Take the directory from the
    # results object instead of guessing.
    run_dir = run_directory(results, config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    return results


def run_directory(results, config: TrainConfig) -> Path:
    """Where Ultralytics actually wrote a run."""
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return Path(config.project) / config.name


def best_weights(config: TrainConfig, results=None) -> Path:
    """Path to the best checkpoint of a finished run.

    Prefers the directory Ultralytics reports. Failing that, searches the
    project tree for the most recent `best.pt` — reconstructing
    `project/name` is not reliable, since Ultralytics may nest the run
    directory or suffix the name when one already exists.
    """
    if results is not None:
        candidate = run_directory(results, config) / "weights" / "best.pt"
        if candidate.exists():
            return candidate

    root = Path(config.project)
    matches = sorted(
        root.glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise TrainingError(
        f"no trained weights found under {root} — has training finished? "
        "(searched **/weights/best.pt)"
    )

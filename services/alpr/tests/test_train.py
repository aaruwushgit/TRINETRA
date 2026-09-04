"""Training configuration and pre-flight checks.

Training itself needs a GPU and is marked accordingly. What runs in CI is the
part that fails silently otherwise: a mistyped config key, a dataset path that
does not exist, an imgsz that Ultralytics rounds without saying so.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alpr.train import (
    TrainConfig,
    TrainingError,
    best_weights,
    provenance,
    run_directory,
    validate_dataset,
)

REPO_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "detector.yaml"


class TestTrainConfig:
    def test_defaults_are_plate_appropriate(self):
        config = TrainConfig()
        # A plate is never upside down.
        assert config.flipud == 0.0
        # Plates *are* photographed rotated and off-axis; Ultralytics defaults
        # both of these to zero.
        assert config.degrees > 0
        assert config.perspective > 0

    def test_committed_config_loads(self):
        config = TrainConfig.from_yaml(REPO_CONFIG)
        assert config.model.startswith("yolov8")
        assert config.epochs > 0

    def test_rejects_unknown_keys(self, tmp_path):
        # A typo would otherwise be ignored and the run would use a default
        # nobody intended.
        path = tmp_path / "c.yaml"
        path.write_text("epochs: 10\nepocs: 999\n")
        with pytest.raises(TrainingError, match="unknown config key"):
            TrainConfig.from_yaml(path)

    def test_extra_is_a_deliberate_passthrough(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("epochs: 10\nextra:\n  cos_lr: true\n")
        assert TrainConfig.from_yaml(path).extra == {"cos_lr": True}

    def test_empty_config_uses_defaults(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("")
        assert TrainConfig.from_yaml(path).epochs == TrainConfig().epochs

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"epochs": 0}, "epochs"),
            ({"batch": 0}, "batch"),
            ({"imgsz": 100}, "multiple of 32"),
            ({"imgsz": 0}, "multiple of 32"),
            ({"flipud": 1.5}, "flipud"),
            ({"mosaic": -0.1}, "mosaic"),
            ({"epochs": 5, "close_mosaic": 10}, "close_mosaic"),
        ],
    )
    def test_rejects_invalid_values(self, kwargs, message):
        with pytest.raises(TrainingError, match=message):
            TrainConfig(**kwargs)

    def test_imgsz_must_be_a_stride_multiple(self):
        # Ultralytics silently rounds a non-multiple, so a run's real
        # resolution would differ from its recorded config.
        TrainConfig(imgsz=640)
        TrainConfig(imgsz=960)
        with pytest.raises(TrainingError):
            TrainConfig(imgsz=650)

    def test_renders_ultralytics_arguments(self):
        args = TrainConfig(epochs=7, close_mosaic=3).to_ultralytics("data.yaml")
        assert args["data"] == "data.yaml"
        assert args["epochs"] == 7
        assert args["flipud"] == 0.0

    def test_extra_overrides_rendered_arguments(self):
        args = TrainConfig(extra={"epochs": 3}).to_ultralytics("d.yaml")
        assert args["epochs"] == 3

    def test_short_smoke_runs_need_close_mosaic_lowered(self):
        # A 1-epoch smoke test with the default close_mosaic=10 would disable
        # mosaic for the entire run — silently training a different recipe
        # than the config describes. Better to refuse than to mislead.
        with pytest.raises(TrainingError, match="close_mosaic"):
            TrainConfig(epochs=1)
        TrainConfig(epochs=1, close_mosaic=0)  # explicit, and honest


class TestProvenance:
    def test_records_versions_and_config(self):
        record = provenance(TrainConfig(), "data.yaml")
        # Ultralytics changes augmentation defaults between minor releases, so
        # a config alone does not pin a run.
        assert "ultralytics_version" in record
        assert record["config"]["seed"] == 0
        assert record["data"] == "data.yaml"


def _dataset(tmp_path, *, splits=("train", "val", "test"), populate=True):
    root = tmp_path / "yolo"
    for split in splits:
        directory = root / "images" / split
        directory.mkdir(parents=True)
        if populate:
            (directory / "a.jpg").write_bytes(b"\xff\xd8\xff")
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "license_plate"},
            }
        )
    )
    return path


class TestValidateDataset:
    def test_accepts_a_phase_1_export(self, tmp_path):
        spec = validate_dataset(_dataset(tmp_path))
        assert spec["names"] == {0: "license_plate"}

    def test_missing_file(self, tmp_path):
        with pytest.raises(TrainingError, match="Run Phase 1 first"):
            validate_dataset(tmp_path / "absent.yaml")

    def test_missing_key(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text(yaml.safe_dump({"train": "images/train"}))
        with pytest.raises(TrainingError, match="missing the 'val' key"):
            validate_dataset(path)

    def test_missing_split_directory(self, tmp_path):
        path = _dataset(tmp_path)
        (tmp_path / "yolo" / "images" / "val" / "a.jpg").unlink()
        (tmp_path / "yolo" / "images" / "val").rmdir()
        with pytest.raises(TrainingError, match="val directory does not exist"):
            validate_dataset(path)

    def test_empty_split_directory(self, tmp_path):
        # Ultralytics would fail on this too, but often minutes into a run.
        path = _dataset(tmp_path, populate=False)
        with pytest.raises(TrainingError, match="is empty"):
            validate_dataset(path)


class _Results:
    def __init__(self, save_dir):
        self.save_dir = str(save_dir)


class TestBestWeights:
    def test_reports_a_run_that_has_not_finished(self, tmp_path):
        config = TrainConfig(project=str(tmp_path / "runs"), name="plates")
        (tmp_path / "runs").mkdir()
        with pytest.raises(TrainingError, match="has training finished"):
            best_weights(config)

    def test_finds_finished_weights(self, tmp_path):
        config = TrainConfig(project=str(tmp_path / "runs"), name="plates")
        weights = tmp_path / "runs" / "plates" / "weights"
        weights.mkdir(parents=True)
        (weights / "best.pt").write_bytes(b"weights")
        assert best_weights(config).name == "best.pt"

    def test_finds_weights_when_ultralytics_nests_the_run_directory(self, tmp_path):
        # Observed on Colab: a relative `project` resolved against Ultralytics'
        # own runs_dir, so weights landed in runs/detect/runs/detect/plates
        # while `project/name` was runs/detect/plates. Reconstructing the path
        # found nothing; searching the tree finds it.
        config = TrainConfig(project=str(tmp_path / "runs" / "detect"), name="plates")
        nested = tmp_path / "runs" / "detect" / "runs" / "detect" / "plates" / "weights"
        nested.mkdir(parents=True)
        (nested / "best.pt").write_bytes(b"weights")
        assert best_weights(config) == nested / "best.pt"

    def test_prefers_the_directory_ultralytics_reports(self, tmp_path):
        config = TrainConfig(project=str(tmp_path / "runs"), name="plates")
        reported = tmp_path / "runs" / "elsewhere" / "weights"
        reported.mkdir(parents=True)
        (reported / "best.pt").write_bytes(b"weights")
        results = _Results(reported.parent)
        assert best_weights(config, results) == reported / "best.pt"

    def test_picks_the_most_recent_run(self, tmp_path):
        import os
        import time

        config = TrainConfig(project=str(tmp_path / "runs"), name="plates")
        for name, age in (("plates", 100), ("plates2", 0)):
            weights = tmp_path / "runs" / name / "weights"
            weights.mkdir(parents=True)
            path = weights / "best.pt"
            path.write_bytes(b"w")
            stamp = time.time() - age
            os.utime(path, (stamp, stamp))
        assert best_weights(config).parent.parent.name == "plates2"


class TestRunDirectory:
    def test_uses_the_reported_save_dir(self, tmp_path):
        config = TrainConfig(project=str(tmp_path), name="plates")
        assert run_directory(_Results("/somewhere/else"), config) == Path("/somewhere/else")

    def test_falls_back_to_project_and_name(self, tmp_path):
        config = TrainConfig(project=str(tmp_path), name="plates")
        assert run_directory(object(), config) == tmp_path / "plates"


@pytest.mark.gpu
def test_train_requires_a_gpu(tmp_path):
    """Opt-in: only meaningful on a CUDA machine."""
    from alpr.train import train

    train(
        TrainConfig(epochs=1, close_mosaic=0, project=str(tmp_path)),
        _dataset(tmp_path),
    )

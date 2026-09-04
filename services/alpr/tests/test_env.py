"""Tests for runtime probing.

`detect_gpus` shells out, so these tests stub the boundary rather than the
function under test — the parsing is the part that has bugs.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from alpr import env


def _fake_smi(monkeypatch, stdout: str) -> None:
    monkeypatch.setattr(env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(args=a, returncode=0, stdout=stdout),
    )


def test_no_driver_means_no_gpus(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _: None)
    assert env.detect_gpus() == []


def test_parses_a_t4(monkeypatch):
    _fake_smi(monkeypatch, "Tesla T4, 15360\n")
    (gpu,) = env.detect_gpus()
    assert gpu.name == "Tesla T4"
    assert gpu.memory_mb == 15360


def test_parses_multiple_gpus(monkeypatch):
    _fake_smi(monkeypatch, "Tesla T4, 15360\nTesla T4, 15360\n")
    assert len(env.detect_gpus()) == 2


def test_ignores_blank_trailing_lines(monkeypatch):
    # nvidia-smi emits a trailing newline; a naive split yields a phantom device.
    _fake_smi(monkeypatch, "Tesla T4, 15360\n\n")
    assert len(env.detect_gpus()) == 1


def test_broken_driver_is_treated_as_no_gpu(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def _boom(*a, **k):
        raise subprocess.CalledProcessError(returncode=1, cmd="nvidia-smi")

    monkeypatch.setattr(env.subprocess, "run", _boom)
    assert env.detect_gpus() == []


def test_require_gpu_raises_without_one(monkeypatch):
    monkeypatch.setattr(env, "detect_gpus", lambda: [])
    with pytest.raises(RuntimeError, match="No CUDA GPU"):
        env.require_gpu()


def test_require_gpu_returns_the_first(monkeypatch):
    monkeypatch.setattr(env, "detect_gpus", lambda: [env.GpuInfo("Tesla T4", 15360)])
    assert env.require_gpu().name == "Tesla T4"


class TestCredentials:
    """Credential lookup.

    Every value here is a dummy. A test asserting a real key puts that key in
    git history, which is how secrets leak out of public repos.
    """

    def test_reads_from_the_environment(self, monkeypatch):
        monkeypatch.setattr(env, "in_colab", lambda: False)
        monkeypatch.setenv(env.KAGGLE_TOKEN_VAR, "dummy-token")
        assert env.get_credential(env.KAGGLE_TOKEN_VAR) == "dummy-token"

    def test_missing_credential_explains_the_fix(self, monkeypatch):
        monkeypatch.setattr(env, "in_colab", lambda: False)
        monkeypatch.delenv(env.KAGGLE_TOKEN_VAR, raising=False)
        with pytest.raises(env.MissingCredential, match="Secrets panel"):
            env.get_credential(env.KAGGLE_TOKEN_VAR)

    def test_empty_value_counts_as_missing(self, monkeypatch):
        # An unset Colab secret returns "", which must not be accepted and
        # then fail confusingly later at the API call.
        monkeypatch.setattr(env, "in_colab", lambda: False)
        monkeypatch.setenv(env.KAGGLE_TOKEN_VAR, "")
        with pytest.raises(env.MissingCredential):
            env.get_credential(env.KAGGLE_TOKEN_VAR)

    def test_source_contains_no_hardcoded_credential(self):
        # Guards the real failure mode: someone adds a convenience default and
        # publishes their key to a public repo. There is no default.
        from pathlib import Path

        source = Path(env.__file__).read_text()
        for marker in ("KGAT", "rf_", "api_key="):
            assert marker not in source, f"possible hardcoded credential: {marker}"

    def test_setup_api_keys_populates_the_environment(self, monkeypatch):
        monkeypatch.setattr(env, "get_credential", lambda name: f"dummy-{name}")
        env.setup_api_keys()
        assert os.environ[env.KAGGLE_TOKEN_VAR] == f"dummy-{env.KAGGLE_TOKEN_VAR}"
        assert os.environ[env.ROBOFLOW_KEY_VAR] == f"dummy-{env.ROBOFLOW_KEY_VAR}"

    def test_setup_api_keys_names_the_missing_one(self, monkeypatch):
        def _raise(name):
            raise env.MissingCredential(f"{name} is not set")

        monkeypatch.setattr(env, "get_credential", _raise)
        with pytest.raises(env.MissingCredential, match=env.KAGGLE_TOKEN_VAR):
            env.setup_api_keys()

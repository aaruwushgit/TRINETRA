"""Runtime environment probing.

Colab sessions are ephemeral and silently vary: a runtime that was a T4
yesterday can be a CPU box today, and the failure shows up hours later as
training that never finishes. Every notebook driver calls `require_gpu()`
in its first cell so that mistake surfaces in seconds instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    """A single visible CUDA device."""

    name: str
    memory_mb: int

    def __str__(self) -> str:
        return f"{self.name} ({self.memory_mb / 1024:.1f} GB)"


def in_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def detect_gpus() -> list[GpuInfo]:
    """Return visible CUDA devices, or an empty list if there are none.

    Shells out to `nvidia-smi` rather than importing torch: this is called
    before the heavy dependencies are installed, and importing torch just to
    read a device name costs seconds on every notebook start.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        # A driver that is present but broken is a no-GPU environment for our
        # purposes; the caller's error message is more useful than a traceback.
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        name, _, memory = line.partition(",")
        if not memory.strip():
            continue
        gpus.append(GpuInfo(name=name.strip(), memory_mb=int(memory.strip())))
    return gpus


def require_gpu() -> GpuInfo:
    """Return the first CUDA device, raising if the runtime has none.

    Raises:
        RuntimeError: when no CUDA device is visible.
    """
    gpus = detect_gpus()
    if not gpus:
        hint = (
            "Runtime > Change runtime type > T4 GPU, then rerun this cell."
            if in_colab()
            else "This step needs a CUDA GPU; run it on Colab."
        )
        raise RuntimeError(f"No CUDA GPU visible. {hint}")
    return gpus[0]


KAGGLE_TOKEN_VAR = "KAGGLE_API_TOKEN"
ROBOFLOW_KEY_VAR = "ROBOFLOW_API_KEY"


class MissingCredential(RuntimeError):
    """A required API credential is not available in this runtime."""


def get_credential(name: str) -> str:
    """Fetch an API credential from Colab's secret store or the environment.

    There is deliberately **no hardcoded fallback**. A key written as a string
    literal is a key in git history, and this repository is public — GitHub
    keeps pushed objects reachable long after the commit is amended away, so
    such a leak is effectively permanent and the key must be rotated. Colab's
    secret store (the 🔑 panel) keeps the value out of both the notebook and
    the repo.

    Raises:
        MissingCredential: when the credential is set nowhere.
    """
    if in_colab():
        try:
            from google.colab import userdata

            value = userdata.get(name)
            if value:
                return value
        except Exception:
            # Secret not granted to this notebook, or no secret store at all.
            # Fall through to the environment rather than failing here.
            pass

    value = os.environ.get(name)
    if value:
        return value

    raise MissingCredential(
        f"{name} is not set. In Colab: open the 🔑 Secrets panel in the left "
        f"sidebar, add {name}, and switch on notebook access. Locally: export "
        f"it in your shell. Never paste it into a cell — the value would be "
        f"saved into the notebook and committed."
    )


def setup_api_keys() -> None:
    """Load the Phase 1 dataset credentials into the environment.

    The Kaggle and Roboflow clients both read their credentials from the
    environment, so this makes them available without either key appearing in
    a notebook cell.

    Raises:
        MissingCredential: naming the first credential that is not configured.
    """
    for name in (KAGGLE_TOKEN_VAR, ROBOFLOW_KEY_VAR):
        os.environ[name] = get_credential(name)

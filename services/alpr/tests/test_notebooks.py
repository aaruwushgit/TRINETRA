"""Guards on the committed notebooks.

Saving a notebook from Colab writes straight to the repo, bypassing
pre-commit entirely — so nbstripout and the gitleaks hook never run on it.
That has already reverted a dependency fix twice and committed cell outputs
once. CI is the only checkpoint such a save cannot skip, so these invariants
live here rather than in a hook.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _cells(path: Path):
    return json.loads(path.read_text())["cells"]


def _source(cell) -> str:
    return "".join(cell["source"])


def test_there_are_notebooks_to_check():
    # A glob that silently matches nothing would make every test below vacuous.
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebook:
    def test_is_valid_json(self, path):
        json.loads(path.read_text())

    def test_has_no_saved_outputs(self, path):
        # Outputs bloat diffs, and a cell that printed a credential would
        # commit it. nbstripout handles local commits; this catches Colab's.
        offenders = [
            i for i, c in enumerate(_cells(path)) if c.get("outputs") or c.get("execution_count")
        ]
        assert not offenders, (
            f"{path.name} has saved output in cell(s) {offenders}. "
            "Clear outputs before saving from Colab (Edit > Clear all outputs)."
        )

    def test_does_not_install_a_nonexistent_extra(self, path):
        # The `gpu` extra was removed in Phase 4: only `paddlepaddle-gpu` was
        # unresolvable, and the CPU build installs from PyPI everywhere. The
        # OCR stack now lives in `[ocr]`. A notebook still asking for `[gpu]`
        # would fail at install time on a fresh runtime.
        for i, cell in enumerate(_cells(path)):
            src = _source(cell)
            if "pip install" in src and ".[gpu]" in src:
                pytest.fail(
                    f"{path.name} cell {i} installs the removed `gpu` extra. "
                    'Use `pip install -e .` or `-e ".[ocr]"`.'
                )

    def test_contains_no_credential_shaped_token(self, path):
        """No long high-entropy token sits in a notebook.

        Detection is by *shape*, never by matching a known key. A test that
        contains a real credential in order to look for it has published that
        credential — the file it lives in is committed to a public repo.
        """
        import re

        text = path.read_text()
        # Vendor prefixes are documented public formats, not secrets.
        for prefix in (r"KGAT_", r"rf_", r"sk-", r"ghp_", r"AKIA"):
            hit = re.search(rf"{prefix}[A-Za-z0-9_\-]{{16,}}", text)
            assert hit is None, (
                f"{path.name} contains a token matching the {prefix!r} format. "
                "Move it to Colab's secret store and rotate it — anything "
                "committed here is public and must be treated as compromised."
            )

    def test_api_keys_are_read_not_literal(self, path):
        """No cell assigns an API key to a literal string."""
        import re

        pattern = re.compile(r"""(api_key|API_KEY|token|TOKEN)\s*=\s*["'][A-Za-z0-9_\-]{15,}["']""")
        for i, cell in enumerate(_cells(path)):
            if cell["cell_type"] != "code":
                continue
            match = pattern.search(_source(cell))
            assert match is None, (
                f"{path.name} cell {i} assigns a literal credential: {match.group(1)}. "
                "Read it with alpr.env.get_credential instead."
            )

"""
Pytest bootstrap — isolates the test suite from the working database.

Why this file exists
--------------------
`tests/test_pipeline.py` has an autouse fixture that runs
`db.query(VehicleEvent).delete()` so each test starts from a known state.
That is reasonable for a test database and catastrophic for a real one: the
suite previously inherited the app's default `sqlite:///./dev.db`, so running
`pytest` silently deleted every vehicle event in the working database. During
this project that destroyed a fully-loaded city-scale demo dataset once
already.

The fix is to point the tests at their own throwaway SQLite file *before* any
backend module is imported. `backend/database.py` builds its engine at import
time from `get_settings()`, which is `lru_cache`d, so the environment has to be
set here — pytest imports `conftest.py` before collecting test modules, which
makes this the only reliable place.

An explicit `DATABASE_URL` in the environment still wins, so CI or a developer
can deliberately target another database.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Only override when the caller has not chosen a database themselves.
if not os.environ.get("DATABASE_URL"):
    _test_db = Path(tempfile.gettempdir()) / "vehicle_intelligence_tests.db"
    # Start each session from a clean file so leftovers from a previous run
    # cannot make a test pass (or fail) for the wrong reason.
    if _test_db.exists():
        _test_db.unlink()
    os.environ["DATABASE_URL"] = f"sqlite:///{_test_db}"

# Keep test runs off the real message broker / cache even if a .env enables them.
os.environ.setdefault("USE_KAFKA", "false")
os.environ.setdefault("USE_REDIS", "false")

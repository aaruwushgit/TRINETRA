#!/usr/bin/env python3
"""
Add the attribute-only identity columns to an existing vehicle_events table.

`init_db()` calls `create_all`, which creates missing *tables* but never adds a
column to a table that already exists. The production database here holds 12M
rows and 6.5 GB, so the five new nullable columns have to be added in place.

This is cheap and safe on SQLite: `ALTER TABLE ... ADD COLUMN` with a NULL
default is a metadata-only change — it rewrites the schema row, not the 12M
data rows — so it completes in milliseconds regardless of table size. The index
on `plate_partial` does have to be built, but only over the rows where it is
non-NULL, which on an existing database is none of them.

Idempotent: already-present columns are skipped, so it is safe to run on a
fresh database, on a partially migrated one, and again after that.

Usage:
  .venv/bin/python scripts/migrate_add_attribute_columns.py
  .venv/bin/python scripts/migrate_add_attribute_columns.py --db-url sqlite:///./scratch.db
  .venv/bin/python scripts/migrate_add_attribute_columns.py --include-sandboxes
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# (column, SQL type). Every one is nullable with no default — that is what
# keeps the ALTER metadata-only.
NEW_COLUMNS = [
    ("vehicle_make", "VARCHAR(50)"),
    ("vehicle_model", "VARCHAR(50)"),
    ("plate_partial", "VARCHAR(20)"),
    ("plate_raw", "VARCHAR(32)"),
    ("attribute_confidence", "FLOAT"),
]

NEW_INDEXES = [
    ("ix_vehicle_events_plate_partial", "vehicle_events (plate_partial)"),
]


def migrate(engine, label: str) -> int:
    from sqlalchemy import text

    added = 0
    with engine.begin() as conn:
        tables = {
            row[0] for row in
            conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).all()
        }
        if "vehicle_events" not in tables:
            print(f"  {label}: no vehicle_events table — nothing to migrate")
            return 0

        existing = {
            row[1] for row in conn.execute(text("PRAGMA table_info(vehicle_events)")).all()
        }
        for name, sql_type in NEW_COLUMNS:
            if name in existing:
                continue
            t0 = time.perf_counter()
            conn.execute(text(f"ALTER TABLE vehicle_events ADD COLUMN {name} {sql_type}"))
            print(f"  {label}: + {name:<22} ({(time.perf_counter() - t0) * 1000:.1f} ms)")
            added += 1

        for index_name, target in NEW_INDEXES:
            t0 = time.perf_counter()
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {target}"))
            print(f"  {label}: index {index_name} ready ({(time.perf_counter() - t0) * 1000:.1f} ms)")

    if added == 0:
        print(f"  {label}: already up to date")
    return added


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-url", default=None, help="override DATABASE_URL")
    ap.add_argument("--include-sandboxes", action="store_true",
                    help="also migrate every sandbox_dbs/*.db (old sandbox runs)")
    args = ap.parse_args()

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    from sqlalchemy import create_engine

    from backend.config import get_settings

    get_settings.cache_clear()
    db_url = args.db_url or get_settings().DATABASE_URL
    if "sqlite" not in db_url:
        print("This migration is written for SQLite. For Postgres, generate an "
              "Alembic revision from the updated model instead.")
        raise SystemExit(2)

    print(f"Migrating {db_url}")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    migrate(engine, "main")
    engine.dispose()

    if args.include_sandboxes:
        sandbox_dir = BASE_DIR / "sandbox_dbs"
        for path in sorted(sandbox_dir.glob("*.db")):
            eng = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
            try:
                migrate(eng, path.name)
            finally:
                eng.dispose()

    print("\nDone.")


if __name__ == "__main__":
    main()

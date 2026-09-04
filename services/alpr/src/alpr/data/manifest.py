"""Reading and writing the dataset manifest.

JSONL, one record per line: appendable without rewriting, diffable in git, and
readable line-by-line so a 100k-image manifest never has to sit in memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from alpr.data.schema import DatasetError, ImageRecord


def write_manifest(records: Iterable[ImageRecord], path: str | Path) -> int:
    """Write records as JSONL. Returns the number written.

    Writes to a temporary file and renames, so an interrupted write leaves the
    previous manifest intact rather than a half-written one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    count = 0
    seen: set[str] = set()
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            for record in records:
                if record.image_id in seen:
                    raise DatasetError(f"duplicate image_id {record.image_id!r}")
                seen.add(record.image_id)
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return count


def iter_manifest(path: str | Path) -> Iterator[ImageRecord]:
    """Stream records from a JSONL manifest.

    Raises:
        DatasetError: on malformed JSON or a record failing validation, naming
            the line number — with 50k lines, "invalid box" alone is useless.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{lineno}: malformed JSON: {exc}") from exc
            try:
                yield ImageRecord.from_dict(payload)
            except (DatasetError, KeyError, TypeError, ValueError) as exc:
                raise DatasetError(f"{path}:{lineno}: {exc}") from exc


def read_manifest(path: str | Path) -> list[ImageRecord]:
    """Load a whole manifest, rejecting duplicate ids."""
    records = list(iter_manifest(path))
    seen: set[str] = set()
    for record in records:
        if record.image_id in seen:
            raise DatasetError(f"duplicate image_id {record.image_id!r} in {path}")
        seen.add(record.image_id)
    return records

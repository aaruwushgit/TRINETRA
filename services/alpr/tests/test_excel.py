"""Excel logging: crash safety, deduplication, and the file-locked case."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from alpr.data.schema import Region
from alpr.excel import (
    COLUMNS,
    ExcelLog,
    ExcelLogError,
    PlateEvent,
    read_journal,
    read_workbook,
    recover,
    write_workbook,
)

T0 = datetime(2026, 8, 9, 12, 0, 0)


def event(plate="MH12AB1234", when=None, **kwargs) -> PlateEvent:
    return PlateEvent(
        plate=plate,
        timestamp=when or T0,
        region=kwargs.pop("region", Region.INDIA),
        display=kwargs.pop("display", "MH 12 AB 1234"),
        confidence=kwargs.pop("confidence", 0.95),
        **kwargs,
    )


class TestPlateEvent:
    def test_json_round_trip(self):
        original = event(track_id=7, frame=120, source="clip.mp4", crop_path="crops/1.jpg")
        assert PlateEvent.from_json(original.to_json()) == original

    def test_row_matches_the_column_order(self):
        assert len(event().row()) == len(COLUMNS)

    def test_row_renders_region_as_its_code(self):
        row = dict(zip([c[0] for c in COLUMNS], event().row(), strict=True))
        assert row["region"] == "IN"

    def test_timezone_aware_timestamps_are_made_writable(self):
        # openpyxl refuses aware datetimes outright; converting to local and
        # stripping is the only way to get the value into a cell.
        aware = event(when=datetime(2026, 8, 9, 12, 0, tzinfo=UTC))
        row = dict(zip([c[0] for c in COLUMNS], aware.row(), strict=True))
        assert row["timestamp"].tzinfo is None


class TestWriteWorkbook:
    def test_writes_a_readable_sheet(self, tmp_path):
        path = write_workbook(tmp_path / "log.xlsx", [event(), event("DAXY123")])
        rows = read_workbook(path)
        assert len(rows) == 2
        assert rows[0]["Plate"] == "MH12AB1234"

    def test_header_row_is_present(self, tmp_path):
        path = write_workbook(tmp_path / "log.xlsx", [event()])
        assert set(read_workbook(path)[0]) == {header for _, header, _ in COLUMNS}

    def test_leaves_no_temp_file(self, tmp_path):
        write_workbook(tmp_path / "log.xlsx", [event()])
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrites_an_existing_workbook(self, tmp_path):
        path = tmp_path / "log.xlsx"
        write_workbook(path, [event()])
        write_workbook(path, [event(), event("DAXY123")])
        assert len(read_workbook(path)) == 2

    def test_reports_a_locked_file_without_losing_data(self, tmp_path, monkeypatch):
        # The mundane real-world failure: the user has the sheet open in Excel,
        # which locks it on Windows. This must not crash a running job.
        path = tmp_path / "log.xlsx"

        def _locked(self, target):
            raise PermissionError(13, "in use")

        monkeypatch.setattr("pathlib.Path.replace", _locked)
        with pytest.raises(ExcelLogError, match="open in another program"):
            write_workbook(path, [event()])
        assert not list(tmp_path.glob("*.tmp"))


class TestExcelLog:
    def test_writes_rows_on_exit(self, tmp_path):
        path = tmp_path / "log.xlsx"
        with ExcelLog(path) as log:
            log.add(event())
            log.add(event("DAXY123", region=Region.GERMANY))
        assert len(read_workbook(path)) == 2

    def test_journals_every_event_immediately(self, tmp_path):
        # The whole point: an event survives a crash before any save.
        path = tmp_path / "log.xlsx"
        with ExcelLog(path, flush_every=1000) as log:
            log.add(event())
            assert not path.exists()  # not yet written to xlsx
            assert len(list(read_journal(log.journal_path))) == 1

    def test_recovers_a_run_that_never_flushed(self, tmp_path):
        path = tmp_path / "log.xlsx"
        log = ExcelLog(path, flush_every=1000)
        log.open()
        log.add(event())
        log.add(event("DAXY123"))
        log._journal.close()  # simulate the process dying

        assert not path.exists()
        recover(path)
        assert len(read_workbook(path)) == 2

    def test_resuming_does_not_relog_earlier_plates(self, tmp_path):
        path = tmp_path / "log.xlsx"
        with ExcelLog(path, cooldown=timedelta(minutes=5)) as log:
            log.add(event(when=T0))

        with ExcelLog(path, cooldown=timedelta(minutes=5)) as log:
            # Same plate, one minute later: the resumed run must know about it.
            assert log.add(event(when=T0 + timedelta(minutes=1))) is False
        assert len(read_workbook(path)) == 1

    def test_deduplicates_within_the_cooldown(self, tmp_path):
        path = tmp_path / "log.xlsx"
        with ExcelLog(path, cooldown=timedelta(minutes=5)) as log:
            assert log.add(event(when=T0)) is True
            assert log.add(event(when=T0 + timedelta(seconds=30))) is False
            assert log.suppressed == 1
        assert len(read_workbook(path)) == 1

    def test_deduplication_can_be_disabled(self, tmp_path):
        path = tmp_path / "log.xlsx"
        with ExcelLog(path, deduplicate=False) as log:
            log.add(event(when=T0))
            log.add(event(when=T0))
        assert len(read_workbook(path)) == 2

    def test_periodic_flush(self, tmp_path):
        path = tmp_path / "log.xlsx"
        with ExcelLog(path, flush_every=2, deduplicate=False) as log:
            log.add(event(when=T0))
            assert not path.exists()
            log.add(event(when=T0))
            assert path.exists()

    def test_rejects_use_before_open(self, tmp_path):
        with pytest.raises(ExcelLogError, match="not open"):
            ExcelLog(tmp_path / "log.xlsx").add(event())

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "log.xlsx"
        with ExcelLog(path) as log:
            log.add(event())
        assert path.exists()

    def test_summary_mentions_suppressions(self, tmp_path):
        with ExcelLog(tmp_path / "log.xlsx", cooldown=timedelta(minutes=5)) as log:
            log.add(event(when=T0))
            log.add(event(when=T0))
            summary = log.summary()
        assert "1 duplicate" in summary

    def test_summary_counts_logged_events_not_flushed_rows(self, tmp_path):
        # Reporting the flushed count made a healthy run read "0 row(s)
        # written" before the first flush — true of the workbook, misleading
        # about the run, since every event was already durable in the journal.
        with ExcelLog(tmp_path / "log.xlsx", flush_every=1000, deduplicate=False) as log:
            log.add(event(when=T0))
            log.add(event(when=T0))
            summary = log.summary()
            assert log.logged == 2
        assert "2 plate(s) logged" in summary
        assert "not yet flushed" in summary


class TestJournal:
    def test_skips_a_torn_final_line(self, tmp_path):
        # A crash mid-write leaves the last line incomplete. Only that line
        # may be dropped — that is why the journal is one event per line.
        journal = tmp_path / "log.xlsx.jsonl"
        journal.write_text(event().to_json() + "\n" + '{"plate": "DAX', encoding="utf-8")
        assert len(list(read_journal(journal))) == 1

    def test_corruption_in_the_middle_is_an_error(self, tmp_path):
        # Mid-file corruption is not a torn write; silently dropping rows
        # would misreport the run.
        journal = tmp_path / "log.xlsx.jsonl"
        journal.write_text(
            "{bad}\n" + event().to_json() + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            list(read_journal(journal))

    def test_missing_journal_yields_nothing(self, tmp_path):
        assert list(read_journal(tmp_path / "absent.jsonl")) == []

    def test_recover_without_a_journal_raises(self, tmp_path):
        with pytest.raises(ExcelLogError, match="nothing to recover"):
            recover(tmp_path / "log.xlsx")

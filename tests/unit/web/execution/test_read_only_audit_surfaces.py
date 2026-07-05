"""Web Landscape read surfaces open the audit DB read-only (option-c slice 1).

The four ``*_for_settings`` execution loaders (accounting, discard summaries,
outputs, diagnostics) and the sessions audit-story route serve HTTP reads from
the Landscape audit database. Slice 1 of ADR-030 flips every one of them to
``LandscapeDB.from_url(..., read_only=True, create_tables=False)`` so web
reads provably never contend for the WAL write lock, never run DDL, and never
create a missing audit DB file.

This module pins:

- each converted loader still serves its reads against a real file-backed
  audit DB, AND the handle it opened carries ``read_only=True``;
- a write attempted through a read-only handle fails loudly at BOTH layers
  (PRAGMA query_only inside ``read_only_connection`` and the SQLite
  ``mode=ro`` file handle underneath it).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import insert
from sqlalchemy.exc import OperationalError

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import runs_table
from elspeth.web.execution.accounting import load_run_accounting_for_settings
from elspeth.web.execution.diagnostics import load_run_diagnostics_for_settings
from elspeth.web.execution.discard_summary import load_discard_summaries_for_settings
from elspeth.web.execution.outputs import load_run_outputs_for_settings

RUN_ID = "ro-surface-run"


@dataclass(frozen=True)
class _SettingsFake:
    landscape_url: str
    landscape_passphrase: str | None = None
    data_dir: Path | None = None

    def get_landscape_url(self) -> str:
        return self.landscape_url


@pytest.fixture
def audit_db_url(tmp_path: Path) -> str:
    """File-backed audit DB seeded with one (empty) run, then closed."""
    url = f"sqlite:///{tmp_path / 'audit.db'}"
    with LandscapeDB.from_url(url) as db:
        RecorderFactory(db).run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id=RUN_ID,
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
    return url


def _settings(url: str, data_dir: Path | None = None) -> _SettingsFake:
    return _SettingsFake(landscape_url=url, data_dir=data_dir)


@pytest.fixture
def from_url_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the kwargs of every ``LandscapeDB.from_url`` call."""
    calls: list[dict[str, Any]] = []
    real = LandscapeDB.from_url.__func__  # type: ignore[attr-defined]  # bound classmethod → underlying function

    def _spy(cls: type[LandscapeDB], /, url: str, **kwargs: Any) -> LandscapeDB:
        calls.append(dict(kwargs))
        return real(cls, url, **kwargs)  # type: ignore[no-any-return]  # delegates to the real classmethod

    monkeypatch.setattr(LandscapeDB, "from_url", classmethod(_spy))
    return calls


def _assert_single_read_only_open(calls: list[dict[str, Any]]) -> None:
    assert len(calls) == 1, f"expected exactly one from_url open, saw {len(calls)}"
    assert calls[0].get("read_only") is True
    assert calls[0].get("create_tables") is False


class TestConvertedLoadersServeReadsReadOnly:
    def test_run_accounting_for_settings(self, audit_db_url: str, from_url_spy: list[dict[str, Any]]) -> None:
        accounting = load_run_accounting_for_settings(_settings(audit_db_url), (RUN_ID, None))

        assert set(accounting) == {RUN_ID}
        assert accounting[RUN_ID].source.rows_processed == 0
        _assert_single_read_only_open(from_url_spy)

    def test_discard_summaries_for_settings(self, audit_db_url: str, from_url_spy: list[dict[str, Any]]) -> None:
        summaries = load_discard_summaries_for_settings(_settings(audit_db_url), (RUN_ID,))

        # A run with zero discards is pruned from the result; the three
        # aggregation SELECTs still executed against the read-only handle.
        assert summaries == {}
        _assert_single_read_only_open(from_url_spy)

    def test_run_outputs_for_settings(self, audit_db_url: str, tmp_path: Path, from_url_spy: list[dict[str, Any]]) -> None:
        response = load_run_outputs_for_settings(
            _settings(audit_db_url, data_dir=tmp_path),
            run_id="web-run",
            landscape_run_id=RUN_ID,
        )

        assert response.landscape_run_id == RUN_ID
        assert response.artifacts == []
        _assert_single_read_only_open(from_url_spy)

    def test_run_diagnostics_for_settings(self, audit_db_url: str, from_url_spy: list[dict[str, Any]]) -> None:
        response = load_run_diagnostics_for_settings(
            _settings(audit_db_url),
            run_id="web-run",
            landscape_run_id=RUN_ID,
            run_status="completed",
        )

        assert response.landscape_run_id == RUN_ID
        assert response.tokens == []
        _assert_single_read_only_open(from_url_spy)

    def test_missing_audit_file_is_not_created_by_loaders(self, tmp_path: Path) -> None:
        """``mode=ro`` cannot create a DB file; the guards return empty first."""
        missing = tmp_path / "never-created.db"
        settings = _settings(f"sqlite:///{missing}")

        assert load_run_accounting_for_settings(settings, (RUN_ID,)) == {}
        assert load_discard_summaries_for_settings(settings, (RUN_ID,)) == {}
        assert not missing.exists()


class TestReadOnlyHandleRejectsWrites:
    def test_write_through_read_only_handle_fails_loudly(self, audit_db_url: str) -> None:
        """INSERT through a read-only handle raises; both defense layers hold."""
        with LandscapeDB.from_url(audit_db_url, read_only=True, create_tables=False) as ro:
            # Layer 1+2 combined: ``connection()`` on a read-only handle routes
            # to ``read_only_connection`` (PRAGMA query_only=ON) on top of the
            # ``mode=ro`` file handle.
            with pytest.raises(OperationalError, match=r"readonly|query_only"), ro.connection() as conn:
                conn.execute(insert(runs_table).values(run_id="forbidden"))

            # Layer 2 alone: even bypassing read_only_connection and going to
            # the raw engine, the SQLite file handle itself is ``mode=ro``.
            with pytest.raises(OperationalError, match="readonly"), ro.engine.begin() as conn:
                conn.exec_driver_sql("DELETE FROM runs")

            # The handle is still good for reads after both rejections.
            with ro.connection() as conn:
                rows = conn.exec_driver_sql("SELECT run_id FROM runs").fetchall()
            assert [row[0] for row in rows] == [RUN_ID]

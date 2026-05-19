"""Column-presence and write-side smoke tests for Phase 4 audit-story fields."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from elspeth.contracts import RunStatus
from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
from tests.fixtures.landscape import make_factory, make_landscape_db


def test_runs_table_has_audit_story_columns() -> None:
    expected_new = {
        "llm_call_count",
        "seeded_from_cache",
        "cache_key",
    }
    actual = {col.name for col in runs_table.columns}

    missing = expected_new - actual

    assert not missing, f"Missing columns on runs_table: {missing}"
    assert "started_at" in actual
    assert "source_data_hash" not in actual
    assert "plugin_versions" not in actual


def test_llm_call_count_is_nullable_integer() -> None:
    col = runs_table.c.llm_call_count
    assert col.nullable


def test_seeded_from_cache_has_server_default() -> None:
    col = runs_table.c.seeded_from_cache
    assert not col.nullable
    assert col.server_default is not None


def test_live_begin_run_writes_non_cache_defaults() -> None:
    db = make_landscape_db()
    factory = make_factory(db)

    run = factory.run_lifecycle.begin_run(
        config={"pipeline": "test"},
        canonical_version="v1",
        run_id="run-live",
    )

    assert run.llm_call_count is None
    assert run.seeded_from_cache is False
    assert run.cache_key is None
    with db.connection() as conn:
        row = conn.execute(select(runs_table).where(runs_table.c.run_id == "run-live")).one()
    assert row.llm_call_count is None
    assert row.seeded_from_cache is False
    assert row.cache_key is None


def test_landscape_write_repository_records_synthesised_cache_run() -> None:
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    run_id = repo.record_synthesised_run(
        pipeline_yaml="source: {}\n",
        rows=[{"url": "ato.gov.au", "rating": 5}],
        source_data_hash="a7f3e2cached",
        llm_call_count=0,
        plugin_versions={"inline_blob": "1.0", "tutorial_summary": "1.0"},
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": True, "cache_key": "a" * 64},
    )

    with db.connection() as conn:
        run_row = conn.execute(select(runs_table).where(runs_table.c.run_id == run_id)).one()
        row_count = conn.execute(select(rows_table.c.row_id).where(rows_table.c.run_id == run_id)).all()
        node_count = conn.execute(select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id)).all()

    assert run_row.status == RunStatus.COMPLETED.value
    assert run_row.llm_call_count == 0
    assert run_row.seeded_from_cache is True
    assert run_row.cache_key == "a" * 64
    assert len(row_count) == 1
    assert len(node_count) == 2


def test_landscape_core_keeps_web_session_identifiers_out_of_audit_schema() -> None:
    """Mirror the staging reset runbook's stop/go grep.

    Landscape replay rows are joined to web-session runs through the web
    session DB's `runs.landscape_run_id`, not by embedding session identifiers
    into Landscape rows.
    """

    landscape_root = Path(__file__).parents[4] / "src" / "elspeth" / "core" / "landscape"
    forbidden = ("session_id", "chat_message_id", "composition_state_id")
    hits: list[str] = []
    for path in landscape_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                hits.append(f"{path.relative_to(landscape_root)}:{token}")

    assert hits == []

"""Column-presence and write-side smoke tests for Phase 4 audit-story fields."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from elspeth.contracts import NodeType, RunStatus
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
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    run_id = repo.record_synthesised_run(
        pipeline_yaml="source: {}\n",
        rows=[{"url": "ato.gov.au", "rating": 5}],
        source_data_hash="a7f3e2cached",
        llm_call_count=0,
        node_specs=[
            SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="inline_blob", plugin_version="1.0"),
            SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="tutorial_summary", plugin_version="1.0"),
        ],
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": True, "cache_key": "a" * 64},
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
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


def test_synthesised_run_records_one_node_per_occurrence_with_plugin_reuse() -> None:
    """Tier-1 audit topology must not collapse when a pipeline reuses a plugin name.

    Pipeline shape under test:
        source (csv) -> transform_a (llm) -> transform_b (llm) -> sink_primary (csv) -> sink_backup (jsonl)

    Two distinct bugs the prior implementation produced from this shape:
      1. dict[name -> version] dedupes by plugin name, dropping reuses.
      2. _node_type derived role from list-index assumed exactly one sink.

    The audit trail must record one nodes row per YAML occurrence, with the
    NodeType taken from the YAML's role (source/transforms/sinks), not from
    position in a deduplicated list.
    """
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    node_specs = [
        SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1.0"),
        SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="2.3"),
        SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="2.3"),
        SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1.0"),
        SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="jsonl", plugin_version="1.4"),
    ]

    run_id = repo.record_synthesised_run(
        pipeline_yaml="# canonical reuse + multi-sink fixture\n",
        rows=[{"url": "ato.gov.au"}],
        source_data_hash="hash-mixed-roles",
        llm_call_count=2,
        node_specs=node_specs,
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": True, "cache_key": "d" * 64},
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )

    with db.connection() as conn:
        recorded = conn.execute(
            select(
                nodes_table.c.plugin_name,
                nodes_table.c.plugin_version,
                nodes_table.c.node_type,
                nodes_table.c.sequence_in_pipeline,
            )
            .where(nodes_table.c.run_id == run_id)
            .order_by(nodes_table.c.sequence_in_pipeline)
        ).all()

    # One nodes row per YAML occurrence — no silent drops.
    assert len(recorded) == len(node_specs), (
        f"Expected {len(node_specs)} nodes rows (one per YAML occurrence) — got {len(recorded)}."
        " Tier-1 audit topology must not collapse by plugin name."
    )

    # Role carried from YAML — not derived from list index.
    actual = [(r.plugin_name, r.plugin_version, r.node_type, r.sequence_in_pipeline) for r in recorded]
    expected = [
        ("csv", "1.0", NodeType.SOURCE.value, 0),
        ("llm", "2.3", NodeType.TRANSFORM.value, 1),
        ("llm", "2.3", NodeType.TRANSFORM.value, 2),
        ("csv", "1.0", NodeType.SINK.value, 3),
        ("jsonl", "1.4", NodeType.SINK.value, 4),
    ]
    assert actual == expected, (
        "Tier-1 audit nodes row mismatch.\n"
        f"  expected: {expected}\n"
        f"  recorded: {actual}\n"
        "Role must come from the YAML's source/transforms/sinks key, not from list position."
    )


def test_synthesised_run_rejects_misshapen_node_specs() -> None:
    """Writer must crash on structural violations rather than emit a partial Tier-1 audit row.

    CLAUDE.md: exactly one source per run; one or more sinks. A synthesised
    audit record that doesn't honour those invariants is a topology lie.
    """
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    repo = LandscapeWriteRepository(make_landscape_db())

    base_kwargs: dict[str, object] = {
        "pipeline_yaml": "# fixture\n",
        "rows": [{"url": "x"}],
        "source_data_hash": "h",
        "llm_call_count": 0,
        "started_at": datetime(2026, 5, 15, tzinfo=UTC),
        "metadata": {"seeded_from_cache": True, "cache_key": "e" * 64},
        "openrouter_catalog_sha256": "0" * 64,
        "openrouter_catalog_source": "bundled",
    }

    with pytest.raises(LandscapeRecordError, match="at least one node"):
        repo.record_synthesised_run(node_specs=[], **base_kwargs)  # type: ignore[arg-type]

    with pytest.raises(LandscapeRecordError, match="first node must be SOURCE"):
        repo.record_synthesised_run(  # type: ignore[arg-type]
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ],
            **base_kwargs,
        )

    with pytest.raises(LandscapeRecordError, match="exactly one SOURCE"):
        repo.record_synthesised_run(  # type: ignore[arg-type]
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="api", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ],
            **base_kwargs,
        )

    with pytest.raises(LandscapeRecordError, match="at least one SINK"):
        repo.record_synthesised_run(  # type: ignore[arg-type]
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="1"),
            ],
            **base_kwargs,
        )


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

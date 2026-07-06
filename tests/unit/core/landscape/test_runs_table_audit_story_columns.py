"""Column-presence and write-side smoke tests for Phase 4 audit-story fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
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
        source_data_hash="1" * 64,
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


def test_synthesised_cache_run_rejects_unexpected_metadata_keys() -> None:
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    with pytest.raises(LandscapeRecordError, match="unexpected metadata keys"):
        repo.record_synthesised_run(
            pipeline_yaml="source: {}\n",
            rows=[{"url": "ato.gov.au"}],
            source_data_hash="a" * 64,
            llm_call_count=0,
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="inline_blob", plugin_version="1.0"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="tutorial_summary", plugin_version="1.0"),
            ],
            started_at=datetime(2026, 5, 15, tzinfo=UTC),
            metadata={
                "seeded_from_cache": True,
                "cache_key": "a" * 64,
                "authorization": "Bearer raw-secret",
            },
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )

    with db.connection() as conn:
        recorded_runs = conn.execute(select(runs_table.c.run_id)).all()

    assert recorded_runs == []


def test_synthesised_single_source_rows_keep_legacy_identity_when_payload_keys_collide() -> None:
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    run_id = repo.record_synthesised_run(
        pipeline_yaml="source: {}\n",
        rows=[
            {
                "url": "ato.gov.au",
                "source_node_index": 99,
                "source_row_index": 88,
                "ingest_sequence": 77,
                "source_data_hash": "payload-field-not-audit-identity",
            }
        ],
        source_data_hash="2" * 64,
        llm_call_count=0,
        node_specs=[
            SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="inline_blob", plugin_version="1.0"),
            SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="tutorial_summary", plugin_version="1.0"),
        ],
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": True, "cache_key": "b" * 64},
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )

    with db.connection() as conn:
        row = conn.execute(select(rows_table).where(rows_table.c.run_id == run_id)).one()

    assert row.source_node_id == f"{run_id}-n0"
    assert row.source_row_index == 0
    assert row.ingest_sequence == 0
    assert row.source_data_hash == "2" * 64


def test_synthesised_single_source_rows_reject_invalid_fallback_source_data_hash() -> None:
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    with pytest.raises(LandscapeRecordError, match="source_data_hash must be 64 lowercase hex chars"):
        repo.record_synthesised_run(
            pipeline_yaml="source: {}\n",
            rows=[{"url": "ato.gov.au"}],
            source_data_hash="not-a-sha",
            llm_call_count=0,
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="inline_blob", plugin_version="1.0"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="tutorial_summary", plugin_version="1.0"),
            ],
            started_at=datetime(2026, 5, 15, tzinfo=UTC),
            metadata={"seeded_from_cache": True, "cache_key": "b" * 64},
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )

    with db.connection() as conn:
        recorded_rows = conn.execute(select(rows_table.c.row_id)).all()

    assert recorded_rows == []


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
        source_data_hash="3" * 64,
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


def test_synthesised_run_records_multi_source_row_identity() -> None:
    """Multi-source cache replay must record explicit per-source row identity."""
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    node_specs = [
        SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1.0"),
        SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="json", plugin_version="1.2"),
        SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1.0"),
    ]

    run_id = repo.record_synthesised_run(
        pipeline_yaml="# canonical multi-source fixture\n",
        rows=[
            {
                "source_node_index": 0,
                "source_row_index": 0,
                "ingest_sequence": 0,
                "source_data_hash": "4" * 64,
            },
            {
                "source_node_index": 1,
                "source_row_index": 0,
                "ingest_sequence": 1,
                "source_data_hash": "5" * 64,
            },
            {
                "source_node_index": 0,
                "source_row_index": 1,
                "ingest_sequence": 2,
                "source_data_hash": "6" * 64,
            },
        ],
        source_data_hash="7" * 64,
        llm_call_count=1,
        node_specs=node_specs,
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": True, "cache_key": "f" * 64},
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )

    with db.connection() as conn:
        rows = conn.execute(
            select(
                rows_table.c.source_node_id,
                rows_table.c.row_index,
                rows_table.c.source_row_index,
                rows_table.c.ingest_sequence,
                rows_table.c.source_data_hash,
            )
            .where(rows_table.c.run_id == run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()

    assert [(row.source_node_id, row.row_index, row.source_row_index, row.ingest_sequence, row.source_data_hash) for row in rows] == [
        (f"{run_id}-n0", 0, 0, 0, "4" * 64),
        (f"{run_id}-n1", 1, 0, 1, "5" * 64),
        (f"{run_id}-n0", 2, 1, 2, "6" * 64),
    ]


def test_synthesised_multi_source_rows_reject_invalid_explicit_source_data_hash() -> None:
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    db = make_landscape_db()
    repo = LandscapeWriteRepository(db)

    with pytest.raises(LandscapeRecordError, match="source_data_hash must be 64 lowercase hex chars"):
        repo.record_synthesised_run(
            pipeline_yaml="# canonical multi-source fixture\n",
            rows=[
                {
                    "source_node_index": 0,
                    "source_row_index": 0,
                    "ingest_sequence": 0,
                    "source_data_hash": "not-a-sha",
                }
            ],
            source_data_hash="a" * 64,
            llm_call_count=1,
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1.0"),
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="json", plugin_version="1.2"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1.0"),
            ],
            started_at=datetime(2026, 5, 15, tzinfo=UTC),
            metadata={"seeded_from_cache": True, "cache_key": "f" * 64},
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )

    with db.connection() as conn:
        recorded_rows = conn.execute(select(rows_table.c.row_id)).all()

    assert recorded_rows == []


def test_synthesised_run_rejects_misshapen_node_specs() -> None:
    """Writer must crash on structural violations rather than emit a partial Tier-1 audit row.

    Canonical settings allow one or more leading sources and one or more
    sinks. A synthesised audit record that doesn't honour those invariants is
    a topology lie.
    """
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    repo = LandscapeWriteRepository(make_landscape_db())

    def record(node_specs: Sequence[SynthesisedNodeSpec], rows: Sequence[Mapping[str, object]] = ({"url": "x"},)) -> str:
        return repo.record_synthesised_run(
            pipeline_yaml="# fixture\n",
            rows=rows,
            source_data_hash="8" * 64,
            llm_call_count=0,
            node_specs=node_specs,
            started_at=datetime(2026, 5, 15, tzinfo=UTC),
            metadata={"seeded_from_cache": True, "cache_key": "e" * 64},
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )

    with pytest.raises(LandscapeRecordError, match="at least one node"):
        record([])

    with pytest.raises(LandscapeRecordError, match="first node must be SOURCE"):
        record(
            [
                SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ]
        )

    with pytest.raises(LandscapeRecordError, match="SOURCE nodes must precede"):
        record(
            [
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="api", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ]
        )

    with pytest.raises(LandscapeRecordError, match="at least one SINK"):
        record(
            [
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.TRANSFORM, plugin_name="llm", plugin_version="1"),
            ]
        )

    with pytest.raises(LandscapeRecordError, match="multi-source synthesised rows require explicit row identity"):
        record(
            [
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="api", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ]
        )


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (
            [{"source_node_index": 0, "source_row_index": 0, "ingest_sequence": 0}],
            "missing .*source_data_hash",
        ),
        (
            [{"source_node_index": 2, "source_row_index": 0, "ingest_sequence": 0, "source_data_hash": "a" * 64}],
            "does not reference a SOURCE node",
        ),
        (
            [{"source_node_index": True, "source_row_index": 0, "ingest_sequence": 0, "source_data_hash": "a" * 64}],
            "source_node_index must be a non-negative integer",
        ),
        (
            [{"source_node_index": 0, "source_row_index": -1, "ingest_sequence": 0, "source_data_hash": "a" * 64}],
            "source_row_index must be a non-negative integer",
        ),
        (
            [{"source_node_index": 0, "source_row_index": 0, "ingest_sequence": "0", "source_data_hash": "a" * 64}],
            "ingest_sequence must be a non-negative integer",
        ),
        (
            [{"source_node_index": 0, "source_row_index": 0, "ingest_sequence": 0, "source_data_hash": ""}],
            "source_data_hash must be 64 lowercase hex chars",
        ),
    ],
)
def test_synthesised_multi_source_rows_reject_malformed_identity(rows: Sequence[Mapping[str, object]], match: str) -> None:
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec

    repo = LandscapeWriteRepository(make_landscape_db())

    with pytest.raises(LandscapeRecordError, match=match):
        repo.record_synthesised_run(
            pipeline_yaml="# fixture\n",
            rows=rows,
            source_data_hash="9" * 64,
            llm_call_count=0,
            node_specs=[
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="csv", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="api", plugin_version="1"),
                SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="csv", plugin_version="1"),
            ],
            started_at=datetime(2026, 5, 15, tzinfo=UTC),
            metadata={"seeded_from_cache": True, "cache_key": "g" * 64},
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
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

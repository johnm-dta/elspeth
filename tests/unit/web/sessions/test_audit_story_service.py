"""Tests for the tutorial audit-story read service."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import update

from elspeth.contracts import NodeType
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.schema import rows_table
from elspeth.core.landscape.write_repository import LandscapeWriteRepository, SynthesisedNodeSpec
from elspeth.web.sessions.audit_story_service import AuditStoryIntegrityError, AuditStoryService
from tests.fixtures.landscape import make_landscape_db


def test_audit_story_reads_real_landscape_rows() -> None:
    db = make_landscape_db()
    write_repo = LandscapeWriteRepository(db)
    run_id = write_repo.record_synthesised_run(
        pipeline_yaml="source: {}\n",
        rows=[{"url": "ato.gov.au", "rating": 5}],
        source_data_hash="a7f3e2cached",
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

    story = AuditStoryService(db).get_run_audit_story(
        run_id,
        public_run_id="session-run-1",
        session_id="session-1",
    )

    assert story.run_id == "session-run-1"
    assert story.session_id == "session-1"
    assert story.llm_call_count == 0
    assert story.source_data_hash == "a7f3e2cached"
    assert "output_file_hash" not in story.model_dump()
    assert story.started_at == datetime(2026, 5, 15, tzinfo=UTC).replace(tzinfo=None) or story.started_at == datetime(
        2026, 5, 15, tzinfo=UTC
    )
    assert story.plugin_versions == {"inline_blob": "1.0", "tutorial_summary": "1.0"}
    assert story.seeded_from_cache is True
    assert story.cache_key == "b" * 64


def test_audit_story_aggregates_multiple_row_source_hashes() -> None:
    db = make_landscape_db()
    write_repo = LandscapeWriteRepository(db)
    run_id = write_repo.record_synthesised_run(
        pipeline_yaml="source: {}\n",
        rows=[{"url": "ato.gov.au"}, {"url": "data.gov.au"}],
        source_data_hash="initial",
        llm_call_count=0,
        node_specs=[
            SynthesisedNodeSpec(node_type=NodeType.SOURCE, plugin_name="inline_blob", plugin_version="1.0"),
            SynthesisedNodeSpec(node_type=NodeType.SINK, plugin_name="tutorial_summary", plugin_version="1.0"),
        ],
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
        metadata={"seeded_from_cache": False, "cache_key": "c" * 64},
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )
    hashes = ("a" * 64, "b" * 64)
    with db.connection() as conn:
        for row_index, source_hash in enumerate(hashes):
            conn.execute(
                update(rows_table)
                .where(rows_table.c.run_id == run_id)
                .where(rows_table.c.row_index == row_index)
                .values(source_data_hash=source_hash)
            )

    story = AuditStoryService(db).get_run_audit_story(
        run_id,
        public_run_id="session-run-1",
        session_id="session-1",
    )

    assert story.source_data_hash == stable_hash({"source_data_hashes": list(hashes)})


def test_audit_story_missing_run_raises_named_error() -> None:
    with pytest.raises(AuditStoryIntegrityError, match="not found"):
        AuditStoryService(make_landscape_db()).get_run_audit_story(
            "missing-run",
            public_run_id="session-run-1",
            session_id="session-1",
        )

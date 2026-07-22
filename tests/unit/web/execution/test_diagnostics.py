"""Tests for web run diagnostics snapshots.

Diagnostics are intentionally a bounded projection over Landscape audit
records. They are for operator visibility and LLM explanation prompts, not a
new audit surface or a payload/context export path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from elspeth.contracts import NodeStateStatus, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.web.execution.diagnostics import load_run_diagnostics_from_db
from elspeth.web.execution.schemas import (
    RunDiagnosticNodeState,
    RunDiagnosticOperation,
    RunDiagnosticSummary,
    RunDiagnosticToken,
)

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_DIAGNOSTIC_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _diagnostic_node_state(**overrides: object) -> RunDiagnosticNodeState:
    payload: dict[str, object] = {
        "state_id": "state-1",
        "token_id": "token-1",
        "node_id": "transform",
        "step_index": 0,
        "attempt": 0,
        "status": "completed",
        "duration_ms": 1.0,
        "started_at": _DIAGNOSTIC_TIME,
        "completed_at": _DIAGNOSTIC_TIME,
    }
    payload.update(overrides)
    return RunDiagnosticNodeState(**payload)


def _diagnostic_token(**overrides: object) -> RunDiagnosticToken:
    payload: dict[str, object] = {
        "token_id": "token-1",
        "row_id": "row-1",
        "row_index": 0,
        "branch_name": None,
        "fork_group_id": None,
        "join_group_id": None,
        "expand_group_id": None,
        "step_in_pipeline": 0,
        "created_at": _DIAGNOSTIC_TIME,
        "terminal_outcome": "success",
        "states": [],
    }
    payload.update(overrides)
    return RunDiagnosticToken(**payload)


def _diagnostic_operation(**overrides: object) -> RunDiagnosticOperation:
    payload: dict[str, object] = {
        "operation_id": "op-1",
        "node_id": "source",
        "operation_type": "source_load",
        "status": "completed",
        "duration_ms": 1.0,
        "started_at": _DIAGNOSTIC_TIME,
        "completed_at": _DIAGNOSTIC_TIME,
        "error_message": None,
    }
    payload.update(overrides)
    return RunDiagnosticOperation(**payload)


def _diagnostic_summary(**overrides: object) -> RunDiagnosticSummary:
    payload: dict[str, object] = {
        "token_count": 1,
        "preview_limit": 50,
        "preview_truncated": False,
        "state_counts": {"completed": 1},
        "operation_counts": {"source_load": 1},
        "latest_activity_at": _DIAGNOSTIC_TIME,
    }
    payload.update(overrides)
    return RunDiagnosticSummary(**payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "not_a_state"),
        ("duration_ms", -1.0),
        ("duration_ms", float("nan")),
    ],
)
def test_diagnostic_node_state_rejects_impossible_contract_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _diagnostic_node_state(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_index", -7),
        ("step_in_pipeline", -2),
        ("terminal_outcome", "not_an_outcome"),
    ],
)
def test_diagnostic_token_rejects_impossible_contract_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _diagnostic_token(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("operation_type", "not_an_operation"),
        ("status", "not_a_status"),
        ("duration_ms", -1.0),
        ("duration_ms", float("nan")),
    ],
)
def test_diagnostic_operation_rejects_impossible_contract_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _diagnostic_operation(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_counts", {"not_a_state": 1}),
        ("state_counts", {"completed": -1}),
        ("operation_counts", {"not_an_operation": 1}),
        ("operation_counts", {"source_load": -1}),
    ],
)
def test_diagnostic_summary_rejects_impossible_contract_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        _diagnostic_summary(**{field: value})


def _register_node(
    factory: RecorderFactory,
    run_id: str,
    node_id: str,
    node_type: NodeType,
    plugin_name: str,
) -> None:
    factory.data_flow.register_node(
        run_id=run_id,
        node_id=node_id,
        plugin_name=plugin_name,
        node_type=node_type,
        plugin_version="1.0",
        config={},
        schema_config=_OBSERVED_SCHEMA,
    )


def _seed_diagnostics_run(db: LandscapeDB, tmp_path, *, web_run_id: str = "web-run-1") -> None:
    factory = RecorderFactory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=web_run_id)
    _register_node(factory, web_run_id, "source", NodeType.SOURCE, "text")
    _register_node(factory, web_run_id, "extract", NodeType.TRANSFORM, "llm_extract")
    _register_node(factory, web_run_id, "json_out", NodeType.SINK, "json")

    first_row = factory.data_flow.create_row(
        web_run_id, "source", 0, {"html": "<h1>A</h1>"}, row_id="row-0", source_row_index=0, ingest_sequence=0
    )
    second_row = factory.data_flow.create_row(
        web_run_id, "source", 1, {"html": "<h1>B</h1>"}, row_id="row-1", source_row_index=1, ingest_sequence=1
    )
    first_token = factory.data_flow.create_token(first_row.row_id, token_id="token-0")
    second_token = factory.data_flow.create_token(second_row.row_id, token_id="token-1")

    first_state = factory.execution.begin_node_state(
        first_token.token_id,
        "extract",
        web_run_id,
        1,
        {"html": "<h1>A</h1>"},
        state_id="state-token-0",
    )
    factory.execution.complete_node_state(
        first_state.state_id,
        NodeStateStatus.COMPLETED,
        output_data={"title": "A"},
        duration_ms=125.0,
    )
    factory.execution.begin_node_state(
        second_token.token_id,
        "extract",
        web_run_id,
        1,
        {"html": "<h1>B</h1>"},
        state_id="state-token-1",
    )
    factory.data_flow.record_token_outcome(
        TokenRef(token_id=first_token.token_id, run_id=web_run_id),
        TerminalOutcome.SUCCESS,
        TerminalPath.DEFAULT_FLOW,
        sink_name="json_out",
    )
    source_operation = factory.execution.begin_operation(web_run_id, "source", "source_load")
    factory.execution.complete_operation(source_operation.operation_id, "completed", duration_ms=15.0)
    factory.execution.register_artifact(
        run_id=web_run_id,
        state_id=first_state.state_id,
        sink_node_id="json_out",
        artifact_type="json",
        path=str(tmp_path / "out.json"),
        content_hash="a" * 64,
        size_bytes=42,
        artifact_id="artifact-1",
    )


def test_diagnostics_returns_bounded_tokens_states_operations_and_artifacts(tmp_path) -> None:
    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    db = LandscapeDB.from_url(db_url)
    try:
        web_run_id = "web-run-1"
        _seed_diagnostics_run(db, tmp_path, web_run_id=web_run_id)

        diagnostics = load_run_diagnostics_from_db(
            db,
            run_id=web_run_id,
            landscape_run_id=web_run_id,
            run_status="running",
            limit=1,
        )

        assert diagnostics.run_id == web_run_id
        assert diagnostics.landscape_run_id == web_run_id
        assert diagnostics.run_status == "running"
        assert diagnostics.summary.token_count == 2
        assert diagnostics.summary.preview_limit == 1
        assert diagnostics.summary.preview_truncated is True
        assert diagnostics.summary.state_counts["completed"] == 1
        assert diagnostics.summary.state_counts["open"] == 1
        assert [token.token_id for token in diagnostics.tokens] == ["token-0"]
        assert diagnostics.tokens[0].row_index == 0
        assert diagnostics.tokens[0].terminal_outcome == "success"
        assert diagnostics.tokens[0].states[0].node_id == "extract"
        assert diagnostics.tokens[0].states[0].status == "completed"
        assert diagnostics.operations[0].operation_type == "source_load"
        assert diagnostics.operations[0].status == "completed"
        assert diagnostics.artifacts[0].path_or_uri.endswith("out.json")
        assert "context_after" not in diagnostics.model_dump_json()
    finally:
        db.close()


def test_diagnostics_rejects_corrupt_landscape_types(tmp_path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        web_run_id = "web-run-1"
        _seed_diagnostics_run(db, tmp_path, web_run_id=web_run_id)

        with db.write_connection() as conn:
            conn.execute(text("UPDATE node_states SET step_index = 1.5 WHERE state_id = 'state-token-0'"))

        with pytest.raises(ValidationError, match="step_index"):
            load_run_diagnostics_from_db(
                db,
                run_id=web_run_id,
                landscape_run_id=web_run_id,
                run_status="running",
                limit=1,
            )
    finally:
        db.close()


def test_diagnostics_summary_counts_share_main_read_snapshot(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        web_run_id = "web-run-1"
        _seed_diagnostics_run(db, tmp_path, web_run_id=web_run_id)
        original_read_only_connection = db.read_only_connection
        read_count = 0

        def counting_read_only_connection():
            nonlocal read_count
            read_count += 1
            return original_read_only_connection()

        monkeypatch.setattr(db, "read_only_connection", counting_read_only_connection)

        diagnostics = load_run_diagnostics_from_db(
            db,
            run_id=web_run_id,
            landscape_run_id=web_run_id,
            run_status="running",
            limit=1,
        )

        assert diagnostics.summary.state_counts["completed"] == 1
        assert diagnostics.summary.operation_counts["source_load"] == 1
        assert read_count == 1
    finally:
        db.close()


def test_diagnostics_surfaces_latest_failed_operation_as_failure_detail(tmp_path) -> None:
    """When a run has a failed operation, failure_detail must point at it.

    Regression test for run 8294aab2 (2026-05-13): a preflight HTTP 400
    killed the pipeline; ``operations.error_message`` had the full chain, but
    the UI rendered only the sanitized class-name from ``runs.error``. The
    diagnostics endpoint must surface the failed operation directly so the
    frontend doesn't have to scan the (paged) operations list, while relying on
    provider-boundary redaction to keep raw provider bodies out of the surfaced
    message.
    """
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        web_run_id = "web-run-1"
        factory = RecorderFactory(db)
        factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=web_run_id)
        _register_node(factory, web_run_id, "transform", NodeType.TRANSFORM, "llm")

        ok_op = factory.execution.begin_operation(web_run_id, "transform", "runtime_preflight")
        factory.execution.complete_operation(ok_op.operation_id, "completed", duration_ms=10.0)

        bad_op = factory.execution.begin_operation(web_run_id, "transform", "runtime_preflight")
        factory.execution.complete_operation(
            bad_op.operation_id,
            "failed",
            error=(
                "pre_flight_failed: llm provider openrouter failed runtime preflight: "
                "LLMClientError: HTTP 400 | provider error body redacted "
                "(body_present=true; chars=58)"
            ),
            duration_ms=995.0,
        )

        diagnostics = load_run_diagnostics_from_db(
            db,
            run_id=web_run_id,
            landscape_run_id=web_run_id,
            run_status="failed",
            limit=50,
        )

        assert diagnostics.failure_detail is not None
        assert diagnostics.failure_detail.operation_id == bad_op.operation_id
        assert diagnostics.failure_detail.node_id == "transform"
        assert diagnostics.failure_detail.operation_type == "runtime_preflight"
        assert "HTTP 400" in diagnostics.failure_detail.error_message
        assert "provider error body redacted" in diagnostics.failure_detail.error_message
        assert "max_output_tokens below minimum" not in diagnostics.failure_detail.error_message
    finally:
        db.close()


def test_diagnostics_failure_detail_none_when_no_failed_operations(tmp_path) -> None:
    """failure_detail must be None for runs without failed operations."""
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        web_run_id = "web-run-1"
        _seed_diagnostics_run(db, tmp_path, web_run_id=web_run_id)

        diagnostics = load_run_diagnostics_from_db(
            db,
            run_id=web_run_id,
            landscape_run_id=web_run_id,
            run_status="completed",
            limit=50,
        )

        assert diagnostics.failure_detail is None

    finally:
        db.close()


def test_diagnostics_empty_when_landscape_run_has_not_started(tmp_path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        diagnostics = load_run_diagnostics_from_db(
            db,
            run_id="web-run-before-begin-run",
            landscape_run_id="web-run-before-begin-run",
            run_status="pending",
            limit=50,
        )

        assert diagnostics.summary.token_count == 0
        assert diagnostics.summary.preview_truncated is False
        assert diagnostics.tokens == []
        assert diagnostics.operations == []
        assert diagnostics.artifacts == []
    finally:
        db.close()

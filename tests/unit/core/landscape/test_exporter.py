"""Tests for LandscapeExporter — compliance-grade audit trail export.

Tests cover:
- Constructor (signing key optional)
- _sign_record (HMAC computation, missing key error)
- export_run (unsigned records, signed with manifest, unknown run ValueError)
- export_run_grouped (groups by record_type)
- _iter_records (record yield order, field mapping for all record types,
  NodeState discriminated union handling)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts.audit import (
    Artifact,
    Batch,
    BatchMember,
    Call,
    Edge,
    Node,
    NodeStateCompleted,
    NodeStateFailed,
    NodeStateOpen,
    NodeStatePending,
    Operation,
    RoutingEvent,
    Row,
    Run,
    SecretResolution,
    Token,
    TokenParent,
)
from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    NodeStateStatus,
    NodeType,
    RoutingMode,
    RunStatus,
    TriggerType,
)
from elspeth.core.landscape.exporter import LandscapeExporter

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DT = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
_DT2 = datetime(2026, 1, 15, 13, 0, 0, tzinfo=UTC)

_RUN = Run(
    run_id="run-1",
    started_at=_DT,
    config_hash="cfg-hash",
    settings_json='{"key": "value"}',
    canonical_version="v1",
    status=RunStatus.COMPLETED,
    completed_at=_DT2,
    reproducibility_grade="A",
)

_NODE = Node(
    node_id="node-1",
    run_id="run-1",
    plugin_name="csv",
    node_type=NodeType.SOURCE,
    plugin_version="1.0",
    determinism=Determinism.DETERMINISTIC,
    config_hash="node-cfg-hash",
    config_json='{"path": "data.csv"}',
    registered_at=_DT,
    schema_hash="schema-hash",
    sequence_in_pipeline=0,
    schema_mode="observed",
    schema_fields=[{"name": "id", "type": "int"}],
)

_EDGE = Edge(
    edge_id="edge-1",
    run_id="run-1",
    from_node_id="node-1",
    to_node_id="node-2",
    label="continue",
    default_mode=RoutingMode.MOVE,
    created_at=_DT,
)

_ROW = Row(
    row_id="row-1",
    run_id="run-1",
    source_node_id="node-1",
    row_index=0,
    source_data_hash="data-hash",
    created_at=_DT,
)

_TOKEN = Token(
    token_id="tok-1",
    row_id="row-1",
    created_at=_DT,
    fork_group_id=None,
    join_group_id=None,
    expand_group_id=None,
    branch_name=None,
    step_in_pipeline=0,
)

_TOKEN_PARENT = TokenParent(
    token_id="tok-1",
    parent_token_id="tok-0",
    ordinal=0,
)

_SECRET = SecretResolution(
    resolution_id="sec-1",
    run_id="run-1",
    timestamp=1705320000.0,
    env_var_name="API_KEY",
    source="keyvault",
    fingerprint="fp-hash",
    vault_url="https://vault.example.com",
    secret_name="api-key",
    resolution_latency_ms=150.0,
)

_OPERATION = Operation(
    operation_id="op-1",
    run_id="run-1",
    node_id="node-1",
    operation_type="source_load",
    started_at=_DT,
    status="completed",
    completed_at=_DT2,
    input_data_ref=None,
    input_data_hash=None,
    output_data_ref="out-ref",
    output_data_hash="abc123def456",
    error_message=None,
    duration_ms=1234.5,
)

_OP_CALL = Call(
    call_id="call-op-1",
    call_index=0,
    call_type=CallType.HTTP,
    status=CallStatus.SUCCESS,
    request_hash="req-hash",
    created_at=_DT,
    state_id=None,
    operation_id="op-1",
    request_ref="req-ref",
    response_hash="resp-hash",
    response_ref="resp-ref",
    error_json=None,
    latency_ms=50.0,
)

_STATE_CALL = Call(
    call_id="call-st-1",
    call_index=0,
    call_type=CallType.LLM,
    status=CallStatus.SUCCESS,
    request_hash="req-hash-2",
    created_at=_DT,
    state_id="state-completed",
    operation_id=None,
    request_ref="req-ref-2",
    response_hash="resp-hash-2",
    response_ref="resp-ref-2",
    error_json=None,
    latency_ms=200.0,
)

_NODE_STATE_OPEN = NodeStateOpen(
    state_id="state-open",
    token_id="tok-1",
    node_id="node-2",
    step_index=1,
    attempt=1,
    status=NodeStateStatus.OPEN,
    input_hash="in-hash",
    started_at=_DT,
    context_before_json='{"prompt": "test"}',
)

_NODE_STATE_PENDING = NodeStatePending(
    state_id="state-pending",
    token_id="tok-1",
    node_id="node-2",
    step_index=1,
    attempt=1,
    status=NodeStateStatus.PENDING,
    input_hash="in-hash",
    started_at=_DT,
    completed_at=_DT2,
    duration_ms=500.0,
    context_before_json='{"prompt": "test"}',
    context_after_json='{"status": "pending"}',
)

_NODE_STATE_COMPLETED = NodeStateCompleted(
    state_id="state-completed",
    token_id="tok-1",
    node_id="node-2",
    step_index=1,
    attempt=1,
    status=NodeStateStatus.COMPLETED,
    input_hash="in-hash",
    started_at=_DT,
    output_hash="out-hash",
    completed_at=_DT2,
    duration_ms=500.0,
    context_before_json='{"prompt": "test"}',
    context_after_json='{"result": "ok"}',
    success_reason_json='{"action": "classified"}',
)

_NODE_STATE_FAILED = NodeStateFailed(
    state_id="state-failed",
    token_id="tok-1",
    node_id="node-2",
    step_index=1,
    attempt=1,
    status=NodeStateStatus.FAILED,
    input_hash="in-hash",
    started_at=_DT,
    completed_at=_DT2,
    duration_ms=100.0,
    error_json='{"error": "timeout"}',
    output_hash=None,
    context_before_json='{"prompt": "test"}',
    context_after_json=None,
)

_ROUTING_EVENT = RoutingEvent(
    event_id="evt-1",
    state_id="state-completed",
    edge_id="edge-1",
    routing_group_id="rg-1",
    ordinal=0,
    mode=RoutingMode.MOVE,
    created_at=_DT,
    reason_hash="reason-hash",
    reason_ref="reason-ref",
)

_BATCH = Batch(
    batch_id="batch-1",
    run_id="run-1",
    aggregation_node_id="node-3",
    attempt=1,
    status=BatchStatus.COMPLETED,
    created_at=_DT,
    completed_at=_DT2,
    trigger_type=TriggerType.COUNT,
    trigger_reason="count >= 10",
)

_BATCH_MEMBER = BatchMember(
    batch_id="batch-1",
    token_id="tok-1",
    ordinal=0,
)

_ARTIFACT = Artifact(
    artifact_id="art-1",
    run_id="run-1",
    produced_by_state_id="state-completed",
    sink_node_id="node-4",
    artifact_type="csv",
    path_or_uri="/output/result.csv",
    content_hash="content-hash",
    size_bytes=1024,
    created_at=_DT,
)


def _make_exporter(
    *,
    signing_key: bytes | None = None,
    run: Run | None = None,
    secret_resolutions: list[Any] | None = None,
    nodes: list[Any] | None = None,
    edges: list[Any] | None = None,
    operations: list[Any] | None = None,
    operation_calls: list[Any] | None = None,
    rows: list[Any] | None = None,
    tokens: list[Any] | None = None,
    token_parents: list[Any] | None = None,
    node_states: list[Any] | None = None,
    routing_events: list[Any] | None = None,
    state_calls: list[Any] | None = None,
    batches: list[Any] | None = None,
    batch_members: list[Any] | None = None,
    artifacts: list[Any] | None = None,
) -> LandscapeExporter:
    """Create an exporter with mocked database and recorder."""
    mock_db = Mock()
    exporter = LandscapeExporter(mock_db, signing_key=signing_key)

    # Mock all recorder methods used by _iter_records
    recorder = exporter._recorder
    object.__setattr__(recorder, "get_run", Mock(return_value=run if run is not None else _RUN))
    object.__setattr__(recorder, "get_secret_resolutions_for_run", Mock(return_value=secret_resolutions or []))
    object.__setattr__(recorder, "get_nodes", Mock(return_value=nodes or []))
    object.__setattr__(recorder, "get_edges", Mock(return_value=edges or []))
    object.__setattr__(recorder, "get_operations_for_run", Mock(return_value=operations or []))
    object.__setattr__(recorder, "get_all_operation_calls_for_run", Mock(return_value=operation_calls or []))
    object.__setattr__(recorder, "get_rows", Mock(return_value=rows or []))
    object.__setattr__(recorder, "get_all_tokens_for_run", Mock(return_value=tokens or []))
    object.__setattr__(recorder, "get_all_token_parents_for_run", Mock(return_value=token_parents or []))
    object.__setattr__(recorder, "get_all_node_states_for_run", Mock(return_value=node_states or []))
    object.__setattr__(recorder, "get_all_routing_events_for_run", Mock(return_value=routing_events or []))
    object.__setattr__(recorder, "get_all_calls_for_run", Mock(return_value=state_calls or []))
    object.__setattr__(recorder, "get_batches", Mock(return_value=batches or []))
    object.__setattr__(recorder, "get_all_batch_members_for_run", Mock(return_value=batch_members or []))
    object.__setattr__(recorder, "get_artifacts", Mock(return_value=artifacts or []))

    return exporter


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    """Tests for exporter initialization."""

    def test_creates_recorder_from_db(self) -> None:
        db = Mock()
        exporter = LandscapeExporter(db)
        assert exporter._db is db
        assert exporter._signing_key is None

    def test_accepts_signing_key(self) -> None:
        db = Mock()
        key = b"secret-key"
        exporter = LandscapeExporter(db, signing_key=key)
        assert exporter._signing_key == key


# ===========================================================================
# Signing
# ===========================================================================


class TestSignRecord:
    """Tests for _sign_record — HMAC-SHA256 signing."""

    def test_produces_hex_signature(self) -> None:
        exporter = _make_exporter(signing_key=b"test-key")
        record = {"record_type": "run", "run_id": "run-1"}
        sig = exporter._sign_record(record)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA-256 hex is 64 chars
        assert all(c in "0123456789abcdef" for c in sig)

    def test_deterministic_signature(self) -> None:
        exporter = _make_exporter(signing_key=b"test-key")
        record = {"record_type": "run", "run_id": "run-1"}
        sig1 = exporter._sign_record(record)
        sig2 = exporter._sign_record(record)
        assert sig1 == sig2

    def test_different_records_different_signatures(self) -> None:
        exporter = _make_exporter(signing_key=b"test-key")
        sig1 = exporter._sign_record({"record_type": "run", "run_id": "run-1"})
        sig2 = exporter._sign_record({"record_type": "run", "run_id": "run-2"})
        assert sig1 != sig2

    def test_no_signing_key_raises(self) -> None:
        exporter = _make_exporter(signing_key=None)
        with pytest.raises(ValueError, match="Signing key not configured"):
            exporter._sign_record({"record_type": "run"})


# ===========================================================================
# export_run — unsigned
# ===========================================================================


class TestExportRunUnsigned:
    """Tests for export_run without signing."""

    def test_unknown_run_raises(self) -> None:
        exporter = _make_exporter(run=None)
        object.__setattr__(exporter._recorder, "get_run", Mock(return_value=None))
        with pytest.raises(ValueError, match="Run not found"):
            list(exporter.export_run("unknown-run"))

    def test_empty_run_yields_run_record_only(self) -> None:
        exporter = _make_exporter()
        records = list(exporter.export_run("run-1"))
        assert len(records) == 1
        assert records[0]["record_type"] == "run"
        assert records[0]["run_id"] == "run-1"
        assert records[0]["status"] == "completed"

    def test_run_record_contains_settings(self) -> None:
        exporter = _make_exporter()
        records = list(exporter.export_run("run-1"))
        assert records[0]["settings"] == {"key": "value"}

    def test_run_record_has_timestamps(self) -> None:
        exporter = _make_exporter()
        records = list(exporter.export_run("run-1"))
        assert records[0]["started_at"] == _DT.isoformat()
        assert records[0]["completed_at"] == _DT2.isoformat()

    def test_no_signature_field_when_unsigned(self) -> None:
        exporter = _make_exporter()
        records = list(exporter.export_run("run-1"))
        for record in records:
            assert "signature" not in record


# ===========================================================================
# export_run — signed
# ===========================================================================


class TestExportRunSigned:
    """Tests for export_run with HMAC signing."""

    def test_sign_without_key_raises(self) -> None:
        exporter = _make_exporter(signing_key=None)
        with pytest.raises(ValueError, match="no signing_key provided"):
            list(exporter.export_run("run-1", sign=True))

    def test_each_record_has_signature(self) -> None:
        exporter = _make_exporter(signing_key=b"test-key")
        records = list(exporter.export_run("run-1", sign=True))
        for record in records:
            assert "signature" in record
            assert len(record["signature"]) == 64

    def test_manifest_emitted_last(self) -> None:
        exporter = _make_exporter(signing_key=b"test-key")
        records = list(exporter.export_run("run-1", sign=True))
        manifest = records[-1]
        assert manifest["record_type"] == "manifest"
        assert manifest["run_id"] == "run-1"
        assert manifest["record_count"] == 1  # Just the run record
        assert manifest["hash_algorithm"] == "sha256"
        assert manifest["signature_algorithm"] == "hmac-sha256"
        assert "final_hash" in manifest
        assert "exported_at" in manifest

    def test_manifest_has_correct_record_count(self) -> None:
        exporter = _make_exporter(
            signing_key=b"test-key",
            nodes=[_NODE],
            edges=[_EDGE],
        )
        records = list(exporter.export_run("run-1", sign=True))
        manifest = records[-1]
        # run + node + edge = 3
        assert manifest["record_count"] == 3


# ===========================================================================
# Record types — field mapping
# ===========================================================================


class TestSecretResolutionRecords:
    """Tests for secret_resolution record serialization."""

    def test_secret_resolution_fields(self) -> None:
        exporter = _make_exporter(secret_resolutions=[_SECRET])
        records = list(exporter.export_run("run-1"))
        sec = [r for r in records if r["record_type"] == "secret_resolution"]
        assert len(sec) == 1
        assert sec[0]["resolution_id"] == "sec-1"
        assert sec[0]["env_var_name"] == "API_KEY"
        assert sec[0]["source"] == "keyvault"
        assert sec[0]["vault_url"] == "https://vault.example.com"
        assert sec[0]["fingerprint"] == "fp-hash"
        assert sec[0]["resolution_latency_ms"] == 150.0


class TestNodeRecords:
    """Tests for node record serialization."""

    def test_node_fields(self) -> None:
        exporter = _make_exporter(nodes=[_NODE])
        records = list(exporter.export_run("run-1"))
        nodes = [r for r in records if r["record_type"] == "node"]
        assert len(nodes) == 1
        n = nodes[0]
        assert n["node_id"] == "node-1"
        assert n["plugin_name"] == "csv"
        assert n["node_type"] == "source"
        assert n["determinism"] == "deterministic"
        assert n["config"] == {"path": "data.csv"}
        assert n["schema_mode"] == "observed"
        assert n["sequence_in_pipeline"] == 0


class TestEdgeRecords:
    """Tests for edge record serialization."""

    def test_edge_fields(self) -> None:
        exporter = _make_exporter(edges=[_EDGE])
        records = list(exporter.export_run("run-1"))
        edges = [r for r in records if r["record_type"] == "edge"]
        assert len(edges) == 1
        e = edges[0]
        assert e["edge_id"] == "edge-1"
        assert e["from_node_id"] == "node-1"
        assert e["to_node_id"] == "node-2"
        assert e["label"] == "continue"
        assert e["default_mode"] == "move"


class TestOperationRecords:
    """Tests for operation + operation-parented call records."""

    def test_operation_fields(self) -> None:
        exporter = _make_exporter(operations=[_OPERATION])
        records = list(exporter.export_run("run-1"))
        ops = [r for r in records if r["record_type"] == "operation"]
        assert len(ops) == 1
        op = ops[0]
        assert op["operation_id"] == "op-1"
        assert op["operation_type"] == "source_load"
        assert op["status"] == "completed"
        assert op["duration_ms"] == 1234.5
        assert op["input_data_hash"] is None
        assert op["output_data_hash"] == "abc123def456"

    def test_operation_call_follows_operation(self) -> None:
        exporter = _make_exporter(
            operations=[_OPERATION],
            operation_calls=[_OP_CALL],
        )
        records = list(exporter.export_run("run-1"))
        calls = [r for r in records if r["record_type"] == "call"]
        assert len(calls) == 1
        c = calls[0]
        assert c["call_id"] == "call-op-1"
        assert c["operation_id"] == "op-1"
        assert c["state_id"] is None  # Operation calls don't have state_id
        assert c["call_type"] == "http"
        assert c["status"] == "success"


class TestRowRecords:
    """Tests for row record serialization."""

    def test_row_fields(self) -> None:
        exporter = _make_exporter(rows=[_ROW])
        records = list(exporter.export_run("run-1"))
        rows = [r for r in records if r["record_type"] == "row"]
        assert len(rows) == 1
        r = rows[0]
        assert r["row_id"] == "row-1"
        assert r["row_index"] == 0
        assert r["source_node_id"] == "node-1"
        assert r["source_data_hash"] == "data-hash"


class TestTokenRecords:
    """Tests for token + parent record serialization."""

    def test_token_fields(self) -> None:
        exporter = _make_exporter(rows=[_ROW], tokens=[_TOKEN])
        records = list(exporter.export_run("run-1"))
        tokens = [r for r in records if r["record_type"] == "token"]
        assert len(tokens) == 1
        t = tokens[0]
        assert t["token_id"] == "tok-1"
        assert t["row_id"] == "row-1"
        assert t["step_in_pipeline"] == 0

    def test_token_parent_fields(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            token_parents=[_TOKEN_PARENT],
        )
        records = list(exporter.export_run("run-1"))
        parents = [r for r in records if r["record_type"] == "token_parent"]
        assert len(parents) == 1
        p = parents[0]
        assert p["parent_token_id"] == "tok-0"
        assert p["ordinal"] == 0


# ===========================================================================
# NodeState discriminated union
# ===========================================================================


class TestNodeStateRecords:
    """Tests for node_state serialization across all state variants."""

    def test_open_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_OPEN],
        )
        records = list(exporter.export_run("run-1"))
        states = [r for r in records if r["record_type"] == "node_state"]
        assert len(states) == 1
        s = states[0]
        assert s["status"] == "open"
        assert s["output_hash"] is None
        assert s["completed_at"] is None
        assert s["duration_ms"] is None
        assert s["context_before_json"] == '{"prompt": "test"}'
        assert s["context_after_json"] is None
        assert s["error_json"] is None
        assert s["success_reason_json"] is None

    def test_pending_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_PENDING],
        )
        records = list(exporter.export_run("run-1"))
        states = [r for r in records if r["record_type"] == "node_state"]
        assert len(states) == 1
        s = states[0]
        assert s["status"] == "pending"
        assert s["output_hash"] is None
        assert s["duration_ms"] == 500.0
        assert s["completed_at"] == _DT2.isoformat()
        assert s["context_after_json"] == '{"status": "pending"}'
        assert s["error_json"] is None
        assert s["success_reason_json"] is None

    def test_completed_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_COMPLETED],
        )
        records = list(exporter.export_run("run-1"))
        states = [r for r in records if r["record_type"] == "node_state"]
        assert len(states) == 1
        s = states[0]
        assert s["status"] == "completed"
        assert s["output_hash"] == "out-hash"
        assert s["duration_ms"] == 500.0
        assert s["error_json"] is None
        assert s["success_reason_json"] == '{"action": "classified"}'

    def test_failed_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_FAILED],
        )
        records = list(exporter.export_run("run-1"))
        states = [r for r in records if r["record_type"] == "node_state"]
        assert len(states) == 1
        s = states[0]
        assert s["status"] == "failed"
        assert s["error_json"] == '{"error": "timeout"}'
        assert s["success_reason_json"] is None


# ===========================================================================
# Routing events and state-parented calls
# ===========================================================================


class TestRoutingEventRecords:
    """Tests for routing_event serialization."""

    def test_routing_event_follows_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_COMPLETED],
            routing_events=[_ROUTING_EVENT],
        )
        records = list(exporter.export_run("run-1"))
        events = [r for r in records if r["record_type"] == "routing_event"]
        assert len(events) == 1
        e = events[0]
        assert e["event_id"] == "evt-1"
        assert e["state_id"] == "state-completed"
        assert e["edge_id"] == "edge-1"
        assert e["mode"] == "move"
        assert e["reason_ref"] == "reason-ref"


class TestStateCallRecords:
    """Tests for state-parented call records."""

    def test_state_call_follows_state(self) -> None:
        exporter = _make_exporter(
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_COMPLETED],
            state_calls=[_STATE_CALL],
        )
        records = list(exporter.export_run("run-1"))
        calls = [r for r in records if r["record_type"] == "call"]
        assert len(calls) == 1
        c = calls[0]
        assert c["call_id"] == "call-st-1"
        assert c["state_id"] == "state-completed"
        assert c["operation_id"] is None  # State calls don't have operation_id
        assert c["call_type"] == "llm"


# ===========================================================================
# Batch records
# ===========================================================================


class TestBatchRecords:
    """Tests for batch + member record serialization."""

    def test_batch_fields(self) -> None:
        exporter = _make_exporter(batches=[_BATCH])
        records = list(exporter.export_run("run-1"))
        batches = [r for r in records if r["record_type"] == "batch"]
        assert len(batches) == 1
        b = batches[0]
        assert b["batch_id"] == "batch-1"
        assert b["aggregation_node_id"] == "node-3"
        assert b["status"] == "completed"
        assert b["trigger_type"] == "count"
        assert b["trigger_reason"] == "count >= 10"

    def test_batch_member_follows_batch(self) -> None:
        exporter = _make_exporter(
            batches=[_BATCH],
            batch_members=[_BATCH_MEMBER],
        )
        records = list(exporter.export_run("run-1"))
        members = [r for r in records if r["record_type"] == "batch_member"]
        assert len(members) == 1
        m = members[0]
        assert m["batch_id"] == "batch-1"
        assert m["token_id"] == "tok-1"
        assert m["ordinal"] == 0


# ===========================================================================
# Artifact records
# ===========================================================================


class TestArtifactRecords:
    """Tests for artifact record serialization."""

    def test_artifact_fields(self) -> None:
        exporter = _make_exporter(artifacts=[_ARTIFACT])
        records = list(exporter.export_run("run-1"))
        arts = [r for r in records if r["record_type"] == "artifact"]
        assert len(arts) == 1
        a = arts[0]
        assert a["artifact_id"] == "art-1"
        assert a["sink_node_id"] == "node-4"
        assert a["artifact_type"] == "csv"
        assert a["content_hash"] == "content-hash"
        assert a["size_bytes"] == 1024


# ===========================================================================
# Record order
# ===========================================================================


class TestRecordOrder:
    """Tests for record yield order in export."""

    def test_order_is_run_secrets_nodes_edges_ops_rows_batches_artifacts(self) -> None:
        """Records should yield in the documented order."""
        exporter = _make_exporter(
            secret_resolutions=[_SECRET],
            nodes=[_NODE],
            edges=[_EDGE],
            operations=[_OPERATION],
            rows=[_ROW],
            tokens=[_TOKEN],
            node_states=[_NODE_STATE_COMPLETED],
            batches=[_BATCH],
            artifacts=[_ARTIFACT],
        )
        records = list(exporter.export_run("run-1"))
        types = [r["record_type"] for r in records]

        # Verify ordering: run first, then cascading types
        assert types[0] == "run"
        run_idx = types.index("run")
        sec_idx = types.index("secret_resolution")
        node_idx = types.index("node")
        edge_idx = types.index("edge")
        op_idx = types.index("operation")
        row_idx = types.index("row")
        batch_idx = types.index("batch")
        art_idx = types.index("artifact")

        assert run_idx < sec_idx < node_idx < edge_idx < op_idx < row_idx < batch_idx < art_idx


# ===========================================================================
# export_run_grouped
# ===========================================================================


class TestExportRunGrouped:
    """Tests for export_run_grouped — groups records by type."""

    def test_groups_by_record_type(self) -> None:
        exporter = _make_exporter(
            nodes=[_NODE],
            edges=[_EDGE],
        )
        groups = exporter.export_run_grouped("run-1")
        assert "run" in groups
        assert "node" in groups
        assert "edge" in groups
        assert len(groups["run"]) == 1
        assert len(groups["node"]) == 1
        assert len(groups["edge"]) == 1

    def test_returns_regular_dict(self) -> None:
        exporter = _make_exporter()
        groups = exporter.export_run_grouped("run-1")
        assert type(groups) is dict  # Not defaultdict

    def test_signed_grouped_includes_manifest(self) -> None:
        exporter = _make_exporter(signing_key=b"key")
        groups = exporter.export_run_grouped("run-1", sign=True)
        assert "manifest" in groups
        assert len(groups["manifest"]) == 1


# ===========================================================================
# Full pipeline export
# ===========================================================================


class TestFullPipelineExport:
    """Integration-style test for a complete pipeline export."""

    def test_full_pipeline_all_record_types(self) -> None:
        """A pipeline with all record types exports correctly."""
        exporter = _make_exporter(
            secret_resolutions=[_SECRET],
            nodes=[_NODE],
            edges=[_EDGE],
            operations=[_OPERATION],
            operation_calls=[_OP_CALL],
            rows=[_ROW],
            tokens=[_TOKEN],
            token_parents=[_TOKEN_PARENT],
            node_states=[_NODE_STATE_COMPLETED],
            routing_events=[_ROUTING_EVENT],
            state_calls=[_STATE_CALL],
            batches=[_BATCH],
            batch_members=[_BATCH_MEMBER],
            artifacts=[_ARTIFACT],
        )

        records = list(exporter.export_run("run-1"))
        type_counts: dict[str, int] = {}
        for r in records:
            rt = r["record_type"]
            type_counts[rt] = type_counts.get(rt, 0) + 1

        assert type_counts == {
            "run": 1,
            "secret_resolution": 1,
            "node": 1,
            "edge": 1,
            "operation": 1,
            "call": 2,  # 1 operation call + 1 state call
            "row": 1,
            "token": 1,
            "token_parent": 1,
            "node_state": 1,
            "routing_event": 1,
            "batch": 1,
            "batch_member": 1,
            "artifact": 1,
        }

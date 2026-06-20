"""Tests for the Landscape model loader layer.

The model loader layer converts SQLAlchemy row objects (where enums are stored
as strings) into strict domain dataclass objects. This is Tier 1 trust --
bad data from our own DB must crash, never coerce.

Tests use SimpleNamespace to simulate SQLAlchemy Row objects. SimpleNamespace
raises AttributeError on missing attributes, matching real Row behavior and
catching field-name typos that bare Mock() would silently accept.
"""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from sqlalchemy.engine import Row as SARow

from elspeth.contracts.audit import (
    _TERMINAL_PAIR_FIELD_CONSTRAINTS,
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
    Token,
    TokenOutcome,
    TokenParent,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.enums import (
    BatchStatus,
    CallStatus,
    CallType,
    Determinism,
    ExportStatus,
    NodeStateStatus,
    NodeType,
    ReproducibilityGrade,
    RoutingMode,
    RunStatus,
    TerminalOutcome,
    TerminalPath,
    TriggerType,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.model_loaders import (
    ArtifactLoader,
    BatchLoader,
    BatchMemberLoader,
    CallLoader,
    EdgeLoader,
    NodeLoader,
    NodeStateLoader,
    OperationLoader,
    RoutingEventLoader,
    RowLoader,
    RunLoader,
    TokenLoader,
    TokenOutcomeLoader,
    TokenParentLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)


def _make_sa_row(**kwargs: object) -> SARow[Any]:
    """Create a SimpleNamespace simulating a SQLAlchemy Row.

    SimpleNamespace raises AttributeError on missing attributes,
    matching real Row behavior and catching loader field-name typos.
    Cast to SARow[Any] so loaders accept it without type errors.
    """
    return cast(SARow[Any], SimpleNamespace(**kwargs))


NOW = datetime.now(UTC)
LATER = NOW + timedelta(seconds=1)
EVEN_LATER = NOW + timedelta(seconds=2)


# ---------------------------------------------------------------------------
# RunLoader
# ---------------------------------------------------------------------------


class TestRunLoader:
    """Tests for RunLoader.load()."""

    def _make_run_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "run_id": "run-1",
            "started_at": NOW,
            "config_hash": "abc123",
            "settings_json": "{}",
            "canonical_version": "v2",
            "status": "running",
            "completed_at": None,
            "reproducibility_grade": None,
            "export_status": None,
            "export_error": None,
            "exported_at": None,
            "export_format": None,
            "export_sink": None,
            # Phase 4 (commit ca9bc05bd) audit-story columns on runs_table.
            # The loader at ``core/landscape/model_loaders.py:109-111`` reads
            # these unconditionally; fixture must supply them so the
            # SimpleNamespace stand-in mirrors the real SA row shape.
            "llm_call_count": None,
            "seeded_from_cache": False,
            "cache_key": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_run_row(
            status="completed",
            completed_at=LATER,
            reproducibility_grade="full_reproducible",
            export_status="pending",
            export_error="some error",
            exported_at=EVEN_LATER,
            export_format="csv",
            export_sink="output",
        )
        loader = RunLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Run)
        assert result.run_id == "run-1"
        assert result.status == RunStatus.COMPLETED
        assert result.export_status == ExportStatus.PENDING
        assert result.started_at == NOW
        assert result.completed_at == LATER
        assert result.exported_at == EVEN_LATER
        assert result.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE
        assert result.export_error == "some error"
        assert result.export_format == "csv"
        assert result.export_sink == "output"

    def test_valid_load_minimal_fields(self) -> None:
        sa_row = self._make_run_row()
        loader = RunLoader()
        result = loader.load(sa_row)

        assert result.status == RunStatus.RUNNING
        assert result.export_status is None
        assert result.completed_at is None

    @pytest.mark.parametrize("status_value", [s.value for s in RunStatus])
    def test_valid_run_status_values(self, status_value: str) -> None:
        # COMPLETED requires completed_at; RUNNING must not have it; FAILED/INTERRUPTED may or may not
        completed_at = None if status_value == RunStatus.RUNNING else LATER
        sa_row = self._make_run_row(status=status_value, completed_at=completed_at)
        loader = RunLoader()
        result = loader.load(sa_row)
        assert result.status == RunStatus(status_value)

    def test_invalid_status_raises_value_error(self) -> None:
        sa_row = self._make_run_row(status="bogus_status")
        loader = RunLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    @pytest.mark.parametrize("export_value", [s.value for s in ExportStatus])
    def test_valid_export_status_values(self, export_value: str) -> None:
        sa_row = self._make_run_row(export_status=export_value)
        loader = RunLoader()
        result = loader.load(sa_row)
        assert result.export_status == ExportStatus(export_value)

    def test_none_export_status_stays_none(self) -> None:
        sa_row = self._make_run_row(export_status=None)
        loader = RunLoader()
        result = loader.load(sa_row)
        assert result.export_status is None

    def test_invalid_export_status_raises_value_error(self) -> None:
        sa_row = self._make_run_row(export_status="bogus_export")
        loader = RunLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_empty_string_export_status_raises_value_error(self) -> None:
        """Empty string should NOT become None -- Tier 1 trust."""
        sa_row = self._make_run_row(export_status="")
        loader = RunLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_completed_without_completed_at_raises_audit_integrity(self) -> None:
        """Completed run with NULL completed_at is a Tier 1 integrity violation."""
        sa_row = self._make_run_row(status="completed", completed_at=None)
        loader = RunLoader()
        with pytest.raises(AuditIntegrityError, match="completed_at"):
            loader.load(sa_row)

    @pytest.mark.parametrize(
        "status",
        [
            RunStatus.COMPLETED_WITH_FAILURES,
            RunStatus.EMPTY,
        ],
    )
    def test_terminal_success_status_without_completed_at_raises_audit_integrity(self, status: RunStatus) -> None:
        """Terminal successful run statuses with NULL completed_at are Tier 1 integrity violations."""
        sa_row = self._make_run_row(status=status.value, completed_at=None)
        loader = RunLoader()
        with pytest.raises(AuditIntegrityError, match=f"status='{status.value}'"):
            loader.load(sa_row)

    def test_running_with_completed_at_raises_audit_integrity(self) -> None:
        """Running run with completed_at set is a Tier 1 integrity violation."""
        sa_row = self._make_run_row(status="running", completed_at=LATER)
        loader = RunLoader()
        with pytest.raises(AuditIntegrityError, match="completed_at"):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# NodeLoader
# ---------------------------------------------------------------------------


class TestNodeLoader:
    """Tests for NodeLoader.load()."""

    def _make_node_row(self, **overrides: object) -> SARow[Any]:
        defaults: dict[str, object] = {
            "node_id": "node-1",
            "run_id": "run-1",
            "plugin_name": "csv",
            "node_type": "source",
            "plugin_version": "1.0.0",
            "source_file_hash": None,
            "determinism": "deterministic",
            "config_hash": "cfg123",
            "config_json": "{}",
            "registered_at": NOW,
            "schema_hash": None,
            "sequence_in_pipeline": 0,
            "schema_mode": None,
            "schema_fields_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_all_fields(self) -> None:
        schema_fields = [{"name": "id", "type": "int"}]
        sa_row = self._make_node_row(
            node_type="transform",
            determinism="non_deterministic",
            schema_hash="sch123",
            sequence_in_pipeline=3,
            schema_mode="fixed",
            schema_fields_json=json.dumps(schema_fields),
        )
        loader = NodeLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Node)
        assert result.node_type == NodeType.TRANSFORM
        assert result.determinism == Determinism.NON_DETERMINISTIC
        assert result.schema_hash == "sch123"
        assert result.sequence_in_pipeline == 3
        assert result.schema_mode == "fixed"
        assert result.schema_fields == tuple(schema_fields)

    def test_valid_load_minimal_fields(self) -> None:
        sa_row = self._make_node_row()
        loader = NodeLoader()
        result = loader.load(sa_row)

        assert result.node_type == NodeType.SOURCE
        assert result.determinism == Determinism.DETERMINISTIC
        assert result.schema_fields is None

    @pytest.mark.parametrize("node_type_value", [t.value for t in NodeType])
    def test_valid_node_type_values(self, node_type_value: str) -> None:
        sa_row = self._make_node_row(node_type=node_type_value)
        loader = NodeLoader()
        result = loader.load(sa_row)
        assert result.node_type == NodeType(node_type_value)

    @pytest.mark.parametrize("det_value", [d.value for d in Determinism])
    def test_valid_determinism_values(self, det_value: str) -> None:
        sa_row = self._make_node_row(determinism=det_value)
        loader = NodeLoader()
        result = loader.load(sa_row)
        assert result.determinism == Determinism(det_value)

    def test_invalid_node_type_raises_value_error(self) -> None:
        sa_row = self._make_node_row(node_type="bogus_type")
        loader = NodeLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_invalid_determinism_raises_value_error(self) -> None:
        sa_row = self._make_node_row(determinism="sometimes")
        loader = NodeLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_schema_fields_json_none_becomes_none(self) -> None:
        sa_row = self._make_node_row(schema_fields_json=None)
        loader = NodeLoader()
        result = loader.load(sa_row)
        assert result.schema_fields is None

    def test_schema_fields_json_valid_json_becomes_tuple(self) -> None:
        fields = [{"name": "amount", "type": "float"}]
        sa_row = self._make_node_row(schema_fields_json=json.dumps(fields))
        loader = NodeLoader()
        result = loader.load(sa_row)
        assert result.schema_fields == tuple(fields)

    def test_schema_fields_json_empty_list(self) -> None:
        sa_row = self._make_node_row(schema_fields_json="[]")
        loader = NodeLoader()
        result = loader.load(sa_row)
        assert result.schema_fields == ()

    def test_schema_fields_json_null_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json="null")
        loader = NodeLoader()
        with pytest.raises(AuditIntegrityError, match="must decode to list"):
            loader.load(sa_row)

    def test_schema_fields_json_object_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json='{"name": "id"}')
        loader = NodeLoader()
        with pytest.raises(AuditIntegrityError, match="must decode to list"):
            loader.load(sa_row)

    def test_schema_fields_json_non_dict_element_raises_value_error(self) -> None:
        sa_row = self._make_node_row(schema_fields_json='[{"name": "id"}, 42]')
        loader = NodeLoader()
        with pytest.raises(AuditIntegrityError, match="must be object/dict"):
            loader.load(sa_row)

    def test_schema_fields_json_unparseable_json_raises(self) -> None:
        """Corrupt JSON in schema_fields_json crashes per Tier 1 trust model."""
        sa_row = self._make_node_row(schema_fields_json="[not valid json")
        loader = NodeLoader()
        with pytest.raises(json.JSONDecodeError):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# EdgeLoader
# ---------------------------------------------------------------------------


class TestEdgeLoader:
    """Tests for EdgeLoader.load()."""

    def _make_edge_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "edge_id": "edge-1",
            "run_id": "run-1",
            "from_node_id": "node-1",
            "to_node_id": "node-2",
            "label": "continue",
            "default_mode": "move",
            "created_at": NOW,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize("mode_value", [m.value for m in RoutingMode])
    def test_valid_routing_modes(self, mode_value: str) -> None:
        sa_row = self._make_edge_row(default_mode=mode_value)
        loader = EdgeLoader()
        result = loader.load(sa_row)
        assert isinstance(result, Edge)
        assert result.default_mode == RoutingMode(mode_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_edge_row()
        loader = EdgeLoader()
        result = loader.load(sa_row)
        assert result.edge_id == "edge-1"
        assert result.from_node_id == "node-1"
        assert result.to_node_id == "node-2"
        assert result.label == "continue"

    def test_invalid_mode_raises_value_error(self) -> None:
        sa_row = self._make_edge_row(default_mode="teleport")
        loader = EdgeLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# RowLoader
# ---------------------------------------------------------------------------


class TestRowLoader:
    """Tests for RowLoader.load()."""

    def _make_row_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "row_id": "row-1",
            "run_id": "run-1",
            "source_node_id": "node-src",
            "row_index": 42,
            # source_row_index (per-source position) and ingest_sequence (global
            # ingest order) are NOT NULL audit columns the loader maps. Use values
            # distinct from row_index to honour the G22 fabrication ban (these are
            # not derived from row_index) and to prove the loader maps each column
            # to the correct Row field rather than conflating them.
            "source_row_index": 5,
            "ingest_sequence": 17,
            "source_data_hash": "hash123",
            "created_at": NOW,
            "source_data_ref": "ref://payload/abc",
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load(self) -> None:
        sa_row = self._make_row_row()
        loader = RowLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Row)
        assert result.row_id == "row-1"
        assert result.row_index == 42
        assert result.source_row_index == 5
        assert result.ingest_sequence == 17
        assert result.source_data_hash == "hash123"
        assert result.source_data_ref == "ref://payload/abc"

    def test_valid_load_with_none_ref(self) -> None:
        sa_row = self._make_row_row(source_data_ref=None)
        loader = RowLoader()
        result = loader.load(sa_row)
        assert result.source_data_ref is None


# ---------------------------------------------------------------------------
# TokenLoader
# ---------------------------------------------------------------------------


class TestTokenLoader:
    """Tests for TokenLoader.load()."""

    def _make_token_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "token_id": "tok-1",
            "row_id": "row-1",
            "run_id": "run-1",
            "created_at": NOW,
            "fork_group_id": None,
            "join_group_id": None,
            "expand_group_id": None,
            "branch_name": None,
            "step_in_pipeline": None,
            # Epoch-11 column: real select(tokens_table) rows always carry it.
            "token_data_ref": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_minimal(self) -> None:
        sa_row = self._make_token_row()
        loader = TokenLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Token)
        assert result.token_id == "tok-1"
        assert result.row_id == "row-1"
        assert result.run_id == "run-1"
        assert result.fork_group_id is None
        assert result.join_group_id is None

    def test_valid_load_with_all_optional_fields(self) -> None:
        sa_row = self._make_token_row(
            fork_group_id="fg-1",
            join_group_id="jg-1",
            expand_group_id="eg-1",
            branch_name="path_a",
            step_in_pipeline=3,
        )
        loader = TokenLoader()
        result = loader.load(sa_row)
        assert result.fork_group_id == "fg-1"
        assert result.join_group_id == "jg-1"
        assert result.expand_group_id == "eg-1"
        assert result.branch_name == "path_a"
        assert result.step_in_pipeline == 3

    def test_load_populates_token_data_ref(self) -> None:
        # The loader must reflect the persisted token_data_ref (epoch-11 column),
        # not silently fabricate None. Expand children and post-coalesce merged
        # tokens carry a non-NULL ref; dropping it would hand a future reader a
        # synthetic None for a value the audit trail actually recorded.
        sa_row = self._make_token_row(token_data_ref="payload-ref-abc123")
        loader = TokenLoader()
        result = loader.load(sa_row)
        assert result.token_data_ref == "payload-ref-abc123"

    def test_load_token_data_ref_none_stays_none(self) -> None:
        # NULL is the legitimate value for fork children (they share the source
        # payload) — None here is real absence, not fabrication.
        sa_row = self._make_token_row(token_data_ref=None)
        loader = TokenLoader()
        result = loader.load(sa_row)
        assert result.token_data_ref is None


# ---------------------------------------------------------------------------
# TokenParentLoader
# ---------------------------------------------------------------------------


class TestTokenParentLoader:
    """Tests for TokenParentLoader.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            token_id="tok-child",
            parent_token_id="tok-parent",
            ordinal=0,
        )
        loader = TokenParentLoader()
        result = loader.load(sa_row)

        assert isinstance(result, TokenParent)
        assert result.token_id == "tok-child"
        assert result.parent_token_id == "tok-parent"
        assert result.ordinal == 0

    def test_valid_load_higher_ordinal(self) -> None:
        sa_row = _make_sa_row(
            token_id="tok-child",
            parent_token_id="tok-parent-2",
            ordinal=5,
        )
        loader = TokenParentLoader()
        result = loader.load(sa_row)
        assert result.ordinal == 5


# ---------------------------------------------------------------------------
# CallLoader
# ---------------------------------------------------------------------------


class TestCallLoader:
    """Tests for CallLoader.load()."""

    def _make_call_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "call_id": "call-1",
            "call_index": 0,
            "call_type": "llm",
            "status": "success",
            "request_hash": "req123",
            "created_at": NOW,
            "state_id": "state-1",
            "operation_id": None,
            "request_ref": None,
            "response_hash": None,
            "response_ref": None,
            "error_json": None,
            "latency_ms": None,
            "resolved_prompt_template_hash": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def test_valid_load_state_parented(self) -> None:
        sa_row = self._make_call_row(state_id="state-1", operation_id=None)
        loader = CallLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Call)
        assert result.call_type == CallType.LLM
        assert result.status == CallStatus.SUCCESS
        assert result.state_id == "state-1"
        assert result.operation_id is None

    def test_maps_resolved_prompt_template_hash(self) -> None:
        """CallLoader must carry the cross-DB prompt-hash anchor (elspeth-543ee35ed3)."""
        digest = "a" * 64
        sa_row = self._make_call_row(call_type="llm", resolved_prompt_template_hash=digest)
        result = CallLoader().load(sa_row)
        assert result.resolved_prompt_template_hash == digest

    def test_valid_load_operation_parented(self) -> None:
        sa_row = self._make_call_row(
            state_id=None,
            operation_id="op-1",
            call_type="http",
            status="error",
        )
        loader = CallLoader()
        result = loader.load(sa_row)

        assert result.call_type == CallType.HTTP
        assert result.status == CallStatus.ERROR
        assert result.state_id is None
        assert result.operation_id == "op-1"

    def test_valid_load_with_response_fields(self) -> None:
        sa_row = self._make_call_row(
            response_hash="resp123",
            response_ref="ref://resp/abc",
            latency_ms=42.5,
        )
        loader = CallLoader()
        result = loader.load(sa_row)
        assert result.response_hash == "resp123"
        assert result.response_ref == "ref://resp/abc"
        assert result.latency_ms == 42.5

    def test_valid_load_with_error_json(self) -> None:
        sa_row = self._make_call_row(
            status="error",
            error_json='{"reason": "timeout"}',
        )
        loader = CallLoader()
        result = loader.load(sa_row)
        assert result.error_json == '{"reason": "timeout"}'

    @pytest.mark.parametrize("call_type_value", [t.value for t in CallType])
    def test_valid_call_type_values(self, call_type_value: str) -> None:
        sa_row = self._make_call_row(call_type=call_type_value)
        loader = CallLoader()
        result = loader.load(sa_row)
        assert result.call_type == CallType(call_type_value)

    @pytest.mark.parametrize("status_value", [s.value for s in CallStatus])
    def test_valid_call_status_values(self, status_value: str) -> None:
        sa_row = self._make_call_row(status=status_value)
        loader = CallLoader()
        result = loader.load(sa_row)
        assert result.status == CallStatus(status_value)

    def test_invalid_call_type_raises_value_error(self) -> None:
        sa_row = self._make_call_row(call_type="carrier_pigeon")
        loader = CallLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_invalid_call_status_raises_value_error(self) -> None:
        sa_row = self._make_call_row(status="maybe")
        loader = CallLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_both_state_and_operation_raises_audit_integrity(self) -> None:
        """Both state_id and operation_id set is a Tier 1 integrity violation."""
        sa_row = self._make_call_row(state_id="state-1", operation_id="op-1")
        loader = CallLoader()
        with pytest.raises(AuditIntegrityError, match="exactly one"):
            loader.load(sa_row)

    def test_neither_state_nor_operation_raises_audit_integrity(self) -> None:
        """Neither state_id nor operation_id set is a Tier 1 integrity violation."""
        sa_row = self._make_call_row(state_id=None, operation_id=None)
        loader = CallLoader()
        with pytest.raises(AuditIntegrityError, match="exactly one"):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# RoutingEventLoader
# ---------------------------------------------------------------------------


class TestRoutingEventLoader:
    """Tests for RoutingEventLoader.load()."""

    def _make_routing_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "event_id": "evt-1",
            "state_id": "state-1",
            "edge_id": "edge-1",
            "routing_group_id": "rg-1",
            "ordinal": 0,
            "mode": "move",
            "created_at": NOW,
            "reason_hash": None,
            "reason_ref": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize("mode_value", [m.value for m in RoutingMode])
    def test_valid_routing_modes(self, mode_value: str) -> None:
        sa_row = self._make_routing_row(mode=mode_value)
        loader = RoutingEventLoader()
        result = loader.load(sa_row)
        assert isinstance(result, RoutingEvent)
        assert result.mode == RoutingMode(mode_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_routing_row(
            reason_hash="hash456",
            reason_ref="ref://reason/abc",
        )
        loader = RoutingEventLoader()
        result = loader.load(sa_row)
        assert result.event_id == "evt-1"
        assert result.reason_hash == "hash456"
        assert result.reason_ref == "ref://reason/abc"

    def test_invalid_mode_raises_value_error(self) -> None:
        sa_row = self._make_routing_row(mode="warp")
        loader = RoutingEventLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# BatchLoader
# ---------------------------------------------------------------------------


class TestBatchLoader:
    """Tests for BatchLoader.load()."""

    def _make_batch_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "batch_id": "batch-1",
            "run_id": "run-1",
            "aggregation_node_id": "agg-node-1",
            "attempt": 1,
            "status": "draft",
            "created_at": NOW,
            "aggregation_state_id": None,
            "retry_of_batch_id": None,
            "trigger_type": None,
            "trigger_reason": None,
            "completed_at": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    @pytest.mark.parametrize("status_value", [s.value for s in BatchStatus])
    def test_valid_batch_status_values(self, status_value: str) -> None:
        sa_row = self._make_batch_row(status=status_value)
        loader = BatchLoader()
        result = loader.load(sa_row)
        assert isinstance(result, Batch)
        assert result.status == BatchStatus(status_value)

    def test_valid_load_all_fields(self) -> None:
        sa_row = self._make_batch_row(
            status="completed",
            aggregation_state_id="agg-state-1",
            retry_of_batch_id="batch-0",
            trigger_type="count",
            trigger_reason="threshold=10",
            completed_at=LATER,
        )
        loader = BatchLoader()
        result = loader.load(sa_row)
        assert result.batch_id == "batch-1"
        assert result.aggregation_state_id == "agg-state-1"
        assert result.retry_of_batch_id == "batch-0"
        assert result.trigger_type == TriggerType.COUNT
        assert isinstance(result.trigger_type, TriggerType)
        assert result.trigger_reason == "threshold=10"
        assert result.created_at == NOW
        assert result.completed_at == LATER

    def test_invalid_status_raises_value_error(self) -> None:
        sa_row = self._make_batch_row(status="cooking")
        loader = BatchLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    @pytest.mark.parametrize("trigger_value", [t.value for t in TriggerType])
    def test_valid_trigger_type_values(self, trigger_value: str) -> None:
        sa_row = self._make_batch_row(trigger_type=trigger_value)
        loader = BatchLoader()
        result = loader.load(sa_row)
        assert result.trigger_type == TriggerType(trigger_value)

    def test_none_trigger_type_stays_none(self) -> None:
        sa_row = self._make_batch_row(trigger_type=None)
        loader = BatchLoader()
        result = loader.load(sa_row)
        assert result.trigger_type is None

    def test_invalid_trigger_type_raises_value_error(self) -> None:
        sa_row = self._make_batch_row(trigger_type="not_a_trigger")
        loader = BatchLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_manual_trigger_type_raises_value_error(self) -> None:
        sa_row = self._make_batch_row(trigger_type="manual")
        loader = BatchLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)


# ---------------------------------------------------------------------------
# NodeStateLoader (discriminated union -- most complex)
# ---------------------------------------------------------------------------


class TestNodeStateLoader:
    """Tests for NodeStateLoader.load() -- discriminated union."""

    def _make_node_state_row(self, **overrides: object) -> SARow[Any]:
        """Create a base node state row with all columns present.

        All completion-related fields default to None (representing OPEN).
        Tests override specific fields to represent other states.
        """
        defaults = {
            "state_id": "state-1",
            "token_id": "tok-1",
            "node_id": "node-1",
            "step_index": 0,
            "attempt": 1,
            "status": "open",
            "input_hash": "in123",
            "started_at": NOW,
            "output_hash": None,
            "completed_at": None,
            "duration_ms": None,
            "context_before_json": None,
            "context_after_json": None,
            "success_reason_json": None,
            "error_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    # === OPEN variant ===

    def test_open_valid(self) -> None:
        sa_row = self._make_node_state_row(status="open")
        loader = NodeStateLoader()
        result = loader.load(sa_row)

        assert isinstance(result, NodeStateOpen)
        assert result.status == NodeStateStatus.OPEN
        assert result.state_id == "state-1"
        assert result.token_id == "tok-1"
        assert result.input_hash == "in123"

    def test_open_with_context_before(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            context_before_json='{"key": "value"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateOpen)
        assert result.context_before_json == '{"key": "value"}'

    def test_open_with_non_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            output_hash="bad_hash",
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL output_hash"):
            loader.load(sa_row)

    def test_open_with_non_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            completed_at=NOW,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL completed_at"):
            loader.load(sa_row)

    def test_open_with_non_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            duration_ms=100.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL duration_ms"):
            loader.load(sa_row)

    def test_open_with_non_null_context_after_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            context_after_json='{"after": true}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL context_after_json"):
            loader.load(sa_row)

    def test_open_with_non_null_error_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            error_json='{"reason": "unexpected"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL error_json"):
            loader.load(sa_row)

    def test_open_with_non_null_success_reason_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="open",
            success_reason_json='{"action": "classified"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL success_reason_json"):
            loader.load(sa_row)

    def test_open_with_all_invalid_completion_fields(self) -> None:
        """All three completion fields present on OPEN -- first check fires."""
        sa_row = self._make_node_state_row(
            status="open",
            output_hash="bad",
            completed_at=NOW,
            duration_ms=50.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="audit integrity violation"):
            loader.load(sa_row)

    # === PENDING variant ===

    def test_pending_valid(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=LATER,
            duration_ms=150.5,
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)

        assert isinstance(result, NodeStatePending)
        assert result.status == NodeStateStatus.PENDING
        assert result.started_at == NOW
        assert result.completed_at == LATER
        assert result.duration_ms == 150.5

    def test_pending_with_context_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            context_before_json='{"before": true}',
            context_after_json='{"after": true}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStatePending)
        assert result.context_before_json == '{"before": true}'
        assert result.context_after_json == '{"after": true}'

    def test_pending_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=None,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL duration_ms"):
            loader.load(sa_row)

    def test_pending_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=None,
            duration_ms=100.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL completed_at"):
            loader.load(sa_row)

    def test_pending_with_non_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            output_hash="unexpected_hash",
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL output_hash"):
            loader.load(sa_row)

    def test_pending_with_non_null_error_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            error_json='{"reason": "premature"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL error_json"):
            loader.load(sa_row)

    def test_pending_with_non_null_success_reason_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="pending",
            completed_at=NOW,
            duration_ms=100.0,
            success_reason_json='{"action": "premature"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL success_reason_json"):
            loader.load(sa_row)

    # === COMPLETED variant ===

    def test_completed_valid(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=LATER,
            duration_ms=200.0,
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)

        assert isinstance(result, NodeStateCompleted)
        assert result.status == NodeStateStatus.COMPLETED
        assert result.output_hash == "out123"
        assert result.started_at == NOW
        assert result.completed_at == LATER
        assert result.duration_ms == 200.0

    def test_completed_with_optional_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=200.0,
            context_before_json='{"b": 1}',
            context_after_json='{"a": 2}',
            success_reason_json='{"action": "classified"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateCompleted)
        assert result.success_reason_json == '{"action": "classified"}'
        assert result.context_before_json == '{"b": 1}'
        assert result.context_after_json == '{"a": 2}'

    def test_completed_with_null_output_hash_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash=None,
            completed_at=NOW,
            duration_ms=100.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL output_hash"):
            loader.load(sa_row)

    def test_completed_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=None,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL duration_ms"):
            loader.load(sa_row)

    def test_completed_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=None,
            duration_ms=100.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL completed_at"):
            loader.load(sa_row)

    def test_completed_with_non_null_error_json_raises(self) -> None:
        """COMPLETED + error_json violates mutual exclusivity."""
        sa_row = self._make_node_state_row(
            status="completed",
            output_hash="out123",
            completed_at=NOW,
            duration_ms=100.0,
            error_json='{"reason": "should not be here"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL error_json"):
            loader.load(sa_row)

    # === FAILED variant ===

    def test_failed_with_null_error_json_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=LATER,
            duration_ms=50.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL error_json"):
            loader.load(sa_row)

    def test_failed_with_error_json(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            error_json='{"reason": "division by zero"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.error_json == '{"reason": "division by zero"}'

    def test_failed_with_output_hash_allowed(self) -> None:
        """FAILED states MAY have output_hash (optional)."""
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            output_hash="partial_out",
            error_json='{"reason": "division by zero"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.output_hash == "partial_out"

    def test_failed_with_output_hash_and_error_json(self) -> None:
        """FAILED allows both output_hash (partial output) and error_json simultaneously."""
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=100.0,
            output_hash="partial-out",
            error_json='{"error": "timeout"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.output_hash == "partial-out"
        assert result.error_json == '{"error": "timeout"}'

    def test_failed_with_context_fields(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            context_before_json='{"b": 1}',
            context_after_json='{"a": 2}',
            error_json='{"reason": "division by zero"}',
        )
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, NodeStateFailed)
        assert result.context_before_json == '{"b": 1}'
        assert result.context_after_json == '{"a": 2}'

    def test_failed_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=None,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL duration_ms"):
            loader.load(sa_row)

    def test_failed_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=None,
            duration_ms=50.0,
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="NULL completed_at"):
            loader.load(sa_row)

    def test_failed_with_non_null_success_reason_json_raises(self) -> None:
        """FAILED + success_reason_json violates mutual exclusivity."""
        sa_row = self._make_node_state_row(
            status="failed",
            completed_at=NOW,
            duration_ms=50.0,
            error_json='{"reason": "division by zero"}',
            success_reason_json='{"action": "should not be here"}',
        )
        loader = NodeStateLoader()
        with pytest.raises(AuditIntegrityError, match="non-NULL success_reason_json"):
            loader.load(sa_row)

    # === Invalid status ===

    def test_unknown_status_string_raises_value_error(self) -> None:
        sa_row = self._make_node_state_row(status="processing")
        loader = NodeStateLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    def test_empty_status_string_raises_value_error(self) -> None:
        sa_row = self._make_node_state_row(status="")
        loader = NodeStateLoader()
        with pytest.raises(ValueError):
            loader.load(sa_row)

    # === Verify all 4 variants route correctly ===

    @pytest.mark.parametrize(
        ("status_str", "expected_type"),
        [
            ("open", NodeStateOpen),
            ("pending", NodeStatePending),
            ("completed", NodeStateCompleted),
            ("failed", NodeStateFailed),
        ],
    )
    def test_status_routes_to_correct_variant(self, status_str: str, expected_type: type) -> None:
        """Each status string produces the correct variant type."""
        extra: dict[str, object] = {}
        if status_str == "pending":
            extra = {"completed_at": NOW, "duration_ms": 100.0}
        elif status_str == "failed":
            extra = {
                "completed_at": NOW,
                "duration_ms": 100.0,
                "error_json": '{"reason": "division by zero"}',
            }
        elif status_str == "completed":
            extra = {
                "output_hash": "out",
                "completed_at": NOW,
                "duration_ms": 100.0,
            }

        sa_row = self._make_node_state_row(status=status_str, **extra)
        loader = NodeStateLoader()
        result = loader.load(sa_row)
        assert isinstance(result, expected_type)


# ---------------------------------------------------------------------------
# ValidationErrorLoader
# ---------------------------------------------------------------------------


class TestValidationErrorLoader:
    """Tests for ValidationErrorLoader.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            error_id="ve-1",
            run_id="run-1",
            node_id="node-src",
            row_hash="hash123",
            row_id="row-1",
            error="Missing required field 'amount'",
            schema_mode="fixed",
            destination="quarantine",
            created_at=NOW,
            row_data_json='{"name": "test"}',
            violation_type="missing_field",
            original_field_name="Amount USD",
            normalized_field_name="amount_usd",
            expected_type="int",
            actual_type=None,
        )
        loader = ValidationErrorLoader()
        result = loader.load(sa_row)

        assert isinstance(result, ValidationErrorRecord)
        assert result.error_id == "ve-1"
        assert result.error == "Missing required field 'amount'"
        assert result.schema_mode == "fixed"
        assert result.row_id == "row-1"
        assert result.row_data_json == '{"name": "test"}'
        assert result.violation_type == "missing_field"
        assert result.original_field_name == "Amount USD"
        assert result.normalized_field_name == "amount_usd"
        assert result.expected_type == "int"
        assert result.actual_type is None

    def test_valid_load_with_none_optionals(self) -> None:
        sa_row = _make_sa_row(
            error_id="ve-2",
            run_id="run-1",
            node_id=None,
            row_hash="hash456",
            row_id=None,
            error="Type error",
            schema_mode="flexible",
            destination="quarantine",
            created_at=NOW,
            row_data_json=None,
            violation_type=None,
            original_field_name=None,
            normalized_field_name=None,
            expected_type=None,
            actual_type=None,
        )
        loader = ValidationErrorLoader()
        result = loader.load(sa_row)
        assert result.node_id is None
        assert result.row_id is None
        assert result.row_data_json is None
        assert result.violation_type is None
        assert result.original_field_name is None


# ---------------------------------------------------------------------------
# TransformErrorLoader
# ---------------------------------------------------------------------------


class TestTransformErrorLoader:
    """Tests for TransformErrorLoader.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            error_id="te-1",
            run_id="run-1",
            token_id="tok-1",
            transform_id="tfm-1",
            row_hash="hash789",
            destination="error_sink",
            created_at=NOW,
            row_data_json='{"x": 1}',
            error_details_json='{"reason": "division by zero"}',
        )
        loader = TransformErrorLoader()
        result = loader.load(sa_row)

        assert isinstance(result, TransformErrorRecord)
        assert result.error_id == "te-1"
        assert result.token_id == "tok-1"
        assert result.transform_id == "tfm-1"
        assert result.error_details_json == '{"reason": "division by zero"}'

    def test_valid_load_with_none_optionals(self) -> None:
        sa_row = _make_sa_row(
            error_id="te-2",
            run_id="run-1",
            token_id="tok-2",
            transform_id="tfm-2",
            row_hash="hash000",
            destination="error_sink",
            created_at=NOW,
            row_data_json=None,
            error_details_json=None,
        )
        loader = TransformErrorLoader()
        result = loader.load(sa_row)
        assert result.row_data_json is None
        assert result.error_details_json is None


# ---------------------------------------------------------------------------
# TokenOutcomeLoader
# ---------------------------------------------------------------------------


class TestTokenOutcomeLoaderTwoAxis:
    """ADR-019 Phase 1: loader runs two-axis cross-checks at read time."""

    def _make_outcome_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "outcome_id": "oc-1",
            "run_id": "run-1",
            "token_id": "tok-1",
            "outcome": "success",
            "path": "default_flow",
            "completed": 1,
            "recorded_at": NOW,
            "sink_name": "output",
            "batch_id": None,
            "fork_group_id": None,
            "join_group_id": None,
            "expand_group_id": None,
            "error_hash": None,
            "context_json": None,
            "expected_branches_json": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    def _valid_fields(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> dict[str, str]:
        constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
        values = {
            "sink_name": "output",
            "batch_id": "batch-1",
            "fork_group_id": "fork-1",
            "join_group_id": "join-1",
            "expand_group_id": "expand-1",
            "error_hash": "e" * 64,
        }
        fields: dict[str, str] = {}
        for field_name in constraints.required:
            exact_value = constraints.exact.get(field_name)
            fields[field_name] = str(exact_value if exact_value is not None else values[field_name])
        for field_name, exact_value in constraints.exact.items():
            fields[field_name] = str(exact_value)
        return fields

    def _row_for_pair(self, pair: tuple[TerminalOutcome | None, TerminalPath], **overrides: object) -> SARow[Any]:
        outcome, path = pair
        row_overrides: dict[str, object] = {
            "outcome": outcome.value if outcome is not None else None,
            "path": path.value,
            "completed": 1 if outcome is not None else 0,
            "sink_name": None,
            "batch_id": None,
            "fork_group_id": None,
            "join_group_id": None,
            "expand_group_id": None,
            "error_hash": None,
        }
        row_overrides.update(self._valid_fields(pair))
        row_overrides.update(overrides)
        return self._make_outcome_row(**row_overrides)

    def test_loads_completed_default_flow(self) -> None:
        result = TokenOutcomeLoader().load(self._row_for_pair((TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW)))

        assert isinstance(result, TokenOutcome)
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW
        assert result.completed is True
        assert result.sink_name == "output"

    def test_loads_buffered(self) -> None:
        result = TokenOutcomeLoader().load(self._row_for_pair((None, TerminalPath.BUFFERED)))

        assert result.outcome is None
        assert result.path == TerminalPath.BUFFERED
        assert result.completed is False
        assert result.batch_id == "batch-1"

    @pytest.mark.parametrize("pair", tuple(_TERMINAL_PAIR_FIELD_CONSTRAINTS))
    def test_every_legal_pair_loads(self, pair: tuple[TerminalOutcome | None, TerminalPath]) -> None:
        outcome, path = pair

        result = TokenOutcomeLoader().load(self._row_for_pair(pair))

        assert result.outcome == outcome
        assert result.path == path
        assert result.completed is (outcome is not None)

    def test_completed_xor_outcome_violation_crashes(self) -> None:
        sa_row = self._make_outcome_row(outcome=None, path="default_flow", completed=1)

        with pytest.raises(AuditIntegrityError, match="completed"):
            TokenOutcomeLoader().load(sa_row)

    def test_illegal_pair_in_db_crashes(self) -> None:
        sa_row = self._make_outcome_row(
            outcome="success",
            path="unrouted",
            completed=1,
            sink_name="output",
        )

        with pytest.raises(AuditIntegrityError, match="_LEGAL_TERMINAL_PAIRS"):
            TokenOutcomeLoader().load(sa_row)

    def test_required_field_missing_crashes(self) -> None:
        sa_row = self._row_for_pair(
            (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW),
            sink_name=None,
        )

        with pytest.raises(AuditIntegrityError, match="requires sink_name"):
            TokenOutcomeLoader().load(sa_row)

    @pytest.mark.parametrize("pair", tuple(_TERMINAL_PAIR_FIELD_CONSTRAINTS))
    def test_every_pair_constraint_row_is_enforced(
        self,
        pair: tuple[TerminalOutcome | None, TerminalPath],
    ) -> None:
        constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
        overrides: dict[str, object]
        if constraints.required:
            overrides = {constraints.required[0]: None}
        elif constraints.exact:
            field_name = next(iter(constraints.exact))
            overrides = {field_name: "wrong-exact-value"}
        else:
            overrides = {constraints.forbidden[0]: "forbidden-extra"}

        with pytest.raises(AuditIntegrityError, match="audit integrity violation"):
            TokenOutcomeLoader().load(self._row_for_pair(pair, **overrides))

    @pytest.mark.parametrize(
        ("completed", "outcome", "path", "match"),
        [
            (1, None, "default_flow", "completed"),
            (0, "success", "buffered", "completed"),
            (0, None, "default_flow", "BUFFERED"),
            (1, "success", None, "path is NULL"),
            (1, "success", "not_a_path", "invalid path"),
            ("1", "success", "default_flow", "invalid completed"),
            (True, "success", "default_flow", "invalid completed"),
        ],
    )
    def test_tampered_shape_crashes(
        self,
        completed: object,
        outcome: object,
        path: object,
        match: str,
    ) -> None:
        sa_row = self._make_outcome_row(completed=completed, outcome=outcome, path=path)

        with pytest.raises(AuditIntegrityError, match=match):
            TokenOutcomeLoader().load(sa_row)

    def test_invalid_outcome_crashes_as_audit_integrity_error(self) -> None:
        sa_row = self._make_outcome_row(outcome="vanished")

        with pytest.raises(AuditIntegrityError, match="invalid outcome"):
            TokenOutcomeLoader().load(sa_row)

    def test_valid_load_with_optional_context_fields(self) -> None:
        sa_row = self._row_for_pair(
            (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT),
            context_json='{"paths": ["a", "b"]}',
            expected_branches_json='["path_a", "path_b"]',
        )

        result = TokenOutcomeLoader().load(sa_row)

        assert result.outcome == TerminalOutcome.TRANSIENT
        assert result.path == TerminalPath.FORK_PARENT
        assert result.fork_group_id == "fork-1"
        assert result.context_json == '{"paths": ["a", "b"]}'
        assert result.expected_branches_json == '["path_a", "path_b"]'


# ---------------------------------------------------------------------------
# ArtifactLoader
# ---------------------------------------------------------------------------


class TestArtifactLoader:
    """Tests for ArtifactLoader.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            artifact_id="art-1",
            run_id="run-1",
            produced_by_state_id="state-1",
            sink_node_id="sink-1",
            artifact_type="csv",
            path_or_uri="/output/results.csv",
            content_hash="hash999",
            size_bytes=1024,
            created_at=NOW,
            idempotency_key=None,
        )
        loader = ArtifactLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Artifact)
        assert result.artifact_id == "art-1"
        assert result.artifact_type == "csv"
        assert result.path_or_uri == "/output/results.csv"
        assert result.size_bytes == 1024
        assert result.idempotency_key is None

    def test_valid_load_with_idempotency_key(self) -> None:
        sa_row = _make_sa_row(
            artifact_id="art-2",
            run_id="run-1",
            produced_by_state_id="state-2",
            sink_node_id="sink-1",
            artifact_type="json",
            path_or_uri="/output/data.json",
            content_hash="hash888",
            size_bytes=2048,
            created_at=NOW,
            idempotency_key="retry-key-42",
        )
        loader = ArtifactLoader()
        result = loader.load(sa_row)
        assert result.idempotency_key == "retry-key-42"


# ---------------------------------------------------------------------------
# BatchMemberLoader
# ---------------------------------------------------------------------------


class TestBatchMemberLoader:
    """Tests for BatchMemberLoader.load()."""

    def test_valid_load(self) -> None:
        sa_row = _make_sa_row(
            batch_id="batch-1",
            run_id="run-1",
            token_id="tok-1",
            ordinal=0,
        )
        loader = BatchMemberLoader()
        result = loader.load(sa_row)

        assert isinstance(result, BatchMember)
        assert result.batch_id == "batch-1"
        assert result.run_id == "run-1"
        assert result.token_id == "tok-1"
        assert result.ordinal == 0

    def test_valid_load_higher_ordinal(self) -> None:
        sa_row = _make_sa_row(
            batch_id="batch-2",
            run_id="run-2",
            token_id="tok-5",
            ordinal=10,
        )
        loader = BatchMemberLoader()
        result = loader.load(sa_row)
        assert result.ordinal == 10


# ---------------------------------------------------------------------------
# OperationLoader
# ---------------------------------------------------------------------------


class TestOperationLoader:
    """Tests for OperationLoader.load().

    Operation uses Literal types validated by __post_init__(), not enums.
    Tests verify that the loader correctly maps all fields and that
    __post_init__ lifecycle invariants still fire through the loader path.
    """

    def _make_operation_row(self, **overrides: object) -> SARow[Any]:
        defaults = {
            "operation_id": "op-1",
            "run_id": "run-1",
            "node_id": "node-1",
            "operation_type": "source_load",
            "started_at": NOW,
            "status": "open",
            "completed_at": None,
            "input_data_ref": None,
            "input_data_hash": None,
            "output_data_ref": None,
            "output_data_hash": None,
            "error_message": None,
            "duration_ms": None,
        }
        defaults.update(overrides)
        return _make_sa_row(**defaults)

    # === Happy paths ===

    def test_open_source_load(self) -> None:
        sa_row = self._make_operation_row()
        loader = OperationLoader()
        result = loader.load(sa_row)

        assert isinstance(result, Operation)
        assert result.operation_id == "op-1"
        assert result.run_id == "run-1"
        assert result.node_id == "node-1"
        assert result.operation_type == "source_load"
        assert result.status == "open"
        assert result.started_at == NOW
        assert result.completed_at is None

    def test_completed_sink_write(self) -> None:
        sa_row = self._make_operation_row(
            operation_type="sink_write",
            status="completed",
            completed_at=LATER,
            duration_ms=250.0,
            input_data_ref="ref://in/abc",
            input_data_hash="inhash",
            output_data_ref="ref://out/xyz",
            output_data_hash="outhash",
        )
        loader = OperationLoader()
        result = loader.load(sa_row)

        assert result.operation_type == "sink_write"
        assert result.status == "completed"
        assert result.started_at == NOW
        assert result.completed_at == LATER
        assert result.duration_ms == 250.0
        assert result.input_data_ref == "ref://in/abc"
        assert result.output_data_ref == "ref://out/xyz"

    def test_failed_with_error(self) -> None:
        sa_row = self._make_operation_row(
            status="failed",
            completed_at=LATER,
            duration_ms=100.0,
            error_message="connection refused",
        )
        loader = OperationLoader()
        result = loader.load(sa_row)

        assert result.status == "failed"
        assert result.error_message == "connection refused"
        assert result.completed_at == LATER

    def test_pending_status(self) -> None:
        sa_row = self._make_operation_row(
            status="pending",
            completed_at=LATER,
            duration_ms=50.0,
        )
        loader = OperationLoader()
        result = loader.load(sa_row)

        assert result.status == "pending"
        assert result.completed_at == LATER

    # === Lifecycle invariant violations (validated by OperationLoader, Tier 1) ===

    def test_invalid_operation_type_raises(self) -> None:
        sa_row = self._make_operation_row(operation_type="kafka_consume")
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="operation_type"):
            loader.load(sa_row)

    def test_invalid_status_raises(self) -> None:
        sa_row = self._make_operation_row(status="running")
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="status"):
            loader.load(sa_row)

    def test_open_with_completed_at_raises(self) -> None:
        sa_row = self._make_operation_row(status="open", completed_at=NOW)
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="completed_at"):
            loader.load(sa_row)

    def test_open_with_duration_ms_raises(self) -> None:
        sa_row = self._make_operation_row(status="open", duration_ms=100.0)
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="duration_ms"):
            loader.load(sa_row)

    def test_open_with_error_message_raises(self) -> None:
        sa_row = self._make_operation_row(status="open", error_message="bad")
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="error_message"):
            loader.load(sa_row)

    def test_completed_with_null_completed_at_raises(self) -> None:
        sa_row = self._make_operation_row(
            status="completed",
            completed_at=None,
            duration_ms=100.0,
        )
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="completed_at"):
            loader.load(sa_row)

    def test_completed_with_null_duration_ms_raises(self) -> None:
        sa_row = self._make_operation_row(
            status="completed",
            completed_at=LATER,
            duration_ms=None,
        )
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="duration_ms"):
            loader.load(sa_row)

    def test_completed_with_error_message_raises(self) -> None:
        sa_row = self._make_operation_row(
            status="completed",
            completed_at=LATER,
            duration_ms=100.0,
            error_message="should not be here",
        )
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="error_message"):
            loader.load(sa_row)

    def test_failed_with_null_error_message_raises(self) -> None:
        sa_row = self._make_operation_row(
            status="failed",
            completed_at=LATER,
            duration_ms=100.0,
            error_message=None,
        )
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="error_message"):
            loader.load(sa_row)

    def test_failed_with_empty_error_message_raises(self) -> None:
        sa_row = self._make_operation_row(
            status="failed",
            completed_at=LATER,
            duration_ms=100.0,
            error_message="",
        )
        loader = OperationLoader()
        with pytest.raises(AuditIntegrityError, match="error_message"):
            loader.load(sa_row)

    # === Runtime operation types accepted ===

    @pytest.mark.parametrize("op_type", ["source_load", "sink_write", "runtime_preflight"])
    def test_runtime_operation_types_accepted(self, op_type: str) -> None:
        sa_row = self._make_operation_row(operation_type=op_type)
        loader = OperationLoader()
        result = loader.load(sa_row)
        assert result.operation_type == op_type

    @pytest.mark.parametrize("status", ["open", "completed", "failed", "pending"])
    def test_all_valid_statuses_accepted(self, status: str) -> None:
        extra: dict[str, object] = {}
        if status in ("completed", "pending"):
            extra = {"completed_at": LATER, "duration_ms": 100.0}
        elif status == "failed":
            extra = {"completed_at": LATER, "duration_ms": 100.0, "error_message": "err"}
        sa_row = self._make_operation_row(status=status, **extra)
        loader = OperationLoader()
        result = loader.load(sa_row)
        assert result.status == status

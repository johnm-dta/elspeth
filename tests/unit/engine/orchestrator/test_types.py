# tests/unit/engine/orchestrator/test_types.py
"""Tests for AggregationFlushResult dataclass.

These tests verify field ordering and semantics after migrating from a 9-element
tuple to a named dataclass. Field ordering bugs would be SILENT - counts could
swap between rows_succeeded and rows_failed without obvious test failures.

The test values (1,2,3,4,5,6,7,8,9) are deliberately distinct to catch any
field ordering mistakes during construction or addition.
"""

from dataclasses import FrozenInstanceError
from types import MappingProxyType
from unittest.mock import Mock

import pytest

from elspeth.contracts import RunStatus
from elspeth.contracts.types import CoalesceName, GateName, NodeID, SinkName
from elspeth.engine.orchestrator.types import (
    AggregationFlushResult,
    ExecutionCounters,
    PipelineConfig,
    ValueSourceFinding,
    ValueSourceValidationError,
)


class TestAggregationFlushResult:
    """Tests for the AggregationFlushResult dataclass."""

    def test_all_fields_accessible_by_name(self) -> None:
        """Verify each field is accessible by name (catches field omission)."""
        result = AggregationFlushResult(
            rows_succeeded=1,
            rows_failed=2,
            rows_routed_success=3,
            rows_routed_failure=0,
            rows_quarantined=4,
            rows_coalesced=5,
            rows_forked=6,
            rows_expanded=7,
            rows_buffered=8,
            routed_destinations={"sink_a": 9},
        )

        # Each field must be accessible and have the correct value
        assert result.rows_succeeded == 1
        assert result.rows_failed == 2
        assert result.rows_routed_success == 3
        assert result.rows_routed_failure == 0
        assert result.rows_quarantined == 4
        assert result.rows_coalesced == 5
        assert result.rows_forked == 6
        assert result.rows_expanded == 7
        assert result.rows_buffered == 8
        assert result.routed_destinations == {"sink_a": 9}

    def test_frozen_dataclass_immutability(self) -> None:
        """Verify frozen=True prevents mutation."""
        result = AggregationFlushResult(rows_succeeded=1)

        with pytest.raises(FrozenInstanceError):
            result.rows_succeeded = 999  # type: ignore[misc]

    def test_default_values(self) -> None:
        """Verify defaults are 0 for counts and empty dict for destinations."""
        result = AggregationFlushResult()

        assert result.rows_succeeded == 0
        assert result.rows_failed == 0
        assert result.rows_routed_success == 0
        assert result.rows_routed_failure == 0
        assert result.rows_quarantined == 0
        assert result.rows_coalesced == 0
        assert result.rows_forked == 0
        assert result.rows_expanded == 0
        assert result.rows_buffered == 0
        assert result.routed_destinations == {}

    def test_addition_operator_sums_all_fields(self) -> None:
        """Verify __add__ correctly sums all fields."""
        result_a = AggregationFlushResult(
            rows_succeeded=1,
            rows_failed=2,
            rows_routed_success=3,
            rows_routed_failure=0,
            rows_quarantined=4,
            rows_coalesced=5,
            rows_forked=6,
            rows_expanded=7,
            rows_buffered=8,
            routed_destinations={"sink_a": 10, "sink_b": 20},
        )
        result_b = AggregationFlushResult(
            rows_succeeded=10,
            rows_failed=20,
            rows_routed_success=30,
            rows_routed_failure=0,
            rows_quarantined=40,
            rows_coalesced=50,
            rows_forked=60,
            rows_expanded=70,
            rows_buffered=80,
            routed_destinations={"sink_b": 30, "sink_c": 40},
        )

        combined = result_a + result_b

        assert combined.rows_succeeded == 11
        assert combined.rows_failed == 22
        assert combined.rows_routed_success == 33
        assert combined.rows_routed_failure == 0
        assert combined.rows_quarantined == 44
        assert combined.rows_coalesced == 55
        assert combined.rows_forked == 66
        assert combined.rows_expanded == 77
        assert combined.rows_buffered == 88
        assert combined.routed_destinations == {"sink_a": 10, "sink_b": 50, "sink_c": 40}

    def test_addition_operator_commutative(self) -> None:
        """Verify a + b == b + a (commutativity)."""
        result_a = AggregationFlushResult(
            rows_succeeded=1,
            rows_failed=2,
            rows_routed_success=3,
            rows_routed_failure=0,
            rows_quarantined=4,
            rows_coalesced=5,
            rows_forked=6,
            rows_expanded=7,
            rows_buffered=8,
            routed_destinations={"sink_a": 10},
        )
        result_b = AggregationFlushResult(
            rows_succeeded=10,
            rows_failed=20,
            rows_routed_success=30,
            rows_routed_failure=0,
            rows_quarantined=40,
            rows_coalesced=50,
            rows_forked=60,
            rows_expanded=70,
            rows_buffered=80,
            routed_destinations={"sink_b": 20},
        )

        assert result_a + result_b == result_b + result_a

    def test_addition_with_zero_result(self) -> None:
        """Verify adding zero-result is identity operation."""
        result = AggregationFlushResult(
            rows_succeeded=1,
            rows_failed=2,
            rows_routed_success=3,
            rows_routed_failure=0,
            rows_quarantined=4,
            rows_coalesced=5,
            rows_forked=6,
            rows_expanded=7,
            rows_buffered=8,
            routed_destinations={"sink_a": 9},
        )
        zero = AggregationFlushResult()

        # Adding zero should return equivalent result
        assert result + zero == result
        assert zero + result == result

    def test_to_dict_returns_plain_dict(self) -> None:
        result = AggregationFlushResult(rows_succeeded=5, rows_routed_success=2, rows_routed_failure=0, routed_destinations={"sink_a": 2})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["routed_destinations"], dict)
        assert d["routed_destinations"] == {"sink_a": 2}

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        result = AggregationFlushResult(routed_destinations={"sink_a": 3})
        json.dumps(result.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# New type tests
# ---------------------------------------------------------------------------


class TestGraphArtifacts:
    """Test GraphArtifacts frozen dataclass with MappingProxyType wrapping."""

    def test_fields_frozen_to_mapping_proxy(self) -> None:
        from elspeth.engine.orchestrator.types import GraphArtifacts

        artifacts = GraphArtifacts(
            edge_map={(NodeID("node1"), "continue"): "edge1"},
            source_id=NodeID("source"),
            source_id_map={"source": NodeID("source")},
            sink_id_map={SinkName("output"): NodeID("sink1")},
            transform_id_map={0: NodeID("t0")},
            config_gate_id_map={GateName("gate1"): NodeID("g1")},
            coalesce_id_map={CoalesceName("merge1"): NodeID("c1")},
        )
        assert isinstance(artifacts.edge_map, MappingProxyType)
        assert isinstance(artifacts.source_id_map, MappingProxyType)
        assert isinstance(artifacts.sink_id_map, MappingProxyType)
        assert isinstance(artifacts.transform_id_map, MappingProxyType)
        assert isinstance(artifacts.config_gate_id_map, MappingProxyType)
        assert isinstance(artifacts.coalesce_id_map, MappingProxyType)

    def test_is_frozen(self) -> None:
        from elspeth.engine.orchestrator.types import GraphArtifacts

        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source"),
            source_id_map={},
            sink_id_map={},
            transform_id_map={},
            config_gate_id_map={},
            coalesce_id_map={},
        )
        with pytest.raises(AttributeError):
            artifacts.source_id = NodeID("other")  # type: ignore[misc]


class TestAggNodeEntry:
    """Test AggNodeEntry named pair."""

    def test_attribute_access(self) -> None:
        from elspeth.engine.orchestrator.types import AggNodeEntry

        mock_transform = Mock()
        entry = AggNodeEntry(transform=mock_transform, node_id=NodeID("agg1"))
        assert entry.transform is mock_transform
        assert entry.node_id == NodeID("agg1")

    def test_is_frozen(self) -> None:
        from elspeth.engine.orchestrator.types import AggNodeEntry

        entry = AggNodeEntry(transform=Mock(), node_id=NodeID("agg1"))
        with pytest.raises(AttributeError):
            entry.node_id = NodeID("other")  # type: ignore[misc]


class TestRunContext:
    """Test RunContext frozen dataclass."""

    def test_mapping_fields_frozen(self) -> None:
        from elspeth.engine.orchestrator.types import AggNodeEntry, RunContext

        run_ctx = RunContext(
            ctx=Mock(),
            processor=Mock(),
            coalesce_executor=None,
            coalesce_node_map={CoalesceName("m1"): NodeID("c1")},
            agg_transform_lookup={"node1": AggNodeEntry(transform=Mock(), node_id=NodeID("agg1"))},
        )
        assert isinstance(run_ctx.coalesce_node_map, MappingProxyType)
        assert isinstance(run_ctx.agg_transform_lookup, MappingProxyType)


class TestLoopContext:
    """Test LoopContext mutable dataclass."""

    def test_mutable_fields_can_be_updated(self) -> None:
        from elspeth.engine.orchestrator.types import LoopContext

        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"output": []},
            processor=Mock(),
            ctx=Mock(),
            config=Mock(),
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )
        # Mutable: counters can be incremented
        loop_ctx.counters.rows_processed += 1
        assert loop_ctx.counters.rows_processed == 1

        # Mutable: pending_tokens can be appended
        loop_ctx.pending_tokens["output"].append((Mock(), None))
        assert len(loop_ctx.pending_tokens["output"]) == 1


class TestExecutionCountersToRunResultRequired:
    """Test that to_run_result requires the status parameter (no implicit default)."""

    def test_status_is_required_parameter(self) -> None:
        """Status has no default — callers must be explicit.

        Phase 2.2 (elspeth-0de989c56d): the biconditional invariant on
        :class:`RunResult` rejects ``status=COMPLETED`` with zero counters
        (``EMPTY`` is the right shape for an ingested-zero-rows run).
        Use ``RUNNING`` / ``INTERRUPTED`` — the only statuses that bypass
        the row-count predicate — for the explicit-status smoke test.
        """
        counters = ExecutionCounters()
        # Must pass status explicitly
        result = counters.to_run_result("run-1", status=RunStatus.RUNNING)
        assert result.status == RunStatus.RUNNING

        result2 = counters.to_run_result("run-1", status=RunStatus.INTERRUPTED)
        assert result2.status == RunStatus.INTERRUPTED


class TestPipelineConfig:
    """PipelineConfig freezes mutable containers in __post_init__."""

    def _make_config(self) -> PipelineConfig:
        """Minimal valid PipelineConfig for testing."""
        source = Mock()
        source.node_id = None
        transform = Mock()
        transform.node_id = None
        transform.on_error = "discard"
        sink = Mock()
        sink.node_id = None
        return PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": sink},
            config={"key": "value"},
            gates=[Mock()],
            aggregation_settings={"agg-1": Mock()},
            coalesce_settings=[Mock()],
        )

    def test_list_fields_frozen_to_tuple(self) -> None:
        config = self._make_config()
        assert isinstance(config.transforms, tuple)
        assert isinstance(config.gates, tuple)
        assert isinstance(config.coalesce_settings, tuple)

    def test_dict_fields_frozen_to_mapping_proxy(self) -> None:
        config = self._make_config()
        assert isinstance(config.sinks, MappingProxyType)
        assert isinstance(config.config, MappingProxyType)
        assert isinstance(config.aggregation_settings, MappingProxyType)

    def test_tuple_fields_reject_append(self) -> None:
        config = self._make_config()
        with pytest.raises(AttributeError):
            config.transforms.append(Mock())  # type: ignore[attr-defined]

    def test_mapping_proxy_fields_reject_assignment(self) -> None:
        config = self._make_config()
        with pytest.raises(TypeError):
            config.sinks["new"] = Mock()  # type: ignore[index]
        with pytest.raises(TypeError):
            config.config["new"] = "value"  # type: ignore[index]

    def test_idempotent_with_already_frozen_inputs(self) -> None:
        source = Mock()
        source.node_id = None
        sink = Mock()
        sink.node_id = None
        frozen_transforms = (Mock(),)
        frozen_sinks = MappingProxyType({"output": sink})
        frozen_config = MappingProxyType({"key": "value"})
        frozen_gates = (Mock(),)
        frozen_agg = MappingProxyType({"agg-1": Mock()})
        frozen_coal = (Mock(),)

        config = PipelineConfig(
            sources={"primary": source},
            transforms=frozen_transforms,
            sinks=frozen_sinks,
            config=frozen_config,
            gates=frozen_gates,
            aggregation_settings=frozen_agg,
            coalesce_settings=frozen_coal,
        )
        assert isinstance(config.transforms, tuple)
        assert isinstance(config.sinks, MappingProxyType)
        assert config.transforms == frozen_transforms
        assert config.sinks == frozen_sinks


class TestValueSourceFinding:
    """Pin the structured-finding contract.

    The walker (``preflight.validate_value_source_compliance``) emits
    these; the composer ``/validate`` consumer reads
    ``component_id``/``field_name``/``reason`` directly. Replacing the
    pre-existing string round-trip eliminated a silent
    ``ValidationError(component_id=None)`` failure mode whenever the
    finding format string drifted.
    """

    def test_construct_with_all_fields(self) -> None:
        finding = ValueSourceFinding(
            component_id="openrouter_node_1",
            field_name="model",
            reason="value 'foo/bar' is not in catalog 'openrouter'",
        )
        assert finding.component_id == "openrouter_node_1"
        assert finding.field_name == "model"
        assert finding.reason.startswith("value")

    def test_format_round_trips_through_human_readable_string(self) -> None:
        finding = ValueSourceFinding(
            component_id="azure_node_1",
            field_name="model",
            reason="value 'a' must equal sibling 'deployment_name' (currently 'b')",
        )
        rendered = finding.format()
        # The format is the operator-facing surface — pin its shape so
        # log lines and ValidationCheck.detail strings stay stable.
        assert rendered == ("component 'azure_node_1' field 'model': value 'a' must equal sibling 'deployment_name' (currently 'b')")

    def test_frozen(self) -> None:
        finding = ValueSourceFinding(
            component_id="x",
            field_name="y",
            reason="z",
        )
        with pytest.raises(FrozenInstanceError):
            finding.component_id = "other"  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("component_id", "field_name", "reason", "match"),
        [
            ("", "model", "reason", "component_id must be non-empty"),
            ("c", "", "reason", "field_name must be non-empty"),
            ("c", "model", "", "reason must be non-empty"),
        ],
    )
    def test_empty_field_rejected(self, component_id: str, field_name: str, reason: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ValueSourceFinding(
                component_id=component_id,
                field_name=field_name,
                reason=reason,
            )


class TestValueSourceValidationError:
    """Pin the structured exception contract.

    Replaces the previous ``findings: tuple[str, ...]`` shape — consumers
    now read ``finding.component_id`` directly without parsing.
    """

    def test_carries_findings_tuple(self) -> None:
        f1 = ValueSourceFinding(component_id="c1", field_name="m", reason="r1")
        f2 = ValueSourceFinding(component_id="c2", field_name="m", reason="r2")
        err = ValueSourceValidationError("two violations", findings=(f1, f2))
        assert err.findings == (f1, f2)
        # Message remains accessible via str() for ValidationCheck.detail.
        assert str(err) == "two violations"

    def test_findings_default_to_empty_tuple(self) -> None:
        err = ValueSourceValidationError("no findings")
        assert err.findings == ()

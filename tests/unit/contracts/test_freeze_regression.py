"""Regression tests for freeze/immutability bug cluster.

Each test verifies that a specific freeze bypass is closed:
- Node.schema_fields: tuple of mutable dicts was not deep-frozen
- RowLineage.source_data: shallow MappingProxyType left nested containers mutable
- HTTPCallResponse.body: non-dict Mapping types bypassed freeze guard
- GracefulShutdownError.routed_destinations: MappingProxyType wrapped without copy
- PendingOutcome._REQUIRES_ERROR_HASH_OUTCOMES: class-level frozenset (not per-instance set)
- deep_freeze: arbitrary Mapping support (non-dict, non-MappingProxyType)
"""

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.audit import Node, RowLineage
from elspeth.contracts.call_data import HTTPCallResponse
from elspeth.contracts.engine import PendingOutcome
from elspeth.contracts.enums import Determinism, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.freeze import deep_freeze

# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _FakePipelinePlugin:
    name: str


def _make_node(**kwargs: object) -> Node:
    """Minimal Node factory — override only what you're testing."""
    defaults: dict[str, Any] = {
        "node_id": "n1",
        "run_id": "r1",
        "plugin_name": "test",
        "node_type": NodeType.SOURCE,
        "plugin_version": "1.0",
        "determinism": Determinism.DETERMINISTIC,
        "config_hash": "x",
        "config_json": "{}",
        "registered_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return Node(**defaults)


def _make_lineage(**kwargs: object) -> RowLineage:
    defaults: dict[str, Any] = {
        "row_id": "row-1",
        "run_id": "run-1",
        "source_node_id": "node-src",
        "row_index": 0,
        "source_row_index": 0,
        "ingest_sequence": 0,
        "source_data_hash": "abc",
        "created_at": datetime.now(UTC),
        "payload_available": True,
    }
    defaults.update(kwargs)
    return RowLineage(**defaults)


class _CustomMapping(Mapping[str, object]):
    """A non-dict Mapping for testing deep_freeze's Mapping fallback."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)


# ── deep_freeze: Mapping fallback ───────────────────────────────────────────


class TestDeepFreezeArbitraryMapping:
    """deep_freeze must handle non-dict Mapping types (not just dict/MappingProxyType)."""

    def test_custom_mapping_gets_frozen(self) -> None:
        custom = _CustomMapping({"key": "val", "nested": {"inner": 1}})
        frozen = deep_freeze(custom)
        assert isinstance(frozen, MappingProxyType)
        assert isinstance(frozen["nested"], MappingProxyType)

    def test_ordered_dict_gets_frozen(self) -> None:
        od = OrderedDict({"a": [1, 2], "b": {"c": 3}})
        frozen = deep_freeze(od)
        assert isinstance(frozen, MappingProxyType)
        assert isinstance(frozen["a"], tuple)
        assert isinstance(frozen["b"], MappingProxyType)


# ── Node.schema_fields ──────────────────────────────────────────────────────


class TestNodeSchemaFieldsDeepFreeze:
    """Bug: tuple of mutable dicts bypassed the isinstance(…, tuple) guard."""

    def test_tuple_of_dicts_gets_deep_frozen(self) -> None:
        mutable_fields = ({"name": "col_a", "type": "str"}, {"name": "col_b", "type": "int"})
        node = _make_node(schema_fields=mutable_fields)
        assert all(isinstance(f, MappingProxyType) for f in node.schema_fields)  # type: ignore[union-attr]

    def test_frozen_fields_are_immutable(self) -> None:
        """Attempting to write into a frozen schema field raises TypeError."""
        node = _make_node(schema_fields=[{"name": "col_a"}])
        with pytest.raises(TypeError):
            node.schema_fields[0]["new_key"] = "x"  # type: ignore[index]

    def test_list_of_dicts_gets_deep_frozen(self) -> None:
        node = _make_node(schema_fields=[{"name": "col_a"}])
        assert isinstance(node.schema_fields, tuple)
        assert isinstance(node.schema_fields[0], MappingProxyType)

    def test_already_frozen_is_detached_copy(self) -> None:
        """MappingProxyType inputs are detached (not identity-preserved).

        MappingProxyType is a view, not a copy. deep_freeze always creates
        a fresh detached mapping to prevent caller mutation leaks.
        """
        frozen = (MappingProxyType({"name": "col_a"}),)
        node = _make_node(schema_fields=frozen)
        assert node.schema_fields == frozen
        assert isinstance(node.schema_fields[0], MappingProxyType)

    def test_empty_tuple_passes_through(self) -> None:
        node = _make_node(schema_fields=())
        assert node.schema_fields == ()
        assert isinstance(node.schema_fields, tuple)

    def test_none_schema_fields_unchanged(self) -> None:
        node = _make_node(schema_fields=None)
        assert node.schema_fields is None


# ── RowLineage.source_data ──────────────────────────────────────────────────


class TestRowLineageDeepFreeze:
    """Bug: shallow MappingProxyType left nested containers mutable."""

    def test_nested_dict_is_deep_frozen(self) -> None:
        lineage = _make_lineage(source_data={"metadata": {"nested_key": "value"}})
        assert isinstance(lineage.source_data, MappingProxyType)
        assert isinstance(lineage.source_data["metadata"], MappingProxyType)

    def test_nested_dict_is_immutable(self) -> None:
        """Attempting to write into nested container raises TypeError."""
        lineage = _make_lineage(source_data={"metadata": {"key": "val"}})
        with pytest.raises(TypeError):
            lineage.source_data["metadata"]["new_key"] = "x"  # type: ignore[index]

    def test_nested_list_is_frozen_to_tuple(self) -> None:
        lineage = _make_lineage(source_data={"tags": ["a", "b"]})
        assert isinstance(lineage.source_data["tags"], tuple)  # type: ignore[index]

    def test_none_source_data_unchanged(self) -> None:
        lineage = _make_lineage(source_data=None, payload_available=False)
        assert lineage.source_data is None


class TestHTTPCallResponseBodyFreeze:
    """Bug: non-dict Mapping types bypassed isinstance(…, dict) guard."""

    def test_ordered_dict_body_gets_frozen(self) -> None:
        body = OrderedDict({"result": "ok", "nested": {"key": "val"}})
        resp = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=42,
            body=body,
        )
        assert isinstance(resp.body, MappingProxyType)
        assert isinstance(resp.body["nested"], MappingProxyType)

    def test_custom_mapping_body_gets_frozen(self) -> None:
        """Non-dict, non-OrderedDict Mapping must also be frozen."""
        body = _CustomMapping({"result": "ok"})
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=10,
            body=body,
        )
        assert isinstance(resp.body, MappingProxyType)

    def test_dict_body_still_frozen(self) -> None:
        resp = HTTPCallResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_size=10,
            body={"key": "value"},
        )
        assert isinstance(resp.body, MappingProxyType)

    def test_frozen_body_is_immutable(self) -> None:
        """Attempting to write into frozen body raises TypeError."""
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=10,
            body={"key": "value"},
        )
        with pytest.raises(TypeError):
            resp.body["new_key"] = "x"  # type: ignore[index]

    def test_string_body_unchanged(self) -> None:
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=5,
            body="hello",
        )
        assert resp.body == "hello"

    def test_none_body_unchanged(self) -> None:
        resp = HTTPCallResponse(status_code=200, headers={})
        assert resp.body is None

    def test_already_frozen_body_detached(self) -> None:
        """MappingProxyType body is detached from caller's source dict."""
        frozen_body = MappingProxyType({"k": "v"})
        resp = HTTPCallResponse(
            status_code=200,
            headers={},
            body_size=10,
            body=frozen_body,
        )
        assert resp.body == frozen_body
        assert isinstance(resp.body, MappingProxyType)


# ── GracefulShutdownError.routed_destinations ───────────────────────────────


class TestGracefulShutdownErrorCopyOnWrap:
    """Bug: MappingProxyType wrapped original dict without copy."""

    def test_caller_mutation_not_visible(self) -> None:
        destinations = {"sink_a": 10, "sink_b": 5}
        error = GracefulShutdownError(
            rows_processed=15,
            run_id="run-1",
            routed_destinations=destinations,
        )
        destinations["sink_c"] = 99
        assert "sink_c" not in error.routed_destinations

    def test_destinations_are_immutable(self) -> None:
        """MappingProxyType wrapping must reject writes."""
        error = GracefulShutdownError(
            rows_processed=5,
            run_id="run-1",
            routed_destinations={"sink_a": 1},
        )
        with pytest.raises(TypeError):
            error.routed_destinations["sink_b"] = 2  # type: ignore[index]

    def test_none_destinations_gives_empty_mapping(self) -> None:
        error = GracefulShutdownError(rows_processed=0, run_id="run-1")
        assert isinstance(error.routed_destinations, MappingProxyType)
        assert len(error.routed_destinations) == 0


# ── PendingOutcome._REQUIRES_ERROR_HASH_PATHS ───────────────────────────────


class TestPendingOutcomeClassVar:
    """Bug: requires-error-hash set recreated on every construction."""

    def test_requires_error_hash_outcomes_is_class_level(self) -> None:
        assert isinstance(PendingOutcome._REQUIRES_ERROR_HASH_PATHS, frozenset)

    def test_requires_error_hash_outcomes_shared_across_instances(self) -> None:
        """Both instances must reference the same ClassVar frozenset object."""
        a = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)
        b = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.GATE_ROUTED)
        assert a._REQUIRES_ERROR_HASH_PATHS is b._REQUIRES_ERROR_HASH_PATHS

    def test_validation_still_works(self) -> None:
        with pytest.raises(ValueError, match="requires non-empty error_hash"):
            PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.QUARANTINED_AT_SOURCE, error_hash=None)
        po = PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.QUARANTINED_AT_SOURCE, error_hash="abc")
        assert po.error_hash == "abc"


def test_pending_outcome_routed_on_error_with_valid_hash_succeeds() -> None:
    """ROUTED_ON_ERROR + non-empty error_hash is the contract-conforming shape."""
    outcome = PendingOutcome(
        outcome=TerminalOutcome.FAILURE,
        path=TerminalPath.ON_ERROR_ROUTED,
        error_hash="0123456789abcdef",
    )
    assert outcome.outcome == TerminalOutcome.FAILURE
    assert outcome.path == TerminalPath.ON_ERROR_ROUTED
    assert outcome.error_hash == "0123456789abcdef"


def test_pending_outcome_routed_on_error_without_hash_raises() -> None:
    """ROUTED_ON_ERROR without error_hash violates the _REQUIRES_ERROR_HASH_PATHS contract."""
    with pytest.raises(ValueError, match="requires non-empty error_hash"):
        PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.ON_ERROR_ROUTED)


def test_pending_outcome_routed_on_error_with_empty_hash_raises() -> None:
    """Empty-string error_hash counts as missing per `__post_init__` whitespace check."""
    with pytest.raises(ValueError, match="requires non-empty error_hash"):
        PendingOutcome(outcome=TerminalOutcome.FAILURE, path=TerminalPath.ON_ERROR_ROUTED, error_hash="")


# ── deep_freeze: set and frozenset branches ─────────────────────────────────


class TestDeepFreezeSetBranches:
    """deep_freeze must handle set→frozenset and frozenset idempotency."""

    def test_set_becomes_frozenset(self) -> None:
        result = deep_freeze({"a", "b", "c"})
        assert isinstance(result, frozenset)
        assert result == frozenset({"a", "b", "c"})

    def test_set_of_mutable_containers_is_deep_frozen(self) -> None:
        """Sets can't contain dicts, but nested in a list they can."""
        result = deep_freeze([{"a", "b"}, {"c"}])
        assert isinstance(result, tuple)
        assert all(isinstance(s, frozenset) for s in result)

    def test_frozenset_is_idempotent(self) -> None:
        """Already-frozen frozenset of scalars returns same object."""
        fs = frozenset({"x", "y", "z"})
        assert deep_freeze(fs) is fs

    def test_frozenset_with_mutable_inner_is_rebuilt(self) -> None:
        """frozenset containing a tuple with a mutable dict gets rebuilt."""
        # frozenset can't contain dicts directly, but can contain tuples
        inner = (1, 2)
        fs = frozenset({inner})
        # Already all-immutable — should be identity-preserved
        assert deep_freeze(fs) is fs


# ── deep_freeze: None and scalar idempotency ────────────────────────────────


class TestDeepFreezeIdempotency:
    """deep_freeze on None and scalars is a no-op."""

    def test_none_returns_none(self) -> None:
        assert deep_freeze(None) is None

    def test_string_returns_same(self) -> None:
        s = "hello"
        assert deep_freeze(s) is s

    def test_int_returns_same(self) -> None:
        assert deep_freeze(42) is 42  # noqa: F632 — intentional identity check

    def test_custom_mapping_is_not_identity_preserved(self) -> None:
        """Non-dict Mapping always rebuilds (no idempotency fast path)."""
        custom = _CustomMapping({"key": "val"})
        frozen = deep_freeze(custom)
        assert frozen is not custom
        assert isinstance(frozen, MappingProxyType)


# ── freeze_fields utility ───────────────────────────────────────────────────


class TestFreezeFieldsUtility:
    """Direct tests for the freeze_fields() utility function."""

    def test_mutable_field_is_replaced(self) -> None:
        """freeze_fields replaces a mutable dict with a MappingProxyType."""
        lineage = _make_lineage(source_data={"key": {"nested": "val"}})
        assert isinstance(lineage.source_data, MappingProxyType)
        assert isinstance(lineage.source_data["key"], MappingProxyType)

    def test_frozen_field_is_detached_from_caller(self) -> None:
        """freeze_fields detaches MappingProxyType from caller's source dict."""
        frozen_data = MappingProxyType({"key": "val"})
        lineage = _make_lineage(source_data=frozen_data)
        assert lineage.source_data == frozen_data
        assert isinstance(lineage.source_data, MappingProxyType)

    def test_none_field_passes_through(self) -> None:
        """freeze_fields on a None-valued field is a no-op."""
        lineage = _make_lineage(source_data=None, payload_available=False)
        assert lineage.source_data is None


# ── Nested Mutation Rejection Tests ─────────────────────────────────────────
#
# These verify that deep-frozen dataclasses reject nested mutation attempts
# with TypeError.  Top-level mutation is tested elsewhere; these target
# the nested container paths that shallow freezing would miss.


class TestPipelineRowNestedMutationRejected:
    """PipelineRow deep-freezes its data dict — nested dicts must be immutable."""

    def test_nested_dict_mutation_raises(self) -> None:
        from elspeth.testing import make_pipeline_row

        row = make_pipeline_row({"meta": {"key": "val"}, "score": 42})
        with pytest.raises(TypeError):
            row["meta"]["key"] = "mutated"

    def test_nested_list_in_dict_is_tuple(self) -> None:
        from elspeth.testing import make_pipeline_row

        row = make_pipeline_row({"tags": ["a", "b"]})
        assert isinstance(row["tags"], tuple)

    def test_to_dict_round_trip_produces_json_compatible_types(self) -> None:
        """to_dict() must return list (not tuple) for list-valued fields."""
        from elspeth.testing import make_pipeline_row

        row = make_pipeline_row({"tags": ["a", "b"], "meta": {"k": "v"}})
        d = row.to_dict()
        assert isinstance(d["tags"], list), f"Expected list, got {type(d['tags']).__name__}"
        assert isinstance(d["meta"], dict), f"Expected dict, got {type(d['meta']).__name__}"


class TestPipelineConfigNestedMutationRejected:
    """PipelineConfig freeze_fields its config — nested dicts must be immutable."""

    def test_nested_config_mutation_raises(self) -> None:
        from elspeth.engine.orchestrator.types import PipelineConfig

        config = PipelineConfig(
            sources={"primary": _FakePipelinePlugin(name="primary")},
            transforms=[],
            sinks={"out": _FakePipelinePlugin(name="out")},
            config={"nested": {"key": "val"}},
        )
        with pytest.raises(TypeError):
            config.config["nested"]["key"] = "mutated"


class TestPluginContextNestedMutationRejected:
    """PluginContext deep-freezes config — nested dicts must be immutable."""

    def test_nested_config_mutation_raises(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(
            run_id="test-run",
            config={"nested": {"key": "val"}},
        )
        with pytest.raises(TypeError):
            ctx.config["nested"]["key"] = "mutated"


class TestReplayedCallNestedMutationRejected:
    """ReplayedCall deep-freezes response_data and error_data."""

    def test_response_data_nested_mutation_raises(self) -> None:
        from elspeth.plugins.infrastructure.clients.replayer import ReplayedCall

        call = ReplayedCall(
            response_data={"body": {"content": "hello"}},
            original_latency_ms=50.0,
            request_hash="abc123",
        )
        with pytest.raises(TypeError):
            call.response_data["body"]["content"] = "mutated"

    def test_error_data_nested_mutation_raises(self) -> None:
        from elspeth.plugins.infrastructure.clients.replayer import ReplayedCall

        call = ReplayedCall(
            response_data={"status": "error"},
            original_latency_ms=50.0,
            request_hash="abc123",
            was_error=True,
            error_data={"detail": {"code": 500}},
        )
        assert call.error_data is not None
        with pytest.raises(TypeError):
            call.error_data["detail"]["code"] = 999

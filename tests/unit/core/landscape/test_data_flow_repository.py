"""Direct unit tests for DataFlowRepository.

Tests exercise the repository directly (not through RecorderFactory delegation)
to verify audit integrity checks, edge cases, and crash paths that the delegation
tests don't cover.

The _make_repo() helper returns (LandscapeDB, DataFlowRepository, RecorderFactory)
— the factory is used for graph setup only (begin_run, register_node),
while the repo is tested directly.

Covers all 3 former mixin domains:
- Token recording: create_row, create_token, record_token_outcome, fork/coalesce/expand
- Graph recording: register_node, get_node (composite PK), get_edge_map
- Error recording: record_validation_error, record_transform_error
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from elspeth.contracts import (
    NodeType,
    RoutingMode,
)
from elspeth.contracts.audit import (
    _TERMINAL_PAIR_FIELD_CONSTRAINTS,
    DISCARD_SINK_NAME,
    TokenRef,
)
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import repr_hash
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.model_loaders import (
    EdgeLoader,
    NodeLoader,
    TokenOutcomeLoader,
    TransformErrorLoader,
    ValidationErrorLoader,
)
from elspeth.core.landscape.schema import (
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)
from tests.fixtures.landscape import make_factory, make_landscape_db

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_ERROR_HASH = "a" * 64


def _make_repo(
    *,
    run_id: str = "run-1",
    payload_store: Any = None,
) -> tuple[LandscapeDB, DataFlowRepository, RecorderFactory]:
    """Create a DataFlowRepository with supporting infrastructure.

    Returns (db, repo, factory) — factory is for graph setup only.
    """
    db = make_landscape_db()
    ops = DatabaseOps(db)
    repo = DataFlowRepository(
        db,
        ops,
        token_outcome_loader=TokenOutcomeLoader(),
        node_loader=NodeLoader(),
        edge_loader=EdgeLoader(),
        validation_error_loader=ValidationErrorLoader(),
        transform_error_loader=TransformErrorLoader(),
        payload_store=payload_store,
    )
    factory = make_factory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="transform",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="transform-1",
        schema_config=_DYNAMIC_SCHEMA,
    )
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv_sink",
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        node_id="sink-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    return db, repo, factory


def _make_repo_with_token(
    *,
    run_id: str = "run-1",
    payload_store: Any = None,
) -> tuple[LandscapeDB, DataFlowRepository, RecorderFactory, str, str]:
    """Create repo with a row and token ready for processing.

    Returns (db, repo, factory, row_id, token_id).
    """
    db, repo, factory = _make_repo(run_id=run_id, payload_store=payload_store)
    row = repo.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1")
    token = repo.create_token("row-1", token_id="tok-1")
    factory.execution.create_batch(run_id=run_id, aggregation_node_id="transform-1", batch_id="batch-1")
    return db, repo, factory, row.row_id, token.token_id


def _record_completed_sink_state_with_artifact(
    factory: RecorderFactory,
    *,
    run_id: str,
    token_id: str,
    sink_node_id: str = "sink-0",
) -> str:
    """Create the I1c node-state and artifact witnesses for direct repo tests."""
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_node_id,
        run_id=run_id,
        step_index=0,
        input_data={},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state.state_id,
        sink_node_id=sink_node_id,
        artifact_type="test",
        path=f"memory://unit/{token_id}",
        content_hash="deadbeef" * 8,
        size_bytes=0,
    )
    return artifact.artifact_id


def _valid_constraint_fields(pair: tuple[TerminalOutcome | None, TerminalPath]) -> dict[str, str]:
    """Return discriminator kwargs satisfying one ADR-019 constraint row."""
    constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
    field_values = {
        "sink_name": "sink-0",
        "batch_id": "batch-1",
        "fork_group_id": "fork-1",
        "join_group_id": "join-1",
        "expand_group_id": "expand-1",
        "error_hash": _ERROR_HASH,
    }
    fields: dict[str, str] = {}
    for field_name in constraints.required:
        exact_value = constraints.exact.get(field_name)
        fields[field_name] = str(exact_value if exact_value is not None else field_values[field_name])
    for field_name, exact_value in constraints.exact.items():
        fields[field_name] = str(exact_value)
    return fields


def _invalid_constraint_fields(pair: tuple[TerminalOutcome | None, TerminalPath]) -> dict[str, str | None]:
    """Return discriminator kwargs violating one requirement for a constraint row."""
    constraints = _TERMINAL_PAIR_FIELD_CONSTRAINTS[pair]
    fields: dict[str, str | None] = dict(_valid_constraint_fields(pair))
    if constraints.required:
        fields[constraints.required[0]] = None
    elif constraints.exact:
        field_name = next(iter(constraints.exact))
        fields[field_name] = "wrong-exact-value"
    else:
        fields[constraints.forbidden[0]] = "forbidden-extra"
    return fields


# ===========================================================================
# H1: Token recording domain — direct repo tests
# ===========================================================================


class TestCreateRow:
    """Tests for DataFlowRepository.create_row — the row ingestion entry point."""

    def test_creates_row_with_canonical_hash(self) -> None:
        """create_row hashes data using stable_hash (canonical)."""
        _db, repo, _fac = _make_repo()
        data = {"name": "Alice", "value": 42}
        row = repo.create_row("run-1", "source-0", 0, data)
        assert row.source_data_hash == stable_hash(data)

    def test_row_id_is_auto_generated_when_not_supplied(self) -> None:
        _db, repo, _fac = _make_repo()
        row = repo.create_row("run-1", "source-0", 0, {"x": 1})
        assert row.row_id is not None
        assert len(row.row_id) > 0

    def test_row_id_is_used_when_supplied(self) -> None:
        _db, repo, _fac = _make_repo()
        row = repo.create_row("run-1", "source-0", 0, {"x": 1}, row_id="custom-id")
        assert row.row_id == "custom-id"

    def test_row_index_is_stored(self) -> None:
        _db, repo, _fac = _make_repo()
        row = repo.create_row("run-1", "source-0", 5, {"x": 1})
        assert row.row_index == 5


class TestCreateToken:
    """Tests for DataFlowRepository.create_token — initial token creation."""

    def test_creates_token_linked_to_row(self) -> None:
        _db, repo, _fac = _make_repo()
        row = repo.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1")
        token = repo.create_token("row-1")
        assert token.row_id == row.row_id
        assert token.token_id is not None

    def test_token_id_is_used_when_supplied(self) -> None:
        _db, repo, _fac = _make_repo()
        repo.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1")
        token = repo.create_token("row-1", token_id="custom-tok")
        assert token.token_id == "custom-tok"


class TestRecordTokenOutcomeTwoAxis:
    """Tests for DataFlowRepository.record_token_outcome via direct repo."""

    def test_facords_completed_outcome(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        outcome_id = repo.record_token_outcome(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink-0",
        )
        assert outcome_id.startswith("out_")

    def test_roundtrip_via_get_token_outcome(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        repo.record_token_outcome(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink-0",
        )
        fetched = repo.get_token_outcome(tok)
        assert fetched is not None
        assert fetched.outcome == TerminalOutcome.SUCCESS
        assert fetched.path == TerminalPath.DEFAULT_FLOW
        assert fetched.completed is True
        assert fetched.sink_name == "sink-0"

    def test_record_buffered(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        repo.record_token_outcome(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id="batch-1",
        )

        fetched = repo.get_token_outcome(tok)

        assert fetched is not None
        assert fetched.outcome is None
        assert fetched.path == TerminalPath.BUFFERED
        assert fetched.completed is False

    def test_record_illegal_pair_crashes(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match=r"Unhandled \(outcome, path\) pair"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
                sink_name="sink-0",
            )

    def test_record_default_flow_requires_sink_name(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match="sink_name"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_record_filter_dropped_requires_no_extra_fields(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match="forbids sink_name"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.FILTER_DROPPED,
                sink_name="sink-0",
            )

    def test_record_expand_parent_requires_expand_group_id(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match="expand_group_id"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.EXPAND_PARENT,
            )

    def test_record_sink_discarded_requires_exact_discard_sink_name(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match=DISCARD_SINK_NAME):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name="not-discard",
                error_hash=_ERROR_HASH,
            )

    def test_record_default_flow_rejects_error_hash(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match="forbids error_hash"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink-0",
                error_hash=_ERROR_HASH,
            )

    def test_record_buffered_rejects_sink_name(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(ValueError, match="forbids sink_name"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=None,
                path=TerminalPath.BUFFERED,
                batch_id="batch-1",
                sink_name="sink-0",
            )

    @pytest.mark.parametrize("pair", tuple(_TERMINAL_PAIR_FIELD_CONSTRAINTS))
    def test_record_accepts_every_constraint_pair(
        self,
        pair: tuple[TerminalOutcome | None, TerminalPath],
    ) -> None:
        _db, repo, fac, _row, tok = _make_repo_with_token()
        outcome, path = pair
        fields = _valid_constraint_fields(pair)
        if pair == (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
            fields["sink_node_id"] = "sink-0"
            fields["artifact_id"] = _record_completed_sink_state_with_artifact(
                fac,
                run_id="run-1",
                token_id=tok,
            )

        outcome_id = repo.record_token_outcome(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            outcome=outcome,
            path=path,
            **fields,
        )

        assert outcome_id.startswith("out_")

    def test_record_failsink_fallback_requires_node_witness(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()

        with pytest.raises(AuditIntegrityError, match=r"I1c.*node_id"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="sink-0",
                error_hash=_ERROR_HASH,
            )

    def test_record_failsink_fallback_rejects_missing_artifact_witness(self) -> None:
        _db, repo, fac, _row, tok = _make_repo_with_token()
        state = fac.execution.begin_node_state(
            token_id=tok,
            node_id="sink-0",
            run_id="run-1",
            step_index=0,
            input_data={},
        )
        fac.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"written": True},
            duration_ms=1.0,
        )

        with pytest.raises(AuditIntegrityError, match=r"I1c.*artifact"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="sink-0",
                sink_node_id="sink-0",
                artifact_id="missing-artifact",
                error_hash=_ERROR_HASH,
            )

    def test_record_discard_rejects_completed_sink_state(self) -> None:
        _db, repo, fac, _row, tok = _make_repo_with_token()
        _record_completed_sink_state_with_artifact(fac, run_id="run-1", token_id=tok)

        with pytest.raises(AuditIntegrityError, match=r"I3.*discard"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash=_ERROR_HASH,
            )

    @pytest.mark.parametrize("pair", tuple(_TERMINAL_PAIR_FIELD_CONSTRAINTS))
    def test_record_rejects_each_constraint_row_violation(
        self,
        pair: tuple[TerminalOutcome | None, TerminalPath],
    ) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        fields = _invalid_constraint_fields(pair)

        with pytest.raises(ValueError, match="Contract violation"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                outcome=pair[0],
                path=pair[1],
                **fields,
            )

    def test_cross_run_contamination_raises(self) -> None:
        """record_token_outcome rejects token from a different run."""
        _db, repo, fac, _row, tok = _make_repo_with_token(run_id="run-1")
        # Create a second run
        fac.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            repo.record_token_outcome(
                ref=TokenRef(token_id=tok, run_id="run-2"),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink-0",
            )


# ===========================================================================
# H1: Graph recording domain — direct repo tests
# ===========================================================================


class TestRegisterNodeDirect:
    """Tests for DataFlowRepository.register_node via direct repo."""

    def test_registers_node_and_retrieves_by_composite_key(self) -> None:
        """register_node stores (node_id, run_id) composite key correctly."""
        _db, repo, _fac = _make_repo()
        # The _make_repo already registered nodes. Register one more for test.
        node = repo.register_node(
            run_id="run-1",
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="2.0",
            config={"key": "val"},
            node_id="transform-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        assert node.node_id == "transform-2"
        assert node.plugin_name == "passthrough"

        # Retrieve via composite key
        fetched = repo.get_node("transform-2", "run-1")
        assert fetched is not None
        assert fetched.node_id == "transform-2"
        assert fetched.plugin_name == "passthrough"

    def test_same_node_id_in_different_runs(self) -> None:
        """Composite PK allows same node_id in different runs."""
        _db, repo, fac = _make_repo(run_id="run-1")
        fac.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        repo.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",  # Same node_id as run-1
            schema_config=_DYNAMIC_SCHEMA,
        )
        node_r1 = repo.get_node("source-0", "run-1")
        node_r2 = repo.get_node("source-0", "run-2")
        assert node_r1 is not None
        assert node_r2 is not None
        # Both exist — composite PK working

    def test_fingerprints_secret_fields_before_persisting_node_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """register_node must not persist raw secrets into nodes.config_json."""
        from elspeth.contracts.security import secret_fingerprint

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fingerprint-key")

        _db, repo, _fac = _make_repo()
        node = repo.register_node(
            run_id="run-1",
            plugin_name="azure_blob",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={
                "account_name": "audit-safe",
                "api_key": "top-secret-key",
                "nested": {
                    "client_secret": "nested-secret",
                    "connection_string": "postgresql://example.test/db?credential=redacted",
                },
            },
            node_id="source-secret",
            schema_config=_DYNAMIC_SCHEMA,
        )

        parsed = json.loads(node.config_json)
        assert parsed["account_name"] == "audit-safe"
        assert "api_key" not in parsed
        assert parsed["api_key_fingerprint"] == secret_fingerprint("top-secret-key", key=b"test-fingerprint-key")
        assert "client_secret" not in parsed["nested"]
        assert parsed["nested"]["client_secret_fingerprint"] == secret_fingerprint(
            "nested-secret",
            key=b"test-fingerprint-key",
        )
        assert "connection_string" not in parsed["nested"]
        assert parsed["nested"]["connection_string_fingerprint"] == secret_fingerprint(
            "postgresql://example.test/db?credential=redacted",
            key=b"test-fingerprint-key",
        )

    def test_from_plugin_instances_configs_are_fingerprinted_before_node_persistence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Graph-built node configs still hit the audit scrubber before persistence."""
        from elspeth.contracts.security import secret_fingerprint
        from elspeth.core.config import SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from tests.fixtures.plugins import CollectSink, ListSource

        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fingerprint-key")

        _db, repo, _fac = _make_repo(run_id="run-graph")
        source = ListSource([], name="list_source", on_success="output")
        source.config = {
            "schema": {"mode": "observed"},
            "api_key": "source-secret",
        }
        sink = CollectSink("output")
        sink.config = {
            "schema": {"mode": "observed"},
            "headers": {
                "authorization_token": "sink-secret",
            },
        }
        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            source_settings=SourceSettings(
                plugin=source.name,
                on_success="output",
                options={},
            ),
            transforms=[],
            sinks={"output": sink},
            aggregations={},
            gates=[],
        )

        plugin_by_name = {
            source.name: source,
            sink.name: sink,
        }
        for node_info in graph.get_nodes():
            schema_config = node_info.output_schema_config
            assert schema_config is not None
            plugin = plugin_by_name[node_info.plugin_name]
            repo.register_node(
                run_id="run-graph",
                node_id=node_info.node_id,
                plugin_name=node_info.plugin_name,
                node_type=node_info.node_type,
                plugin_version=plugin.plugin_version,
                config=node_info.config,
                determinism=plugin.determinism,
                schema_config=schema_config,
            )

        source_node_id = next(node.node_id for node in graph.get_nodes() if node.node_type == NodeType.SOURCE)
        sink_node_id = next(node.node_id for node in graph.get_nodes() if node.node_type == NodeType.SINK)
        source_node = repo.get_node(source_node_id, "run-graph")
        sink_node = repo.get_node(sink_node_id, "run-graph")
        assert source_node is not None
        assert sink_node is not None

        parsed_source = json.loads(source_node.config_json)
        parsed_sink = json.loads(sink_node.config_json)

        assert "api_key" not in parsed_source
        assert parsed_source["api_key_fingerprint"] == secret_fingerprint("source-secret", key=b"test-fingerprint-key")
        assert "authorization_token" not in parsed_sink["headers"]
        assert parsed_sink["headers"]["authorization_token_fingerprint"] == secret_fingerprint(
            "sink-secret",
            key=b"test-fingerprint-key",
        )


class TestRegisterEdgeAndEdgeMapDirect:
    """Tests for DataFlowRepository edge registration and edge map via direct repo."""

    def test_register_edge_and_get_edge_map(self) -> None:
        _db, repo, _fac = _make_repo()
        edge = repo.register_edge(
            run_id="run-1",
            from_node_id="transform-1",
            to_node_id="sink-0",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        assert edge.edge_id is not None
        edge_map = repo.get_edge_map("run-1")
        assert edge_map[("transform-1", "continue")] == edge.edge_id

    def test_get_edge_map_run_isolation(self) -> None:
        """get_edge_map only returns edges from the specified run."""
        _db, repo, fac = _make_repo(run_id="run-1")
        fac.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        fac.data_flow.register_node(
            run_id="run-2",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        fac.data_flow.register_node(
            run_id="run-2",
            plugin_name="sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )

        repo.register_edge(run_id="run-1", from_node_id="transform-1", to_node_id="sink-0", label="continue", mode=RoutingMode.MOVE)
        repo.register_edge(run_id="run-2", from_node_id="source-0", to_node_id="sink-0", label="default", mode=RoutingMode.MOVE)

        map_r1 = repo.get_edge_map("run-1")
        assert ("transform-1", "continue") in map_r1
        assert ("source-0", "default") not in map_r1

    def test_get_edge_map_raises_on_empty(self) -> None:
        """get_edge_map raises AuditIntegrityError when run has no edges."""
        _db, repo, _fac = _make_repo()
        with pytest.raises(AuditIntegrityError, match="no edges registered"):
            repo.get_edge_map("run-1")


# ===========================================================================
# H1: Error recording domain — direct repo tests
# ===========================================================================


class TestRecordValidationErrorDirect:
    """Tests for DataFlowRepository.record_validation_error via direct repo."""

    def test_returns_verr_prefixed_id(self) -> None:
        _db, repo, _fac = _make_repo()
        error_id = repo.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"name": "alice"},
            error="Field missing",
            schema_mode="strict",
            destination="quarantine",
        )
        assert error_id.startswith("verr_")

    def test_roundtrip_via_get_validation_errors_for_run(self) -> None:
        _db, repo, _fac = _make_repo()
        repo.record_validation_error(
            run_id="run-1",
            node_id="source-0",
            row_data={"x": 1},
            error="bad field",
            schema_mode="observed",
            destination="quarantine",
        )
        errors = repo.get_validation_errors_for_run("run-1")
        assert len(errors) == 1
        assert errors[0].error == "bad field"


class TestRecordTransformErrorDirect:
    """Tests for DataFlowRepository.record_transform_error via direct repo."""

    def test_returns_terr_prefixed_id(self) -> None:
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        error_id = repo.record_transform_error(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "test_error", "field": "amount", "error": "ZeroDivisionError"},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

    def test_invalid_error_reason_crashes_at_tier1_boundary(self) -> None:
        """Invalid TransformErrorCategory crashes — Tier 1 write guard.

        TypedDict has zero runtime enforcement. If a plugin passes
        {"reason": "banana"}, the Literal type annotation does nothing.
        The Tier 1 write boundary must validate before persisting.
        """
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        with pytest.raises(AuditIntegrityError, match="Invalid TransformErrorCategory"):
            repo.record_transform_error(
                ref=TokenRef(token_id=tok, run_id="run-1"),
                transform_id="transform-1",
                row_data={"name": "test"},
                error_details={"reason": "banana_error", "error": "this is not a real category"},  # type: ignore[typeddict-item]  # intentionally invalid reason
                destination="quarantine",
            )

    def test_valid_error_reason_passes_tier1_validation(self) -> None:
        """Valid TransformErrorCategory is accepted at the Tier 1 boundary."""
        _db, repo, _fac, _row, tok = _make_repo_with_token()
        error_id = repo.record_transform_error(
            ref=TokenRef(token_id=tok, run_id="run-1"),
            transform_id="transform-1",
            row_data={"name": "test"},
            error_details={"reason": "api_error", "error": "timeout"},
            destination="quarantine",
        )
        assert error_id.startswith("terr_")

    def test_cross_run_contamination_raises(self) -> None:
        """record_transform_error rejects token from a different run."""
        _db, repo, fac, _row, tok = _make_repo_with_token(run_id="run-1")
        fac.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-2")
        fac.data_flow.register_node(
            run_id="run-2",
            plugin_name="transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-1",
            schema_config=_DYNAMIC_SCHEMA,
        )

        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            repo.record_transform_error(
                ref=TokenRef(token_id=tok, run_id="run-2"),
                transform_id="transform-1",
                row_data={"name": "test"},
                error_details={"reason": "test_error", "field": "f", "error": "E"},
                destination="quarantine",
            )


# ===========================================================================
# H1: _validate_outcome_fields exhaustive guard test
# ===========================================================================


class TestValidateOutcomeFieldsExhaustive:
    """Test the M1 exhaustive else clause on _validate_outcome_fields."""

    def test_rejects_unknown_outcome_string(self) -> None:
        """Unknown outcome variants raise ValueError (M1 exhaustive guard)."""
        _db, repo, _fac = _make_repo()
        with pytest.raises(ValueError, match=r"Unhandled \(outcome, path\) pair"):
            repo._validate_outcome_fields(
                cast(TerminalOutcome, "IMAGINARY_OUTCOME"),
                TerminalPath.DEFAULT_FLOW,
                sink_name=None,
                batch_id=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                error_hash=None,
            )


# ===========================================================================
# H2: Atomic transaction rollback tests
# ===========================================================================


def _count_tokens(db: LandscapeDB) -> int:
    """Count total tokens in the database."""
    with db.engine.connect() as conn:
        return conn.execute(select(tokens_table)).rowcount or len(conn.execute(select(tokens_table)).fetchall())


def _count_token_outcomes(db: LandscapeDB) -> int:
    """Count total token outcomes in the database."""
    with db.engine.connect() as conn:
        return len(conn.execute(select(token_outcomes_table)).fetchall())


def _count_token_parents(db: LandscapeDB) -> int:
    """Count total token_parents records in the database."""
    with db.engine.connect() as conn:
        return len(conn.execute(select(token_parents_table)).fetchall())


class TestForkTokenAtomicity:
    """fork_token must be all-or-nothing: children + parent outcome together."""

    def test_fork_rollback_on_failure_leaves_zero_partial_state(self) -> None:
        """If transaction fails mid-way, no children and no parent outcome persist."""
        db, repo, _fac, row_id, tok_id = _make_repo_with_token()
        tokens_before = _count_tokens(db)
        outcomes_before = _count_token_outcomes(db)
        parents_before = _count_token_parents(db)

        # Inject failure: patch _db.connection to raise after child inserts
        original_connection = repo._db.connection
        call_count = 0

        @contextmanager
        def failing_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                nonlocal call_count
                call_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal call_count
                    call_count += 1
                    # Let child token + parent relationship inserts through (2 per child)
                    # Fail when recording the parent FORKED outcome (5th call for 2 branches)
                    if call_count >= 5:
                        raise RuntimeError("Injected failure mid-transaction")
                    return original_execute(stmt, *args, **kwargs)

                conn.execute = patched_execute
                yield conn

        repo._db.connection = failing_connection  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Injected failure"):
            repo.fork_token(
                parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
                row_id=row_id,
                branches=["a", "b"],
            )

        # Verify: zero partial state — all counts unchanged
        assert _count_tokens(db) == tokens_before
        assert _count_token_outcomes(db) == outcomes_before
        assert _count_token_parents(db) == parents_before


class TestCoalesceTokensAtomicity:
    """coalesce_tokens must be all-or-nothing: merged token + parent links together."""

    def test_coalesce_rollback_on_failure_leaves_zero_partial_state(self) -> None:
        """If transaction fails mid-way, no merged token and no parent links persist."""
        db, repo, _fac, row_id, tok_id = _make_repo_with_token()

        # Fork first to get two child tokens to coalesce
        children, _fg = repo.fork_token(
            parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
            row_id=row_id,
            branches=["a", "b"],
        )
        child_ids = [c.token_id for c in children]

        tokens_before = _count_tokens(db)
        parents_before = _count_token_parents(db)

        # Inject failure: raise after merged token insert but before parent links
        original_connection = repo._db.connection
        call_count = 0

        @contextmanager
        def failing_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                nonlocal call_count
                call_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal call_count
                    call_count += 1
                    # Let merged token insert through (1st call), fail on parent link (2nd)
                    if call_count >= 2:
                        raise RuntimeError("Injected failure mid-transaction")
                    return original_execute(stmt, *args, **kwargs)

                conn.execute = patched_execute
                yield conn

        repo._db.connection = failing_connection  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Injected failure"):
            repo.coalesce_tokens(
                parent_refs=[TokenRef(token_id=cid, run_id="run-1") for cid in child_ids],
                row_id=row_id,
            )

        # Verify: zero partial state
        assert _count_tokens(db) == tokens_before
        assert _count_token_parents(db) == parents_before


class TestExpandTokenAtomicity:
    """expand_token must be all-or-nothing: children + parent outcome together."""

    def test_expand_rollback_on_failure_leaves_zero_partial_state(self) -> None:
        """If transaction fails mid-way, no child tokens and no parent outcome persist."""
        db, repo, _fac, row_id, tok_id = _make_repo_with_token()
        tokens_before = _count_tokens(db)
        outcomes_before = _count_token_outcomes(db)
        parents_before = _count_token_parents(db)

        # Inject failure: raise after child inserts but before parent outcome
        original_connection = repo._db.connection
        call_count = 0

        @contextmanager
        def failing_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                nonlocal call_count
                call_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal call_count
                    call_count += 1
                    # For 3 children: 6 calls (token insert + parent link each)
                    # 7th call is the parent EXPANDED outcome — fail here
                    if call_count >= 7:
                        raise RuntimeError("Injected failure mid-transaction")
                    return original_execute(stmt, *args, **kwargs)

                conn.execute = patched_execute
                yield conn

        repo._db.connection = failing_connection  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Injected failure"):
            repo.expand_token(
                parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
                row_id=row_id,
                count=3,
                step_in_pipeline=2,
            )

        # Verify: zero partial state
        assert _count_tokens(db) == tokens_before
        assert _count_token_outcomes(db) == outcomes_before
        assert _count_token_parents(db) == parents_before


class TestForkTokenRowcountValidation:
    """fork_token must validate rowcount on every insert — phantom tokens are audit corruption."""

    def test_fork_raises_on_zero_rowcount_token_insert(self) -> None:
        """If a token insert silently affects zero rows, AuditIntegrityError is raised."""
        _db, repo, _fac, row_id, tok_id = _make_repo_with_token()

        original_connection = repo._db.connection

        @contextmanager
        def zero_rowcount_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                insert_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal insert_count
                    result = original_execute(stmt, *args, **kwargs)
                    # Only intercept INSERT statements (not SELECT for validation)
                    if hasattr(stmt, "is_insert") and stmt.is_insert:
                        insert_count += 1
                        # First insert is child token — return zero rowcount
                        if insert_count == 1:
                            mock_result = MagicMock()
                            mock_result.rowcount = 0
                            return mock_result
                    return result

                conn.execute = patched_execute
                yield conn

        repo._db.connection = zero_rowcount_connection  # type: ignore[method-assign]

        with pytest.raises(AuditIntegrityError, match="zero rows"):
            repo.fork_token(
                parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
                row_id=row_id,
                branches=["a"],
            )


class TestCoalesceTokensRowcountValidation:
    """coalesce_tokens must validate rowcount on every insert."""

    def test_coalesce_raises_on_zero_rowcount_token_insert(self) -> None:
        """If merged token insert affects zero rows, AuditIntegrityError is raised."""
        _db, repo, _fac, row_id, tok_id = _make_repo_with_token()

        # Fork first to get children to coalesce
        children, _fg = repo.fork_token(
            parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
            row_id=row_id,
            branches=["a", "b"],
        )
        child_ids = [c.token_id for c in children]

        original_connection = repo._db.connection

        @contextmanager
        def zero_rowcount_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                insert_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal insert_count
                    result = original_execute(stmt, *args, **kwargs)
                    if hasattr(stmt, "is_insert") and stmt.is_insert:
                        insert_count += 1
                        if insert_count == 1:
                            mock_result = MagicMock()
                            mock_result.rowcount = 0
                            return mock_result
                    return result

                conn.execute = patched_execute
                yield conn

        repo._db.connection = zero_rowcount_connection  # type: ignore[method-assign]

        with pytest.raises(AuditIntegrityError, match="zero rows"):
            repo.coalesce_tokens(
                parent_refs=[TokenRef(token_id=cid, run_id="run-1") for cid in child_ids],
                row_id=row_id,
            )


class TestExpandTokenRowcountValidation:
    """expand_token must validate rowcount on every insert."""

    def test_expand_raises_on_zero_rowcount_token_insert(self) -> None:
        """If child token insert affects zero rows, AuditIntegrityError is raised."""
        _db, repo, _fac, row_id, tok_id = _make_repo_with_token()

        original_connection = repo._db.connection

        @contextmanager
        def zero_rowcount_connection():
            with original_connection() as conn:
                original_execute = conn.execute
                insert_count = 0

                def patched_execute(stmt, *args: Any, **kwargs: Any):
                    nonlocal insert_count
                    result = original_execute(stmt, *args, **kwargs)
                    if hasattr(stmt, "is_insert") and stmt.is_insert:
                        insert_count += 1
                        if insert_count == 1:
                            mock_result = MagicMock()
                            mock_result.rowcount = 0
                            return mock_result
                    return result

                conn.execute = patched_execute
                yield conn

        repo._db.connection = zero_rowcount_connection  # type: ignore[method-assign]

        with pytest.raises(AuditIntegrityError, match="zero rows"):
            repo.expand_token(
                parent_ref=TokenRef(token_id=tok_id, run_id="run-1"),
                row_id=row_id,
                count=2,
            )


# ===========================================================================
# H3: create_row quarantine fallback tests
# ===========================================================================


class TestCreateRowQuarantined:
    """Tests for create_row quarantine fallback paths (Tier 3 boundary)."""

    def test_quarantined_with_nan_uses_repr_hash(self) -> None:
        """create_row(quarantined=True) uses repr_hash when data contains NaN."""
        _db, repo, _fac = _make_repo()
        data = {"v": float("nan")}
        row = repo.create_row("run-1", "source-0", 0, data, quarantined=True)
        assert row.source_data_hash == repr_hash(data)

    def test_quarantined_with_infinity_uses_repr_hash(self) -> None:
        """create_row(quarantined=True) uses repr_hash when data contains Infinity."""
        _db, repo, _fac = _make_repo()
        data = {"v": float("inf")}
        row = repo.create_row("run-1", "source-0", 0, data, quarantined=True)
        assert row.source_data_hash == repr_hash(data)

    def test_quarantined_normal_data_still_uses_canonical_hash(self) -> None:
        """create_row(quarantined=True) with normal data uses stable_hash (not repr)."""
        _db, repo, _fac = _make_repo()
        data = {"v": 42}
        row = repo.create_row("run-1", "source-0", 0, data, quarantined=True)
        assert row.source_data_hash == stable_hash(data)

    def test_non_quarantined_with_nan_crashes(self) -> None:
        """create_row(quarantined=False) with NaN crashes — Tier 2 guarantee."""
        _db, repo, _fac = _make_repo()
        with pytest.raises(ValueError):
            repo.create_row("run-1", "source-0", 0, {"v": float("nan")})

    def test_quarantined_nan_payload_uses_repr_fallback(self) -> None:
        """create_row(quarantined=True) with payload_store falls back to repr payload for NaN data."""
        mock_store = MagicMock()
        mock_store.store.return_value = "payload-ref-123"
        _db, repo, _fac = _make_repo(payload_store=mock_store)

        data = {"v": float("nan")}
        row = repo.create_row("run-1", "source-0", 0, data, quarantined=True)

        # Verify payload_store.store() was called with repr fallback bytes
        mock_store.store.assert_called_once()
        stored_bytes = mock_store.store.call_args[0][0]
        parsed = json.loads(stored_bytes.decode("utf-8"))
        assert "_repr" in parsed
        assert row.source_data_ref == "payload-ref-123"

    def test_quarantined_normal_data_payload_uses_canonical(self) -> None:
        """create_row(quarantined=True) with payload_store uses canonical JSON for normal data."""
        mock_store = MagicMock()
        mock_store.store.return_value = "payload-ref-456"
        _db, repo, _fac = _make_repo(payload_store=mock_store)

        data = {"v": 42}
        repo.create_row("run-1", "source-0", 0, data, quarantined=True)

        mock_store.store.assert_called_once()
        stored_bytes = mock_store.store.call_args[0][0]
        parsed = json.loads(stored_bytes.decode("utf-8"))
        assert parsed == {"v": 42}
        assert "_repr" not in parsed


# ===========================================================================
# H1: TokenRef validation — _validate_token_run_ownership with bundled refs
# ===========================================================================


class TestValidateTokenRunOwnership:
    """Tests for _validate_token_run_ownership accepting TokenRef.

    These test the validation at the point where TokenRef is first verified
    against the audit database. They ensure the cross-run contamination
    check works correctly with the bundled type.
    """

    def test_valid_ref_passes(self) -> None:
        """A TokenRef where token belongs to the specified run should pass."""
        _db, repo, _fac, _row, tok = _make_repo_with_token(run_id="run-1")
        ref = TokenRef(token_id=tok, run_id="run-1")
        # Should not raise
        repo._validate_token_run_ownership(ref)

    def test_mismatched_run_raises_audit_integrity_error(self) -> None:
        """A TokenRef with wrong run_id should raise AuditIntegrityError."""
        _db, repo, _fac, _row, tok = _make_repo_with_token(run_id="run-1")
        ref = TokenRef(token_id=tok, run_id="wrong-run-id")
        with pytest.raises(AuditIntegrityError, match="Cross-run contamination"):
            repo._validate_token_run_ownership(ref)

    def test_nonexistent_token_raises(self) -> None:
        """A TokenRef with a token_id not in the DB should raise."""
        _db, repo, _fac = _make_repo()
        ref = TokenRef(token_id="nonexistent-token", run_id="run-1")
        with pytest.raises(AuditIntegrityError):
            repo._validate_token_run_ownership(ref)

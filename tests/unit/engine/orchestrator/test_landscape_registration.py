from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import create_autospec, patch

import pytest

from elspeth import __version__ as ENGINE_VERSION
from elspeth.contracts import Determinism, NodeType
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.payload_store import PayloadStore
from elspeth.contracts.plugin_policy_audit import WebPluginPolicyEvidence
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.events import EventBusProtocol
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
from elspeth.engine.orchestrator.landscape_registration import (
    NodeAuditMetadata,
    register_nodes_with_landscape,
    resolve_node_audit_metadata,
    resolve_source_contracts_by_node_id,
)
from elspeth.engine.orchestrator.run_lifecycle import RunLifecycleCoordinator
from elspeth.engine.spans import SpanFactory

_SCHEMA_CONFIG = SchemaConfig(mode="observed", fields=None)


def _policy_evidence() -> WebPluginPolicyEvidence:
    return WebPluginPolicyEvidence(
        schema_version=1,
        policy_hash="a" * 64,
        snapshot_hash="b" * 64,
        authorized_plugin_ids=("sink:json", "source:csv"),
        available_plugin_ids=("sink:json", "source:csv"),
        control_modes=(("llm", "recommend"),),
        selected_implementations=(("llm", None),),
        selected_profile_aliases=(),
        plugin_code_identities=(
            ("sink:json", "1.0.0", "sha256:1111111111111111"),
            ("source:csv", "1.0.0", "sha256:2222222222222222"),
        ),
        binding_generation_fingerprint="c" * 64,
        decision_codes=("policy_allowed",),
    )


def test_web_policy_audit_failure_prevents_run_started_telemetry() -> None:
    ceremony = create_autospec(RunCeremony, instance=True)
    lifecycle = RunLifecycleCoordinator(
        db=create_autospec(LandscapeDB, instance=True),
        events=create_autospec(EventBusProtocol, instance=True),
        ceremony=ceremony,
        checkpoints=create_autospec(CheckpointCoordinator, instance=True),
        span_factory=create_autospec(SpanFactory, instance=True),
        canonical_version="v1",
    )
    run_lifecycle = create_autospec(RunLifecycleRepository, instance=True)
    run_lifecycle.begin_run.side_effect = RuntimeError("policy audit insert failed")
    factory = SimpleNamespace(run_lifecycle=run_lifecycle)
    source = SimpleNamespace(name="csv", output_schema=SimpleNamespace(model_json_schema=lambda: {"type": "object"}))
    config = SimpleNamespace(sources={"input": source}, config={})

    with (
        patch("elspeth.engine.orchestrator.run_lifecycle.RecorderFactory", return_value=factory),
        pytest.raises(RuntimeError, match="policy audit insert failed"),
    ):
        lifecycle.initialize_database_phase(
            config,
            create_autospec(PayloadStore, instance=True),
            None,
            openrouter_catalog_sha256="d" * 64,
            openrouter_catalog_source="bundled",
            web_plugin_policy_evidence=_policy_evidence(),
        )

    ceremony.emit_telemetry.assert_not_called()
    assert run_lifecycle.begin_run.call_args.kwargs["web_plugin_policy_evidence"] == _policy_evidence()


class _Plugin:
    def __init__(
        self,
        *,
        name: str,
        plugin_version: str,
        determinism: Determinism,
        source_file_hash: str | None,
        node_id: str | None = None,
        schema_contract: object | None = None,
    ) -> None:
        self.name = name
        self.plugin_version = plugin_version
        self.determinism = determinism
        self.source_file_hash = source_file_hash
        self.node_id = node_id
        self.schema_contract = schema_contract

    def get_schema_contract(self) -> object | None:
        return self.schema_contract


class _Graph:
    def __init__(self, *node_infos: SimpleNamespace) -> None:
        self._node_infos = {str(info.node_id): info for info in node_infos}

    def topological_order(self) -> list[str]:
        return list(self._node_infos)

    def get_node_info(self, node_id: str) -> SimpleNamespace:
        return self._node_infos[str(node_id)]


class _DataFlow:
    def __init__(self) -> None:
        self.register_node_calls: list[dict[str, Any]] = []

    def register_node(self, **kwargs: Any) -> None:
        self.register_node_calls.append(kwargs)


def _node_info(
    node_id: str,
    node_type: NodeType,
    *,
    plugin_name: str | None = None,
    config: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        node_id=NodeID(node_id),
        node_type=node_type,
        plugin_name=plugin_name or node_type.value,
        config=config or {},
        output_schema_config=_SCHEMA_CONFIG,
    )


def test_resolve_node_audit_metadata_covers_plugin_and_structural_nodes() -> None:
    source = _Plugin(
        name="csv",
        plugin_version="source:1",
        determinism=Determinism.IO_READ,
        source_file_hash="sha256:1111111111111111",
    )
    transform = _Plugin(
        name="clean",
        plugin_version="transform:1",
        determinism=Determinism.DETERMINISTIC,
        source_file_hash="sha256:2222222222222222",
    )
    aggregation = _Plugin(
        name="sum",
        plugin_version="aggregation:1",
        determinism=Determinism.SEEDED,
        source_file_hash="sha256:3333333333333333",
        node_id="agg",
    )
    sink = _Plugin(
        name="json",
        plugin_version="sink:1",
        determinism=Determinism.IO_WRITE,
        source_file_hash="sha256:4444444444444444",
    )
    config = SimpleNamespace(
        sources={"input": source},
        transforms=(transform, aggregation),
        sinks={"out": sink},
    )
    graph = _Graph(
        _node_info("src", NodeType.SOURCE, plugin_name="csv", config={"source_name": "input"}),
        _node_info("gate", NodeType.GATE, plugin_name="config_gate"),
        _node_info("xform", NodeType.TRANSFORM, plugin_name="clean"),
        _node_info("agg", NodeType.AGGREGATION, plugin_name="sum"),
        _node_info("coalesce", NodeType.COALESCE, plugin_name="coalesce"),
        _node_info("queue", NodeType.QUEUE, plugin_name="queue"),
        _node_info("sink", NodeType.SINK, plugin_name="json"),
    )

    metadata = resolve_node_audit_metadata(
        config,
        graph,
        source_id_map={"input": NodeID("src")},
        transform_id_map={0: NodeID("xform")},
        sink_id_map={SinkName("out"): NodeID("sink")},
        config_gate_node_ids={NodeID("gate")},
        aggregation_node_ids={NodeID("agg")},
        coalesce_node_ids={NodeID("coalesce")},
    )

    assert metadata[NodeID("src")] == NodeAuditMetadata(
        plugin_version="source:1",
        determinism=Determinism.IO_READ,
        source_file_hash="sha256:1111111111111111",
    )
    assert metadata[NodeID("xform")].plugin_version == "transform:1"
    assert metadata[NodeID("agg")].plugin_version == "aggregation:1"
    assert metadata[NodeID("sink")].plugin_version == "sink:1"
    engine_metadata = NodeAuditMetadata(
        plugin_version=f"engine:{ENGINE_VERSION}",
        determinism=Determinism.DETERMINISTIC,
        source_file_hash=None,
    )
    assert metadata[NodeID("gate")] == engine_metadata
    assert metadata[NodeID("coalesce")] == engine_metadata
    assert metadata[NodeID("queue")] == engine_metadata


def test_resolve_node_audit_metadata_fails_closed_on_unmapped_plugin_node() -> None:
    config = SimpleNamespace(sources={}, transforms=(), sinks={})
    graph = _Graph(_node_info("xform", NodeType.TRANSFORM, plugin_name="clean"))

    with pytest.raises(OrchestrationInvariantError, match="requires plugin-backed audit metadata"):
        resolve_node_audit_metadata(
            config,
            graph,
            source_id_map={},
            transform_id_map={},
            sink_id_map={},
            config_gate_node_ids=set(),
            aggregation_node_ids=set(),
            coalesce_node_ids=set(),
        )


def test_resolve_source_contracts_by_node_id_uses_source_id_map() -> None:
    schema_contract = object()
    source = _Plugin(
        name="csv",
        plugin_version="source:1",
        determinism=Determinism.IO_READ,
        source_file_hash=None,
        schema_contract=schema_contract,
    )
    config = SimpleNamespace(sources={"input": source})

    assert resolve_source_contracts_by_node_id(config, {"input": NodeID("src")}) == {NodeID("src"): schema_contract}


def test_resolve_source_contracts_by_node_id_fails_closed_on_stale_source_map() -> None:
    config = SimpleNamespace(sources={})

    with pytest.raises(OrchestrationInvariantError, match="Source ID map references source"):
        resolve_source_contracts_by_node_id(config, {"missing": NodeID("src")})


def test_register_nodes_with_landscape_uses_resolved_audit_metadata() -> None:
    schema_contract = object()
    graph = _Graph(_node_info("src", NodeType.SOURCE, plugin_name="csv", config={"source_name": "input"}))
    data_flow = _DataFlow()
    factory = SimpleNamespace(data_flow=data_flow)

    register_nodes_with_landscape(
        factory,
        "run-1",
        graph,
        ["src"],
        {
            NodeID("src"): NodeAuditMetadata(
                plugin_version="resolved:1",
                determinism=Determinism.IO_READ,
                source_file_hash="sha256:aaaaaaaaaaaaaaaa",
            )
        },
        {NodeID("src"): schema_contract},
    )

    assert data_flow.register_node_calls == [
        {
            "run_id": "run-1",
            "node_id": "src",
            "plugin_name": "csv",
            "node_type": NodeType.SOURCE,
            "plugin_version": "resolved:1",
            "config": {"source_name": "input"},
            "determinism": Determinism.IO_READ,
            "schema_config": _SCHEMA_CONFIG,
            "output_contract": schema_contract,
            "source_file_hash": "sha256:aaaaaaaaaaaaaaaa",
        }
    ]

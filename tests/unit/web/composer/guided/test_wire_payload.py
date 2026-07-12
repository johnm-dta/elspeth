"""Tests for the STEP_4_WIRE turn payload data model (P2/B2)."""

from __future__ import annotations

from elspeth.web.composer.guided.emitters import _build_wire_topology, build_step_4_wire_turn
from elspeth.web.composer.guided.protocol import (
    TurnType,
    WireStageData,
    WireTopology,
    validate_payload,
)
from elspeth.web.composer.state import CompositionState, EdgeSpec, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec


def _canonical_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="inline_blob",
            on_success="chain_in",
            options={"schema": {"mode": "observed"}},
            on_validation_failure="reject",
        ),
        nodes=(
            NodeSpec(
                id="scrape",
                node_type="transform",
                plugin="web_scrape",
                input="chain_in",
                on_success="scraped",
                on_error=None,
                options={"url_field": "url"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="mapper",
                node_type="transform",
                plugin="field_mapper",
                input="scraped",
                on_success="jsonl_out",
                on_error=None,
                options={"mapping": {"body": "text"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="jsonl_out",
                plugin="json",
                options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Tutorial", description="Wire payload fixture"),
        version=1,
    )


def _contract_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="t1",
            options={"path": "/data/input.csv", "schema": {"mode": "fixed", "fields": ["text: str"]}},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "schema": {"mode": "observed"},
                    "operations": [{"target": "text", "expression": "row['text']"}],
                    "required_input_fields": ["text"],
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "outputs/main.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Contracts", description=""),
        version=1,
    )


def _queue_state() -> CompositionState:
    """Canonical two-source declared queue fan-in.

    Two named sources publish the ``inbound`` connection; the declared queue
    node ``inbound`` (id == input, plugin None, implicit output under its id)
    consumes them and a single ordinary transform consumes the queue.
    """

    def _src() -> SourceSpec:
        return SourceSpec(
            plugin="csv",
            on_success="inbound",
            options={"schema": {"mode": "observed"}},
            on_validation_failure="discard",
        )

    return CompositionState(
        source=None,
        sources={"orders": _src(), "refunds": _src()},
        nodes=(
            NodeSpec(
                id="inbound",
                node_type="queue",
                plugin=None,
                input="inbound",
                on_success=None,
                on_error=None,
                options={"description": "Orders and refunds interleave here"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="normalize",
                node_type="transform",
                plugin="passthrough",
                input="inbound",
                on_success="combined",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(
                name="combined",
                plugin="json",
                options={"schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Queue fan-in", description=""),
        version=1,
    )


class TestWireStageDataShape:
    def test_wire_stage_data_keys(self) -> None:
        payload: WireStageData = {
            "topology": {"sources": {}, "nodes": [], "outputs": []},
            "edge_contracts": [],
            "semantic_contracts": [],
            "warnings": [],
        }

        assert set(payload) == {
            "topology",
            "edge_contracts",
            "semantic_contracts",
            "warnings",
        }

    def test_wire_topology_keys(self) -> None:
        topology: WireTopology = {
            "sources": {},
            "nodes": [],
            "outputs": [],
        }

        assert set(topology) == {"sources", "nodes", "outputs"}


class TestConfirmWiringValidation:
    def test_valid_wire_payload_passes(self) -> None:
        payload = {
            "topology": {"sources": {}, "nodes": [], "outputs": []},
            "edge_contracts": [],
            "semantic_contracts": [],
            "warnings": [],
        }

        assert validate_payload(TurnType.CONFIRM_WIRING, payload) is None

    def test_missing_topology_rejected(self) -> None:
        err = validate_payload(
            TurnType.CONFIRM_WIRING,
            {
                "edge_contracts": [],
                "semantic_contracts": [],
                "warnings": [],
            },
        )

        assert err is not None
        assert "topology" in err

    def test_topology_must_be_mapping_with_expected_keys(self) -> None:
        err = validate_payload(
            TurnType.CONFIRM_WIRING,
            {
                "topology": {},
                "edge_contracts": [],
                "semantic_contracts": [],
                "warnings": [],
            },
        )

        assert err is not None
        assert "payload.topology" in err
        assert "sources" in err
        assert "nodes" in err
        assert "outputs" in err

    def test_missing_warnings_rejected(self) -> None:
        err = validate_payload(
            TurnType.CONFIRM_WIRING,
            {
                "topology": {"sources": {}, "nodes": [], "outputs": []},
                "edge_contracts": [],
                "semantic_contracts": [],
            },
        )

        assert err is not None
        assert "warnings" in err


class TestBuildWireTopology:
    def test_topology_reads_connection_labels(self) -> None:
        topo = _build_wire_topology(_canonical_state())

        assert topo["sources"] == {
            "source": {
                "id": "source",
                "plugin": "inline_blob",
                "on_success": "chain_in",
                "on_validation_failure": "reject",
            }
        }
        assert topo["nodes"][0]["input"] == "chain_in"
        assert topo["nodes"][0]["on_success"] == "scraped"
        assert topo["nodes"][1]["input"] == "scraped"
        assert topo["nodes"][1]["on_success"] == "jsonl_out"
        assert topo["outputs"] == [
            {
                "id": "output:jsonl_out",
                "sink_name": "jsonl_out",
                "plugin": "json",
                "on_write_failure": "discard",
            }
        ]

    def test_topology_node_subset_drops_options(self) -> None:
        topo = _build_wire_topology(_canonical_state())

        assert set(topo["nodes"][0]) == {
            "id",
            "node_type",
            "plugin",
            "input",
            "on_success",
            "on_error",
            "routes",
            "fork_to",
            "branches",
        }

    def test_topology_never_reads_state_edges(self) -> None:
        state = _canonical_state()
        stale_editor_edge_state = state.with_edge(
            EdgeSpec(
                id="stale-editor-edge",
                from_node="source",
                to_node="wrong_target",
                edge_type="on_success",
                label="wrong_label",
            )
        )

        assert _build_wire_topology(stale_editor_edge_state) == _build_wire_topology(state)

    def test_output_sink_name_is_preserved_as_connection_label(self) -> None:
        topo = _build_wire_topology(_canonical_state())

        assert topo["outputs"][0]["sink_name"] == "jsonl_out"

    def test_named_source_id_matches_validation_contract_id(self) -> None:
        state = CompositionState(
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        ).with_named_source(
            "refunds",
            SourceSpec(
                plugin="inline_blob",
                on_success="refund_in",
                options={"schema": {"mode": "observed"}},
                on_validation_failure="reject",
            ),
        )

        topo = _build_wire_topology(state)

        assert topo["sources"]["refunds"]["id"] == "source:refunds"

    def test_topology_includes_failure_routes_and_coalesce_branches(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="inline_blob",
                on_success="branch_in",
                options={"schema": {"mode": "observed"}},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="fork",
                    node_type="gate",
                    plugin=None,
                    input="branch_in",
                    on_success=None,
                    on_error=None,
                    options={},
                    condition="true",
                    routes={"true": "fork"},
                    fork_to=("path_a", "path_b"),
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="merge",
                    node_type="coalesce",
                    plugin=None,
                    input="branches",
                    on_success="main",
                    on_error=None,
                    options={},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches={"a": "path_a_done", "b": "path_b_done"},
                    policy="require_all",
                    merge="nested",
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="json",
                    options={"path": "main.json", "schema": {"mode": "observed"}},
                    on_write_failure="failures",
                ),
                OutputSpec(
                    name="quarantine",
                    plugin="json",
                    options={"path": "quarantine.json", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
                OutputSpec(
                    name="failures",
                    plugin="json",
                    options={"path": "failures.json", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(name="Failure routes", description=""),
            version=1,
        )

        topo = _build_wire_topology(state)

        assert topo["sources"]["source"]["on_validation_failure"] == "quarantine"
        assert topo["nodes"][1]["branches"] == {"a": "path_a_done", "b": "path_b_done"}
        assert topo["outputs"][0]["on_write_failure"] == "failures"


class TestBuildStep4WireTurn:
    def test_turn_type_and_step(self) -> None:
        turn = build_step_4_wire_turn(_canonical_state())

        assert turn["type"] == TurnType.CONFIRM_WIRING.value
        assert turn["step_index"] == 4

    def test_payload_merges_topology_and_contracts(self) -> None:
        state = _contract_state()
        validation = state.validate()

        turn = build_step_4_wire_turn(state)
        payload = turn["payload"]

        assert payload["topology"] == _build_wire_topology(state)
        assert payload["edge_contracts"] == [ec.to_dict() for ec in validation.edge_contracts]
        assert payload["semantic_contracts"] == []
        assert payload["warnings"] == [warning.to_dict() for warning in validation.warnings]

    def test_edge_contracts_use_from_to_keys_not_from_id_to_id(self) -> None:
        payload = build_step_4_wire_turn(_contract_state())["payload"]

        edge_contract = payload["edge_contracts"][0]

        assert edge_contract["from"] == "source"
        assert edge_contract["to"] == "t1"
        assert "from_id" not in edge_contract
        assert "to_id" not in edge_contract

    def test_payload_validates(self) -> None:
        turn = build_step_4_wire_turn(_contract_state())

        assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None

    def test_revise_kwargs_fold_into_payload(self) -> None:
        turn = build_step_4_wire_turn(
            _canonical_state(),
            catalog=object(),
            advisor_findings="advisor says the wiring is coherent",
            signoff_outcome="approved",
        )

        assert turn["payload"]["advisor_findings"] == "advisor says the wiring is coherent"
        assert turn["payload"]["signoff_outcome"] == "approved"

    def test_initial_confirm_omits_revise_keys(self) -> None:
        payload = build_step_4_wire_turn(_canonical_state())["payload"]

        assert "advisor_findings" not in payload
        assert "signoff_outcome" not in payload


class TestQueueTransportPin:
    """The wire transport DTO carries ``node_type`` as a generic ``str``. A queue
    node must therefore flow through ``_build_wire_topology`` unchanged, with NO
    queue-specific emitter branch, and the resulting payload must still validate.
    """

    def test_queue_node_flows_through_topology_unchanged(self) -> None:
        topo = _build_wire_topology(_queue_state())

        queue = next(node for node in topo["nodes"] if node["node_type"] == "queue")
        assert queue == {
            "id": "inbound",
            "node_type": "queue",
            "plugin": None,
            "input": "inbound",
            "on_success": None,
            "on_error": None,
            "routes": None,
            "fork_to": None,
            "branches": None,
        }

    def test_both_sources_publish_the_queue_connection(self) -> None:
        topo = _build_wire_topology(_queue_state())

        assert topo["sources"]["orders"]["on_success"] == "inbound"
        assert topo["sources"]["refunds"]["on_success"] == "inbound"
        # The ordinary downstream consumer reads the queue's id, unchanged.
        normalize = next(node for node in topo["nodes"] if node["id"] == "normalize")
        assert normalize["input"] == "inbound"

    def test_step_4_wire_turn_emits_and_validates_queue_topology(self) -> None:
        turn = build_step_4_wire_turn(_queue_state())

        assert turn["type"] == TurnType.CONFIRM_WIRING.value
        assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None
        node_types = {node["node_type"] for node in turn["payload"]["topology"]["nodes"]}
        assert "queue" in node_types


class TestHonestGapRendering:
    def test_fork_topology_does_not_fabricate_edge_contract_rows(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="inline_blob",
                on_success="chain_in",
                options={"schema": {"mode": "observed"}},
                on_validation_failure="reject",
            ),
            nodes=(
                NodeSpec(
                    id="fork",
                    node_type="gate",
                    plugin=None,
                    input="chain_in",
                    on_success=None,
                    on_error=None,
                    options={},
                    condition="true",
                    routes={"true": "fork"},
                    fork_to=("branch_a", "branch_b"),
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="jsonl_out",
                    plugin="json",
                    options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(name="Honest gaps", description=""),
            version=1,
        )

        payload = build_step_4_wire_turn(state)["payload"]
        node = payload["topology"]["nodes"][0]

        assert node["fork_to"] == ["branch_a", "branch_b"]
        assert ("fork", "branch_a") not in {(row["from"], row["to"]) for row in payload["edge_contracts"]}
        assert ("fork", "branch_b") not in {(row["from"], row["to"]) for row in payload["edge_contracts"]}

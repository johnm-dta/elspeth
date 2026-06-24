"""Tests for the STEP_4_WIRE turn payload data model (P2/B2)."""

from __future__ import annotations

from elspeth.web.composer.guided.emitters import _build_wire_topology, build_step_4_wire_turn
from elspeth.web.composer.guided.protocol import (
    TurnType,
    WireStageData,
    WireTopology,
    validate_payload,
)
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec


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
        }

    def test_topology_never_reads_state_edges(self) -> None:
        topo = _build_wire_topology(_canonical_state())

        assert topo["nodes"][0]["input"] == "chain_in"
        assert topo["nodes"][1]["input"] == "scraped"
        assert topo["outputs"][0]["sink_name"] == "jsonl_out"

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

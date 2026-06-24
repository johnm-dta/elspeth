"""Tests for the STEP_4_WIRE turn payload data model (P2/B2)."""

from __future__ import annotations

from elspeth.web.composer.guided.protocol import (
    TurnType,
    WireStageData,
    WireTopology,
    validate_payload,
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

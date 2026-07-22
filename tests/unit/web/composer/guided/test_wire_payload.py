"""Tests for the authoritative arbitrary-DAG Step-4 wire payload."""

from __future__ import annotations

from copy import deepcopy

import pytest

from elspeth.web.composer.guided.protocol import TurnType, WireStageData, validate_payload


def _wire_payload() -> WireStageData:
    return {
        "proposal_id": "00000000-0000-4000-8000-000000000001",
        "draft_hash": "d" * 64,
        "sources": [
            {
                "stable_id": "00000000-0000-4000-8000-000000000002",
                "label": "source-1",
                "plugin": "csv",
                "on_validation_failure": "discard",
                "guaranteed_fields": ["text"],
                "row_cardinality": {
                    "input": "none",
                    "output": "zero_or_many",
                    "expected_output_count": None,
                },
            }
        ],
        "nodes": [],
        "outputs": [
            {
                "stable_id": "00000000-0000-4000-8000-000000000003",
                "label": "output-1",
                "plugin": "json",
                "on_write_failure": "discard",
                "required_fields": ["text"],
                "business_schema": {
                    "mode": "observed",
                    "fields": [],
                    "guaranteed_fields": [],
                    "required_fields": [],
                },
            }
        ],
        "connections": [
            {
                "stable_id": "00000000-0000-4000-8000-000000000004",
                "from_endpoint": {
                    "kind": "source",
                    "stable_id": "00000000-0000-4000-8000-000000000002",
                },
                "to_endpoint": {
                    "kind": "output",
                    "stable_id": "00000000-0000-4000-8000-000000000003",
                },
                "flow": {"kind": "source_success", "branch": None},
                "schema_contract": None,
            }
        ],
        "semantic_contracts": [],
        "warnings": [],
        "blockers": [],
        "can_confirm": True,
    }


def test_wire_stage_data_has_only_the_candidate_derived_contract() -> None:
    payload = _wire_payload()

    assert set(payload) == {
        "proposal_id",
        "draft_hash",
        "sources",
        "nodes",
        "outputs",
        "connections",
        "semantic_contracts",
        "warnings",
        "blockers",
        "can_confirm",
    }
    assert "topology" not in payload
    assert "edge_contracts" not in payload


def test_valid_arbitrary_dag_wire_payload_passes() -> None:
    assert validate_payload(TurnType.CONFIRM_WIRING, _wire_payload()) is None


@pytest.mark.parametrize("removed_field", ("advisor_findings", "signoff_outcome", "passes_remaining"))
def test_wire_payload_rejects_removed_advisor_signoff_fields(removed_field: str) -> None:
    payload = deepcopy(_wire_payload())
    payload[removed_field] = 0 if removed_field == "passes_remaining" else "legacy"  # type: ignore[literal-required]

    error = validate_payload(TurnType.CONFIRM_WIRING, payload)

    assert error is not None
    assert "unexpected" in error


@pytest.mark.parametrize(
    "missing",
    (
        "proposal_id",
        "draft_hash",
        "sources",
        "nodes",
        "outputs",
        "connections",
        "semantic_contracts",
        "warnings",
        "blockers",
        "can_confirm",
    ),
)
def test_each_required_wire_field_is_required(missing: str) -> None:
    payload = deepcopy(_wire_payload())
    del payload[missing]  # type: ignore[misc]

    error = validate_payload(TurnType.CONFIRM_WIRING, payload)

    assert error is not None
    assert missing in error


def test_connection_preserves_stable_endpoints_and_flow() -> None:
    connection = _wire_payload()["connections"][0]

    assert connection["from_endpoint"]["kind"] == "source"
    assert connection["to_endpoint"]["kind"] == "output"
    assert connection["flow"] == {"kind": "source_success", "branch": None}

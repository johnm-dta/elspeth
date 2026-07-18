"""Tests for guided-mode protocol types."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.web.composer.guided.protocol import (
    ControlSignal,
    GuidedStep,
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposePipelinePayload,
    SchemaFormPayload,
    SingleSelectPayload,
    Turn,
    TurnResponse,
    TurnType,
    validate_current_turn,
    validate_payload,
)


class TestTurnType:
    def test_six_turn_types_defined(self) -> None:
        expected = {
            "inspect_and_confirm",
            "single_select",
            "multi_select_with_custom",
            "schema_form",
            "propose_pipeline",
            "confirm_wiring",
        }
        assert {t.value for t in TurnType} == expected

    def test_turn_type_is_str_enum(self) -> None:
        assert TurnType.SINGLE_SELECT.value == "single_select"
        assert TurnType("single_select") is TurnType.SINGLE_SELECT


class TestGuidedStep:
    def test_step_4_wire_defined(self) -> None:
        assert GuidedStep.STEP_4_WIRE.value == "step_4_wire"
        assert GuidedStep("step_4_wire") is GuidedStep.STEP_4_WIRE


class TestPayloadShapes:
    def test_inspect_and_confirm_payload_required_keys(self) -> None:
        payload: InspectAndConfirmPayload = {
            "observed": {"columns": ["a", "b"], "samples": [{"a": 1}], "warnings": []},
        }
        assert payload["observed"]["columns"] == ["a", "b"]

    def test_single_select_payload(self) -> None:
        payload: SingleSelectPayload = {
            "question": "Pick one",
            "options": [{"id": "a", "label": "A", "hint": None}],
            "allow_custom": False,
        }
        assert payload["allow_custom"] is False

    def test_multi_select_payload(self) -> None:
        payload: MultiSelectWithCustomPayload = {
            "question": "Pick many",
            "options": [{"id": "a", "label": "A", "hint": None}],
            "default_chosen": ["a"],
            "escape_label": "Or: let source decide",
        }
        assert payload["escape_label"] == "Or: let source decide"

    def test_schema_form_payload(self) -> None:
        payload: SchemaFormPayload = {
            "mode": "plugin_options",
            "plugin": "csv",
            "knobs": {"fields": []},
            "prefilled": {},
        }
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "csv"

    def test_propose_pipeline_payload_is_exported(self) -> None:
        payload: ProposePipelinePayload = {
            "proposal_id": "00000000-0000-4000-8000-000000000401",
            "draft_hash": "d" * 64,
            "summary": "One transform and one output.",
            "rationale": "The reviewed contract requires validation.",
            "blockers": [],
            "graph": {"sources": [], "edges": []},
            "nodes": [],
            "outputs": [],
            "edit_targets": [],
        }
        assert payload["proposal_id"].endswith("0401")


class TestTurnResponse:
    def test_control_signal_values(self) -> None:
        assert {s.value for s in ControlSignal} == {
            "exit_to_freeform",
            "request_advisor",
            "reject",
            "back",
            "passthrough",
        }

    def test_turn_response_minimal(self) -> None:
        resp: TurnResponse = {
            "operation_id": "00000000-0000-4000-8000-000000000411",
            "turn_token": "a" * 64,
            "chosen": ["jsonl"],
            "edited_values": None,
            "custom_inputs": None,
            "proposal_id": None,
            "draft_hash": None,
            "edit_target": None,
            "control_signal": None,
        }
        assert resp["chosen"] == ["jsonl"]

    def test_turn_response_with_control_signal(self) -> None:
        resp: TurnResponse = {
            "operation_id": "00000000-0000-4000-8000-000000000411",
            "turn_token": None,
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "proposal_id": None,
            "draft_hash": None,
            "edit_target": None,
            "control_signal": ControlSignal.EXIT_TO_FREEFORM,
        }
        assert resp["control_signal"] is ControlSignal.EXIT_TO_FREEFORM
        # StrEnum members serialize as their .value for canonical-JSON / wire
        # purposes — document the equivalence so the round-trip contract is
        # captured in test code.
        assert ControlSignal.EXIT_TO_FREEFORM.value == "exit_to_freeform"


class TestTurn:
    def test_turn_carries_type_and_payload(self) -> None:
        turn: Turn = {
            "type": "single_select",
            "step_index": 1,
            "payload": {
                "question": "X?",
                "options": [],
                "allow_custom": False,
            },
        }
        assert turn["type"] == "single_select"
        assert turn["step_index"] == 1


class TestLegalTurnMatrix:
    def test_step_1_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_1_SOURCE)
        assert TurnType.INSPECT_AND_CONFIRM in legal
        assert TurnType.SINGLE_SELECT in legal
        assert TurnType.SCHEMA_FORM in legal
        assert TurnType.PROPOSE_PIPELINE not in legal

    def test_step_2_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_2_SINK)
        assert TurnType.SINGLE_SELECT in legal
        assert TurnType.MULTI_SELECT_WITH_CUSTOM in legal
        assert TurnType.SCHEMA_FORM in legal

    def test_step_3_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_3_TRANSFORMS)
        assert TurnType.PROPOSE_PIPELINE in legal
        assert TurnType.SINGLE_SELECT in legal

    def test_step_4_wire_confirm_wiring_only(self) -> None:
        from elspeth.web.composer.guided.protocol import legal_turn_types_for

        assert legal_turn_types_for(GuidedStep.STEP_4_WIRE) == frozenset({TurnType.CONFIRM_WIRING})


class TestPayloadValidation:
    @pytest.mark.parametrize(
        ("step", "turn"),
        [
            (
                GuidedStep.STEP_1_SOURCE,
                {
                    "type": "inspect_and_confirm",
                    "step_index": 0,
                    "payload": {"observed": {"columns": [1], "samples": [], "warnings": []}},
                },
            ),
            (
                GuidedStep.STEP_1_SOURCE,
                {
                    "type": "single_select",
                    "step_index": 0,
                    "payload": {
                        "question": "Choose",
                        "options": [{"id": "csv", "label": "CSV", "hint": None, "canary": True}],
                        "allow_custom": False,
                    },
                },
            ),
            (
                GuidedStep.STEP_2_SINK,
                {
                    "type": "multi_select_with_custom",
                    "step_index": 1,
                    "payload": {
                        "question": "Choose fields",
                        "options": [],
                        "default_chosen": [1],
                        "escape_label": None,
                    },
                },
            ),
            (
                GuidedStep.STEP_1_SOURCE,
                {
                    "type": "schema_form",
                    "step_index": 0,
                    "payload": {
                        "mode": "plugin_options",
                        "plugin": "csv",
                        "knobs": {
                            "fields": [
                                {
                                    "name": "path",
                                    "label": "Path",
                                    "kind": "credential-canary",
                                    "required": True,
                                    "nullable": False,
                                }
                            ]
                        },
                        "prefilled": {},
                    },
                },
            ),
            (
                GuidedStep.STEP_4_WIRE,
                {
                    "type": "confirm_wiring",
                    "step_index": 3,
                    "payload": {
                        "topology": {
                            "sources": {
                                "source": {
                                    "id": 7,
                                    "plugin": "csv",
                                    "on_success": None,
                                    "on_validation_failure": "discard",
                                }
                            },
                            "nodes": [],
                            "outputs": [],
                        },
                        "edge_contracts": [],
                        "semantic_contracts": [],
                        "warnings": [],
                    },
                },
            ),
        ],
        ids=["inspect", "single-select", "multi-select", "schema-form", "confirm-wiring"],
    )
    def test_validate_current_turn_rejects_recursively_malformed_nonproposal_payloads(
        self,
        step: GuidedStep,
        turn: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="current-schema turn payload is invalid"):
            validate_current_turn(step, turn)

    def test_validate_single_select_ok(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        ok = validate_payload(
            TurnType.SINGLE_SELECT,
            {"question": "Q?", "options": [], "allow_custom": False},
        )
        assert ok is None

    def test_validate_single_select_missing_field(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(TurnType.SINGLE_SELECT, {"question": "Q?"})
        assert err is not None
        assert "options" in err

    def test_validate_unknown_turn_type_rejected(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        with pytest.raises(ValueError):
            validate_payload("not_a_turn_type", {})  # type: ignore[arg-type]

    def test_schema_form_plugin_options_golden_validates(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.SCHEMA_FORM,
            {
                "mode": "plugin_options",
                "plugin": "csv",
                "knobs": {"fields": []},
                "prefilled": {},
            },
        )
        assert err is None

    def test_schema_form_requires_knobs_fields(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.SCHEMA_FORM,
            {
                "mode": "plugin_options",
                "plugin": "csv",
                "knobs": {},
                "prefilled": {},
            },
        )
        assert err is not None
        assert "payload.knobs" in err
        assert "fields" in err

    def test_confirm_wiring_minimal_wire_payload_validates(self) -> None:
        payload = {
            "topology": {"sources": {}, "nodes": [], "outputs": []},
            "edge_contracts": [],
            "semantic_contracts": [],
            "warnings": [],
        }
        assert validate_payload(TurnType.CONFIRM_WIRING, payload) is None

    def test_confirm_wiring_accepts_queue_node_type_generically(self) -> None:
        # _WireNodeTopo.node_type is a generic str transport field, not the
        # closed composition vocabulary, so a queue node validates without any
        # queue-specific schema branch (elspeth-a5b86149d4 / elspeth-6421ffa028).
        payload = {
            "topology": {
                "sources": {},
                "nodes": [
                    {
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
                ],
                "outputs": [],
            },
            "edge_contracts": [],
            "semantic_contracts": [],
            "warnings": [],
        }
        assert validate_payload(TurnType.CONFIRM_WIRING, payload) is None

    def test_confirm_wiring_payload_missing_key_rejected(self) -> None:
        err = validate_payload(TurnType.CONFIRM_WIRING, {"topology": {}})
        assert err is not None
        assert "confirm_wiring" in err
        assert "edge_contracts" in err
        assert "semantic_contracts" in err
        assert "warnings" in err

    def test_schema_form_plugin_options_requires_plugin(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.SCHEMA_FORM,
            {
                "mode": "plugin_options",
                "knobs": {"fields": []},
                "prefilled": {},
            },
        )
        assert err is not None
        assert "plugin" in err

    # ---------------------------------------------------------------------------
    # Recursive nested-shape validation (S4 uplift)
    # ---------------------------------------------------------------------------

    def test_inspect_and_confirm_golden_validates(self) -> None:
        """Full valid INSPECT_AND_CONFIRM payload passes recursively."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.INSPECT_AND_CONFIRM,
            {"observed": {"columns": ["a", "b"], "samples": [{"a": 1}], "warnings": []}},
        )
        assert err is None

    def test_inspect_and_confirm_missing_nested_key(self) -> None:
        """``observed`` present but missing ``columns`` — error path-rooted."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.INSPECT_AND_CONFIRM,
            {"observed": {"samples": [], "warnings": []}},  # missing "columns"
        )
        assert err is not None
        # Path-rooted: must mention "payload.observed"
        assert "payload.observed" in err
        assert "columns" in err

    def test_inspect_and_confirm_observed_not_mapping(self) -> None:
        """``observed`` is a scalar, not a Mapping — type error is path-rooted."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.INSPECT_AND_CONFIRM,
            {"observed": "not-a-mapping"},
        )
        assert err is not None
        assert "payload.observed" in err
        assert "mapping" in err

    def test_top_level_missing_key_error_takes_priority(self) -> None:
        """Missing top-level key is reported before nested checks run."""
        from elspeth.web.composer.guided.protocol import validate_payload

        # "observed" is missing entirely — nested check cannot run.
        err = validate_payload(TurnType.INSPECT_AND_CONFIRM, {})
        assert err is not None
        assert "observed" in err
        # Path-rooted nested error should NOT appear when top-level is missing.
        assert "payload.observed" not in err

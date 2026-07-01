"""Tests for guided-mode protocol types."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import (
    ControlSignal,
    GuidedStep,
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposeChainPayload,
    SchemaFormPayload,
    SingleSelectPayload,
    Turn,
    TurnResponse,
    TurnType,
    validate_payload,
)


class TestTurnType:
    def test_seven_turn_types_defined(self) -> None:
        expected = {
            "inspect_and_confirm",
            "single_select",
            "multi_select_with_custom",
            "schema_form",
            "propose_chain",
            "recipe_offer",
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

    def test_propose_chain_payload(self) -> None:
        payload: ProposeChainPayload = {
            "steps": [
                {"plugin": "type_coerce", "options": {}, "rationale": "needed for gate"},
            ],
            "why": "bridge str to float",
            "blockers": [],
        }
        assert len(payload["steps"]) == 1

    def test_recipe_offer_payload_is_recipe_decision_schema_form(self) -> None:
        """RECIPE_OFFER keeps its turn discriminator but shares schema_form payloads."""
        from elspeth.web.composer.guided.protocol import validate_payload

        payload: SchemaFormPayload = {
            "mode": "recipe_decision",
            "knobs": {
                "fields": [
                    {
                        "name": "classifier_template",
                        "label": "Classifier template",
                        "description": "Jinja2 template",
                        "kind": "text",
                        "required": True,
                        "nullable": False,
                    }
                ]
            },
            "prefilled": {"source_blob_id": "abc-uuid", "output_path": "out.jsonl"},
            "recipe_context": {
                "recipe_name": "classify-rows-llm-jsonl",
                "description": "Classify CSV rows",
                "alternatives": ["build_manually"],
            },
        }
        assert validate_payload(TurnType.RECIPE_OFFER, payload) is None
        assert payload["recipe_context"]["recipe_name"] == "classify-rows-llm-jsonl"

    def test_recipe_offer_legacy_payload_rejected(self) -> None:
        """The old recipe_offer shape is not accepted alongside the new contract."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {  # type: ignore[arg-type]
                "recipe_name": "x",
                "slots": {},
                "alternatives": [],
            },
        )
        assert err is not None
        assert "mode" in err


class TestTurnResponse:
    def test_control_signal_values(self) -> None:
        assert {s.value for s in ControlSignal} == {
            "exit_to_freeform",
            "request_advisor",
            "reject",
            "back",
        }

    def test_turn_response_minimal(self) -> None:
        resp: TurnResponse = {
            "chosen": ["jsonl"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        assert resp["chosen"] == ["jsonl"]

    def test_turn_response_with_control_signal(self) -> None:
        resp: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
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
        assert TurnType.PROPOSE_CHAIN not in legal

    def test_step_2_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_2_SINK)
        assert TurnType.SINGLE_SELECT in legal
        assert TurnType.MULTI_SELECT_WITH_CUSTOM in legal
        assert TurnType.SCHEMA_FORM in legal

    def test_step_2_5_recipe_offer_only(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_2_5_RECIPE_MATCH)
        assert legal == frozenset({TurnType.RECIPE_OFFER})

    def test_step_3_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_3_TRANSFORMS)
        assert TurnType.PROPOSE_CHAIN in legal
        assert TurnType.SINGLE_SELECT in legal

    def test_step_4_wire_confirm_wiring_only(self) -> None:
        from elspeth.web.composer.guided.protocol import legal_turn_types_for

        assert legal_turn_types_for(GuidedStep.STEP_4_WIRE) == frozenset({TurnType.CONFIRM_WIRING})


class TestPayloadValidation:
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

    def test_schema_form_recipe_decision_requires_recipe_context(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.SCHEMA_FORM,
            {
                "mode": "recipe_decision",
                "knobs": {"fields": []},
                "prefilled": {},
            },
        )
        assert err is not None
        assert "recipe_context" in err

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

    def test_recipe_offer_golden_validates(self) -> None:
        """Full valid RECIPE_OFFER recipe-decision payload passes."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "mode": "recipe_decision",
                "knobs": {
                    "fields": [
                        {
                            "name": "x",
                            "label": "X",
                            "kind": "text",
                            "required": True,
                            "nullable": False,
                        },
                    ],
                },
                "prefilled": {},
                "recipe_context": {
                    "recipe_name": "r",
                    "description": "Recipe",
                    "alternatives": ["build_manually"],
                },
            },
        )
        assert err is None

    def test_recipe_offer_empty_knobs_valid(self) -> None:
        """Empty knob fields list is valid when resolver covered all required slots."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "mode": "recipe_decision",
                "knobs": {"fields": []},
                "prefilled": {"x": 1},
                "recipe_context": {
                    "recipe_name": "r",
                    "description": "Recipe",
                    "alternatives": [],
                },
            },
        )
        assert err is None

    def test_recipe_offer_knobs_missing_fields_key(self) -> None:
        """A knobs mapping missing ``fields`` is rejected path-locally."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "mode": "recipe_decision",
                "knobs": {},
                "prefilled": {},
                "recipe_context": {
                    "recipe_name": "r",
                    "description": "Recipe",
                    "alternatives": [],
                },
            },
        )
        assert err is not None
        assert "payload.knobs" in err
        assert "fields" in err

    def test_recipe_offer_knobs_not_mapping(self) -> None:
        """``knobs`` must be a Mapping, not a sequence or scalar."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "mode": "recipe_decision",
                "knobs": "not-a-mapping",
                "prefilled": {},
                "recipe_context": {
                    "recipe_name": "r",
                    "description": "Recipe",
                    "alternatives": [],
                },
            },
        )
        assert err is not None
        assert "payload.knobs" in err
        assert "mapping" in err

    def test_recipe_offer_recipe_context_missing_key(self) -> None:
        """recipe_context must carry the metadata banner fields."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "mode": "recipe_decision",
                "knobs": {"fields": []},
                "prefilled": {},
                "recipe_context": {
                    "recipe_name": "r",
                    "alternatives": [],
                },
            },
        )
        assert err is not None
        assert "payload.recipe_context" in err
        assert "description" in err

    def test_top_level_missing_key_error_takes_priority(self) -> None:
        """Missing top-level key is reported before nested checks run."""
        from elspeth.web.composer.guided.protocol import validate_payload

        # "observed" is missing entirely — nested check cannot run.
        err = validate_payload(TurnType.INSPECT_AND_CONFIRM, {})
        assert err is not None
        assert "observed" in err
        # Path-rooted nested error should NOT appear when top-level is missing.
        assert "payload.observed" not in err


class TestBuildStep3ProposeChainTurn:
    """Tests for the propose_chain emitter (Task 4.5)."""

    def test_emits_propose_chain_with_step_index_3(self) -> None:
        from elspeth.web.composer.guided.emitters import build_step_3_propose_chain_turn
        from elspeth.web.composer.guided.state_machine import ChainProposal

        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "passthrough",
                    "options": {"schema": {"mode": "observed"}},
                    "rationale": "no-op chain",
                },
            ),
            why="rows already conform",
        )

        turn = build_step_3_propose_chain_turn(proposal)

        assert turn["type"] == TurnType.PROPOSE_CHAIN.value
        # STEP_3_TRANSFORMS is the 4th step (0-based index 3).
        assert turn["step_index"] == 3

    def test_payload_carries_steps_why_and_empty_blockers(self) -> None:
        from elspeth.web.composer.guided.emitters import build_step_3_propose_chain_turn
        from elspeth.web.composer.guided.state_machine import ChainProposal

        proposal = ChainProposal(
            steps=({"plugin": "passthrough", "options": {"schema": {"mode": "observed"}}, "rationale": "r"},),
            why="why-text",
        )

        turn = build_step_3_propose_chain_turn(proposal)
        payload = turn["payload"]

        assert set(payload.keys()) == {"steps", "why", "blockers"}
        assert payload["why"] == "why-text"
        # MVP-scope decision: blockers is always [] for server-emitted
        # propose_chain — solve_chain raises rather than returning a
        # partial proposal with blockers populated.
        assert payload["blockers"] == []
        assert payload["steps"][0]["plugin"] == "passthrough"

    def test_payload_validates_against_propose_chain_schema(self) -> None:
        from elspeth.web.composer.guided.emitters import build_step_3_propose_chain_turn
        from elspeth.web.composer.guided.protocol import validate_payload
        from elspeth.web.composer.guided.state_machine import ChainProposal

        proposal = ChainProposal(
            steps=({"plugin": "passthrough", "options": {"schema": {"mode": "observed"}}, "rationale": "r"},),
            why="ok",
        )
        turn = build_step_3_propose_chain_turn(proposal)
        assert validate_payload(TurnType.PROPOSE_CHAIN, turn["payload"]) is None

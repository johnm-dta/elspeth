"""Tests for guided-mode protocol types."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import (
    ControlSignal,
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposeChainPayload,
    RecipeOfferPayload,
    SchemaFormPayload,
    SingleSelectPayload,
    Turn,
    TurnResponse,
    TurnType,
)


class TestTurnType:
    def test_six_turn_types_defined(self) -> None:
        expected = {
            "inspect_and_confirm",
            "single_select",
            "multi_select_with_custom",
            "schema_form",
            "propose_chain",
            "recipe_offer",
        }
        assert {t.value for t in TurnType} == expected

    def test_turn_type_is_str_enum(self) -> None:
        assert TurnType.SINGLE_SELECT.value == "single_select"
        assert TurnType("single_select") is TurnType.SINGLE_SELECT


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
            "plugin": "csv",
            "schema_block": {"path": {"type": "string"}},
            "prefilled": {},
        }
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

    def test_recipe_offer_payload(self) -> None:
        payload: RecipeOfferPayload = {
            "recipe_name": "classify-rows-llm-jsonl",
            "slots": {},
            "alternatives": [],
            "unsatisfied_slots": [],
        }
        assert payload["recipe_name"] == "classify-rows-llm-jsonl"

    def test_recipe_offer_payload_with_unsatisfied_slots(self) -> None:
        """unsatisfied_slots carries the editable schema for unfilled required slots.

        ``required`` is intentionally absent from the wire shape: the
        RecipeMatch invariant guarantees every entry is required.
        """
        from elspeth.web.composer.guided.protocol import validate_payload

        payload: RecipeOfferPayload = {
            "recipe_name": "classify-rows-llm-jsonl",
            "slots": {"source_blob_id": "abc-uuid", "output_path": "out.jsonl"},
            "alternatives": ["build_manually"],
            "unsatisfied_slots": [
                {
                    "name": "classifier_template",
                    "slot_type": "str",
                    "description": "Jinja2 template",
                },
                {
                    "name": "model",
                    "slot_type": "str",
                    "description": "LLM model identifier",
                },
            ],
        }
        assert validate_payload(TurnType.RECIPE_OFFER, payload) is None
        assert payload["unsatisfied_slots"][0]["name"] == "classifier_template"

    def test_recipe_offer_payload_missing_unsatisfied_slots_rejected(self) -> None:
        """A payload missing the new required key surfaces a validation error."""
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
        assert "unsatisfied_slots" in err


class TestTurnResponse:
    def test_control_signal_values(self) -> None:
        assert {s.value for s in ControlSignal} == {
            "exit_to_freeform",
            "request_advisor",
            "reject",
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
        """Full valid RECIPE_OFFER payload with unsatisfied slots passes."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "recipe_name": "r",
                "slots": {},
                "alternatives": [],
                "unsatisfied_slots": [
                    {"name": "x", "slot_type": "str", "description": "hint"},
                ],
            },
        )
        assert err is None

    def test_recipe_offer_empty_unsatisfied_slots_valid(self) -> None:
        """Empty unsatisfied_slots list is valid (resolver covered all required slots)."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "recipe_name": "r",
                "slots": {"x": 1},
                "alternatives": [],
                "unsatisfied_slots": [],
            },
        )
        assert err is None

    def test_recipe_offer_unsatisfied_slot_missing_key(self) -> None:
        """An unsatisfied slot entry missing ``slot_type`` — path-rooted error."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "recipe_name": "r",
                "slots": {},
                "alternatives": [],
                "unsatisfied_slots": [
                    {"name": "x", "description": "d"},  # missing "slot_type"
                ],
            },
        )
        assert err is not None
        # Path-rooted: "payload.unsatisfied_slots[0] missing required keys: ..."
        assert "payload.unsatisfied_slots[0]" in err
        assert "slot_type" in err

    def test_recipe_offer_unsatisfied_slots_not_sequence(self) -> None:
        """``unsatisfied_slots`` is a Mapping, not a Sequence — type error is path-rooted."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "recipe_name": "r",
                "slots": {},
                "alternatives": [],
                "unsatisfied_slots": {"name": "x"},  # Mapping instead of Sequence
            },
        )
        assert err is not None
        assert "payload.unsatisfied_slots" in err
        assert "sequence" in err

    def test_recipe_offer_unsatisfied_slot_not_mapping(self) -> None:
        """An unsatisfied slot entry is a string, not a Mapping — type error is path-rooted."""
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(
            TurnType.RECIPE_OFFER,
            {
                "recipe_name": "r",
                "slots": {},
                "alternatives": [],
                "unsatisfied_slots": ["not-a-mapping"],
            },
        )
        assert err is not None
        assert "payload.unsatisfied_slots[0]" in err
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

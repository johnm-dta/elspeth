"""Tests for guided-mode protocol types."""

from __future__ import annotations

from elspeth.web.composer.guided.protocol import (
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposeChainPayload,
    RecipeOfferPayload,
    SchemaFormPayload,
    SingleSelectPayload,
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
        }
        assert payload["recipe_name"] == "classify-rows-llm-jsonl"

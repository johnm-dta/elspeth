"""Tests for guided-mode turn emitters (Task 10.0 — Gap 6 wire shape).

Existing emitter coverage lives alongside protocol tests (test_protocol.py's
``TestBuildStep3ProposeChainTurn``); this module specifically pins the
``build_step_2_5_recipe_offer_turn`` emitter that consumes the new
``RecipeMatch.unsatisfied_slots`` field and projects each ``SlotSpec`` to the
``_RecipeSlotInput`` wire shape.
"""

from __future__ import annotations

from types import SimpleNamespace

from elspeth.web.composer.guided.emitters import (
    _step_index,
    build_step_1_schema_form_turn,
    build_step_2_5_recipe_offer_turn,
    build_step_2_schema_form_turn,
    build_step_3_schema_form_turn,
    build_step_4_wire_turn,
)
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType, validate_payload
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.recipes import SlotSpec, get_recipe
from elspeth.web.composer.source_inspection import SourceInspectionFacts
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.sessions.routes._helpers import _guided_step_index


class _Catalog:
    def get_schema(self, plugin_type: str, plugin_name: str) -> SimpleNamespace:
        return SimpleNamespace(
            json_schema={"properties": {"path": {"type": "string"}}},
            knob_schema={
                "fields": [
                    {
                        "name": "path",
                        "label": "Path",
                        "kind": "text",
                        "required": True,
                        "nullable": False,
                    }
                ]
            },
        )


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestBuildSchemaFormTurns:
    def test_step_1_schema_form_uses_knob_schema_payload(self) -> None:
        turn = build_step_1_schema_form_turn("csv", _Catalog())

        assert turn["type"] == TurnType.SCHEMA_FORM.value
        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "csv"
        assert payload["knobs"]["fields"][0]["name"] == "path"
        assert "schema_block" not in payload
        assert validate_payload(TurnType.SCHEMA_FORM, payload) is None

    def test_step_1_schema_form_prefills_from_inspection_facts(self) -> None:
        facts = SourceInspectionFacts(
            source_kind="csv",
            redacted_identity={"filename": "input.csv"},
            byte_range_inspected=(0, 32),
            sample_row_count=1,
            observed_headers=("name", "age"),
            inferred_types={"name": "str", "age": "int"},
            url_candidates=(),
            warnings=(),
        )

        turn = build_step_1_schema_form_turn("csv", _Catalog(), inspection_facts=facts)

        schema_prefill = turn["payload"]["prefilled"]["schema"]
        assert schema_prefill == {"mode": "flexible", "fields": ["name: str", "age: int"]}

    def test_step_2_schema_form_uses_sink_knobs(self) -> None:
        turn = build_step_2_schema_form_turn("json", _Catalog())

        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "json"
        assert payload["knobs"]["fields"][0]["label"] == "Path"
        assert payload["prefilled"] == {"schema": {"mode": "observed"}}

    def test_step_3_schema_form_uses_transform_knobs_and_options_prefill(self) -> None:
        turn = build_step_3_schema_form_turn(
            plugin="type_coerce",
            options={"conversions": {"age": "int"}},
            catalog=_Catalog(),
        )

        payload = turn["payload"]
        assert payload["mode"] == "plugin_options"
        assert payload["plugin"] == "type_coerce"
        assert payload["knobs"]["fields"][0]["name"] == "path"
        assert payload["prefilled"] == {"conversions": {"age": "int"}}


class TestBuildStep25RecipeOfferTurn:
    def test_emits_recipe_offer_type(self) -> None:
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={"source_blob_id": "abc"},
            unsatisfied_slots={},
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        assert turn["type"] == TurnType.RECIPE_OFFER.value
        # STEP_2_5_RECIPE_MATCH ordinal in the wizard sequence.
        assert turn["step_index"] == 2

    def test_payload_carries_recipe_decision_knobs_prefill_and_context(
        self,
    ) -> None:
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={"a": 1},
            unsatisfied_slots={
                "x": SlotSpec(slot_type="str", description="hint", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        payload = turn["payload"]
        assert set(payload.keys()) == {
            "mode",
            "knobs",
            "prefilled",
            "recipe_context",
        }
        assert payload["mode"] == "recipe_decision"
        assert payload["prefilled"] == {"a": 1}
        assert payload["knobs"]["fields"][0]["name"] == "x"
        recipe = get_recipe("classify-rows-llm-jsonl")
        assert recipe is not None
        assert payload["recipe_context"] == {
            "recipe_name": "classify-rows-llm-jsonl",
            "description": recipe.description,
            "alternatives": ["build_manually"],
        }

    def test_payload_validates_against_recipe_offer_schema(self) -> None:
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={},
            unsatisfied_slots={
                "x": SlotSpec(slot_type="str", description="d", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        assert validate_payload(TurnType.RECIPE_OFFER, turn["payload"]) is None

    def test_unsatisfied_slot_entries_lower_to_knob_fields(
        self,
    ) -> None:
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={},
            unsatisfied_slots={
                "model": SlotSpec(
                    slot_type="str",
                    description="LLM model identifier",
                    required=True,
                ),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        fields = list(turn["payload"]["knobs"]["fields"])
        assert len(fields) == 1
        field = fields[0]
        assert field["name"] == "model"
        assert field["kind"] == "text"
        assert field["description"] == "LLM model identifier"
        assert field["required"] is True

    def test_unsatisfied_slots_empty_when_resolver_covers_all_required(self) -> None:
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={"source_blob_id": "abc"},
            unsatisfied_slots={},
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        assert list(turn["payload"]["knobs"]["fields"]) == []

    def test_unsatisfied_slots_preserves_iteration_order(self) -> None:
        """Frontend rendering order matters; iteration order of the source
        mapping is preserved (dict iteration is insertion-ordered in
        CPython ≥3.7)."""
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={},
            unsatisfied_slots={
                "first": SlotSpec(slot_type="str", description="1", required=True),
                "second": SlotSpec(slot_type="int", description="2", required=True),
                "third": SlotSpec(slot_type="float", description="3", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        names = [entry["name"] for entry in turn["payload"]["knobs"]["fields"]]
        assert names == ["first", "second", "third"]

    def test_unsatisfied_slot_numeric_types_project_to_wire(self) -> None:
        """slot_type field carries the raw SlotType literal so the frontend
        can pick the right <input type=...>."""
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={},
            unsatisfied_slots={
                "threshold": SlotSpec(slot_type="float", description="d", required=True),
                "count": SlotSpec(slot_type="int", description="d", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        by_name = {e["name"]: e for e in turn["payload"]["knobs"]["fields"]}
        assert by_name["threshold"]["kind"] == "number-float"
        assert by_name["count"]["kind"] == "number-int"


class TestStep4WireEmitter:
    def test_step_4_wire_index_matches_guided_order(self) -> None:
        assert _step_index(GuidedStep.STEP_4_WIRE) == 4
        assert _guided_step_index(GuidedStep.STEP_4_WIRE) == 4

    def test_builds_confirm_wiring_skeleton_payload(self) -> None:
        turn = build_step_4_wire_turn(validation=_empty_state().validate())

        assert turn["type"] == TurnType.CONFIRM_WIRING.value
        assert turn["step_index"] == 4
        assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None
        assert turn["payload"]["topology"] == {}
        assert turn["payload"]["edge_contracts"] == []
        assert turn["payload"]["semantic_contracts"] == []

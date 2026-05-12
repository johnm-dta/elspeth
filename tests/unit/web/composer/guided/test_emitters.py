"""Tests for guided-mode turn emitters (Task 10.0 — Gap 6 wire shape).

Existing emitter coverage lives alongside protocol tests (test_protocol.py's
``TestBuildStep3ProposeChainTurn``); this module specifically pins the
``build_step_2_5_recipe_offer_turn`` emitter that consumes the new
``RecipeMatch.unsatisfied_slots`` field and projects each ``SlotSpec`` to the
``_RecipeSlotInput`` wire shape.
"""

from __future__ import annotations

from elspeth.web.composer.guided.emitters import build_step_2_5_recipe_offer_turn
from elspeth.web.composer.guided.protocol import TurnType, validate_payload
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.recipes import SlotSpec


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

    def test_payload_carries_recipe_name_slots_alternatives_and_unsatisfied(
        self,
    ) -> None:
        match = RecipeMatch(
            recipe_name="r",
            slots={"a": 1},
            unsatisfied_slots={
                "x": SlotSpec(slot_type="str", description="hint", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        payload = turn["payload"]
        assert set(payload.keys()) == {
            "recipe_name",
            "slots",
            "alternatives",
            "unsatisfied_slots",
        }
        assert payload["recipe_name"] == "r"
        assert payload["slots"] == {"a": 1}
        assert payload["alternatives"] == ["build_manually"]

    def test_payload_validates_against_recipe_offer_schema(self) -> None:
        match = RecipeMatch(
            recipe_name="r",
            slots={},
            unsatisfied_slots={
                "x": SlotSpec(slot_type="str", description="d", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        assert validate_payload(TurnType.RECIPE_OFFER, turn["payload"]) is None

    def test_unsatisfied_slot_entries_carry_name_type_description(
        self,
    ) -> None:
        """Each entry is a _RecipeSlotInput TypedDict literal — the three keys
        the frontend needs to render the editable form.

        ``required`` is intentionally absent from the wire shape: the
        RecipeMatch invariant guarantees every entry is required, so the field
        would carry only dead information (always True).
        """
        match = RecipeMatch(
            recipe_name="r",
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
        entries = list(turn["payload"]["unsatisfied_slots"])
        assert len(entries) == 1
        entry = entries[0]
        assert entry["name"] == "model"
        assert entry["slot_type"] == "str"
        assert entry["description"] == "LLM model identifier"
        assert "required" not in entry

    def test_unsatisfied_slots_empty_when_resolver_covers_all_required(self) -> None:
        match = RecipeMatch(
            recipe_name="r",
            slots={"source_blob_id": "abc"},
            unsatisfied_slots={},
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        assert list(turn["payload"]["unsatisfied_slots"]) == []

    def test_unsatisfied_slots_preserves_iteration_order(self) -> None:
        """Frontend rendering order matters; iteration order of the source
        mapping is preserved (dict iteration is insertion-ordered in
        CPython ≥3.7)."""
        match = RecipeMatch(
            recipe_name="r",
            slots={},
            unsatisfied_slots={
                "first": SlotSpec(slot_type="str", description="1", required=True),
                "second": SlotSpec(slot_type="int", description="2", required=True),
                "third": SlotSpec(slot_type="float", description="3", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        names = [entry["name"] for entry in turn["payload"]["unsatisfied_slots"]]
        assert names == ["first", "second", "third"]

    def test_unsatisfied_slot_numeric_types_project_to_wire(self) -> None:
        """slot_type field carries the raw SlotType literal so the frontend
        can pick the right <input type=...>."""
        match = RecipeMatch(
            recipe_name="r",
            slots={},
            unsatisfied_slots={
                "threshold": SlotSpec(slot_type="float", description="d", required=True),
                "count": SlotSpec(slot_type="int", description="d", required=True),
            },
        )
        turn = build_step_2_5_recipe_offer_turn(match)
        by_name = {e["name"]: e for e in turn["payload"]["unsatisfied_slots"]}
        assert by_name["threshold"]["slot_type"] == "float"
        assert by_name["count"]["slot_type"] == "int"

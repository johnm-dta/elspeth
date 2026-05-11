"""Tests for guided-mode skill prompt loading + structural assertions.

Skill prompts are LLM behaviour, not code — these tests verify the file
exists, loads, and contains the load-bearing protocol-definition strings.
They do not assert the prompt's behavioural quality; that is verified by
real-LLM tests in test_chain_solver.py.
"""

from __future__ import annotations

from elspeth.web.composer.guided.prompts import load_guided_skill


class TestGuidedSkill:
    def test_loads(self) -> None:
        text = load_guided_skill()
        assert len(text) > 0

    def test_under_size_target(self) -> None:
        text = load_guided_skill()
        assert text.count("\n") <= 100, "guided skill should be ≤100 lines (target ≤80)"

    def test_mentions_six_turn_types(self) -> None:
        text = load_guided_skill()
        for turn_type in (
            "inspect_and_confirm",
            "single_select",
            "multi_select_with_custom",
            "schema_form",
            "propose_chain",
            "recipe_offer",
        ):
            assert turn_type in text, f"missing turn type: {turn_type}"

    def test_anti_fabrication_clause_present(self) -> None:
        text = load_guided_skill()
        # The hard rule that survives from freeform skill (spec §8.1.4)
        assert "anti-fabrication" in text.lower() or "do not invent" in text.lower()

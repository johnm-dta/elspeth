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


class TestStep3ContextBlock:
    def test_renders_source_and_sink(self) -> None:
        from elspeth.web.composer.guided.prompts import build_step_3_context_block
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved,
            SinkResolved,
            SourceResolved,
        )

        ctx = build_step_3_context_block(
            source=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("price", "qty"),
                sample_rows=({"price": "1.99", "qty": "2"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl"},
                        required_fields=("avg_price",),
                        schema_mode="fixed",
                    ),
                )
            ),
            recipe_match=None,
        )
        assert "source:" in ctx
        assert "csv" in ctx
        assert "price" in ctx
        assert "avg_price" in ctx
        assert "recipe_match: null" in ctx

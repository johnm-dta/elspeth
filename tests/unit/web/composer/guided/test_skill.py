"""Tests for guided-mode skill prompt loading + structural assertions.

Skill prompts are LLM behaviour, not code — these tests verify the file
exists, loads, and contains the load-bearing protocol-definition strings.
They do not assert the prompt's behavioural quality; that is verified by
real-LLM tests in test_chain_solver.py.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.prompts import (
    load_guided_skill,
    load_step_chat_skill,
)
from elspeth.web.composer.guided.protocol import GuidedStep


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


class TestStepChatSkill:
    """Per-step skill briefings — base + one step's playbook only.

    Used by the per-step chat solver to scope the LLM's context to the user's
    current wizard position.  The contract: each step skill includes the
    always-applies preamble + that step's playbook AND ONLY that step's
    playbook (no leakage from other steps).
    """

    @pytest.mark.parametrize("step", list(GuidedStep))
    def test_each_step_loads(self, step: GuidedStep) -> None:
        text = load_step_chat_skill(step)
        assert len(text) > 0

    @pytest.mark.parametrize("step", list(GuidedStep))
    def test_each_step_includes_base_preamble(self, step: GuidedStep) -> None:
        text = load_step_chat_skill(step)
        # The base preamble is the always-applies content; the anti-fabrication
        # rule is the canonical marker.
        assert "anti-fabrication" in text.lower() or "do not invent" in text.lower(), (
            f"{step.value} skill missing base preamble (anti-fabrication clause)"
        )

    def test_each_step_smaller_than_full_skill(self) -> None:
        full = load_guided_skill()
        for step in GuidedStep:
            scoped = load_step_chat_skill(step)
            assert len(scoped) < len(full), (
                f"per-step skill for {step.value} ({len(scoped)}) should be smaller than full composed skill ({len(full)})"
            )

    def test_step_skills_are_scoped(self) -> None:
        """Each step's skill must contain its own playbook header AND NOT
        the other steps' playbook headers — otherwise scoping is illusory."""
        markers = {
            GuidedStep.STEP_1_SOURCE: "Step 1 — Source",
            GuidedStep.STEP_2_SINK: "Step 2 — Sink",
            GuidedStep.STEP_2_5_RECIPE_MATCH: "Step 2.5 — Recipe",
            GuidedStep.STEP_3_TRANSFORMS: "Step 3 — Transform",
        }
        for step, marker in markers.items():
            text = load_step_chat_skill(step)
            assert marker in text, f"{step.value} skill missing its own marker {marker!r}"
            for other_step, other_marker in markers.items():
                if other_step is step:
                    continue
                assert other_marker not in text, f"{other_marker!r} leaked into {step.value} skill — scoping violated"


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

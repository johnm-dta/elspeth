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
        # NOTE: load_guided_skill() concatenates base.md + EVERY step skill, but
        # the runtime never sends that whole block to a model — the per-step chat
        # solver loads base.md + ONE step (load_step_chat_skill), so the real
        # prompt the light model sees is bounded by the largest single step, not
        # this sum. The largest per-step prompt is base + step_3_transforms at
        # ~189 lines; step_1's surface-or-record additions pushed the concatenated
        # total to 288. The cap is therefore a loose ceiling on aggregate bloat
        # across all stages, not a per-prompt budget — raised to 300 to absorb the
        # step_1 surface-or-record doctrine while still catching unbounded growth.
        assert text.count("\n") <= 300, "guided skill should be ≤300 lines (aggregate of all step skills)"

    def test_anti_fabrication_clause_present(self) -> None:
        text = load_guided_skill()
        # The hard anti-fabrication rule survives the LLM-primary rewrite as
        # "Don't invent things." (base.md) — match on the load-bearing verb so the
        # assertion is robust to wording, not pinned to a phrase that drifted.
        assert "invent" in text.lower() or "anti-fabrication" in text.lower()


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
        # Loader contract: every per-step skill is base.md + that step's playbook.
        # Anchor on base.md's H1 title — the stable, load-bearing marker of the
        # base preamble — rather than a prose phrase that the LLM-primary rewrite
        # reworded.
        assert "# Guided Pipeline Composer" in text, f"{step.value} skill missing base preamble"

    def test_each_step_smaller_than_full_skill(self) -> None:
        full = load_guided_skill()
        for step in GuidedStep:
            scoped = load_step_chat_skill(step)
            assert len(scoped) < len(full), (
                f"per-step skill for {step.value} ({len(scoped)}) should be smaller than full composed skill ({len(full)})"
            )

    def test_step_skills_are_scoped(self) -> None:
        """Each step's skill must contain its own playbook header AND NOT
        the other steps' playbook headers — otherwise scoping is illusory.

        Markers are each step file's actual H3 header. The LLM-primary rewrite
        moved some from "Step N — X" to "This stage: X"; the no-leakage property
        is what this test really guards, and it holds for any distinct headers.
        """
        markers = {
            GuidedStep.STEP_1_SOURCE: "### Step 1 — Source",
            GuidedStep.STEP_2_SINK: "### This stage: the output",
            GuidedStep.STEP_2_5_RECIPE_MATCH: "### Step 2.5 — (skipped)",
            GuidedStep.STEP_3_TRANSFORMS: "### This stage: the transforms",
            GuidedStep.STEP_4_WIRE: "### Step 4 — Wiring constraints",
        }
        for step, marker in markers.items():
            text = load_step_chat_skill(step)
            assert marker in text, f"{step.value} skill missing its own marker {marker!r}"
            for other_step, other_marker in markers.items():
                if other_step is step:
                    continue
                assert other_marker not in text, f"{other_marker!r} leaked into {step.value} skill — scoping violated"

    def test_step_4_wire_skill_mentions_wiring_constraints(self) -> None:
        text = load_step_chat_skill(GuidedStep.STEP_4_WIRE)
        for marker in ("wiring", "chain_in", '"main"', "select_only"):
            assert marker in text

    def test_step_3_only_declares_proven_mapper_guarantees(self) -> None:
        text = load_step_chat_skill(GuidedStep.STEP_3_TRANSFORMS)
        assert "mapping **keys**" in text
        assert "immediate upstream contract guarantees" in text
        assert "mapped output **targets**" in text
        assert "unproven source" in text

    def test_full_guided_skill_includes_wiring_constraint_labels(self) -> None:
        assert "chain_in" in load_guided_skill()


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
                name="source",
                plugin="csv",
                options={},
                observed_columns=("price", "qty"),
                sample_rows=({"price": "1.99", "qty": "2"},),
                on_validation_failure="discard",
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        name="main",
                        plugin="json",
                        options={"path": "out.jsonl"},
                        required_fields=("avg_price",),
                        schema_mode="fixed",
                        on_write_failure="discard",
                    ),
                )
            ),
        )
        assert "source:" in ctx
        assert "csv" in ctx
        assert "price" in ctx
        assert "avg_price" in ctx
        assert "recipe_match" not in ctx

    def test_step_3_context_redacts_sample_row_values(self) -> None:
        from elspeth.web.composer.guided.prompts import build_step_3_context_block
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved,
            SinkResolved,
            SourceResolved,
        )

        ctx = build_step_3_context_block(
            source=SourceResolved(
                name="source",
                plugin="csv",
                options={},
                observed_columns=("email", "api_key", "profile_url", "note"),
                sample_rows=(
                    {
                        "email": "person@example.test",
                        "api_key": "sk-test-secret-row-value",
                        "profile_url": "https://example.test/private?token=secret",
                        "note": "customer asked for refunds",
                    },
                ),
                on_validation_failure="discard",
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        name="main",
                        plugin="json",
                        options={},
                        required_fields=("email_hash",),
                        schema_mode="fixed",
                        on_write_failure="discard",
                    ),
                )
            ),
        )

        assert "person@example.test" not in ctx
        assert "sk-test-secret-row-value" not in ctx
        assert "https://example.test/private" not in ctx
        assert "customer asked for refunds" not in ctx
        assert "<sample:email-like>" in ctx
        assert "<sample:secret-like>" in ctx
        assert "<sample:url>" in ctx
        assert "<sample:string:" in ctx

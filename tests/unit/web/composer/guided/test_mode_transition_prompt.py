"""Unit tests for build_mode_transition_system_prompt.

Phase 5 Task 5.2 — progressive disclosure transition prompt.
Codex #17 — freeform_skill parameter threading (dependency-inversion fix).
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.prompts import (
    build_mode_transition_system_prompt,
)

# Sentinel freeform skill used across tests — unique enough to identify as
# the supplied value without depending on production skill content.
_SENTINEL_FREEFORM = "SENTINEL_FREEFORM_SKILL_CONTENT_XYZ"


class TestBuildModeTransitionSystemPrompt:
    """Tests for build_mode_transition_system_prompt."""

    @pytest.mark.parametrize(
        "reason",
        [
            "user_pressed_exit",
            "protocol_violation",
            "solver_exhausted",
            "completed_pipeline",
        ],
    )
    def test_layered_content_all_reasons(self, reason: str) -> None:
        """Returned prompt contains transition context, reason, and freeform content."""
        result = build_mode_transition_system_prompt(
            terminal_reason=reason,
            freeform_skill=_SENTINEL_FREEFORM,
        )

        # (a) rules-lifted signal present (spec §8.2 line 522)
        assert "LIFTED" in result, "Missing LIFTED signal"

        # (b) the terminal reason string present
        assert reason in result, f"Missing reason string: {reason}"

        # (c) supplied freeform_skill present verbatim
        assert _SENTINEL_FREEFORM in result, "Missing freeform_skill content in result"

    def test_layer_ordering(self) -> None:
        """Freeform-skill content appears before transition context."""
        result = build_mode_transition_system_prompt(
            terminal_reason="user_pressed_exit",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        transition_pos = result.find("## Mode Transition")
        freeform_pos = result.find(_SENTINEL_FREEFORM)

        assert transition_pos != -1, "transition header not found"
        assert freeform_pos != -1, "freeform-skill marker not found"

        assert freeform_pos < transition_pos, "Freeform content must appear before transition header"

    def test_transition_prompt_does_not_embed_guided_non_mutation_rule(self) -> None:
        """The first freeform turn must not restate guided-mode mutation bans."""
        result = build_mode_transition_system_prompt(
            terminal_reason="user_pressed_exit",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        assert "You **cannot** mutate pipeline state" not in result
        assert "Anything else is rejected" not in result

    def test_transition_header_literal(self) -> None:
        """Transition header matches spec §8.2 exactly."""
        result = build_mode_transition_system_prompt(
            terminal_reason="solver_exhausted",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        assert "## Mode Transition — Guided → Freeform" in result

    def test_reason_in_transition_block(self) -> None:
        """The reason appears inside the transition header block (not elsewhere)."""
        result = build_mode_transition_system_prompt(
            terminal_reason="completed_pipeline",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        assert "reason: completed_pipeline" in result

    def test_freeform_skill_supplied_verbatim(self) -> None:
        """The freeform_skill argument appears in the output exactly as supplied.

        This is the core contract for Codex #17: the caller-supplied (fully
        processed) freeform skill reaches the returned prompt unchanged —
        deployment overlay and advisor-strip are the caller's responsibility.
        """
        custom_skill = "CUSTOM_DEPLOYMENT_OVERLAY_CONTENT\nadvisor_stripped: true"
        result = build_mode_transition_system_prompt(
            terminal_reason="completed_pipeline",
            freeform_skill=custom_skill,
        )

        assert custom_skill in result

    def test_freeform_skill_is_first_layer(self) -> None:
        """Freeform instructions must be the first layer of the returned prompt."""
        result = build_mode_transition_system_prompt(
            terminal_reason="completed_pipeline",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        assert result.startswith(_SENTINEL_FREEFORM), "Freeform skill must be the first layer of the returned prompt"

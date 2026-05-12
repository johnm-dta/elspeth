"""Unit tests for build_mode_transition_system_prompt.

Phase 5 Task 5.2 — progressive disclosure transition prompt.
Codex #17 — freeform_skill parameter threading (dependency-inversion fix).
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.prompts import (
    build_mode_transition_system_prompt,
    load_guided_skill,
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
        """Returned prompt contains guided-skill content, LIFTED signal, reason, and freeform content."""
        result = build_mode_transition_system_prompt(
            terminal_reason=reason,
            freeform_skill=_SENTINEL_FREEFORM,
        )

        # (a) guided-skill content present (guided_pipeline.md mentions "guided mode")
        assert "guided mode" in result.lower(), "Missing guided-skill content marker"

        # (b) rules-lifted signal present (spec §8.2 line 522)
        assert "LIFTED" in result, "Missing LIFTED signal"

        # (c) the terminal reason string present
        assert reason in result, f"Missing reason string: {reason}"

        # (d) supplied freeform_skill present verbatim
        assert _SENTINEL_FREEFORM in result, "Missing freeform_skill content in result"

    def test_layer_ordering(self) -> None:
        """Guided-skill content appears before transition header, before freeform-skill content."""
        result = build_mode_transition_system_prompt(
            terminal_reason="user_pressed_exit",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        guided_pos = result.lower().find("guided mode")
        transition_pos = result.find("## Mode Transition")
        freeform_pos = result.find(_SENTINEL_FREEFORM)

        assert guided_pos != -1, "guided-skill marker not found"
        assert transition_pos != -1, "transition header not found"
        assert freeform_pos != -1, "freeform-skill marker not found"

        assert guided_pos < transition_pos, "Guided content must appear before transition header"
        assert transition_pos < freeform_pos, "Transition header must appear before freeform content"

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

    def test_guided_skill_from_load_guided_skill(self) -> None:
        """The guided section equals load_guided_skill() exactly."""
        guided = load_guided_skill()
        result = build_mode_transition_system_prompt(
            terminal_reason="completed_pipeline",
            freeform_skill=_SENTINEL_FREEFORM,
        )

        assert result.startswith(guided), "Guided skill must be the first layer of the returned prompt"

"""Unit tests for build_mode_transition_system_prompt and _load_freeform_skill.

Phase 5 Task 5.2 — progressive disclosure transition prompt.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.prompts import (
    _load_freeform_skill,
    build_mode_transition_system_prompt,
)


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
        result = build_mode_transition_system_prompt(terminal_reason=reason)

        # (a) guided-skill content present (guided_pipeline.md mentions "guided mode")
        assert "guided mode" in result.lower(), "Missing guided-skill content marker"

        # (b) rules-lifted signal present (spec §8.2 line 522)
        assert "LIFTED" in result, "Missing LIFTED signal"

        # (c) the terminal reason string present
        assert reason in result, f"Missing reason string: {reason}"

        # (d) freeform-skill content present (pipeline_composer.md contains "Audit Primacy")
        assert "Audit Primacy" in result, "Missing freeform-skill content marker"

    def test_layer_ordering(self) -> None:
        """Guided-skill content appears before transition header, before freeform-skill content."""
        result = build_mode_transition_system_prompt(terminal_reason="user_pressed_exit")

        guided_pos = result.lower().find("guided mode")
        transition_pos = result.find("## Mode Transition")
        freeform_pos = result.find("Audit Primacy")

        assert guided_pos != -1, "guided-skill marker not found"
        assert transition_pos != -1, "transition header not found"
        assert freeform_pos != -1, "freeform-skill marker not found"

        assert guided_pos < transition_pos, "Guided content must appear before transition header"
        assert transition_pos < freeform_pos, "Transition header must appear before freeform content"

    def test_transition_header_literal(self) -> None:
        """Transition header matches spec §8.2 exactly."""
        result = build_mode_transition_system_prompt(terminal_reason="solver_exhausted")

        assert "## Mode Transition — Guided → Freeform" in result

    def test_reason_in_transition_block(self) -> None:
        """The reason appears inside the transition header block (not elsewhere)."""
        result = build_mode_transition_system_prompt(terminal_reason="completed_pipeline")

        assert "reason: completed_pipeline" in result


class TestLoadFreeformSkill:
    def test_is_cached(self) -> None:
        """Two calls return the same object (lru_cache contract)."""
        first = _load_freeform_skill()
        second = _load_freeform_skill()

        assert first is second

    def test_returns_non_empty_string(self) -> None:
        content = _load_freeform_skill()
        assert isinstance(content, str)
        assert len(content) > 0

    def test_contains_audit_primacy_marker(self) -> None:
        content = _load_freeform_skill()
        assert "Audit Primacy" in content

"""Tests for WorkflowProfile - frozen value type + closed-enum discriminator."""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.web.composer.guided.profile import (
    EMPTY_PROFILE,
    TUTORIAL_PROFILE,
    WorkflowProfileKind,
    profile_for_kind,
)


class TestWorkflowProfileShape:
    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            EMPTY_PROFILE.coaching = True  # type: ignore[misc]

    def test_empty_profile_is_live_guided_default(self) -> None:
        assert EMPTY_PROFILE.entry_seed is None
        assert EMPTY_PROFILE.coaching is False
        assert EMPTY_PROFILE.advisor_checkpoints is False
        assert EMPTY_PROFILE.recipe_match is True
        assert EMPTY_PROFILE.bookends is False

    def test_tutorial_profile_enables_coaching_advisor_bookends(self) -> None:
        assert TUTORIAL_PROFILE.coaching is True
        assert TUTORIAL_PROFILE.advisor_checkpoints is True
        assert TUTORIAL_PROFILE.recipe_match is True
        assert TUTORIAL_PROFILE.bookends is True
        assert isinstance(TUTORIAL_PROFILE.entry_seed, str)
        assert TUTORIAL_PROFILE.entry_seed.strip() != ""


class TestWorkflowProfileKind:
    def test_kind_values_are_closed(self) -> None:
        assert WorkflowProfileKind.LIVE.value == "live"
        assert WorkflowProfileKind.TUTORIAL.value == "tutorial"
        assert {k.value for k in WorkflowProfileKind} == {"live", "tutorial"}

    def test_profile_for_kind_maps_live_to_empty(self) -> None:
        assert profile_for_kind(WorkflowProfileKind.LIVE) is EMPTY_PROFILE

    def test_profile_for_kind_maps_tutorial(self) -> None:
        assert profile_for_kind(WorkflowProfileKind.TUTORIAL) is TUTORIAL_PROFILE

    def test_unknown_kind_string_rejected_by_enum(self) -> None:
        with pytest.raises(ValueError):
            WorkflowProfileKind("bespoke")

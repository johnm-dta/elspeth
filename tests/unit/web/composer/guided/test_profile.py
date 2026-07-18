"""Tests for WorkflowProfile - frozen value type + closed-enum discriminator."""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.profile import (
    EMPTY_PROFILE,
    TUTORIAL_PROFILE,
    WorkflowProfile,
    WorkflowProfileKind,
    profile_for_kind,
)


class TestWorkflowProfileShape:
    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            EMPTY_PROFILE.coaching = True  # type: ignore[misc]

    def test_empty_profile_is_live_guided_default(self) -> None:
        assert EMPTY_PROFILE.coaching is False
        assert EMPTY_PROFILE.advisor_checkpoints is True
        assert EMPTY_PROFILE.bookends is False

    def test_tutorial_profile_enables_coaching_bookends(self) -> None:
        assert TUTORIAL_PROFILE.coaching is True
        # Tutorial is the explicit demo bypass: it skips the nondeterministic
        # terminal advisor sign-off for a known-good passive walkthrough.
        assert TUTORIAL_PROFILE.advisor_checkpoints is False
        assert TUTORIAL_PROFILE.bookends is True


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


class TestWorkflowProfileSerialisation:
    def test_empty_profile_round_trips(self) -> None:
        assert WorkflowProfile.from_dict(EMPTY_PROFILE.to_dict()) == EMPTY_PROFILE

    def test_tutorial_profile_round_trips(self) -> None:
        assert WorkflowProfile.from_dict(TUTORIAL_PROFILE.to_dict()) == TUTORIAL_PROFILE

    def test_to_dict_emits_all_three_keys(self) -> None:
        assert set(EMPTY_PROFILE.to_dict()) == {
            "coaching",
            "advisor_checkpoints",
            "bookends",
        }

    def test_from_dict_rejects_missing_key(self) -> None:
        d = TUTORIAL_PROFILE.to_dict()
        del d["advisor_checkpoints"]
        with pytest.raises(InvariantError, match=r"WorkflowProfile\.from_dict"):
            WorkflowProfile.from_dict(d)

    def test_from_dict_uses_direct_key_not_get_default(self) -> None:
        # An empty dict must crash, never silently fabricate a profile.
        with pytest.raises(InvariantError, match=r"WorkflowProfile\.from_dict"):
            WorkflowProfile.from_dict({})

    def test_from_dict_rejects_unknown_key(self) -> None:
        # A forked/tampered blob with an injected field must be rejected, not
        # silently ignored - the closed schema is the tamper boundary.
        d = {**TUTORIAL_PROFILE.to_dict(), "stages": ["smuggled"]}
        with pytest.raises(InvariantError, match=r"unexpected keys"):
            WorkflowProfile.from_dict(d)

    def test_from_dict_rejects_non_bool_advisor_checkpoints(self) -> None:
        # The JSON string "false" is TRUTHY - a present-but-mistyped gate must
        # raise, never silently flip the server-owned advisor gate.
        d = {**TUTORIAL_PROFILE.to_dict(), "advisor_checkpoints": "false"}
        with pytest.raises(InvariantError, match=r"advisor_checkpoints must be bool"):
            WorkflowProfile.from_dict(d)

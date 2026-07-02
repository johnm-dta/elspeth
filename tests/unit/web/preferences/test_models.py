"""Tests for the ComposerPreferences Pydantic models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.preferences.models import (
    ComposerPreferences,
    UpdateComposerPreferencesRequest,
)


def test_composer_preferences_valid() -> None:
    """A well-formed payload constructs cleanly."""
    payload = ComposerPreferences(
        default_mode="guided",
        banner_dismissed_at=None,
        tutorial_completed_at=None,
        tutorial_stage=None,
        tutorial_session_id=None,
        tutorial_run_id=None,
        tutorial_source_data_hash=None,
        updated_at=datetime.now(UTC),
    )
    assert payload.default_mode == "guided"
    assert payload.tutorial_completed_at is None
    assert payload.tutorial_stage is None


def test_composer_preferences_rejects_invalid_mode() -> None:
    """Tier-3 boundary: only 'guided' or 'freeform' are accepted."""
    with pytest.raises(ValidationError):
        ComposerPreferences(
            default_mode="kiosk",  # type: ignore[arg-type]
            banner_dismissed_at=None,
            tutorial_completed_at=None,
            tutorial_stage=None,
            tutorial_session_id=None,
            tutorial_run_id=None,
            tutorial_source_data_hash=None,
            updated_at=datetime.now(UTC),
        )


def test_update_request_accepts_full_payload() -> None:
    payload = UpdateComposerPreferencesRequest(default_mode="freeform")
    assert payload.default_mode == "freeform"
    assert payload.banner_dismissed_at is None
    assert payload.tutorial_completed_at is None


def test_update_request_accepts_only_banner_field() -> None:
    """Partial PATCH: caller sets only banner_dismissed_at."""
    stamp = datetime.now(UTC)
    payload = UpdateComposerPreferencesRequest(banner_dismissed_at=stamp)
    assert payload.default_mode is None
    assert payload.banner_dismissed_at == stamp


def test_update_request_rejects_invalid_mode() -> None:
    with pytest.raises(ValidationError):
        UpdateComposerPreferencesRequest(default_mode="kiosk")  # type: ignore[arg-type]


def test_update_request_accepts_empty_payload_as_noop() -> None:
    """An empty PATCH payload is a no-op; the request succeeds without changes."""
    payload = UpdateComposerPreferencesRequest()
    assert payload.default_mode is None
    assert payload.banner_dismissed_at is None
    assert payload.tutorial_completed_at is None


def test_composer_preferences_accepts_tutorial_completed_at() -> None:
    stamp = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    payload = ComposerPreferences(
        default_mode="guided",
        banner_dismissed_at=None,
        tutorial_completed_at=stamp,
        tutorial_stage=None,
        tutorial_session_id=None,
        tutorial_run_id=None,
        tutorial_source_data_hash=None,
        updated_at=datetime.now(UTC),
    )
    assert payload.tutorial_completed_at == stamp


def test_update_request_accepts_tutorial_completed_at() -> None:
    stamp = datetime(2026, 5, 15, 12, 0, tzinfo=UTC)
    payload = UpdateComposerPreferencesRequest(tutorial_completed_at=stamp)
    assert payload.default_mode is None
    assert payload.banner_dismissed_at is None
    assert payload.tutorial_completed_at == stamp
    assert "tutorial_completed_at" in payload.model_fields_set


def test_update_request_distinguishes_absent_from_explicit_null_tutorial() -> None:
    absent = UpdateComposerPreferencesRequest()
    explicit_null = UpdateComposerPreferencesRequest(tutorial_completed_at=None)

    assert "tutorial_completed_at" not in absent.model_fields_set
    assert "tutorial_completed_at" in explicit_null.model_fields_set


def test_update_request_rejects_unknown_field() -> None:
    """extra='forbid': a typo in the field name surfaces as ValidationError, not silent no-op."""
    with pytest.raises(ValidationError):
        UpdateComposerPreferencesRequest(default_modd="freeform")  # type: ignore[call-arg]


def test_composer_preferences_rejects_unknown_field() -> None:
    """extra='forbid' on the response model too (codebase convention)."""
    with pytest.raises(ValidationError):
        ComposerPreferences(
            default_mode="guided",
            banner_dismissed_at=None,
            tutorial_completed_at=None,
            tutorial_stage=None,
            tutorial_session_id=None,
            tutorial_run_id=None,
            tutorial_source_data_hash=None,
            updated_at=datetime.now(UTC),
            extra_key="boom",  # type: ignore[call-arg]
        )


def test_composer_preferences_rejects_invalid_tutorial_stage() -> None:
    """Tier-3 boundary: only the closed TutorialStage set (or None) is accepted;
    'welcome' is deliberately outside the set — it is never persisted."""
    with pytest.raises(ValidationError):
        ComposerPreferences(
            default_mode="guided",
            banner_dismissed_at=None,
            tutorial_completed_at=None,
            tutorial_stage="welcome",  # type: ignore[arg-type]
            tutorial_session_id="sess-1",
            tutorial_run_id=None,
            tutorial_source_data_hash=None,
            updated_at=datetime.now(UTC),
        )


def test_update_request_accepts_tutorial_progress_fields() -> None:
    payload = UpdateComposerPreferencesRequest(
        tutorial_stage="guided",
        tutorial_session_id="sess-1",
    )
    assert payload.tutorial_stage == "guided"
    assert payload.tutorial_session_id == "sess-1"
    assert "tutorial_stage" in payload.model_fields_set
    assert "tutorial_run_id" not in payload.model_fields_set


def test_update_request_rejects_invalid_tutorial_stage() -> None:
    with pytest.raises(ValidationError):
        UpdateComposerPreferencesRequest(tutorial_stage="welcome")  # type: ignore[arg-type]


def test_update_request_distinguishes_absent_from_explicit_null_stage() -> None:
    absent = UpdateComposerPreferencesRequest()
    explicit_null = UpdateComposerPreferencesRequest(tutorial_stage=None)

    assert "tutorial_stage" not in absent.model_fields_set
    assert "tutorial_stage" in explicit_null.model_fields_set

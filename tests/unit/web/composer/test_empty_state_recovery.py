"""Unit tests for §7.6 Option C empty-state recovery tracker.

The tracker is a per-compose-call single-fire guard. These tests pin the
contract independently of the service.py wiring; integration tests in
test_service.py cover the trigger predicate + loop re-entry semantics.
"""

from __future__ import annotations

from elspeth.web.composer.empty_state_recovery import (
    _RECOVERY_NUDGE_CONTENT,
    EmptyStateRecoveryTracker,
)


def test_initial_state_has_not_fired() -> None:
    tracker = EmptyStateRecoveryTracker()
    assert tracker.has_fired() is False


def test_record_fire_flips_state() -> None:
    tracker = EmptyStateRecoveryTracker()
    tracker.record_fire()
    assert tracker.has_fired() is True


def test_record_fire_is_idempotent() -> None:
    """Calling record_fire twice is safe — still fired, no exception."""
    tracker = EmptyStateRecoveryTracker()
    tracker.record_fire()
    tracker.record_fire()
    assert tracker.has_fired() is True


def test_two_trackers_are_independent() -> None:
    """Per-compose-call instances must not share state."""
    a = EmptyStateRecoveryTracker()
    b = EmptyStateRecoveryTracker()
    a.record_fire()
    assert a.has_fired() is True
    assert b.has_fired() is False


def test_nudge_content_carries_stable_marker() -> None:
    """The marker is the load-bearing contract for cohort scoring and
    audit-DB queries — pin it at the test layer."""
    assert "[ELSPETH-RECOVERY-NUDGE]" in _RECOVERY_NUDGE_CONTENT


def test_nudge_content_describes_minimal_shape() -> None:
    """Nudge guides the model toward a minimal-shape pipeline; the design
    rejected plugin-specific content as not generalising. These tokens
    pin the GENERIC framing so a future edit can't silently drift to
    plugin-specific without being noticed in review."""
    text = _RECOVERY_NUDGE_CONTENT.lower()
    assert "minimal" in text
    assert "source" in text
    assert "transform" in text or "pass-through" in text
    assert "sink" in text or "output" in text
    assert "preview_pipeline" in _RECOVERY_NUDGE_CONTENT

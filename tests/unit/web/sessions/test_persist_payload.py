"""Tests for the persist-payload dataclasses (spec §5.2.1)."""

from __future__ import annotations

import pytest

from elspeth.web.sessions._persist_payload import (
    AuditOutcome,
    RedactedToolRow,
    StatePayload,
)


def test_audit_outcome_success_shape():
    """Success: assistant_id set, no unwind failure."""
    outcome = AuditOutcome(
        assistant_id="abc",
        unwind_audit_failed=False,
    )
    assert outcome.assistant_id == "abc"
    assert outcome.unwind_audit_failed is False
    assert outcome.current_state_id is None


def test_audit_outcome_unwind_failure_shape():
    """Tool failed AND audit unwind failed: assistant_id=None,
    flag set. Caller will raise the captured plugin crash."""
    outcome = AuditOutcome(
        assistant_id=None,
        unwind_audit_failed=True,
    )
    assert outcome.assistant_id is None
    assert outcome.unwind_audit_failed is True
    assert outcome.current_state_id is None


def test_audit_outcome_rejects_ambiguous_shape():
    """assistant_id=set + unwind_audit_failed=True is contradictory:
    the unwind path runs only when the tool already failed, so no
    assistant message could have been produced. The dataclass rejects
    the combination at construction time."""
    with pytest.raises(ValueError, match="incompatible"):
        AuditOutcome(
            assistant_id="abc",  # produced by a successful path
            unwind_audit_failed=True,  # claimed by an unwind path
        )


def test_audit_outcome_no_tier1_violation_field():
    """Sanity: the tier1_violation flag-return path was deleted in
    Stage 4 of the plan revision; persist_compose_turn now raises
    AuditIntegrityError directly. Closes finding H1."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(AuditOutcome)}
    assert fields == {"assistant_id", "unwind_audit_failed", "current_state_id"}


def test_redacted_tool_row_with_state_advance():
    from elspeth.web.sessions.protocol import CompositionStateData

    row = RedactedToolRow(
        tool_call_id="tc_1",
        content='{"ok": true}',
        composition_state_payload=StatePayload(
            # B1 (Phase 1 plan-review synthesis): no ``version=``.
            # Version is allocated inside _session_write_lock by
            # ``_insert_composition_state`` (Task 10), not supplied by
            # the caller. Removing the field at the dataclass level
            # forecloses the dual-allocator race that fabricated
            # Tier-1 violations on contention loss.
            data=CompositionStateData(
                source={"kind": "test"},
                nodes=[],
                edges=[],
                outputs=[],
                metadata_={},
                is_valid=True,
                validation_errors=None,
            ),
            derived_from_state_id="prev_state_id",
        ),
    )
    assert row.composition_state_payload is not None
    assert row.composition_state_payload.data.is_valid is True
    assert row.composition_state_payload.derived_from_state_id == "prev_state_id"


def test_state_payload_has_no_version_field():
    """B1 (Phase 1 plan-review synthesis): ``StatePayload`` MUST NOT
    carry a caller-supplied ``version`` field. Version is allocated
    inside _session_write_lock by
    ``_insert_composition_state`` via
    ``SELECT COALESCE(MAX(version), 0) + 1 FROM composition_states
    WHERE session_id = :sid`` (Task 10).

    Pre-B1 the field existed; the compose loop in Phase 3 read
    ``MAX(version)`` outside the lock and dispatched a precomputed
    version into the locked helper. Two concurrent allocators could
    both compute ``MAX+1``; the loser's INSERT triggered
    ``uq_composition_state_version`` → ``IntegrityError`` → the
    locked-path handler incremented ``tool_row_integrity_violation_total``
    on what was structurally a contention loss, fabricating a Tier-1
    audit-integrity violation. SLO threshold is 0; the alert fires on a
    non-event.

    The fix is structural — version simply isn't a payload field — so
    no new caller can reintroduce the race by accident. This test pins
    the contract so a refactor that re-adds the field fails fast."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(StatePayload)}
    assert "version" not in fields, (
        "B1 regression: StatePayload must not carry a caller-supplied "
        "version field — version is allocated by "
        "_insert_composition_state under _session_write_lock"
    )
    assert fields == {"data", "derived_from_state_id"}, f"unexpected StatePayload fields: {fields}"

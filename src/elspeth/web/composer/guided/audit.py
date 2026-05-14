"""Guided-mode audit event emission.

Per Errata C4: the four event types (turn_emitted, turn_answered,
step_advanced, dropped_to_freeform) are recorded as
:class:`~elspeth.contracts.composer_audit.ComposerToolInvocation` records
with a ``tool_name`` discriminator. No new audit primitive, no schema
migration, no ``record_guided_event()`` method.

Tier 1 (audit-trust). Coercion forbidden — invalid input crashes.
Events go through the existing recorder via ``recorder.record(invocation)``.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §9.1.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolRecorder,
    ComposerToolStatus,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalReason

# Closed allowlists for discriminator fields. Checked before construction so
# the audit trail never records a record with an invalid discriminator value.
_VALID_EMITTERS: frozenset[str] = frozenset({"server", "llm"})
_VALID_ADVANCE_REASONS: frozenset[str] = frozenset({"recipe_applied", "user_advanced", "auto_advanced"})


def _build_invocation(
    *,
    tool_name: str,
    payload: Mapping[str, Any],
    composition_version: int,
    actor: str,
    now: datetime,
) -> ComposerToolInvocation:
    """Construct a ComposerToolInvocation for a synthetic guided event.

    Guided events are observations — there is no result payload (the
    ``arguments_canonical`` IS the record). ``result_canonical`` and
    ``result_hash`` are therefore ``None`` on a SUCCESS record, which
    deviates from the typical pattern (where None signals incomplete
    dispatch). The deviation is deliberate per Errata C4: these events
    have no LLM tool-call result; they are audit observations, not
    tool invocations in the traditional sense.

    ``version_before == version_after``: guided events observe state
    but do not mutate it. ``latency_ms == 0``: synthetic events have
    no measurable processing latency. ``started_at == finished_at``
    captures the wall-clock instant of emission.
    """
    args_canonical = canonical_json(payload)
    args_hash = stable_hash(payload)
    return ComposerToolInvocation(
        tool_call_id=uuid.uuid4().hex,
        tool_name=tool_name,
        arguments_canonical=args_canonical,
        arguments_hash=args_hash,
        result_canonical=None,
        result_hash=None,
        status=ComposerToolStatus.SUCCESS,
        error_class=None,
        error_message=None,
        version_before=composition_version,
        version_after=composition_version,
        started_at=now,
        finished_at=now,
        latency_ms=0,
        actor=actor,
    )


def emit_turn_emitted(
    recorder: ComposerToolRecorder,
    *,
    step: GuidedStep,
    turn_type: TurnType,
    payload_hash: str,
    payload_payload_id: str,
    emitter: str,
    composition_version: int,
    actor: str,
) -> None:
    """Record a ``guided_turn_emitted`` audit event.

    Fires when the guided-mode server (or LLM acting within the guided
    protocol) emits a turn to the user.

    Args:
        recorder: The composition session's active recorder.
        step: The wizard step at which the turn is emitted.
        turn_type: The taxonomy type of the turn payload.
        payload_hash: SHA-256 hex digest of the turn payload blob.
        payload_payload_id: Payload-store ID of the turn payload blob.
        emitter: Who constructed this turn — ``"server"`` or ``"llm"``.
        composition_version: Current ``CompositionState.version``.
        actor: Stable identity of the driving actor.

    Raises:
        ValueError: If ``emitter`` is not one of ``{"server", "llm"}``.
    """
    if emitter not in _VALID_EMITTERS:
        raise ValueError(f"emitter must be one of {sorted(_VALID_EMITTERS)}, got {emitter!r}")
    payload: dict[str, Any] = {
        "step_index": step.value,
        "turn_type": turn_type.value,
        "payload_hash": payload_hash,
        "payload_payload_id": payload_payload_id,
        "emitter": emitter,
    }
    now = datetime.now(UTC)
    invocation = _build_invocation(
        tool_name="guided_turn_emitted",
        payload=payload,
        composition_version=composition_version,
        actor=actor,
        now=now,
    )
    recorder.record(invocation)


def emit_turn_answered(
    recorder: ComposerToolRecorder,
    *,
    step: GuidedStep,
    turn_type: TurnType,
    response_hash: str,
    response_payload_id: str,
    control_signal: str | None,
    composition_version: int,
    actor: str,
) -> None:
    """Record a ``guided_turn_answered`` audit event.

    Fires when the user submits a response to a guided turn.

    Args:
        recorder: The composition session's active recorder.
        step: The wizard step at which the turn was answered.
        turn_type: The taxonomy type of the answered turn.
        response_hash: SHA-256 hex digest of the user's response payload.
        response_payload_id: Payload-store ID of the response payload.
        control_signal: Out-of-band signal if the user sent one; ``None``
            otherwise. Absent from the canonical record when ``None`` so
            the absence itself is unambiguous.
        composition_version: Current ``CompositionState.version``.
        actor: Stable identity of the driving actor.
    """
    payload: dict[str, Any] = {
        "step_index": step.value,
        "turn_type": turn_type.value,
        "response_hash": response_hash,
        "response_payload_id": response_payload_id,
    }
    if control_signal is not None:
        payload["control_signal"] = control_signal
    now = datetime.now(UTC)
    invocation = _build_invocation(
        tool_name="guided_turn_answered",
        payload=payload,
        composition_version=composition_version,
        actor=actor,
        now=now,
    )
    recorder.record(invocation)


def emit_step_advanced(
    recorder: ComposerToolRecorder,
    *,
    prev: GuidedStep,
    next_: GuidedStep,
    reason: str,
    composition_version: int,
    actor: str,
) -> None:
    """Record a ``guided_step_advanced`` audit event.

    Fires when the wizard advances from one step to the next.

    Args:
        recorder: The composition session's active recorder.
        prev: The step the wizard is leaving.
        next_: The step the wizard is entering.
        reason: Why the advance happened. Must be one of
            ``{"recipe_applied", "user_advanced", "auto_advanced"}``.
        composition_version: Current ``CompositionState.version``.
        actor: Stable identity of the driving actor.

    Raises:
        ValueError: If ``reason`` is not in the closed allowlist.
    """
    if reason not in _VALID_ADVANCE_REASONS:
        raise ValueError(f"reason must be one of {sorted(_VALID_ADVANCE_REASONS)}, got {reason!r}")
    payload: dict[str, Any] = {
        "prev_step": prev.value,
        "next_step": next_.value,
        "reason": reason,
    }
    now = datetime.now(UTC)
    invocation = _build_invocation(
        tool_name="guided_step_advanced",
        payload=payload,
        composition_version=composition_version,
        actor=actor,
        now=now,
    )
    recorder.record(invocation)


def emit_dropped_to_freeform(
    recorder: ComposerToolRecorder,
    *,
    prev: GuidedStep,
    drop_reason: TerminalReason,
    validation_result: Mapping[str, Any] | None,
    composition_version: int,
    actor: str,
) -> None:
    """Record a ``guided_dropped_to_freeform`` audit event.

    Fires when the guided wizard exits to freeform composition, for any
    reason (user exit, protocol violation, solver exhaustion).

    Args:
        recorder: The composition session's active recorder.
        prev: The step at which the wizard exited.
        drop_reason: Why guided mode was abandoned.
        validation_result: Pipeline validation output at exit time, when
            present (e.g. for SOLVER_EXHAUSTED). ``None`` is recorded as
            absent — the absence is meaningful (no validation was run),
            not a gap to fill with a fabricated default.
        composition_version: Current ``CompositionState.version``.
        actor: Stable identity of the driving actor.
    """
    payload: dict[str, Any] = {
        "prev_step": prev.value,
        "drop_reason": drop_reason.value,
    }
    if validation_result is not None:
        payload["validation_result"] = dict(validation_result)
    now = datetime.now(UTC)
    invocation = _build_invocation(
        tool_name="guided_dropped_to_freeform",
        payload=payload,
        composition_version=composition_version,
        actor=actor,
        now=now,
    )
    recorder.record(invocation)


def emit_hidden_field_rejected(
    recorder: ComposerToolRecorder,
    *,
    session_id: str,
    plugin_kind: str,
    plugin_name: str,
    field: str,
    predicate: Mapping[str, Any],
    actual_state: Mapping[str, Any],
    composition_version: int,
    actor: str,
) -> None:
    """Record a ``guided_hidden_field_rejected`` audit event.

    Fires before returning HTTP 400 when a schema-form submission includes a
    field hidden by its ``visible_when`` predicate.
    """
    payload: dict[str, Any] = {
        "session_id": session_id,
        "plugin_kind": plugin_kind,
        "plugin_name": plugin_name,
        "field": field,
        "predicate": dict(predicate),
        "actual_state": dict(actual_state),
    }
    now = datetime.now(UTC)
    invocation = _build_invocation(
        tool_name="guided_hidden_field_rejected",
        payload=payload,
        composition_version=composition_version,
        actor=actor,
        now=now,
    )
    recorder.record(invocation)

"""Tests for guided-mode audit emit helpers.

Audit-tier (Tier 1) per CLAUDE.md. Coercion forbidden — every field is
either present or the function raises.

Per Errata C4: the four event types are recorded as ComposerToolInvocation
records with a tool_name discriminator, NOT via GuidedAuditEvent.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from elspeth.contracts.composer_audit import ComposerToolInvocation
from elspeth.web.composer.guided.audit import (
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
)
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalReason


class _FakeRecorder:
    """Captures ComposerToolInvocation records without DB side effects.

    Implements the ComposerToolRecorder protocol structurally (both
    `record` and `resolve_session`). The `record_llm_call` assertion guard
    verifies guided events never route through the LLM-call path.
    """

    def __init__(self) -> None:
        self.invocations: list[ComposerToolInvocation] = []

    def record(self, invocation: ComposerToolInvocation) -> None:
        self.invocations.append(invocation)

    def resolve_session(self, session_id: str) -> None:
        pass  # no-op for in-memory fake; satisfies the Protocol

    def record_llm_call(self, *_: object, **__: object) -> None:
        raise AssertionError("guided events must not route through record_llm_call")


class TestEmitTurnEmitted:
    def test_records_step_and_type(self) -> None:
        rec = _FakeRecorder()
        emit_turn_emitted(
            rec,
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc",
            payload_payload_id="payload-1",
            emitter="server",
            composition_version=1,
            actor="test-actor",
        )
        assert len(rec.invocations) == 1
        inv = rec.invocations[0]
        assert inv.tool_name == "guided_turn_emitted"
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        assert decoded["step_index"] == "step_1_source"
        assert inv.actor == "test-actor"
        assert inv.version_before == inv.version_after
        assert inv.result_canonical is None
        assert inv.result_hash is None

    def test_invalid_emitter_raises(self) -> None:
        rec = _FakeRecorder()
        with pytest.raises(ValueError, match="emitter"):
            emit_turn_emitted(
                rec,
                step=GuidedStep.STEP_1_SOURCE,
                turn_type=TurnType.SINGLE_SELECT,
                payload_hash="abc",
                payload_payload_id="payload-1",
                emitter="robot",  # invalid
                composition_version=1,
                actor="test-actor",
            )
        assert len(rec.invocations) == 0  # nothing recorded on validation failure


class TestEmitTurnAnswered:
    def test_records_response_hash(self) -> None:
        rec = _FakeRecorder()
        emit_turn_answered(
            rec,
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            response_hash="xyz",
            response_payload_id="payload-2",
            control_signal=None,
            composition_version=2,
            actor="test-actor",
        )
        assert len(rec.invocations) == 1
        inv = rec.invocations[0]
        assert inv.tool_name == "guided_turn_answered"
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        assert decoded["response_hash"] == "xyz"
        assert "validation_result" not in decoded  # no spurious keys
        assert inv.version_before == inv.version_after == 2
        assert inv.result_canonical is None
        assert inv.result_hash is None


class TestEmitStepAdvanced:
    def test_records_prev_and_next(self) -> None:
        rec = _FakeRecorder()
        emit_step_advanced(
            rec,
            prev=GuidedStep.STEP_1_SOURCE,
            next_=GuidedStep.STEP_2_SINK,
            reason="user_advanced",
            composition_version=3,
            actor="test-actor",
        )
        assert len(rec.invocations) == 1
        inv = rec.invocations[0]
        assert inv.tool_name == "guided_step_advanced"
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        assert decoded["prev_step"] == "step_1_source"
        assert decoded["next_step"] == "step_2_sink"
        assert decoded["reason"] == "user_advanced"
        assert inv.version_before == inv.version_after
        assert inv.result_canonical is None

    def test_invalid_reason_raises(self) -> None:
        rec = _FakeRecorder()
        with pytest.raises(ValueError, match="reason"):
            emit_step_advanced(
                rec,
                prev=GuidedStep.STEP_1_SOURCE,
                next_=GuidedStep.STEP_2_SINK,
                reason="teleported",  # invalid
                composition_version=1,
                actor="test-actor",
            )
        assert len(rec.invocations) == 0


class TestEmitDroppedToFreeform:
    def test_records_drop_reason(self) -> None:
        rec = _FakeRecorder()
        emit_dropped_to_freeform(
            rec,
            prev=GuidedStep.STEP_3_TRANSFORMS,
            drop_reason=TerminalReason.SOLVER_EXHAUSTED,
            # Realistic shape: ValidationSummary -> {"is_valid", "errors":[ValidationEntry.to_dict()]}.
            validation_result={
                "is_valid": False,
                "errors": [{"component": "source", "message": "schema mismatch", "severity": "high"}],
            },
            composition_version=5,
            actor="test-actor",
        )
        assert len(rec.invocations) == 1
        inv = rec.invocations[0]
        assert inv.tool_name == "guided_dropped_to_freeform"
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        assert decoded["drop_reason"] == "solver_exhausted"
        assert decoded["prev_step"] == "step_3_transforms"
        # Structured outcome retained; free-form ``message`` stripped by allowlist.
        assert decoded["validation_result"] == {
            "is_valid": False,
            "errors": [{"component": "source", "severity": "high"}],
        }
        assert inv.result_canonical is None

    def test_redacts_validation_error_message_egress(self) -> None:
        """Free-form validator ``message`` text MUST NOT reach the audit record.

        The guided-event channel is structurally exempt from the redaction
        MANIFEST (the persistence projection at sessions/routes/_helpers.py
        fail-opens for non-MANIFEST tool_names), so this synthetic event must
        be safe by construction. ``ValidationEntry.message`` echoes filesystem
        paths and raw plugin/pydantic exception text (see _common.py path / exc
        messages); only the structured outcome (``is_valid`` + per-error
        ``component`` / ``severity``) is safe to persist.
        """
        rec = _FakeRecorder()
        emit_dropped_to_freeform(
            rec,
            prev=GuidedStep.STEP_3_TRANSFORMS,
            drop_reason=TerminalReason.SOLVER_EXHAUSTED,
            validation_result={
                "is_valid": False,
                "errors": [
                    {
                        "component": "source",
                        "message": "Path violation (S2): '/etc/secrets/db.pem' is outside the data dir",
                        "severity": "high",
                    },
                    {
                        "component": "node:enrich",
                        "message": "Invalid options for transform 'llm': api_key=sk-LEAKED-7f3a unauthorized",
                        "severity": "high",
                    },
                ],
            },
            composition_version=5,
            actor="test-actor",
        )
        inv = rec.invocations[0]
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        # Structured outcome survives; component attribution + severity kept.
        assert decoded["validation_result"] == {
            "is_valid": False,
            "errors": [
                {"component": "source", "severity": "high"},
                {"component": "node:enrich", "severity": "high"},
            ],
        }
        # The whole canonical record (hashed into the Tier-1 audit trail) is
        # free of the sensitive free-form text.
        blob = inv.arguments_canonical
        for leaked in ("/etc/secrets/db.pem", "sk-LEAKED-7f3a", "Path violation", "Invalid options"):
            assert leaked not in blob, leaked

    def test_user_pressed_exit_has_no_validation_result(self) -> None:
        rec = _FakeRecorder()
        emit_dropped_to_freeform(
            rec,
            prev=GuidedStep.STEP_2_SINK,
            drop_reason=TerminalReason.USER_PRESSED_EXIT,
            validation_result=None,
            composition_version=1,
            actor="test-actor",
        )
        inv = rec.invocations[0]
        decoded: dict[str, Any] = json.loads(inv.arguments_canonical)
        # validation_result must be absent (None means "never sent"), not present-as-null
        assert "validation_result" not in decoded


class TestArgumentsHashInvariant:
    """Tier-1 audit invariant: arguments_hash == sha256(arguments_canonical)."""

    def test_arguments_hash_matches_canonical(self) -> None:
        rec = _FakeRecorder()
        emit_turn_emitted(
            rec,
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="deadbeef",
            payload_payload_id="payload-99",
            emitter="llm",
            composition_version=7,
            actor="audit-verifier",
        )
        inv = rec.invocations[0]
        # Reproduce the hash independently using only hashlib — no canonical_json import.
        # This ensures the test doesn't tautologically depend on the same helper.
        expected_hash = hashlib.sha256(inv.arguments_canonical.encode("utf-8")).hexdigest()
        assert inv.arguments_hash == expected_hash

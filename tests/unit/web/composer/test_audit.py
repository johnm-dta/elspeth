"""Unit tests for the web composer's BufferingRecorder + audit envelope helper.

Pin the invariants:
- record() / invocations snapshot semantics (mutation after snapshot doesn't affect snapshot)
- audit_envelope produces the correct {_kind: "audit", invocation: {...}} shape
- thread-safe append under concurrent record() calls
- resolve_session is a no-op (Protocol conformance)
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.web.composer.audit import BufferingRecorder, audit_envelope, begin_dispatch, dispatch_with_audit


def _make_invocation(seq: int) -> ComposerToolInvocation:
    payload = {"seq": seq}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    t = datetime(2026, 5, 4, 12, 0, seq % 60, tzinfo=UTC)
    return ComposerToolInvocation(
        tool_call_id=f"tc-{seq}",
        tool_name="upsert_node",
        arguments_canonical=canon,
        arguments_hash=h,
        result_canonical=canon,
        result_hash=h,
        status=ComposerToolStatus.SUCCESS,
        error_class=None,
        error_message=None,
        version_before=seq,
        version_after=seq + 1,
        started_at=t,
        finished_at=t,
        latency_ms=1,
        actor="test",
    )


def test_record_appends_in_order() -> None:
    rec = BufferingRecorder()
    for i in range(5):
        rec.record(_make_invocation(i))
    invs = rec.invocations
    assert len(invs) == 5
    assert [inv.tool_call_id for inv in invs] == [f"tc-{i}" for i in range(5)]


def test_invocations_returns_immutable_snapshot() -> None:
    """A snapshot taken via .invocations should not reflect later mutations."""
    rec = BufferingRecorder()
    rec.record(_make_invocation(0))
    snapshot = rec.invocations
    rec.record(_make_invocation(1))
    rec.record(_make_invocation(2))
    # Original snapshot unchanged.
    assert len(snapshot) == 1
    assert snapshot[0].tool_call_id == "tc-0"
    # New snapshot reflects later state.
    assert len(rec.invocations) == 3


def test_concurrent_records_all_landed() -> None:
    """Thread-safe append: 100 concurrent records land all 100 entries."""
    rec = BufferingRecorder()
    threads = [threading.Thread(target=rec.record, args=(_make_invocation(i),)) for i in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(rec.invocations) == 100


def test_resolve_session_is_no_op() -> None:
    """The web recorder's session_id is known up front; resolve is no-op."""
    rec = BufferingRecorder()
    # Should not raise; returns None.
    result = rec.resolve_session("abc123def456")
    assert result is None
    # No record was created by resolve.
    assert rec.invocations == ()


def test_audit_envelope_kind_discriminator() -> None:
    """Every audit envelope must carry the _kind: 'audit' discriminator."""
    inv = _make_invocation(1)
    env = audit_envelope(inv)
    assert env["_kind"] == "audit"
    assert "invocation" in env
    inv_payload = env["invocation"]
    assert isinstance(inv_payload, dict)
    assert inv_payload["tool_call_id"] == "tc-1"
    assert inv_payload["status"] == "success"


def test_audit_envelope_invocation_is_json_serializable() -> None:
    """The envelope must round-trip through json.dumps without raising."""
    inv = _make_invocation(1)
    env = audit_envelope(inv)
    serialized = json.dumps(env)
    round_tripped = json.loads(serialized)
    assert round_tripped["_kind"] == "audit"
    assert round_tripped["invocation"]["tool_call_id"] == "tc-1"


@pytest.mark.asyncio
async def test_dispatch_with_audit_records_plugin_crash_when_success_payload_extraction_fails() -> None:
    """A bad successful result shape must not bypass the audit recorder."""

    @dataclass(frozen=True, slots=True)
    class _UpdatedState:
        version: int

    class _BadResult:
        updated_state = _UpdatedState(version=2)

        def to_dict(self) -> list[str]:
            return ["not", "a", "mapping"]

    async def _dispatch() -> _BadResult:
        return _BadResult()

    recorder = BufferingRecorder()
    audit = begin_dispatch(
        "tc-bad-result",
        "set_metadata",
        {"patch": {"name": "bad-result"}},
        version_before=1,
        actor="assistant",
    )

    with pytest.raises(TypeError, match=r"result.to_dict\(\) returned list"):
        await dispatch_with_audit(
            recorder=recorder,
            audit=audit,
            do_dispatch=_dispatch,
            version_after_provider=lambda result: result.updated_state.version,
            arg_error_payload_factory=lambda _exc: {"error": "unused"},
        )

    invocations = recorder.invocations
    assert len(invocations) == 1
    inv = invocations[0]
    assert inv.status == ComposerToolStatus.PLUGIN_CRASH
    assert inv.tool_call_id == "tc-bad-result"
    assert inv.error_class == "TypeError"
    assert inv.error_message == "TypeError"
    assert inv.version_before == 1
    assert inv.version_after is None

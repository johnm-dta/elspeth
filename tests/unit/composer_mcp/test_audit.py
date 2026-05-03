"""Unit tests for the JSONL events sidecar recorder + integrity verifier.

Pin the invariants reviewers identified:
- pre-resolution buffering and flush on first concrete session_id
- per-record fsync (durable on return)
- byte-integrity verifier (recomputed sha256 over stored canonical bytes)
- tampering with hash alone OR canonical alone is detected
- _drain_buffer_to holds buffer-lock across the disk write
- recorder.record() raising propagates (audit primacy: don't fail silently)
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.composer_mcp.audit import (
    JsonlEventRecorder,
    events_sidecar_path,
    verify_events_sidecar_integrity,
)
from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus


def _make_invocation(seq: int, *, args: dict | None = None) -> ComposerToolInvocation:
    payload = args if args is not None else {"seq": seq}
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


def test_pre_resolution_buffer_flushes_on_first_record(tmp_path: Path) -> None:
    """Records pushed before session_id resolves should land in the sidecar
    once the next record is made under a resolved session_id."""
    sid_holder: list[str | None] = [None]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    rec.record(_make_invocation(2))
    # Sidecar should not exist yet.
    assert not list(tmp_path.glob("*.events.jsonl"))
    sid_holder[0] = "abc123def456"
    rec.record(_make_invocation(3))
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    assert sidecar.exists()
    lines = sidecar.read_text().splitlines()
    assert len(lines) == 3
    # Order preserved: seq 1, 2, 3.
    parsed = [json.loads(line) for line in lines]
    assert [p["tool_call_id"] for p in parsed] == ["tc-1", "tc-2", "tc-3"]


def test_resolve_session_drains_buffer_without_new_record(tmp_path: Path) -> None:
    """When the LLM stops after new_session, resolve_session() must flush
    the pre-resolution buffer."""
    sid_holder: list[str | None] = [None]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    sid_holder[0] = "abc123def456"
    rec.resolve_session("abc123def456")
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    assert sidecar.exists()
    assert len(sidecar.read_text().splitlines()) == 1


def test_resolve_session_idempotent(tmp_path: Path) -> None:
    """Repeated resolve_session() with the same id must not duplicate records."""
    sid_holder: list[str | None] = ["abc123def456"]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    rec.resolve_session("abc123def456")
    rec.resolve_session("abc123def456")
    rec.resolve_session("abc123def456")
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    assert len(sidecar.read_text().splitlines()) == 1


def test_integrity_check_passes_unmodified_sidecar(tmp_path: Path) -> None:
    """Freshly-written records must pass byte-integrity verification."""
    sid_holder: list[str | None] = ["abc123def456"]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    for seq in range(5):
        rec.record(_make_invocation(seq))
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    verify_events_sidecar_integrity(sidecar)  # Must not raise.


def test_integrity_check_catches_canonical_tamper_without_rehash(tmp_path: Path) -> None:
    """Mutating arguments_canonical without recomputing the hash → ValueError."""
    sid_holder: list[str | None] = ["abc123def456"]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    line = sidecar.read_text().rstrip()
    obj = json.loads(line)
    obj["arguments_canonical"] = '{"_tampered":true}'  # Hash unchanged.
    sidecar.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="arguments_hash mismatch"):
        verify_events_sidecar_integrity(sidecar)


def test_integrity_check_catches_hash_tamper_alone(tmp_path: Path) -> None:
    """Mutating arguments_hash without rewriting the canonical → ValueError."""
    sid_holder: list[str | None] = ["abc123def456"]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    line = sidecar.read_text().rstrip()
    obj = json.loads(line)
    obj["arguments_hash"] = "deadbeef" * 8
    sidecar.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="arguments_hash mismatch"):
        verify_events_sidecar_integrity(sidecar)


def test_integrity_check_byte_integrity_only_documented_gap(tmp_path: Path) -> None:
    """Pin the documented narrow semantics: a consistent
    canonical+hash rewrite is NOT detected by this verifier (byte-integrity,
    not canonical-form integrity).

    If this test ever fails because the verifier was hardened, update the
    docstring and remove this xfail-ish assertion.
    """
    sid_holder: list[str | None] = ["abc123def456"]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    rec.record(_make_invocation(1))
    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    line = sidecar.read_text().rstrip()
    obj = json.loads(line)
    # Rewrite both canonical AND hash consistently.
    new_canonical = '{"_consistently_tampered":true}'
    obj["arguments_canonical"] = new_canonical
    obj["arguments_hash"] = hashlib.sha256(new_canonical.encode("utf-8")).hexdigest()
    sidecar.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")
    # Verifier passes — the bytes match the hash. The docstring says
    # this gap is acknowledged; this test pins it.
    verify_events_sidecar_integrity(sidecar)


def test_concurrent_records_preserve_order_under_resolved_session(tmp_path: Path) -> None:
    """Concurrent record() calls under a resolved session_id must serialize via
    _event_lock so each line is intact (no interleaving)."""
    sid = "abc123def456"
    rec = JsonlEventRecorder(tmp_path, lambda: sid)

    threads = [threading.Thread(target=rec.record, args=(_make_invocation(i),)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    sidecar = events_sidecar_path(tmp_path, sid)
    lines = sidecar.read_text().splitlines()
    assert len(lines) == 20
    # Each line should be valid JSON (no interleaved writes).
    for line in lines:
        json.loads(line)
    # Verify byte-integrity of every line.
    verify_events_sidecar_integrity(sidecar)


def test_drain_buffer_holds_lock_across_disk_write(tmp_path: Path) -> None:
    """Pre-resolution records, when drained concurrent with a fresh record,
    must land before the post-resolution record in file order.

    Tests the QA-review fix where _drain_buffer_to holds the buffer-lock
    across _append. Without that fix, a concurrent record could land
    BEFORE the drained batch.
    """
    sid_holder: list[str | None] = [None]
    rec = JsonlEventRecorder(tmp_path, lambda: sid_holder[0])
    # Buffer 10 records pre-resolution.
    for i in range(10):
        rec.record(_make_invocation(i))
    # Now resolve and concurrently emit a post-resolution record.
    sid_holder[0] = "abc123def456"
    barrier = threading.Barrier(2)

    def post_record() -> None:
        barrier.wait()
        rec.record(_make_invocation(99))

    def drain_resolve() -> None:
        barrier.wait()
        rec.resolve_session("abc123def456")

    t1 = threading.Thread(target=post_record)
    t2 = threading.Thread(target=drain_resolve)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    sidecar = events_sidecar_path(tmp_path, "abc123def456")
    lines = sidecar.read_text().splitlines()
    seqs = [json.loads(line)["version_before"] for line in lines]
    # Pre-resolution batch (0..9) must appear contiguously and the
    # post-resolution record (seq=99) must come AFTER all of them.
    pre_indices = [i for i, s in enumerate(seqs) if s < 90]
    post_indices = [i for i, s in enumerate(seqs) if s == 99]
    assert pre_indices == list(range(10))
    assert post_indices == [10]


def test_recorder_failure_propagates(tmp_path: Path) -> None:
    """If the recorder cannot write to disk (e.g. parent path is a regular
    file), the exception must propagate — silent failure violates audit
    primacy.
    """
    sid = "abc123def456"
    # Force the sidecar to a path we cannot create by writing a non-directory
    # file at the parent path so mkdir(parents=True) fails.
    bad_parent = tmp_path / "blocker"
    bad_parent.write_text("not a dir")
    rec = JsonlEventRecorder(bad_parent / "subdir", lambda: sid)
    with pytest.raises((NotADirectoryError, FileExistsError, OSError)):
        rec.record(_make_invocation(1))

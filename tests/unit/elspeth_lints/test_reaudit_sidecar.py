"""Round-trip tests for reaudit sidecar entry serialization.

Covers ``_entry_to_dict`` / ``_entry_from_dict`` — the AllowlistEntry
serialization boundary used to persist reaudit sidecars. v1 entries bind via
``file_fingerprint``; v2 entries bind via ``scope_fingerprint`` and carry a
``judge_signature_version``. Both must survive the round trip intact.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict
from elspeth_lints.core.reaudit_sidecar import _entry_from_dict, _entry_to_dict


def _roundtrip(entry: AllowlistEntry) -> AllowlistEntry:
    payload = _entry_to_dict(entry)
    return _entry_from_dict(payload, sidecar_path=Path("test.sidecar"), line_no=1)


def test_entry_dict_roundtrip_preserves_v2_scope_binding() -> None:
    """A v2 entry round-trips with scope_fingerprint and judge_signature_version."""
    scope_fp = "a" * 64
    entry = AllowlistEntry(
        key="web/x.py:R1:fn:fp=aa",
        owner="alice",
        reason="permitted boundary",
        safety="contained",
        expires=None,
        file_fingerprint=None,
        scope_fingerprint=scope_fp,
        judge_signature_version=2,
        ast_path="Module.body[0]",
        pattern=None,
        source_file="test.yaml",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model="some-model",
        judge_rationale="rationale",
        judge_confidence=None,
        judge_model_verdict=JudgeVerdict.ACCEPTED,
        judge_policy_hash="policyhash",
        judge_metadata_signature="hmac-sha256:v2:" + "0" * 64,
    )

    payload = _entry_to_dict(entry)
    assert payload["scope_fingerprint"] == scope_fp
    assert payload["judge_signature_version"] == 2
    assert payload["file_fingerprint"] is None

    restored = _entry_from_dict(payload, sidecar_path=Path("test.sidecar"), line_no=1)
    assert restored.scope_fingerprint == scope_fp
    assert restored.judge_signature_version == 2
    assert restored.file_fingerprint is None


def test_entry_dict_roundtrip_preserves_v1_file_binding() -> None:
    """A v1 entry (file_fingerprint, no scope_fingerprint/version) round-trips intact."""
    file_fp = "b" * 64
    entry = AllowlistEntry(
        key="web/x.py:R1:fn:fp=bb",
        owner="alice",
        reason="permitted boundary",
        safety="contained",
        expires=None,
        file_fingerprint=file_fp,
        scope_fingerprint=None,
        judge_signature_version=None,
        ast_path="Module.body[0]",
        pattern=None,
        source_file="test.yaml",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model="some-model",
        judge_rationale="rationale",
        judge_confidence=None,
        judge_model_verdict=JudgeVerdict.ACCEPTED,
        judge_policy_hash="policyhash",
        judge_metadata_signature="hmac-sha256:v1:" + "0" * 64,
    )

    restored = _roundtrip(entry)
    assert restored.file_fingerprint == file_fp
    assert restored.scope_fingerprint is None
    assert restored.judge_signature_version is None


def test_sidecar_round_trips_judge_transport() -> None:
    """The additive judge_transport field survives the sidecar round trip."""
    entry = AllowlistEntry(
        key="web/x.py:R1:fn:fp=aa",
        owner="alice",
        reason="permitted boundary",
        safety="contained",
        expires=None,
        file_fingerprint=None,
        scope_fingerprint="a" * 64,
        judge_signature_version=2,
        judge_transport="claude_agent_sdk",
        ast_path="Module.body[0]",
        pattern=None,
        source_file="test.yaml",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model="some-model",
        judge_rationale="rationale",
        judge_confidence=None,
        judge_model_verdict=JudgeVerdict.ACCEPTED,
        judge_policy_hash="policyhash",
        judge_metadata_signature="hmac-sha256:v2:" + "0" * 64,
    )

    restored = _roundtrip(entry)
    assert restored.judge_transport == "claude_agent_sdk"

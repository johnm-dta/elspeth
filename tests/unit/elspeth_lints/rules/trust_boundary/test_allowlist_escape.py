"""Allowlist escape tests for trust-boundary honesty gates."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.protocols import Finding
from elspeth_lints.rules.trust_boundary.scope import rule as scope_rule
from elspeth_lints.rules.trust_boundary.tests import rule as tests_rule
from elspeth_lints.rules.trust_boundary.tier import rule as tier_rule

_TEST_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"
_RECORDED_AT = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)
_POLICY_HASH = "sha256:" + ("ab" * 32)


def _write_source(root: Path, relative: str, body: str) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("from elspeth.contracts.trust_boundary import trust_boundary\n\n" + body.lstrip(), encoding="utf-8")
    return target


def _canonical_key(finding: Finding) -> str:
    key = finding.canonical_key
    return key() if callable(key) else key


def _write_signed_allowlist(
    allowlist_dir: Path,
    *,
    finding: Finding,
    source_file: Path,
    expires: date = date(2099, 1, 1),
) -> None:
    allowlist_dir.mkdir(parents=True, exist_ok=True)
    key = _canonical_key(finding)
    file_fingerprint = hashlib.sha256(source_file.read_bytes()).hexdigest()
    ast_path = getattr(finding, "ast_path", f"decorator:{finding.line}:{finding.column}")
    judge_rationale = "Synthetic judge accepted this exact honesty-gate false positive for test coverage."
    signature = compute_judge_metadata_signature(
        key=key,
        file_fingerprint=file_fingerprint,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=_RECORDED_AT,
        judge_model="test-judge",
        judge_rationale=judge_rationale,
        judge_policy_hash=_POLICY_HASH,
        hmac_key=_TEST_HMAC_KEY.encode("utf-8"),
    )
    (allowlist_dir / "honesty.yaml").write_text(
        "\n".join(
            [
                "allow_hits:",
                f"- key: {key}",
                "  owner: test-owner",
                "  reason: Synthetic trust-boundary false positive exemption.",
                "  safety: Exact signed fixture only; no per-file blanket rule.",
                f"  expires: {expires.isoformat()}",
                f"  file_fingerprint: {file_fingerprint}",
                f"  ast_path: {ast_path}",
                "  judge_verdict: ACCEPTED",
                f"  judge_recorded_at: {_RECORDED_AT.isoformat()}",
                "  judge_model: test-judge",
                f"  judge_rationale: {judge_rationale}",
                f"  judge_policy_hash: {_POLICY_HASH}",
                f"  judge_metadata_signature: '{signature}'",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_scope_honesty_gate_accepts_exact_signed_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "src" / "elspeth"
    source = _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=3,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return 42
""",
    )
    raw = scope_rule.scan_root(root)
    assert [finding.rule_id for finding in raw] == ["TBS2"]

    allowlist_dir = tmp_path / "allowlist"
    _write_signed_allowlist(allowlist_dir, finding=raw[0], source_file=source)
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_HMAC_KEY)

    assert scope_rule.scan_root(root, allowlist_dir_override=allowlist_dir) == []


def test_tests_honesty_gate_accepts_exact_signed_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    root = repo_root / "src" / "elspeth"
    source = _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=3,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return payload["x"]
""",
    )
    raw = tests_rule.scan_root(root, repo_root=repo_root)
    assert [finding.rule_id for finding in raw] == ["TBE1"]

    allowlist_dir = tmp_path / "allowlist"
    _write_signed_allowlist(allowlist_dir, finding=raw[0], source_file=source)
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_HMAC_KEY)

    assert tests_rule.scan_root(root, repo_root=repo_root, allowlist_dir_override=allowlist_dir) == []


def test_tier_honesty_gate_accepts_exact_signed_allowlist(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "src" / "elspeth"
    source = _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=2,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return payload["x"]
""",
    )
    raw = tier_rule.scan_root(root)
    assert [finding.rule_id for finding in raw] == ["TBT1"]

    allowlist_dir = tmp_path / "allowlist"
    _write_signed_allowlist(allowlist_dir, finding=raw[0], source_file=source)
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_HMAC_KEY)

    assert tier_rule.scan_root(root, allowlist_dir_override=allowlist_dir) == []


def test_trust_boundary_allowlist_rejects_pre_judge_entries(tmp_path: Path) -> None:
    root = tmp_path / "src" / "elspeth"
    source = _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=3,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return 42
""",
    )
    raw = scope_rule.scan_root(root)
    key = _canonical_key(raw[0])
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "honesty.yaml").write_text(
        "\n".join(
            [
                "allow_hits:",
                f"- key: {key}",
                "  owner: test-owner",
                "  reason: Unsigned pre-judge entries must not suppress honesty gates.",
                "  safety: This fixture intentionally lacks judge metadata.",
                "  expires: 2099-01-01",
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        scope_rule.scan_root(root, allowlist_dir_override=allowlist_dir)
    except ValueError as exc:
        assert "judge_verdict" in str(exc)
    else:
        raise AssertionError(f"{source}: unsigned trust-boundary allowlist entry was accepted")


def test_trust_boundary_allowlist_does_not_suppress_expired_exact_entry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "src" / "elspeth"
    source = _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=3,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return 42
""",
    )
    raw = scope_rule.scan_root(root)
    assert [finding.rule_id for finding in raw] == ["TBS2"]

    allowlist_dir = tmp_path / "allowlist"
    _write_signed_allowlist(allowlist_dir, finding=raw[0], source_file=source, expires=date(2020, 1, 1))
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_HMAC_KEY)

    findings = scope_rule.scan_root(root, allowlist_dir_override=allowlist_dir)

    assert [finding.rule_id for finding in findings] == [
        "TBS2",
        "allowlist.stale_entry",
        "allowlist.expired_entry",
    ]


def test_trust_boundary_allowlist_rejects_per_file_rules(tmp_path: Path) -> None:
    root = tmp_path / "src" / "elspeth"
    _write_source(
        root,
        "boundary.py",
        """
@trust_boundary(
    tier=3,
    source="external payload",
    source_param="payload",
    suppresses=("R1",),
    invariant="raises ValueError on malformed payload",
)
def handler(payload):
    return 42
""",
    )
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    (allowlist_dir / "honesty.yaml").write_text(
        "\n".join(
            [
                "per_file_rules:",
                "- pattern: '*.py'",
                "  rules:",
                "  - TBS2",
                "  reason: Blanket honesty-gate suppression is forbidden.",
                "  expires: 2099-01-01",
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        scope_rule.scan_root(root, allowlist_dir_override=allowlist_dir)
    except ValueError as exc:
        assert "per_file_rules" in str(exc)
    else:
        raise AssertionError("per-file trust-boundary allowlist rule was accepted")

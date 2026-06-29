"""Adversarial-boundary tests for the new-entry judge-coverage gate (C1).

Convergent panel finding C1: judge enforcement is voluntary in the
prototype because the loader treats all-``None`` ``judge_*`` fields
as honest pre-judge representation. An agent who hand-edits a YAML
entry without judge metadata produces output observationally
identical to a grandfathered pre-judge entry — the audit trail is
silently eroded.

The ``check_judge_coverage`` function closes this loop on a PR diff
against a baseline ref. These tests probe the boundaries of:

1. Rotation grandfathering: an entry whose fingerprint shifted in
   HEAD still matches its baseline counterpart by the
   ``(file, rule, symbol, owner, reason)`` discriminator. The
   policy is operator-confirmed (2026-05-23): rotation = mechanical
   rename, not a re-justification trigger.
2. Genuine new entries lacking judge fields surface as violations.
3. Genuine new entries WITH the atomic quartet are accepted.
4. Pre-judge entries that existed in baseline stay grandfathered
   (not flagged), but the SAME pre-judge entry shape added new in
   this PR is a violation (the gate's whole point).
5. Baseline-file-absent / directory-absent / git-rev-bad failure
   modes produce actionable diagnostics, not silent passes.
6. The ``allow_classes:`` and private-``entries:`` legacy shapes
   surface as ``UNRECOGNIZED_ENTRY_SHAPE`` rather than silently
   bypassing coverage enforcement.

Test discipline per M5: parameterise invariant violations first.
Each "happy path" assertion is paired with a "what breaks the
contract" probe (rotation policy is the canonical example —
verifying grandfathering is half the story; verifying that a
genuinely-different-owner entry is NOT grandfathered is the other
half).
"""

from __future__ import annotations

import inspect
import subprocess
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, compute_judge_metadata_signature
from elspeth_lints.core.judge import DEFAULT_JUDGE_MODEL, JUDGE_POLICY_HASH
from elspeth_lints.core.judge_coverage import (
    JUDGE_METADATA_MUTATED,
    PER_FILE_RULE_REQUIRES_JUDGE,
    UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC,
    JudgeCoverageError,
    JudgeCoverageReport,
    _discriminator,
    _git_show,
    _judge_binding_identity,
    _ls_tree_yaml_files,
    _missing_judge_fields,
    check_judge_coverage,
    check_one_directory,
)

_FAKE_JUDGE_METADATA_SIGNATURE = "hmac-sha256:v1:" + "0" * 64
_TEST_JUDGE_METADATA_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"


def _valid_v1_judge_signature(
    *,
    key: str,
    file_fingerprint: str,
    ast_path: str,
    recorded_at: datetime,
    rationale: str,
) -> str:
    return compute_judge_metadata_signature(
        key=key,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=recorded_at,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_rationale=rationale,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=1,
        file_fingerprint=file_fingerprint,
        hmac_key=_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"),
    )


def _valid_v2_judge_signature(
    *,
    key: str,
    scope_fingerprint: str,
    ast_path: str,
    recorded_at: datetime,
    rationale: str,
    judge_transport: str = "openrouter",
) -> str:
    return compute_judge_metadata_signature(
        key=key,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=recorded_at,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_rationale=rationale,
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=scope_fingerprint,
        judge_transport=judge_transport,
        hmac_key=_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"),
    )


def test_judge_gates_do_not_reintroduce_private_yaml_loaders() -> None:
    """C1/C3 gates share allowlist_io for YAML parsing and allow_hits iteration."""
    from elspeth_lints.core import judge_coverage, override_rate

    assert "_load_yaml_strict" not in vars(judge_coverage)
    assert "_load_yaml_strict" not in vars(override_rate)
    assert "yaml.safe_load" not in inspect.getsource(judge_coverage)
    assert "yaml.safe_load" not in inspect.getsource(override_rate)


# =========================================================================
# Discriminator: rotation policy unit tests (no git, no filesystem)
# =========================================================================


def _make_entry(
    *,
    key: str,
    owner: str = "alice",
    reason: str = "permitted boundary",
    expires=None,
    judge_verdict=None,
    judge_recorded_at=None,
    judge_model=None,
    judge_policy_hash=None,
    judge_rationale=None,
    judge_model_verdict=None,
    judge_metadata_signature=None,
):
    """Construct an AllowlistEntry for discriminator/judge-field tests."""
    from elspeth_lints.core.allowlist import AllowlistEntry

    return AllowlistEntry(
        key=key,
        owner=owner,
        reason=reason,
        safety="contained",
        expires=expires,
        file_fingerprint=None,
        ast_path=None,
        pattern=None,
        source_file="test.yaml",
        judge_verdict=judge_verdict,
        judge_recorded_at=judge_recorded_at,
        judge_model=judge_model,
        judge_policy_hash=judge_policy_hash,
        judge_rationale=judge_rationale,
        judge_model_verdict=judge_model_verdict,
        judge_metadata_signature=judge_metadata_signature,
    )


def test_discriminator_strips_fingerprint_suffix() -> None:
    """The fp=<hex> segment must not be part of the identity tuple."""
    rotated = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa")
    same_after_rotate = _make_entry(key="web/x.py:R1:fn:fp=bbbbbbbbbbbbbbbb")
    assert _discriminator(rotated) == _discriminator(same_after_rotate)


def test_discriminator_distinguishes_different_symbols() -> None:
    """Same file + rule but different symbol is a different entry."""
    a = _make_entry(key="web/x.py:R1:fn_a:fp=aaaaaaaaaaaaaaaa")
    b = _make_entry(key="web/x.py:R1:fn_b:fp=aaaaaaaaaaaaaaaa")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_distinguishes_different_reasons() -> None:
    """Same key but different reason is a different audit assertion.

    Without this, two distinct overrides justifying different aspects
    of the same suppression would collapse — destroying the audit
    signal that the operator wrote them deliberately.
    """
    a = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", reason="reason A")
    b = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", reason="reason B")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_distinguishes_different_owners() -> None:
    """Same key but different owner is a different audit assertion."""
    a = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", owner="alice")
    b = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", owner="bob")
    assert _discriminator(a) != _discriminator(b)


def test_discriminator_normalises_whitespace_in_reason() -> None:
    """Line-wrap reformatting of reason must not look like a new entry.

    YAML's folded-scalar quirks (``>``, ``>-``) can rewrap text on
    save; the discriminator collapses internal whitespace so a
    reformat-only diff is grandfathered.
    """
    a = _make_entry(
        key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa",
        reason="multi\nline   reason",
    )
    b = _make_entry(
        key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa",
        reason="multi line reason",
    )
    assert _discriminator(a) == _discriminator(b)


def test_discriminator_distinguishes_different_expires() -> None:
    """Renewing only the expiry date is a fresh suppression decision, not the same entry.

    ``expires`` is part of the ``per_file_rules`` identity but was historically
    excluded from the ``allow_hits`` discriminator — letting an agent renew a
    suppression by editing only the date, with NO fresh judge review (the C1
    date-renewal bypass). Folding ``expires`` in makes a renewal look like a new
    entry that must carry judge metadata.
    """
    from datetime import date

    a = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", expires=date(2026, 8, 1))
    b = _make_entry(key="web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", expires=date(2026, 11, 1))
    assert _discriminator(a) != _discriminator(b)


@pytest.mark.parametrize(
    "field, value",
    [
        ("owner", "x"),
        ("owner", "."),
        ("reason", "x"),
        ("reason", "."),
    ],
)
def test_discriminator_rejects_trivial_owner_reason_anchors(field: str, value: str) -> None:
    """Grandfathering must fail closed when owner/reason cannot disambiguate entries."""
    kwargs = {"key": "web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa", "owner": "qa", "reason": "real boundary reason"}
    kwargs[field] = value
    entry = _make_entry(**kwargs)

    with pytest.raises(JudgeCoverageError, match=field):
        _discriminator(entry)


# =========================================================================
# Missing-fields validation: atomic-quartet invariant probes
# =========================================================================


def test_missing_judge_fields_reports_all_absent_for_pre_judge() -> None:
    """A bare entry with no judge_* fields fails on the full quartet."""
    entry = _make_entry(key="web/x.py:R1:fn:fp=aa")
    assert _missing_judge_fields(entry) == (
        "judge_verdict",
        "judge_recorded_at",
        "judge_model",
        "judge_policy_hash",
        "judge_rationale",
        "judge_metadata_signature",
    )


def test_missing_judge_fields_requires_metadata_signature() -> None:
    """A judged entry without the HMAC marker is field-present but not authentic."""
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import JudgeVerdict

    entry = _make_entry(
        key="web/x.py:R1:fn:fp=aa",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale="rationale",
    )
    assert _missing_judge_fields(entry) == ("judge_metadata_signature",)


def test_missing_judge_fields_empty_for_complete_entry() -> None:
    """An entry with all atomic-quartet fields passes."""
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import JudgeVerdict

    entry = _make_entry(
        key="web/x.py:R1:fn:fp=aa",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale="rationale",
        judge_metadata_signature=_FAKE_JUDGE_METADATA_SIGNATURE,
    )
    assert _missing_judge_fields(entry) == ()


def test_judge_binding_identity_includes_scope_fingerprint_for_v2_entry() -> None:
    """A v2 entry's binding identity carries scope_fingerprint (v2 binds via scope)."""
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict

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
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale="rationale",
        judge_metadata_signature="hmac-sha256:v2:" + "0" * 64,
    )
    assert scope_fp in _judge_binding_identity(entry)


def _v2_entry(*, judge_transport: str | None = None):
    """Construct a judged v2 AllowlistEntry, mirroring the binding-identity test above."""
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import AllowlistEntry, JudgeVerdict

    return AllowlistEntry(
        key="web/x.py:R1:fn:fp=aa",
        owner="alice",
        reason="permitted boundary",
        safety="contained",
        expires=None,
        file_fingerprint=None,
        scope_fingerprint="a" * 64,
        judge_signature_version=2,
        ast_path="Module.body[0]",
        pattern=None,
        source_file="test.yaml",
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, tzinfo=UTC),
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_policy_hash=JUDGE_POLICY_HASH,
        judge_rationale="rationale",
        judge_transport=judge_transport,
        judge_metadata_signature="hmac-sha256:v2:" + "0" * 64,
    )


def test_judge_metadata_payload_includes_transport() -> None:
    """``judge_transport`` (how the verdict was produced) is authenticity-bearing metadata.

    It must appear in the coverage/rotation diff identity so a re-justify under a
    different transport surfaces as a metadata change rather than silently passing.
    """
    from elspeth_lints.core.judge_coverage import _judge_metadata_payload

    entry = _v2_entry(judge_transport="claude_agent_sdk")
    payload = _judge_metadata_payload(entry)
    assert payload is not None
    assert "claude_agent_sdk" in payload


def test_judge_transport_is_metadata_not_binding() -> None:
    """``judge_transport`` lives in the metadata cluster, not the source-binding cluster.

    Two entries differing ONLY in ``judge_transport`` must share an identical
    ``_judge_binding_identity`` (source binding unchanged — a fresh justify may
    legitimately change transport without that being a binding-diff) while their
    ``_judge_metadata_payload`` differs (the change is an authenticity-bearing
    metadata-diff the coverage/rotation gate protects).
    """
    from elspeth_lints.core.judge_coverage import _judge_metadata_payload

    openrouter = _v2_entry(judge_transport="openrouter")
    agent_sdk = _v2_entry(judge_transport="claude_agent_sdk")

    # Source binding is unchanged by a transport difference.
    assert _judge_binding_identity(openrouter) == _judge_binding_identity(agent_sdk)
    assert "openrouter" not in _judge_binding_identity(openrouter)

    # Metadata identity reflects the transport difference.
    assert _judge_metadata_payload(openrouter) != _judge_metadata_payload(agent_sdk)


# =========================================================================
# End-to-end: git diff over a real fixture repo
# =========================================================================


def _init_git_fixture(tmp_path: Path) -> Path:
    """Create a minimal repo with config/cicd/enforce_tier_model/."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    enforce_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    enforce_dir.mkdir(parents=True)
    return enforce_dir


def _commit(tmp_path: Path, msg: str) -> str:
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", msg],
        check=True,
    )
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_e2e_pre_judge_baseline_with_rotation_is_grandfathered(tmp_path: Path) -> None:
    """An entry whose fingerprint rotated between baseline and HEAD is grandfathered.

    This is the load-bearing rotation-policy assertion. The same
    (file, rule, symbol, owner, reason) entry with a different fp=
    must NOT be flagged as new, because the operator confirmed
    rotation = mechanical rename.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial: pre-judge entry")

    # Simulate a rotation: same logical entry, fresh fingerprint.
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=bbbbbbbbbbbbbbbb
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    _commit(tmp_path, "rotation: fingerprint shifted")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 0
    assert report.violations == ()
    assert report.passes


def test_e2e_new_pre_judge_entry_added_in_pr_is_flagged(tmp_path: Path) -> None:
    """A genuinely-new entry without judge metadata is the C1 violation."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:existing:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing boundary
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial: one existing entry")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:existing:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing boundary
          safety: contained
        - key: web/x.py:R5:freshly_added:fp=cccccccccccccccc
          owner: alice
          reason: hand-added without judge run
          safety: contained
    """)
    )
    _commit(tmp_path, "PR: add unjudged entry")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 2
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].entry_key.endswith("freshly_added:fp=cccccccccccccccc")
    assert "judge_verdict" in report.violations[0].missing_fields
    assert not report.passes


def test_e2e_unjudged_entry_colliding_with_baseline_discriminator_is_flagged(tmp_path: Path) -> None:
    """A NEW unjudged entry sharing an existing (file,rule,symbol,owner,reason) is NOT grandfathered.

    The fp= suffix is stripped from the discriminator for rotation stability, so a
    second unjudged suppression at the same symbol (different fp — e.g. a second
    ``.get()`` in the same method) collides with the baseline discriminator. Without
    count-limiting, both grandfather and the new suppression lands with NO judge run
    (C1 collision-add bypass). Grandfathering must be limited to the number of
    pre-judge baseline entries per discriminator; the excess is a new entry that must
    carry judge metadata.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial: one pre-judge entry")

    # Agent adds a SECOND unjudged entry at the SAME discriminator (same
    # file/rule/symbol/owner/reason), different fp — masquerading as a rotation.
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
        - key: web/x.py:R1:fn:fp=cccccccccccccccc
          owner: alice
          reason: legitimate boundary
          safety: contained
    """)
    )
    _commit(tmp_path, "PR: add second unjudged entry at same discriminator")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 2
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert "judge_verdict" in report.violations[0].missing_fields
    assert not report.passes


def test_e2e_renewal_by_expires_edit_requires_judge(tmp_path: Path) -> None:
    """Renewing a suppression by editing only ``expires`` requires fresh judge metadata.

    ``expires`` is part of the discriminator, so a date-only renewal looks like a new
    entry and must carry judge metadata rather than being grandfathered (the C1
    date-renewal bypass).
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
          expires: 2026-08-01
    """)
    )
    baseline = _commit(tmp_path, "initial: bounded pre-judge entry")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
          expires: 2026-11-01
    """)
    )
    _commit(tmp_path, "PR: renew by editing only the expiry date")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert not report.passes


def test_e2e_new_entry_with_full_judge_quartet_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A new entry that records signed judge metadata satisfies the gate."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: empty allowlist")
    key = "web/x.py:R5:judged:fp=dddddddddddddddd"
    file_fingerprint = "0" * 64
    recorded_at = datetime(2026, 5, 23, 12, tzinfo=UTC)
    rationale = "model reasoned that this boundary is legitimate."
    signature = _valid_v1_judge_signature(
        key=key,
        file_fingerprint=file_fingerprint,
        ast_path="body[0]",
        recorded_at=recorded_at,
        rationale=rationale,
    )

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: {key}
          owner: alice
          reason: new judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '{recorded_at.isoformat()}'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: {rationale}
          file_fingerprint: '{file_fingerprint}'
          ast_path: body[0]
          judge_metadata_signature: '{signature}'
    """)
    )
    _commit(tmp_path, "PR: judged new entry")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert report.violations == ()
    assert report.passes


def test_e2e_new_entry_with_forged_judge_signature_is_flagged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signature-shaped judge quartet is not proof of review."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: empty allowlist")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:forged:fp=dddddddddddddddd
          owner: alice
          reason: forged judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: fabricated rationale from PR YAML
          file_fingerprint: '{"0" * 64}'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    _commit(tmp_path, "PR: forged judged new entry")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC,)
    assert not report.passes


def test_e2e_keyless_fork_mode_rejects_new_signed_entry(tmp_path: Path) -> None:
    """Fork PRs cannot prove a newly-signed entry is authentic without the HMAC key."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: empty allowlist")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:forged:fp=dddddddddddddddd
          owner: fork-contributor
          reason: forged signed entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: fake rationale from fork PR
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    _commit(tmp_path, "fork PR: add forged judged entry")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
        forbid_unverified_judge_metadata=True,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC,)
    assert not report.passes


def test_e2e_keyless_fork_mode_grandfathers_unchanged_signed_entry(tmp_path: Path) -> None:
    """Fork PRs can inspect an existing signed entry without accepting new signatures."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:existing:fp=dddddddddddddddd
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: previously accepted rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    baseline = _commit(tmp_path, "initial: signed allowlist entry")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
        forbid_unverified_judge_metadata=True,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 0
    assert report.violations == ()
    assert report.passes


def test_check_judge_coverage_accepts_single_enforce_directory_root(tmp_path: Path) -> None:
    """CI can point the fork-only gate at one signed allowlist directory."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: empty allowlist")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:forged:fp=dddddddddddddddd
          owner: fork-contributor
          reason: forged signed entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: fake rationale from fork PR
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    _commit(tmp_path, "fork PR: add forged judged entry")

    reports = check_judge_coverage(
        allowlist_root=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
        forbid_unverified_judge_metadata=True,
    )

    assert set(reports) == {"enforce_tier_model"}
    report = reports["enforce_tier_model"]
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC,)


def test_e2e_grandfathered_judge_metadata_mutation_is_flagged(tmp_path: Path) -> None:
    """Editing judge metadata on an existing entry must not be grandfathered."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=dddddddddddddddd
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: original rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    baseline = _commit(tmp_path, "initial: judged entry")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=dddddddddddddddd
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: mutated rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    _commit(tmp_path, "PR: mutate judge rationale")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 0
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (JUDGE_METADATA_MUTATED,)
    assert not report.passes


def test_e2e_same_rationale_rejudged_after_fingerprint_drift_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh judge record can replace an old one after source/fingerprint drift."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    new_key = "web/x.py:R5:judged:fp=bbbbbbbbbbbbbbbb"
    new_file_fingerprint = "1" * 64
    new_recorded_at = datetime(2026, 5, 24, 12, tzinfo=UTC)
    new_rationale = "same accepted rationale"
    new_signature = _valid_v1_judge_signature(
        key=new_key,
        file_fingerprint=new_file_fingerprint,
        ast_path="body[1]",
        recorded_at=new_recorded_at,
        rationale=new_rationale,
    )
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: same accepted rationale
          file_fingerprint: '{"0" * 64}'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    baseline = _commit(tmp_path, "initial: judged entry")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: {new_key}
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '{new_recorded_at.isoformat()}'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: {new_rationale}
          file_fingerprint: '{new_file_fingerprint}'
          ast_path: body[1]
          judge_metadata_signature: '{new_signature}'
    """)
    )
    _commit(tmp_path, "PR: rejudge after fingerprint drift")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert report.violations == ()
    assert report.passes


def test_e2e_rejudge_with_forged_signature_is_metadata_mutation(tmp_path: Path) -> None:
    """A binding drift is not a fresh rejudge unless the new signature verifies."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: same accepted rationale
          file_fingerprint: '{"0" * 64}'
          ast_path: body[0]
          judge_metadata_signature: '{_FAKE_JUDGE_METADATA_SIGNATURE}'
    """)
    )
    baseline = _commit(tmp_path, "initial: judged entry")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=bbbbbbbbbbbbbbbb
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-24T12:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: forged fresh rationale
          file_fingerprint: '{"1" * 64}'
          ast_path: body[1]
          judge_metadata_signature: '{"hmac-sha256:v1:" + "1" * 64}'
    """)
    )
    _commit(tmp_path, "PR: forged rejudge after fingerprint drift")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 0
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (JUDGE_METADATA_MUTATED,)
    assert not report.passes


def test_e2e_v2_rejudged_after_scope_drift_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A v2 fresh judge record can replace an old v2 record after scope drift."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    new_key = "web/x.py:R5:judged:fp=bbbbbbbbbbbbbbbb"
    new_scope_fingerprint = "b" * 64
    new_recorded_at = datetime(2026, 5, 24, 12, tzinfo=UTC)
    new_rationale = "same accepted rationale"
    new_signature = _valid_v2_judge_signature(
        key=new_key,
        scope_fingerprint=new_scope_fingerprint,
        ast_path="body[1]",
        recorded_at=new_recorded_at,
        rationale=new_rationale,
    )
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: same accepted rationale
          judge_signature_version: 2
          scope_fingerprint: '{"a" * 64}'
          ast_path: body[0]
          judge_transport: openrouter
          judge_metadata_signature: '{"hmac-sha256:v2:" + "0" * 64}'
    """)
    )
    baseline = _commit(tmp_path, "initial: v2 judged entry")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: {new_key}
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '{new_recorded_at.isoformat()}'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: {new_rationale}
          judge_signature_version: 2
          scope_fingerprint: '{new_scope_fingerprint}'
          ast_path: body[1]
          judge_transport: openrouter
          judge_metadata_signature: '{new_signature}'
    """)
    )
    _commit(tmp_path, "PR: v2 rejudge after scope drift")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.new_entry_count == 1
    assert report.violations == ()
    assert report.passes


def test_e2e_v2_rejudge_with_forged_signature_is_metadata_mutation(tmp_path: Path) -> None:
    """A v2 HMAC-shaped signature string is not enough to prove rejudge."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T12:00:00+00:00'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: same accepted rationale
          judge_signature_version: 2
          scope_fingerprint: '{"a" * 64}'
          ast_path: body[0]
          judge_transport: openrouter
          judge_metadata_signature: '{"hmac-sha256:v2:" + "0" * 64}'
    """)
    )
    baseline = _commit(tmp_path, "initial: v2 judged entry")

    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R5:judged:fp=bbbbbbbbbbbbbbbb
          owner: alice
          reason: existing judged entry
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-24T12:00:00+00:00'
          judge_model: {DEFAULT_JUDGE_MODEL}
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: forged fresh rationale
          judge_signature_version: 2
          scope_fingerprint: '{"b" * 64}'
          ast_path: body[1]
          judge_transport: openrouter
          judge_metadata_signature: '{"hmac-sha256:v2:" + "1" * 64}'
    """)
    )
    _commit(tmp_path, "PR: forged v2 rejudge after scope drift")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.new_entry_count == 0
    assert len(report.violations) == 1
    assert report.violations[0].missing_fields == (JUDGE_METADATA_MUTATED,)
    assert not report.passes


def test_e2e_baseline_absent_directory_treats_all_entries_as_new(tmp_path: Path) -> None:
    """A directory added in this PR has no baseline — every entry must be judged."""
    enforce_dir = _init_git_fixture(tmp_path)
    # Initial commit with empty repo state for the enforce_tier_model dir.
    (tmp_path / "README.md").write_text("seed\n")
    baseline = _commit(tmp_path, "initial: just a readme, no enforce dir")

    # Now the directory + entries appear in this PR.
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fresh:fp=ee
          owner: alice
          reason: new
          safety: contained
    """)
    )
    _commit(tmp_path, "PR: add enforce dir")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert not report.passes


def test_check_judge_coverage_rejects_bad_baseline_ref(tmp_path: Path) -> None:
    """An unknown baseline ref must error, not silently pass.

    The CI agent must never get a green gate because the workflow
    passed a typo for the baseline ref.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref="nonexistent-ref-xyz",
            repo_root=tmp_path,
        )
    assert "baseline-ref" in str(exc_info.value)


def test_check_judge_coverage_leniently_loads_historical_baseline_without_safety(tmp_path: Path) -> None:
    """Historical baseline entries missing later-required fields still grandfather.

    The baseline ref is read-only diff context. It must remain parseable when a
    newer HEAD schema adds fields such as ``safety``; otherwise C1 crashes before
    it can compare the current strict HEAD entries against their baseline
    counterpart.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
    """),
        encoding="utf-8",
    )
    baseline = _commit(tmp_path, "initial: historical pre-safety entry")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
          safety: contained
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "PR: preserve entry under current schema")

    report = check_one_directory(
        allowlist_dir=enforce_dir,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    assert report.head_entry_count == 1
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 0
    assert report.violations == ()


def test_check_judge_coverage_keeps_head_safety_required(tmp_path: Path) -> None:
    """Lenient baseline loading must not weaken current HEAD validation."""
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n", encoding="utf-8")
    baseline = _commit(tmp_path, "initial: empty allowlist")

    yaml_path.write_text(
        textwrap.dedent("""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aaaaaaaaaaaaaaaa
          owner: alice
          reason: legitimate boundary
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "PR: add current entry without safety")

    with pytest.raises(JudgeCoverageError, match="safety"):
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )


def test_check_judge_coverage_skips_standalone_legacy_allow_classes_shape(tmp_path: Path) -> None:
    """Standalone legacy allowlist formats are outside the C1 judge surface.

    ``audit_evidence.nominal_base`` uses ``allow_classes:`` and has no
    per-finding judge metadata representation. C1 should not route that
    directory into judge-coverage unless it also contains a standard
    ``allow_hits`` / ``per_file_rules`` surface.
    """
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "T"],
        check=True,
    )
    enforce_root = tmp_path / "config" / "cicd"
    legacy = enforce_root / "enforce_audit_evidence_nominal"
    legacy.mkdir(parents=True)
    (legacy / "errors.yaml").write_text(
        textwrap.dedent("""\
        allow_classes:
        - key: x.py:AEN1:Class
          owner: alice
          reason: legacy
          safety: contained
    """)
    )
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert reports == {}


def test_check_judge_coverage_skips_standalone_custom_entries_shape(tmp_path: Path) -> None:
    """Custom governance ``entries:`` manifests are not judgeable allow_hits.

    The telemetry-backfill trailer allowlist stores commit SHA exemptions under
    ``entries:``. Those records are audited by their own CI backstop and should
    not fail the allowlist-judge metadata gate.
    """
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    enforce_root = tmp_path / "config" / "cicd"
    custom = enforce_root / "enforce_telemetry_backfill_trailer"
    custom.mkdir(parents=True)
    (custom / "cohorts.yaml").write_text(
        textwrap.dedent("""\
        entries:
        - commit_sha: 0123456789abcdef0123456789abcdef01234567
          cohort: b2
          reason: pre-hook cohort backfill
          owner: codex
          expires: 2026-06-30
    """),
        encoding="utf-8",
    )
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert reports == {}


def test_check_judge_coverage_reports_mixed_allow_hits_and_allow_classes(tmp_path: Path) -> None:
    """A legacy entry shape remains visible even beside valid ``allow_hits``."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    enforce_root = tmp_path / "config" / "cicd"
    mixed = enforce_root / "enforce_mixed"
    mixed.mkdir(parents=True)
    (mixed / "mixed.yaml").write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: x.py:R1:fn:fp=aa
          owner: alice
          reason: real boundary reason
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T00:00:00+00:00'
          judge_model: m
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: judge accepted the suppression
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
        allow_classes:
        - key: x.py:AEN1:Legacy
          owner: bob
          reason: legacy class allowlist
          safety: contained
    """),
        encoding="utf-8",
    )
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    report = reports["enforce_mixed"]
    assert report.head_entry_count == 2
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].entry_key == "allow_classes[0]::x.py:AEN1:Legacy"
    assert report.violations[0].missing_fields == ("UNRECOGNIZED_ENTRY_SHAPE",)


def test_check_judge_coverage_flags_new_per_file_rule_as_separate_category(tmp_path: Path) -> None:
    """A newly-created per-file wildcard rule must not bypass judge coverage."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    enforce_root = tmp_path / "config" / "cicd"
    enforce_dir = enforce_root / "enforce_tier_model"
    enforce_dir.mkdir(parents=True)
    (enforce_dir / "core.yaml").write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial")

    (enforce_dir / "core.yaml").write_text(
        textwrap.dedent("""\
        per_file_rules:
        - pattern: core/config.py
          rules: [R1, R6]
          reason: configuration loader validates user YAML at the boundary
          expires: null
          max_hits: 4
    """),
        encoding="utf-8",
    )
    _commit(tmp_path, "PR: add per-file rule")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    report = reports["enforce_tier_model"]
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 0
    assert report.new_entry_count == 1
    assert len(report.violations) == 1
    assert report.violations[0].entry_key == "per_file_rules[0]::pattern=core/config.py::rules=R1,R6"
    assert report.violations[0].missing_fields == (PER_FILE_RULE_REQUIRES_JUDGE,)


def test_check_judge_coverage_grandfathers_existing_per_file_rule(tmp_path: Path) -> None:
    """Existing per-file rules remain visible but do not fail C1 retroactively."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "T"], check=True)
    enforce_root = tmp_path / "config" / "cicd"
    enforce_dir = enforce_root / "enforce_tier_model"
    enforce_dir.mkdir(parents=True)
    (enforce_dir / "core.yaml").write_text(
        textwrap.dedent("""\
        per_file_rules:
        - pattern: core/config.py
          rules: [R1, R6]
          reason: configuration loader validates user YAML at the boundary
          expires: null
          max_hits: 4
    """),
        encoding="utf-8",
    )
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )

    report = reports["enforce_tier_model"]
    assert report.head_entry_count == 1
    assert report.grandfathered_count == 1
    assert report.new_entry_count == 0
    assert report.violations == ()


def test_check_judge_coverage_skips_empty_legacy_shape(tmp_path: Path) -> None:
    """Empty legacy stubs do not create entry-shaped violations."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@e.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "T"],
        check=True,
    )
    enforce_root = tmp_path / "config" / "cicd"
    legacy = enforce_root / "enforce_audit_evidence_nominal"
    legacy.mkdir(parents=True)
    (legacy / "errors.yaml").write_text("allow_classes: []\n")
    baseline = _commit(tmp_path, "initial")

    reports = check_judge_coverage(
        allowlist_root=enforce_root,
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert reports == {}


def test_check_judge_coverage_root_not_a_dir_errors(tmp_path: Path) -> None:
    """A missing allowlist root produces a clear error, not silent zero-report."""
    with pytest.raises(JudgeCoverageError) as exc_info:
        check_judge_coverage(
            allowlist_root=tmp_path / "no_such_dir",
            baseline_ref="HEAD",
            repo_root=tmp_path,
        )
    assert "not a directory" in str(exc_info.value)


def test_report_passes_returns_true_when_no_violations() -> None:
    """JudgeCoverageReport.passes is the inverse of violations being non-empty."""
    empty = JudgeCoverageReport(
        head_entry_count=0,
        grandfathered_count=0,
        new_entry_count=0,
        violations=(),
    )
    assert empty.passes


# =========================================================================
# C7-4a: OSError during routing must fail closed, not silent-skip
# =========================================================================


def test_directory_routing_oserror_raises_judge_coverage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable YAML file during the allow_hits routing check fails the gate (C7-4a).

    The prior behaviour caught ``OSError`` and silently continued —
    indistinguishable from "this file has no allow_hits". A silent
    skip on read failure means the gate would route the *whole
    directory* out of inspection when only one file was unreadable.
    We surface as ``JudgeCoverageError`` (CLI exit 2 — gate broken).
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")

    from typing import Any

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == yaml_path:
            raise OSError("permission denied (simulated)")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_judge_coverage(
            allowlist_root=tmp_path / "config" / "cicd",
            baseline_ref="HEAD",
            repo_root=tmp_path,
        )
    assert "web.yaml" in str(exc_info.value)
    assert "routing for allowlist entries" in str(exc_info.value)


def test_directory_routing_oserror_cli_exit_code_is_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI surfaces the OSError-routing failure as exit 2 (C7-4a)."""
    from elspeth_lints.core.cli import main

    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")

    from typing import Any

    real_read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == yaml_path:
            raise OSError("permission denied (simulated)")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    exit_code = main(
        [
            "check-judge-coverage",
            "--allowlist-root",
            str(tmp_path / "config" / "cicd"),
            "--baseline-ref",
            "HEAD",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


# =========================================================================
# C7-4b: CRLF + start-of-file allow_hits detection
# =========================================================================


def test_directory_routing_detects_crlf_line_endings(tmp_path: Path) -> None:
    """A YAML file with CRLF line endings and ``allow_hits:`` mid-file is routed in (C7-4b).

    The prior substring-match for ``"\\nallow_hits:"`` missed CRLF
    (``"\\r\\nallow_hits:"``). A missed routing decision silently
    excludes the whole directory from the gate's purview — the
    operator sees the gate pass on a corpus the gate never actually
    inspected.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # CRLF line endings with allow_hits NOT at start-of-file.
    yaml_content = (
        "# leading comment\r\n"
        "allow_hits:\r\n"
        "- key: web/x.py:R1:fn:fp=aa\r\n"
        "  owner: alice\r\n"
        "  reason: real boundary reason\r\n"
        "  safety: contained\r\n"
        "  judge_verdict: ACCEPTED\r\n"
        "  judge_recorded_at: '2026-05-23T00:00:00+00:00'\r\n"
        "  judge_model: m\r\n"
        f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'\r\n"
        "  judge_rationale: r\r\n"
        "  file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'\r\n"
        "  ast_path: body[0]\r\n"
    )
    yaml_path.write_bytes(yaml_content.encode("utf-8"))
    baseline = _commit(tmp_path, "initial: CRLF allow_hits")

    reports = check_judge_coverage(
        allowlist_root=tmp_path / "config" / "cicd",
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    # The enforce_tier_model directory MUST have been routed in.
    assert "enforce_tier_model" in reports
    assert reports["enforce_tier_model"].head_entry_count == 1


def test_directory_routing_detects_allow_hits_at_start_of_file(tmp_path: Path) -> None:
    """A YAML file beginning with ``allow_hits:`` at byte 0 is routed in (C7-4b).

    The prior substring-match required a preceding newline. A
    YAML file authored with ``allow_hits:`` as the very first
    token (no header comment, no blank line) would be silently
    excluded from inspection.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # allow_hits: as the very first bytes of the file.
    yaml_path.write_text(
        "allow_hits:\n"
        "- key: web/x.py:R1:fn:fp=aa\n"
        "  owner: alice\n"
        "  reason: real boundary reason\n"
        "  safety: contained\n"
        "  judge_verdict: ACCEPTED\n"
        "  judge_recorded_at: '2026-05-23T00:00:00+00:00'\n"
        "  judge_model: m\n"
        f"  judge_policy_hash: '{JUDGE_POLICY_HASH}'\n"
        "  judge_rationale: r\n"
        "  file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'\n"
        "  ast_path: body[0]\n"
    )
    baseline = _commit(tmp_path, "initial: start-of-file allow_hits")

    reports = check_judge_coverage(
        allowlist_root=tmp_path / "config" / "cicd",
        baseline_ref=baseline,
        repo_root=tmp_path,
    )
    assert "enforce_tier_model" in reports
    assert reports["enforce_tier_model"].head_entry_count == 1


# =========================================================================
# C7-4c: baseline parse failure must fail the gate (not silently empty)
# =========================================================================


def test_baseline_parse_failure_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A baseline YAML that doesn't parse fails the gate; it does NOT degrade to empty-baseline (C7-4c).

    The prior behaviour silently caught ``ValueError`` and treated
    the unparseable baseline file as contributing zero entries.
    That converted every HEAD entry to "new", which floods the
    operator with bogus "missing judge metadata" violations and
    hides the real problem (baseline corruption / schema drift).

    Fail-closed: raise ``JudgeCoverageError`` with the offending
    baseline path. The operator fixes the baseline (or picks a
    different ref), not the cascading symptoms.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # Baseline content is malformed (YAML mapping must be a dict;
    # a scalar string at root is rejected by _load_yaml_strict).
    yaml_path.write_text("just a scalar string at root\n")
    baseline = _commit(tmp_path, "initial: malformed-mapping baseline")

    # HEAD content is a well-formed allow_hits with one entry.
    yaml_path.write_text(
        textwrap.dedent(f"""\
        allow_hits:
        - key: web/x.py:R1:fn:fp=aa
          owner: alice
          reason: legitimate boundary
          safety: contained
          judge_verdict: ACCEPTED
          judge_recorded_at: '2026-05-23T00:00:00+00:00'
          judge_model: anthropic/claude-opus-4-7
          judge_policy_hash: '{JUDGE_POLICY_HASH}'
          judge_rationale: rationale
          file_fingerprint: '0000000000000000000000000000000000000000000000000000000000000000'
          ast_path: body[0]
    """)
    )
    _commit(tmp_path, "PR: fix baseline shape")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "baseline" in str(exc_info.value)
    assert "web.yaml" in str(exc_info.value)


def test_baseline_yamlerror_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A baseline file with un-tokenisable YAML (yaml.YAMLError) fails the gate (C7-4c + C7-5).

    Combines the baseline-parse-must-fail-closed contract (C7-4c)
    with the YAMLError-not-subclass-of-ValueError fix (C7-5): a
    baseline whose YAML cannot even be tokenised must still produce
    a ``JudgeCoverageError``, not an uncaught traceback.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    # ``:- invalid yaml -:`` is a tokeniser-level error (yaml.YAMLError).
    yaml_path.write_text("key: [unclosed\n")
    baseline = _commit(tmp_path, "initial: tokenizer-error baseline")

    # Replace HEAD with valid content.
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "PR: clean head")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "baseline" in str(exc_info.value)
    assert "failed to parse" in str(exc_info.value)


# =========================================================================
# C7-5: yaml.YAMLError catch at HEAD parse path
# =========================================================================


def test_head_yamlerror_raises_judge_coverage_error(tmp_path: Path) -> None:
    """A HEAD file with un-tokenisable YAML raises JudgeCoverageError (C7-5).

    ``yaml.safe_load`` raises ``yaml.YAMLError`` for malformed
    input — a sibling of Exception, NOT a ValueError subclass. The
    prior handler caught only ``ValueError`` and let YAMLError
    escape as a raw traceback (exit-1 "gate broken" via wrong
    channel) rather than the documented exit-2 with a structured
    JudgeCoverageError message.
    """
    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial: clean baseline")

    # Replace HEAD with a tokeniser-error file.
    yaml_path.write_text("key: [unclosed\n")
    _commit(tmp_path, "PR: malformed HEAD")

    with pytest.raises(JudgeCoverageError) as exc_info:
        check_one_directory(
            allowlist_dir=enforce_dir,
            baseline_ref=baseline,
            repo_root=tmp_path,
        )
    assert "web.yaml" in str(exc_info.value)
    assert "failed to parse" in str(exc_info.value)


def test_head_yamlerror_cli_exit_code_is_two(tmp_path: Path) -> None:
    """CLI returns exit 2 for HEAD YAMLError (C7-5 surface)."""
    from elspeth_lints.core.cli import main

    enforce_dir = _init_git_fixture(tmp_path)
    yaml_path = enforce_dir / "web.yaml"
    yaml_path.write_text("allow_hits: []\n")
    _commit(tmp_path, "initial")
    yaml_path.write_text("key: [unclosed\n")
    _commit(tmp_path, "PR: malformed HEAD")

    exit_code = main(
        [
            "check-judge-coverage",
            "--allowlist-root",
            str(tmp_path / "config" / "cicd"),
            "--baseline-ref",
            "HEAD~1",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 2


# =========================================================================
# M7-14/M7-15: git plumbing failures are structured and locale-stable
# =========================================================================


def test_git_show_returns_none_for_true_baseline_path_absence(tmp_path: Path) -> None:
    """A missing path at a valid baseline ref is the only semantic ``None`` case."""
    enforce_dir = _init_git_fixture(tmp_path)
    (enforce_dir / "web.yaml").write_text("allow_hits: []\n")
    baseline = _commit(tmp_path, "initial")

    assert (
        _git_show(
            baseline_ref=baseline,
            rel_path="config/cicd/enforce_tier_model/missing.yaml",
            repo_root=tmp_path,
        )
        is None
    )


def test_git_show_bad_baseline_ref_raises_judge_coverage_error(tmp_path: Path) -> None:
    """An invalid baseline ref is a gate error, not an empty baseline file."""
    _init_git_fixture(tmp_path)
    (tmp_path / "README.md").write_text("seed\n")
    _commit(tmp_path, "initial")

    with pytest.raises(JudgeCoverageError, match="baseline-ref"):
        _git_show(
            baseline_ref="definitely-not-a-ref",
            rel_path="config/cicd/enforce_tier_model/web.yaml",
            repo_root=tmp_path,
        )


def test_git_commands_force_c_locale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Git subprocesses run with ``LC_ALL=C`` and do not depend on localized stderr."""
    calls: list[dict[str, str]] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs.get("env", {}))
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        _ls_tree_yaml_files(
            baseline_ref="HEAD",
            rel_dir="config/cicd/enforce_tier_model",
            repo_root=tmp_path,
        )
        == []
    )
    assert calls
    assert all(call["LC_ALL"] == "C" for call in calls)

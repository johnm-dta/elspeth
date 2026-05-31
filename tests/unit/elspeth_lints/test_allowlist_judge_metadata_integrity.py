"""Cross-field integrity checks for an allowlist entry's judge metadata.

The five ``judge_*`` fields on :class:`AllowlistEntry` are an atomic audit
record. Either *all* are absent (the pre-judge era representation, where an
entry was written before the cicd-judge gate existed) or *all* required
fields are present (a fully-recorded judge interaction). Partial shapes —
``judge_verdict`` set with ``judge_recorded_at`` missing, or
``judge_model_verdict`` set on a non-override entry — are corruption: half-
written audit state, a botched merge, a partial revert.

Per the project's Tier-1 doctrine (the allowlist YAML is *our own data*,
written by ``elspeth-lints justify`` or hand-edited by an operator inside
our trust boundary), inconsistent shape must crash on load rather than
silently propagate into the gate's decision. These tests pin the loader's
behaviour against the four invariants enforced by
``_validate_judge_metadata_atomic``.

See ``B2`` in the cicd-judge-cli prototype review.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    _parse_allow_hits,
    compute_judge_metadata_signature,
    load_allowlist,
    verify_entry_binding_against_finding,
)
from elspeth_lints.core.judge import JUDGE_POLICY_HASH
from elspeth_lints.core.source_excerpt import RedactionRecord

_TEST_JUDGE_METADATA_HMAC_KEY = "test-judge-metadata-hmac-key-2026-05-24"


def _expected_judge_metadata_signature(
    *,
    key: str,
    file_fingerprint: str,
    ast_path: str,
    judge_verdict: str = "ACCEPTED",
    judge_model_verdict: str | None = None,
    judge_recorded_at: str = "2026-05-23T00:00:00+00:00",
    judge_model: str = "anthropic/claude-opus-4-7",
    judge_rationale: str = "judge accepted the suppression",
    judge_policy_hash: str = JUDGE_POLICY_HASH,
    judge_excerpt_redactions: tuple[dict[str, int | str], ...] = (),
) -> str:
    """Return the v1 judge-metadata HMAC used by signed fixture YAML."""
    payload = {
        "version": 1,
        "key": key,
        "file_fingerprint": file_fingerprint,
        "ast_path": ast_path,
        "judge_verdict": judge_verdict,
        "judge_model_verdict": judge_model_verdict,
        "judge_recorded_at": judge_recorded_at,
        "judge_model": judge_model,
        "judge_rationale": judge_rationale,
        "judge_policy_hash": judge_policy_hash,
        "judge_excerpt_redactions": list(judge_excerpt_redactions),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hmac.new(_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    return f"hmac-sha256:v1:{digest}"


# ---------------------------------------------------------------------------
# Invariant 1: OVERRIDDEN_BY_OPERATOR entries must record judge_model_verdict
# ---------------------------------------------------------------------------


def test_override_entry_without_judge_model_verdict_crashes(tmp_path: Path) -> None:
    """An override entry must record what the model said.

    Without ``judge_model_verdict``, the override is fabrication of "we
    overrode something" without recording what. The
    "override-rate-by-underlying-verdict" meta-metric (a key signal of
    judge-gate health) becomes unqueryable.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: operator
            reason: shipping under deadline
            safety: low
            judge_verdict: OVERRIDDEN_BY_OPERATOR
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: model said BLOCKED but we proceed
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
            # judge_model_verdict deliberately omitted — corruption.
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*judge_model_verdict.*absent"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# Invariant 2: non-override entries must NOT carry judge_model_verdict
# ---------------------------------------------------------------------------


def test_non_override_entry_with_judge_model_verdict_crashes(tmp_path: Path) -> None:
    """For a non-override entry, model verdict and entry verdict agree.

    Recording a separate ``judge_model_verdict`` on a non-override entry
    fabricates a divergence-signal that doesn't exist by construction.
    The downstream aggregator that distinguishes "operator overrode
    BLOCKED" from "operator overrode ACCEPTED" would receive a
    contradictory record.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            judge_model_verdict: ACCEPTED  # fabricated divergence on a non-override entry
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    # ACCEPTED (not BLOCKED) so this hits invariant 2 specifically rather
    # than the BLOCKED-asymmetry guard (invariant 5).
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*non-override.*judge_model_verdict"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# Invariant 3: the verdict + recorded_at + model + rationale quartet is atomic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing_field",
    ["judge_recorded_at", "judge_model", "judge_rationale", "judge_policy_hash"],
)
def test_partial_post_judge_entry_crashes(tmp_path: Path, missing_field: str) -> None:
    """A ``judge_verdict`` without the full companion-set is corruption.

    The four fields are written atomically by ``elspeth-lints justify``;
    a partial shape indicates the file was hand-edited mid-write or a
    git merge resolved to a shape that doesn't correspond to any single
    valid state. The loader must crash so the corruption is visible to
    the operator, not silently propagate.
    """
    fields = {
        "judge_verdict": "ACCEPTED",
        "judge_recorded_at": "2026-05-01T10:00:00+00:00",
        "judge_model": "claude-opus-4-7",
        "judge_rationale": "judge agrees",
        "judge_policy_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
    }
    del fields[missing_field]
    body = "\n".join(f"            {key}: {value}" for key, value in fields.items())
    yaml = textwrap.dedent(f"""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
{body}
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=rf"allow_hits\[0\].*{missing_field}"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# Invariant 4: pre-judge entries must not carry stray judge_* fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stray_field, stray_value",
    [
        ("judge_recorded_at", "2026-05-01T10:00:00+00:00"),
        ("judge_model", "claude-opus-4-7"),
        ("judge_rationale", "stale rationale text"),
        ("judge_model_verdict", "ACCEPTED"),
        ("judge_policy_hash", "sha256:0000000000000000000000000000000000000000000000000000000000000000"),
    ],
)
def test_pre_judge_entry_with_stray_field_crashes(tmp_path: Path, stray_field: str, stray_value: str) -> None:
    """A pre-judge entry has ``judge_verdict=None`` and no other judge fields.

    A stray ``judge_recorded_at`` (or any other ``judge_*``) without
    ``judge_verdict`` is corruption: probably the trace of a partial
    revert that removed the verdict but left the surrounding metadata.
    Either the entry predates the judge (and should have all fields
    absent) or it postdates the judge (and should have all required
    fields present). The hybrid shape doesn't correspond to any valid
    history.
    """
    yaml = textwrap.dedent(f"""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: pre-judge era entry
            safety: low
            # judge_verdict deliberately omitted
            {stray_field}: {stray_value}
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    # The message names both the index (allow_hits[0]) and the stray
    # field; order in the rendered string is implementation-detail.
    with pytest.raises(ValueError) as exc_info:
        load_allowlist(path, valid_rule_ids=set())
    message = str(exc_info.value)
    assert "allow_hits[0]" in message
    assert stray_field in message
    assert "pre-judge" in message


# ---------------------------------------------------------------------------
# Positive paths: every well-shaped entry round-trips correctly
# ---------------------------------------------------------------------------


def test_fully_populated_override_entry_round_trips(tmp_path: Path) -> None:
    """An OVERRIDDEN_BY_OPERATOR entry with all five judge fields loads cleanly."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: operator-jdoe
            reason: shipping under deadline
            safety: medium
            judge_verdict: OVERRIDDEN_BY_OPERATOR
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: "model said: this should be fixed in code"
            judge_model_verdict: BLOCKED
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    al = load_allowlist(path, valid_rule_ids=set())
    assert len(al.entries) == 1
    entry = al.entries[0]
    assert entry.judge_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR
    assert entry.judge_model_verdict is JudgeVerdict.BLOCKED
    assert entry.judge_model == "claude-opus-4-7"
    assert entry.judge_rationale == "model said: this should be fixed in code"
    assert entry.judge_recorded_at is not None
    assert entry.judge_recorded_at.tzinfo is not None


def test_pre_judge_entry_round_trips(tmp_path: Path) -> None:
    """A pre-judge-era entry (no judge_* fields at all) loads cleanly.

    This is the historical-corpus shape: entries written before the
    cicd-judge gate existed. Per the fabrication-decision test,
    ``None`` for every judge_* field is the honest representation of
    "this entry predates the judge."
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: historical-agent
            reason: ancient suppression
            safety: low
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    al = load_allowlist(path, valid_rule_ids=set())
    assert len(al.entries) == 1
    entry = al.entries[0]
    assert entry.judge_verdict is None
    assert entry.judge_recorded_at is None
    assert entry.judge_model is None
    assert entry.judge_rationale is None
    assert entry.judge_model_verdict is None


def test_fully_populated_non_override_entry_round_trips(tmp_path: Path) -> None:
    """A non-override post-judge entry omits judge_model_verdict but has the quartet."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: test-agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    al = load_allowlist(path, valid_rule_ids=set())
    assert len(al.entries) == 1
    entry = al.entries[0]
    assert entry.judge_verdict is JudgeVerdict.ACCEPTED
    assert entry.judge_model_verdict is None  # absence is the signal: no divergence


def test_non_utc_judge_recorded_at_round_trips_as_timezone_aware(tmp_path: Path) -> None:
    """The loader accepts aware ISO timestamps with non-UTC offsets."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: test-agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+09:30
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    entry = load_allowlist(path, valid_rule_ids=set()).entries[0]

    assert entry.judge_recorded_at is not None
    assert entry.judge_recorded_at.isoformat() == "2026-05-01T10:00:00+09:30"


def test_naive_judge_recorded_at_crashes(tmp_path: Path) -> None:
    """Judge timestamps must carry an explicit timezone offset."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: test-agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: '2026-05-01T10:00:00'
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=r"allow_hits\[0\]\.judge_recorded_at.*timezone offset"):
        load_allowlist(path, valid_rule_ids=set())


@pytest.mark.parametrize(
    "field",
    ["judge_verdict", "judge_model_verdict"],
)
def test_invalid_judge_verdict_enum_values_crash(tmp_path: Path, field: str) -> None:
    """Verdict fields are closed enums; unknown values are corruption."""
    values = {
        "judge_verdict": "OVERRIDDEN_BY_OPERATOR" if field == "judge_model_verdict" else "NOT_A_VERDICT",
        "judge_model_verdict": "NOT_A_VERDICT" if field == "judge_model_verdict" else "BLOCKED",
    }
    yaml = textwrap.dedent(f"""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: test-agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: {values["judge_verdict"]}
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            judge_model_verdict: {values["judge_model_verdict"]}
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=rf"allow_hits\[0\]\.{field}.*one of"):
        load_allowlist(path, valid_rule_ids=set())


@pytest.mark.parametrize(
    "field",
    ["owner", "reason", "judge_model", "judge_policy_hash", "judge_rationale", "file_fingerprint", "ast_path"],
)
def test_empty_string_audit_fields_crash(tmp_path: Path, field: str) -> None:
    """Required audit anchors and judge binding fields may not be empty strings."""
    values = {
        "owner": "test-agent",
        "reason": "tier-3 boundary",
        "judge_model": "claude-opus-4-7",
        "judge_policy_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "judge_rationale": "judge agrees",
        "file_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000",
        "ast_path": "body[0]",
    }
    values[field] = ""
    yaml = textwrap.dedent(f"""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: {values["owner"]!r}
            reason: {values["reason"]!r}
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: {values["judge_model"]!r}
            judge_policy_hash: {values["judge_policy_hash"]!r}
            judge_rationale: {values["judge_rationale"]!r}
            file_fingerprint: {values["file_fingerprint"]!r}
            ast_path: {values["ast_path"]!r}
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=rf"allow_hits\[0\]\.{field}.*non-empty"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# C8-1: BLOCKED never persists. It is the in-memory runtime verdict the
# gate uses to reject a candidate suppression; a BLOCKED value on disk is
# corruption (botched hand-edit, partial revert, tampering). Two gates
# catch it: ``_optional_judge_verdict`` rejects on parse, and
# ``_validate_judge_metadata_atomic`` reaffirms (defense-in-depth).
# ---------------------------------------------------------------------------


def test_judge_verdict_blocked_on_disk_crashes(tmp_path: Path) -> None:
    """A persisted ``judge_verdict: BLOCKED`` is corruption.

    The CLI declines to write BLOCKED entries (``JudgeVerdict.BLOCKED`` is
    documented "Reserved for in-memory representation only"). Mechanical
    enforcement here makes that docstring load-bearing.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: BLOCKED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge said BLOCKED but somebody wrote the entry anyway
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*judge_verdict.*BLOCKED.*in-memory"):
        load_allowlist(path, valid_rule_ids=set())


def test_judge_model_verdict_blocked_on_non_override_entry_crashes(tmp_path: Path) -> None:
    """``judge_model_verdict: BLOCKED`` is only legal on OVERRIDDEN entries.

    On a non-override entry, ``judge_model_verdict`` should be absent
    (per invariant 2). If it is set to BLOCKED outside an OVERRIDDEN
    entry, that is corruption regardless of which sibling invariant
    catches it first.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            judge_model_verdict: BLOCKED
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    # Two invariants could catch this; either is acceptable. Invariant 5
    # fires first (BLOCKED rejection); if it didn't, invariant 2 would
    # (non-override entry carrying judge_model_verdict). The shared
    # context for both is "allow_hits[0]" and "judge_model_verdict".
    with pytest.raises(ValueError) as exc_info:
        load_allowlist(path, valid_rule_ids=set())
    message = str(exc_info.value)
    assert "allow_hits[0]" in message
    assert "judge_model_verdict" in message


def test_override_with_judge_model_verdict_blocked_round_trips(tmp_path: Path) -> None:
    """OVERRIDDEN entries with ``judge_model_verdict: BLOCKED`` are the override shape.

    This is the *only* legal place a BLOCKED value lives on disk: the
    operator overrode the model's BLOCKED verdict and we record what the
    model said for the override-rate meta-metric.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: operator-jdoe
            reason: shipping under deadline
            safety: medium
            judge_verdict: OVERRIDDEN_BY_OPERATOR
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: model said BLOCKED; operator proceeds anyway
            judge_model_verdict: BLOCKED
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    al = load_allowlist(path, valid_rule_ids=set())
    assert len(al.entries) == 1
    entry = al.entries[0]
    assert entry.judge_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR
    assert entry.judge_model_verdict is JudgeVerdict.BLOCKED


def test_judge_model_verdict_rejects_operator_override_value(tmp_path: Path) -> None:
    """``judge_model_verdict`` records the model's answer, never the operator action."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: operator
            reason: shipping under deadline
            safety: low
            judge_verdict: OVERRIDDEN_BY_OPERATOR
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: model said BLOCKED but we proceed
            judge_model_verdict: OVERRIDDEN_BY_OPERATOR
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=r"allow_hits\[0\]\.judge_model_verdict.*model verdict"):
        load_allowlist(path, valid_rule_ids=set())


@pytest.mark.parametrize(
    "field, value",
    [
        ("owner", "x"),
        ("owner", "."),
        ("reason", "x"),
        ("reason", "."),
    ],
)
def test_owner_and_reason_must_be_substantive_discriminator_anchors(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """The C1 grandfathering discriminator cannot rely on trivial anchors."""
    values = {
        "owner": "qa",
        "reason": "real boundary reason",
    }
    values[field] = value
    yaml = textwrap.dedent(f"""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: {values["owner"]!r}
            reason: {values["reason"]!r}
            safety: low
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=rf"allow_hits\[0\]\.{field}.*substantive"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# C8-4: a judge verdict with empty/whitespace-only rationale is audit-
# broken. _optional_string rejects empty strings at parse-time, but a
# whitespace-only rationale ("   ") slips past it; the atomic validator
# catches it as defense-in-depth.
# ---------------------------------------------------------------------------


def test_whitespace_only_rationale_on_post_judge_entry_crashes(tmp_path: Path) -> None:
    """A whitespace-only judge_rationale on a judge-gated entry is audit-broken.

    The rationale is the "why" of the audit record. An auditor asking
    "why did the judge accept this?" must get a substantive answer, not
    a blank string. ``_optional_string`` accepts only non-empty strings,
    so a literal ``""`` would crash there; a whitespace-only value
    (a quoted ``"   "``) passes the string-shape check but is still
    empty after strip. The atomic validator catches it.
    """
    # Quoted scalar preserves the whitespace; PyYAML returns the literal
    # three spaces as the string value, which is truthy (passes
    # _optional_string) but strips to empty (caught by invariant 7).
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: "   "
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*judge_rationale.*empty"):
        load_allowlist(path, valid_rule_ids=set())


# ---------------------------------------------------------------------------
# C8-3: quartet-transplant defence (binding fields + source-drift gate).
#
# The C8-3 attack vector: a judge quartet (verdict + rationale + model +
# recorded_at) accepted on a SAFE entry A is copied verbatim onto an entry B
# keyed at DANGEROUS code, and the allowlist gate has no way to detect the
# rebind because the quartet is plain text and not bound to the code it was
# judged against. The fix wires two binding fields (file_fingerprint +
# ast_path) end-to-end:
#
#   * Writer (``elspeth-lints justify``) computes both at write time.
#   * Loader (``load_allowlist`` with ``source_root``) recomputes the live
#     file_fingerprint and rejects mismatches at load — catching cross-file
#     transplant and source drift.
#   * Matcher (tier_model's ``_match_finding``) asserts the live finding's
#     ast_path equals the persisted one — catching in-file AST-node
#     transplant.
#
# These tests pin the invariants enforced at the loader gate; the writer
# tests live in test_justify.py and the matcher integration tests live in
# test_trust_tier_model_rule.py (covered via the C8-3 ast-mismatch case).
# ---------------------------------------------------------------------------


def test_post_judge_entry_missing_file_fingerprint_crashes(tmp_path: Path) -> None:
    """A judge-gated entry without ``file_fingerprint`` is transplantable."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            ast_path: "body[0]"
            # file_fingerprint deliberately omitted — quartet is unbound.
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*binding fields are missing.*file_fingerprint"):
        load_allowlist(path, valid_rule_ids=set())


def test_post_judge_entry_missing_ast_path_crashes(tmp_path: Path) -> None:
    """A judge-gated entry without ``ast_path`` is transplantable within a file."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            # ast_path deliberately omitted — in-file transplant possible.
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*binding fields are missing.*ast_path"):
        load_allowlist(path, valid_rule_ids=set())


def test_pre_judge_entry_with_stray_binding_field_crashes(tmp_path: Path) -> None:
    """A pre-judge entry must not carry binding fields (invariant 4).

    Binding fields are written ONLY by ``justify`` alongside a judge
    verdict. Their presence on a verdict-less entry is the same class
    of partial-revert corruption as a stray ``judge_rationale``.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: agent
            reason: pre-judge era entry
            safety: low
            # judge_verdict deliberately omitted
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"allow_hits\[0\].*file_fingerprint.*pre-judge"):
        load_allowlist(path, valid_rule_ids=set())


def _write_source(root: Path, file_path: str, content: str) -> Path:
    """Lay out one source file under ``root`` and return its absolute path."""
    target = root / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _write_judge_gated_yaml(
    *,
    yaml_path: Path,
    key: str,
    file_fingerprint: str,
    ast_path: str,
    signature_key: str | None = _TEST_JUDGE_METADATA_HMAC_KEY,
    judge_excerpt_redactions: tuple[dict[str, int | str], ...] = (),
) -> None:
    """Write one judge-gated allowlist entry to ``yaml_path``."""
    lines = [
        "allow_hits:",
        f"  - key: {key}",
        "    owner: test-agent",
        "    reason: tier-3 boundary",
        "    safety: low",
        "    judge_verdict: ACCEPTED",
        "    judge_recorded_at: '2026-05-23T00:00:00+00:00'",
        "    judge_model: anthropic/claude-opus-4-7",
        f"    judge_policy_hash: '{JUDGE_POLICY_HASH}'",
        "    judge_rationale: judge accepted the suppression",
        f"    file_fingerprint: '{file_fingerprint}'",
        f"    ast_path: '{ast_path}'",
    ]
    if judge_excerpt_redactions:
        lines.append("    judge_excerpt_redactions:")
        for redaction in judge_excerpt_redactions:
            lines.extend(
                [
                    f"      - pattern: {redaction['pattern']}",
                    f"        byte_count: {redaction['byte_count']}",
                    f"        redacted_hash: {redaction['redacted_hash']}",
                ]
            )
    if signature_key is not None:
        assert signature_key == _TEST_JUDGE_METADATA_HMAC_KEY
        signature = _expected_judge_metadata_signature(
            key=key,
            file_fingerprint=file_fingerprint,
            ast_path=ast_path,
            judge_excerpt_redactions=judge_excerpt_redactions,
        )
        lines.append(f"    judge_metadata_signature: '{signature}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_transplanted_quartet_across_files_fails_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C8-3 cross-file transplant: copying quartet+binding from file A onto an entry keyed at file B fails.

    Setup: two source files (a.py, b.py) with different bytes. A
    judge-gated allowlist entry is written for a finding in a.py with
    a's live file_fingerprint. An attacker copies that entry's whole
    quartet + binding fields onto an entry whose key points at b.py.
    The loader recomputes b.py's bytes hash, compares against the
    persisted (a-derived) file_fingerprint, and refuses to load.

    This is THE CORE SECURITY TEST: it proves the binding wiring
    actually closes the transplant attack the C8-3 ticket exists to
    close.
    """
    source_root = tmp_path / "src"
    a_path = _write_source(source_root, "a.py", "# safe file: empty module\n")
    b_path = _write_source(source_root, "b.py", "import os\nos.system('rm -rf /')\n")
    a_fp = hashlib.sha256(a_path.read_bytes()).hexdigest()
    b_fp = hashlib.sha256(b_path.read_bytes()).hexdigest()
    assert a_fp != b_fp  # sanity: the files MUST differ
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    allowlist = tmp_path / "allowlist.yaml"
    # The transplant: key points at b.py, but file_fingerprint is a's.
    # That is exactly what an attacker would write after pasting a's
    # accepted quartet onto a synthesized b-keyed entry.
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="b.py:R1:dangerous:fp=deadbeef",
        file_fingerprint=a_fp,
        ast_path="body[0]",
    )

    with pytest.raises(ValueError) as exc_info:
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)
    message = str(exc_info.value)
    assert "file_fingerprint mismatch" in message
    assert "b.py" in message
    assert a_fp in message  # the persisted (wrong) fingerprint
    assert b_fp in message  # the live (correct) fingerprint


def test_source_drift_after_judge_verdict_fails_at_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """C8-3 source-drift: editing the source after the verdict invalidates the binding.

    Scenario: the judge accepted a suppression at a particular file
    state; the source has since been edited (refactor, change to
    surrounding code, anything that mutates the bytes). The judge's
    rationale no longer describes the live code, so the entry must
    be re-justified. The load-time gate makes that requirement
    mechanically enforced rather than aspirational.
    """
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "drifty.py", "# original content\n")
    original_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="drifty.py:R1:fn:fp=somehash",
        file_fingerprint=original_fp,
        ast_path="body[0]",
    )

    # First load: source unchanged, binding holds, load succeeds.
    al = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)
    assert len(al.entries) == 1

    # Now drift the source. The judge's rationale was written about
    # the ORIGINAL bytes; the live bytes are different.
    src_path.write_text("# completely different content with bug\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"file_fingerprint mismatch.*drifty\.py"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)


def test_signed_judge_metadata_round_trips_with_source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A source-root production load accepts a valid HMAC-bound judge record."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "signed.py", "payload = {'name': 'Ada'}\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    key = "signed.py:R1:fn:fp=somehash"
    ast_path = "body[0]"
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key=key,
        file_fingerprint=file_fp,
        ast_path=ast_path,
    )
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    loaded = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)

    assert len(loaded.entries) == 1
    assert loaded.entries[0].judge_metadata_signature == _expected_judge_metadata_signature(
        key=key,
        file_fingerprint=file_fp,
        ast_path=ast_path,
    )


def test_judge_excerpt_redactions_round_trip_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Writer-emitted redaction records are parsed as structured audit data."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "redacted.py", "token = 'AWS_ACCESS_KEY_REDACTED_PLACEHOLDER'\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    key = "redacted.py:R1:fn:fp=somehash"
    ast_path = "body[0]"
    redactions = (
        {
            "pattern": "aws_access_key",
            "byte_count": 20,
            "redacted_hash": "0123456789abcdef",
        },
    )
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key=key,
        file_fingerprint=file_fp,
        ast_path=ast_path,
        judge_excerpt_redactions=redactions,
    )
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    loaded = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)

    assert loaded.entries[0].judge_excerpt_redactions == (
        RedactionRecord(pattern_name="aws_access_key", byte_count=20, redacted_hash="0123456789abcdef"),
    )


def test_malformed_judge_excerpt_redactions_fail_closed(tmp_path: Path) -> None:
    """Malformed redaction metadata must not remain presence-only YAML."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a.py:R1:fn:fp=somehash"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
            file_fingerprint: "0000000000000000000000000000000000000000000000000000000000000000"
            ast_path: "body[0]"
            judge_excerpt_redactions:
              - pattern: aws_access_key
                byte_count: -1
                redacted_hash: 0123456789abcdef
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)

    with pytest.raises(ValueError, match=r"judge_excerpt_redactions\[0\]\.byte_count.*positive"):
        load_allowlist(path, valid_rule_ids=set())


def test_post_judge_entry_missing_signature_fails_at_source_root_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production loads must reject unsigned post-judge metadata."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "unsigned.py", "payload = {'name': 'Ada'}\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="unsigned.py:R1:fn:fp=somehash",
        file_fingerprint=file_fp,
        ast_path="body[0]",
        signature_key=None,
    )
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    with pytest.raises(ValueError, match=r"allow_hits\[0\].*judge_metadata_signature.*missing"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text.replace("judge_rationale: judge accepted the suppression", "judge_rationale: softened rationale"),
        lambda text: text.replace("judge_model: anthropic/claude-opus-4-7", "judge_model: anthropic/claude-sonnet-4"),
        lambda text: text.replace(JUDGE_POLICY_HASH, "sha256:" + "1" * 64),
        lambda text: text.replace(
            "judge_recorded_at: '2026-05-23T00:00:00+00:00'",
            "judge_recorded_at: '2026-05-24T00:00:00+00:00'",
        ),
        lambda text: text.replace(
            "judge_verdict: ACCEPTED",
            "judge_verdict: OVERRIDDEN_BY_OPERATOR\n    judge_model_verdict: ACCEPTED",
        ),
    ],
)
def test_tampered_judge_metadata_signature_fails_at_source_root_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[str], str],
) -> None:
    """Editing the verdict quartet without recomputing the HMAC is rejected."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "tampered.py", "payload = {'name': 'Ada'}\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="tampered.py:R1:fn:fp=somehash",
        file_fingerprint=file_fp,
        ast_path="body[0]",
    )
    original = allowlist.read_text(encoding="utf-8")
    tampered = mutate(original)
    assert tampered != original
    allowlist.write_text(tampered, encoding="utf-8")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    with pytest.raises(ValueError, match=r"allow_hits\[0\].*judge_metadata_signature.*mismatch"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)


def test_signed_judge_metadata_requires_hmac_key_at_source_root_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signature cannot be verified honestly when the deployment key is absent."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "needs_key.py", "payload = {'name': 'Ada'}\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="needs_key.py:R1:fn:fp=somehash",
        file_fingerprint=file_fp,
        ast_path="body[0]",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)

    with pytest.raises(ValueError, match=r"ELSPETH_JUDGE_METADATA_HMAC_KEY"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)


def test_source_root_load_can_skip_hmac_recompute_for_untrusted_fork_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fork PR static analysis can inspect signed entries without the private key."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "fork_ci.py", "payload = {'name': 'Ada'}\n")
    file_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="fork_ci.py:R1:fn:fp=somehash",
        file_fingerprint=file_fp,
        ast_path="body[0]",
    )
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE", "shape-only-when-key-missing")

    loaded = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)

    assert len(loaded.entries) == 1
    assert loaded.entries[0].judge_metadata_signature is not None


def test_load_without_source_root_skips_recompute_but_still_requires_binding(tmp_path: Path) -> None:
    """Loaders that don't have a source root (override_rate, judge_coverage) still enforce co-presence.

    ``source_root=None`` skips the live-bytes recompute (which would be
    meaningless without a tree to read), but invariant 8 (binding co-
    presence) still fires from ``_validate_judge_metadata_atomic``.
    This is the contract: aggregate-shape loaders see fewer guarantees
    than rule loaders, but never see broken-shape entries.
    """
    # Missing binding fields: invariant 8 fires even without source_root.
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "a.py:R1:fn:fp=1"
            owner: agent
            reason: tier-3 boundary
            safety: low
            judge_verdict: ACCEPTED
            judge_recorded_at: 2026-05-01T10:00:00+00:00
            judge_model: claude-opus-4-7
            judge_policy_hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
            judge_rationale: judge agrees
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"binding fields are missing"):
        load_allowlist(path, valid_rule_ids=set())  # source_root defaults to None


def test_shape_only_downgrade_emits_warning_on_source_root_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A source-root load in shape-only mode with the key absent warns loudly.

    Silently degrading the HMAC verification control violates the no-silent-
    failures doctrine: a reviewer must never mistake a green shape-only run for a
    cryptographically verified one. The warning fires once per load, at the load
    boundary, regardless of how many entries are present.
    """
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE", "shape-only-when-key-missing")
    monkeypatch.delenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", raising=False)
    enforce_dir = tmp_path / "enforce_tier_model"
    enforce_dir.mkdir()
    (enforce_dir / "web.yaml").write_text("allow_hits: []\n")
    source_root = tmp_path / "src"
    source_root.mkdir()

    load_allowlist(enforce_dir, valid_rule_ids={"R1", "R5"}, source_root=source_root)

    err = capsys.readouterr().err
    assert "DOWNGRADED to shape-only" in err
    assert "CANNOT detect forged or tampered judge metadata" in err


def test_required_mode_source_root_load_is_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In the default ``required`` mode (key present) there is no downgrade warning."""
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE", "required")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", "x" * 32)
    enforce_dir = tmp_path / "enforce_tier_model"
    enforce_dir.mkdir()
    (enforce_dir / "web.yaml").write_text("allow_hits: []\n")
    source_root = tmp_path / "src"
    source_root.mkdir()

    load_allowlist(enforce_dir, valid_rule_ids={"R1", "R5"}, source_root=source_root)

    assert "DOWNGRADED" not in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Additive v2 binding fields: scope_fingerprint + judge_signature_version
#
# Parse with ``source_root=None`` so signature verification and the
# file_fingerprint live-source check are both skipped (both gate on
# ``source_root``), letting a shape-only dummy signature suffice.
#
# NOTE: since Task 4, the atomic validator is version-aware — a v2 entry
# binds via scope_fingerprint and must NOT carry the v1 file_fingerprint.
# The round-trip below therefore uses the *valid* v2 shape (scope-only).
# ---------------------------------------------------------------------------

_VALID_FINGERPRINT = "0" * 64


def _valid_post_judge_entry() -> dict[str, object]:
    """A complete, currently-valid post-judge (v1) allow_hits entry dict."""
    return {
        "key": "a:b:c:fp=1",
        "owner": "test-agent",
        "reason": "tier-3 boundary",
        "safety": "low",
        "judge_verdict": "ACCEPTED",
        "judge_recorded_at": "2026-05-01T10:00:00+00:00",
        "judge_model": "claude-opus-4-7",
        "judge_policy_hash": "sha256:" + _VALID_FINGERPRINT,
        "judge_rationale": "judge agrees",
        "judge_metadata_signature": "hmac-sha256:v1:" + _VALID_FINGERPRINT,
        "file_fingerprint": _VALID_FINGERPRINT,
        "ast_path": "body[0]",
    }


def test_scope_fingerprint_and_signature_version_round_trip() -> None:
    """A valid v2 entry (scope-only binding) round-trips the additive fields."""
    scope_fp = "a" * 64
    entry = _valid_post_judge_entry()
    del entry["file_fingerprint"]
    entry["scope_fingerprint"] = scope_fp
    entry["judge_signature_version"] = 2
    entry["judge_metadata_signature"] = "hmac-sha256:v2:" + _VALID_FINGERPRINT
    data = {"allow_hits": [entry]}

    entries = _parse_allow_hits(data, source_file="x.yaml", source_root=None)

    assert len(entries) == 1
    assert entries[0].scope_fingerprint == scope_fp
    assert entries[0].judge_signature_version == 2
    assert entries[0].file_fingerprint is None


def test_invalid_judge_signature_version_crashes() -> None:
    """A version other than 1 or 2 is corruption and crashes on load."""
    entry = _valid_post_judge_entry()
    entry["judge_signature_version"] = 3
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match=r"1 or 2"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_boolean_judge_signature_version_crashes() -> None:
    """A YAML bool must not be read as version 1 (bool is an int subclass)."""
    entry = _valid_post_judge_entry()
    entry["judge_signature_version"] = True
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match=r"not a boolean"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


# ---------------------------------------------------------------------------
# Versioned signature payload (Task 4): v1 binds file_fingerprint, v2 binds
# scope_fingerprint, and the version lives INSIDE the signed payload so a
# v1<->v2 flip is unforgeable. These exercise ``compute_judge_metadata_signature``
# directly with an injected test key (the real function, not the hand-rolled
# v1-only ``_expected_judge_metadata_signature`` above).
# ---------------------------------------------------------------------------

_TEST_KEY = b"x" * 32
_AWARE_DT = datetime(2026, 5, 23, tzinfo=UTC)


def _sig(**overrides: object) -> str:
    base: dict[str, object] = {
        "key": "core/x.py:R6:C:m:fp=abc",
        "ast_path": "body[0]/body[0]",
        "judge_verdict": JudgeVerdict.ACCEPTED,
        "judge_recorded_at": _AWARE_DT,
        "judge_model": "anthropic/claude-opus",
        "judge_rationale": "external call boundary",
        "judge_policy_hash": "sha256:" + "0" * 64,
        "hmac_key": _TEST_KEY,
    }
    base.update(overrides)
    return compute_judge_metadata_signature(**base)  # type: ignore[arg-type]


def test_v2_signature_binds_scope_fingerprint() -> None:
    sig = _sig(signature_version=2, scope_fingerprint="a" * 64)
    assert sig.startswith("hmac-sha256:v2:")


def test_v1_signature_binds_file_fingerprint_and_keeps_v1_prefix() -> None:
    sig = _sig(signature_version=1, file_fingerprint="b" * 64)
    assert sig.startswith("hmac-sha256:v1:")


def test_v1_and_v2_signatures_differ_for_same_logical_entry() -> None:
    v1 = _sig(signature_version=1, file_fingerprint="b" * 64)
    v2 = _sig(signature_version=2, scope_fingerprint="a" * 64)
    assert not hmac.compare_digest(v1, v2)


def test_v2_requires_scope_fingerprint() -> None:
    with pytest.raises(ValueError, match="scope_fingerprint"):
        _sig(signature_version=2, scope_fingerprint=None)


def test_v1_requires_file_fingerprint() -> None:
    with pytest.raises(ValueError, match="file_fingerprint"):
        _sig(signature_version=1, file_fingerprint=None)


def test_default_signature_version_is_v1() -> None:
    """Omitting ``signature_version`` keeps the v1 prefix (load-bearing default).

    Direct callers that predate the v2 migration (notably the justify write
    path) omit the version argument; they must keep emitting v1 signatures.
    """
    sig = _sig(file_fingerprint="b" * 64)
    assert sig.startswith("hmac-sha256:v1:")


def test_unknown_signature_version_crashes() -> None:
    with pytest.raises(ValueError, match="unknown signature_version"):
        _sig(signature_version=3, file_fingerprint="b" * 64)


# ---------------------------------------------------------------------------
# Version-aware atomic validator (Task 4): invariant 8 dispatches on
# ``judge_signature_version`` to require the correct binding field per
# version and forbid the other; invariant 4 treats scope_fingerprint /
# judge_signature_version as stray on a pre-judge entry. Parsed with
# ``source_root=None`` so signature verification + file_fingerprint live-source
# recompute are both skipped (a shape-only dummy signature suffices).
# ---------------------------------------------------------------------------


def test_v2_entry_with_scope_fingerprint_and_no_file_fingerprint_validates() -> None:
    """A v2 post-judge entry binds via scope_fingerprint, not file_fingerprint."""
    entry = _valid_post_judge_entry()
    del entry["file_fingerprint"]
    entry["judge_signature_version"] = 2
    entry["scope_fingerprint"] = "a" * 64
    entry["judge_metadata_signature"] = "hmac-sha256:v2:" + _VALID_FINGERPRINT
    data = {"allow_hits": [entry]}

    entries = _parse_allow_hits(data, source_file="x.yaml", source_root=None)

    assert len(entries) == 1
    assert entries[0].judge_signature_version == 2
    assert entries[0].scope_fingerprint == "a" * 64
    assert entries[0].file_fingerprint is None


def test_v2_entry_with_file_fingerprint_crashes() -> None:
    """A v2 entry must NOT carry the v1 whole-file file_fingerprint."""
    entry = _valid_post_judge_entry()
    entry["judge_signature_version"] = 2
    entry["scope_fingerprint"] = "a" * 64
    entry["judge_metadata_signature"] = "hmac-sha256:v2:" + _VALID_FINGERPRINT
    # file_fingerprint left present from the valid-entry template — corruption.
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match="must not carry"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_v2_entry_missing_scope_fingerprint_crashes() -> None:
    """A v2 post-judge entry without scope_fingerprint is unbound."""
    entry = _valid_post_judge_entry()
    del entry["file_fingerprint"]
    entry["judge_signature_version"] = 2
    entry["judge_metadata_signature"] = "hmac-sha256:v2:" + _VALID_FINGERPRINT
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match=r"binding fields are missing.*scope_fingerprint"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_v1_entry_with_scope_fingerprint_crashes() -> None:
    """A v1 (or absent-version) entry carrying scope_fingerprint is corruption."""
    entry = _valid_post_judge_entry()
    entry["scope_fingerprint"] = "a" * 64
    # judge_signature_version absent -> treated as v1; file_fingerprint present.
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match="corruption"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_pre_judge_entry_with_stray_scope_fingerprint_crashes() -> None:
    """A pre-judge entry must not carry scope_fingerprint (invariant 4)."""
    entry = {
        "key": "a:b:c:fp=1",
        "owner": "test-agent",
        "reason": "pre-judge era entry",
        "safety": "low",
        "scope_fingerprint": "a" * 64,
    }
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match=r"scope_fingerprint.*pre-judge"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_pre_judge_entry_with_stray_signature_version_crashes() -> None:
    """A pre-judge entry must not carry judge_signature_version (invariant 4)."""
    entry = {
        "key": "a:b:c:fp=1",
        "owner": "test-agent",
        "reason": "pre-judge era entry",
        "safety": "low",
        "judge_signature_version": 2,
    }
    data = {"allow_hits": [entry]}

    with pytest.raises(ValueError, match=r"judge_signature_version.*pre-judge"):
        _parse_allow_hits(data, source_file="x.yaml", source_root=None)


def test_v2_entry_at_match_time_verifies_scope_fingerprint() -> None:
    """A v2 entry is verified at match time against the live scope_fingerprint.

    ``verify_entry_binding_against_finding`` compares the persisted
    enclosing-scope ``scope_fingerprint`` the judge inspected against the
    value the scanner stamped on the live finding. A match passes; a drift
    crashes (the function/class the judge inspected changed, re-justify is
    required). The dedicated v1/v2 match-binding matrix lives in
    ``test_allowlist_match_binding.py``; this case guards the v2 path from
    the judge-metadata-integrity surface.
    """
    entry = AllowlistEntry(
        key="core/x.py:R6:C:m:fp=abc",
        owner="test-agent",
        reason="tier-3 boundary",
        safety="low",
        expires=None,
        ast_path="body[0]",
        scope_fingerprint="a" * 64,
        file_fingerprint=None,
        judge_signature_version=2,
        judge_verdict=JudgeVerdict.ACCEPTED,
    )

    # Matching live scope_fingerprint: no raise.
    verify_entry_binding_against_finding(entry, file_path="core/x.py", ast_path="body[0]", scope_fingerprint="a" * 64)

    # Drifted live scope_fingerprint: crash-loud.
    with pytest.raises(ValueError, match="scope_fingerprint"):
        verify_entry_binding_against_finding(entry, file_path="core/x.py", ast_path="body[0]", scope_fingerprint="b" * 64)


# ---------------------------------------------------------------------------
# Load-time source binding: v2 is file-exists only (scope verified at match
# time); v1 retains the whole-file byte-hash recompute.
# ---------------------------------------------------------------------------


def _write_v2_judge_gated_yaml(
    *,
    yaml_path: Path,
    key: str,
    scope_fingerprint: str,
    ast_path: str,
) -> None:
    """Write one v2 judge-gated allowlist entry (scope-bound, no file_fingerprint)."""
    signature = compute_judge_metadata_signature(
        key=key,
        ast_path=ast_path,
        judge_verdict=JudgeVerdict.ACCEPTED,
        judge_recorded_at=datetime(2026, 5, 23, 0, 0, 0, tzinfo=UTC),
        judge_model="anthropic/claude-opus-4-7",
        judge_rationale="judge accepted the suppression",
        judge_policy_hash=JUDGE_POLICY_HASH,
        signature_version=2,
        scope_fingerprint=scope_fingerprint,
        hmac_key=_TEST_JUDGE_METADATA_HMAC_KEY.encode("utf-8"),
    )
    lines = [
        "allow_hits:",
        f"  - key: {key}",
        "    owner: test-agent",
        "    reason: tier-3 boundary",
        "    safety: low",
        "    judge_verdict: ACCEPTED",
        "    judge_recorded_at: '2026-05-23T00:00:00+00:00'",
        "    judge_model: anthropic/claude-opus-4-7",
        f"    judge_policy_hash: '{JUDGE_POLICY_HASH}'",
        "    judge_rationale: judge accepted the suppression",
        "    judge_signature_version: 2",
        f"    scope_fingerprint: '{scope_fingerprint}'",
        f"    ast_path: '{ast_path}'",
        f"    judge_metadata_signature: '{signature}'",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_v2_entry_loads_when_file_present_even_if_bytes_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A v2 entry binds by scope, not whole-file bytes; an unrelated edit must not crash the load.

    This is the relief this change delivers: under v1, any edit to the
    source file invalidated the whole-file file_fingerprint and crashed the
    load. v2 has no load-time whole-file hash (scope is verified at match
    time), so editing an unrelated line leaves the load green.

    The v2 LOAD path verifies only file-exists + the HMAC signature, so the
    ``scope_fingerprint`` value is arbitrary here (no scanning needed); the
    signature is computed over that arbitrary value with the test key.
    """
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "scoped.py", "def f():\n    return 1\n")
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    allowlist = tmp_path / "allowlist.yaml"
    _write_v2_judge_gated_yaml(
        yaml_path=allowlist,
        key="scoped.py:R1:f:fp=somehash",
        scope_fingerprint="a" * 64,
        ast_path="body[0]",
    )

    # First load succeeds with the file as written.
    loaded = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)
    assert len(loaded.entries) == 1
    assert loaded.entries[0].judge_signature_version == 2
    assert loaded.entries[0].scope_fingerprint == "a" * 64
    assert loaded.entries[0].file_fingerprint is None

    # Mutate an UNRELATED line in the source. Under v1 this would flip the
    # whole-file fingerprint and crash; under v2 the load must stay green.
    src_path.write_text("def f():\n    return 1  # unrelated comment\n", encoding="utf-8")
    reloaded = load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)
    assert len(reloaded.entries) == 1


def test_v2_entry_crashes_at_load_when_source_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The version-independent file-exists guard still fires for v2 entries."""
    source_root = tmp_path / "src"
    source_root.mkdir(parents=True, exist_ok=True)
    # Deliberately do NOT create gone.py under source_root.
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    allowlist = tmp_path / "allowlist.yaml"
    _write_v2_judge_gated_yaml(
        yaml_path=allowlist,
        key="gone.py:R1:f:fp=somehash",
        scope_fingerprint="a" * 64,
        ast_path="body[0]",
    )

    with pytest.raises(ValueError, match="does not exist"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)


def test_v1_entry_still_crashes_on_whole_file_byte_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The retained v1 path is unchanged: a v1 entry whose source bytes changed crashes."""
    source_root = tmp_path / "src"
    src_path = _write_source(source_root, "v1drift.py", "# original content\n")
    original_fp = hashlib.sha256(src_path.read_bytes()).hexdigest()
    monkeypatch.setenv("ELSPETH_JUDGE_METADATA_HMAC_KEY", _TEST_JUDGE_METADATA_HMAC_KEY)

    allowlist = tmp_path / "allowlist.yaml"
    _write_judge_gated_yaml(
        yaml_path=allowlist,
        key="v1drift.py:R1:fn:fp=somehash",
        file_fingerprint=original_fp,
        ast_path="body[0]",
    )

    # Source unchanged: v1 binding holds.
    load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)

    # Drift the bytes: the v1 whole-file recompute must crash.
    src_path.write_text("# changed content\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file_fingerprint mismatch"):
        load_allowlist(allowlist, valid_rule_ids=set(), source_root=source_root)

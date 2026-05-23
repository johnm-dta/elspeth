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
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import JudgeVerdict, load_allowlist

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
    ["judge_recorded_at", "judge_model", "judge_rationale"],
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
) -> None:
    """Write one judge-gated allowlist entry to ``yaml_path``."""
    yaml_path.write_text(
        textwrap.dedent(f"""\
            allow_hits:
              - key: {key}
                owner: test-agent
                reason: tier-3 boundary
                safety: low
                judge_verdict: ACCEPTED
                judge_recorded_at: '2026-05-23T00:00:00+00:00'
                judge_model: anthropic/claude-opus-4
                judge_rationale: judge accepted the suppression
                file_fingerprint: '{file_fingerprint}'
                ast_path: '{ast_path}'
            """)
    )


def test_transplanted_quartet_across_files_fails_at_load(tmp_path: Path) -> None:
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


def test_source_drift_after_judge_verdict_fails_at_load(tmp_path: Path) -> None:
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
            judge_rationale: judge agrees
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    with pytest.raises(ValueError, match=r"binding fields are missing"):
        load_allowlist(path, valid_rule_ids=set())  # source_root defaults to None

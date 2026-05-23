"""Unit tests for ``elspeth_lints.rules.trust_tier.tier_model.rotate``.

These exercise the pure classification function ``plan_rotations`` plus
the ``apply_plan`` writer. The end-to-end ``scan_for_rotations`` is covered
implicitly via the CLI smoke (a one-shot dry-run against the real allowlist
ran as part of slice 1 verification); these tests target the algorithm.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import AllowlistEntry, PerFileRule
from elspeth_lints.rules.trust_tier.tier_model.rotate import (
    AmbiguousGroup,
    Rotation,
    apply_plan,
    fingerprint_of,
    identity_prefix,
    plan_rotations,
)
from elspeth_lints.rules.trust_tier.tier_model.rotate import (
    StaleEntry as StaleEntryForTest,
)
from elspeth_lints.rules.trust_tier.tier_model.rule import Finding


def _make_finding(*, file_path: str, rule_id: str, symbol: tuple[str, ...], fingerprint: str) -> Finding:
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line=1,
        col=1,
        symbol_context=symbol,
        fingerprint=fingerprint,
        code_snippet="<test>",
        message="<test>",
    )


def _make_entry(
    *,
    file_path: str,
    rule_id: str,
    symbol: tuple[str, ...],
    fingerprint: str,
    owner: str = "team",
    reason: str = "boundary",
    safety: str = "validated",
    expires: date | None = None,
    source_file: str = "test.yaml",
) -> AllowlistEntry:
    symbol_part = ":".join(symbol) if symbol else "_module_"
    key = f"{file_path}:{rule_id}:{symbol_part}:fp={fingerprint}"
    return AllowlistEntry(
        key=key,
        owner=owner,
        reason=reason,
        safety=safety,
        expires=expires,
        source_file=source_file,
    )


# ---------- parse helpers ----------


def test_identity_prefix_strips_fingerprint_suffix() -> None:
    key = "web/auth.py:R1:Provider:authenticate:fp=abcd1234"
    assert identity_prefix(key) == "web/auth.py:R1:Provider:authenticate"


def test_fingerprint_of_extracts_hex() -> None:
    key = "web/auth.py:R1:Provider:authenticate:fp=abcd1234"
    assert fingerprint_of(key) == "abcd1234"


def test_identity_prefix_raises_without_fp_marker() -> None:
    with pytest.raises(ValueError, match=":fp="):
        identity_prefix("not-a-canonical-key")


def test_fingerprint_of_raises_without_fp_marker() -> None:
    with pytest.raises(ValueError, match=":fp="):
        fingerprint_of("not-a-canonical-key")


# ---------- plan_rotations classifications ----------


def test_one_to_one_same_fp_is_unchanged() -> None:
    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa")
    entry = _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa")
    plan = plan_rotations(findings=[finding], allowlist_entries=[entry])
    assert plan.unchanged_count == 1
    assert plan.rotations == ()
    assert plan.ambiguous == ()
    assert plan.stale_entries == ()


def test_one_to_one_different_fp_is_rotation() -> None:
    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="bbbb")
    entry = _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa", source_file="web.yaml")
    plan = plan_rotations(findings=[finding], allowlist_entries=[entry])
    assert len(plan.rotations) == 1
    rotation = plan.rotations[0]
    assert rotation.old_key.endswith(":fp=aaaa")
    assert rotation.new_key.endswith(":fp=bbbb")
    assert rotation.entry_source_file == "web.yaml"
    assert plan.unchanged_count == 0


def test_finding_covered_by_per_file_rule_is_not_classified_as_new() -> None:
    """Drop-in compatibility: wildcard per_file_rules suppress findings.

    Without this filter, every wildcard-allowlisted finding would be
    classified as 'new' and the tool would exit non-zero on a tree the
    production check considers clean.
    """
    finding = _make_finding(file_path="web/telemetry/foo.py", rule_id="R4", symbol=("emit",), fingerprint="abcd")
    pf_rule = PerFileRule(
        pattern="web/telemetry/*",
        rules=("R4",),
        reason="W5 telemetry-only exemption",
        expires=None,
        max_hits=10,
        source_file="web.yaml",
    )
    plan = plan_rotations(findings=[finding], allowlist_entries=[], per_file_rules=[pf_rule])
    assert plan.new_finding_count == 0
    assert not plan.has_new_findings


def test_finding_with_no_entry_counts_as_new() -> None:
    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa")
    plan = plan_rotations(findings=[finding], allowlist_entries=[])
    assert plan.new_finding_count == 1
    assert plan.has_new_findings
    assert plan.new_findings[0].canonical_key.endswith(":fp=aaaa")
    assert plan.new_findings[0].rule_id == "R1"
    assert plan.rotations == ()


def test_entry_with_no_finding_is_stale() -> None:
    entry = _make_entry(
        file_path="a.py",
        rule_id="R1",
        symbol=("ghost",),
        fingerprint="aaaa",
        owner="web-auth",
        reason="legacy",
    )
    plan = plan_rotations(findings=[], allowlist_entries=[entry])
    assert plan.stale_entry_count == 1
    stale = plan.stale_entries[0]
    assert stale.key.endswith(":fp=aaaa")
    assert stale.owner == "web-auth"
    assert stale.reason == "legacy"


def test_n_to_n_without_symmetric_flag_is_ambiguous() -> None:
    findings = [_make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("aaaa", "bbbb")]
    entries = [_make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("cccc", "dddd")]
    plan = plan_rotations(findings=findings, allowlist_entries=entries, allow_symmetric_pairing=False)
    assert plan.rotations == ()
    assert len(plan.ambiguous) == 1
    group = plan.ambiguous[0]
    assert isinstance(group, AmbiguousGroup)
    assert group.finding_count == 2
    assert group.entry_count == 2


def test_n_to_n_with_default_symmetric_pairing_rotates_deterministically() -> None:
    """Drop-in compatibility check: the prior tool paired N:N groups by default."""
    findings = [_make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("dddd", "cccc")]
    entries = [_make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("bbbb", "aaaa")]
    # No allow_symmetric_pairing kwarg => uses the new default (True).
    plan = plan_rotations(findings=findings, allowlist_entries=entries)
    assert plan.ambiguous == ()
    assert len(plan.rotations) == 2


def test_n_to_n_with_symmetric_flag_produces_sorted_pairing() -> None:
    findings = [
        _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f)
        for f in ("dddd", "cccc")  # intentionally not sorted
    ]
    entries = [
        _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f, source_file="web.yaml")
        for f in ("bbbb", "aaaa")  # intentionally not sorted
    ]
    plan = plan_rotations(findings=findings, allowlist_entries=entries, allow_symmetric_pairing=True)
    assert plan.ambiguous == ()
    assert len(plan.rotations) == 2
    # Sorted pairing: aaaa<->cccc, bbbb<->dddd (entries sorted by key, findings by canonical_key)
    old_to_new = {r.old_key: r.new_key for r in plan.rotations}
    aaaa_key = "a.py:R1:foo:fp=aaaa"
    bbbb_key = "a.py:R1:foo:fp=bbbb"
    cccc_key = "a.py:R1:foo:fp=cccc"
    dddd_key = "a.py:R1:foo:fp=dddd"
    assert old_to_new[aaaa_key] == cccc_key
    assert old_to_new[bbbb_key] == dddd_key


def test_asymmetric_n_to_m_remains_ambiguous_even_with_flag() -> None:
    findings = [
        _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa"),
    ]
    entries = [_make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("bbbb", "cccc")]
    plan = plan_rotations(findings=findings, allowlist_entries=entries, allow_symmetric_pairing=True)
    assert len(plan.ambiguous) == 1
    assert plan.rotations == ()


def test_todo_entries_are_surfaced_regardless_of_match_state() -> None:
    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa")
    matched_todo = _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa", owner="TODO", reason="TODO -- review")
    plan = plan_rotations(findings=[finding], allowlist_entries=[matched_todo])
    assert plan.unchanged_count == 1
    assert plan.todo_entry_count == 1
    assert plan.todo_entries[0].owner == "TODO"


def test_symmetric_pairing_with_some_matching_fps_marks_those_unchanged() -> None:
    findings = [_make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("aaaa", "dddd")]
    entries = [_make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint=f) for f in ("aaaa", "bbbb")]
    plan = plan_rotations(findings=findings, allowlist_entries=entries, allow_symmetric_pairing=True)
    # Sorted entries: aaaa, bbbb; sorted findings: aaaa, dddd.
    # Pair 0: aaaa<->aaaa = unchanged. Pair 1: bbbb<->dddd = rotation.
    assert plan.unchanged_count == 1
    assert len(plan.rotations) == 1
    assert plan.rotations[0].old_key.endswith(":fp=bbbb")
    assert plan.rotations[0].new_key.endswith(":fp=dddd")


# ---------- apply_plan ----------


def test_apply_plan_rewrites_yaml_text_surgically(tmp_path: Path) -> None:
    yaml_path = tmp_path / "web.yaml"
    yaml_path.write_text(
        """# preserved comment
allow_hits:
- key: a.py:R1:foo:fp=aaaa
  owner: team
  # inline note preserved
  reason: boundary
""",
        encoding="utf-8",
    )
    rotation = Rotation(
        old_key="a.py:R1:foo:fp=aaaa",
        new_key="a.py:R1:foo:fp=bbbb",
        entry_source_file=str(yaml_path),
    )
    plan = _empty_plan_with(rotations=(rotation,))
    applied = apply_plan(plan)
    assert set(applied) == {str(yaml_path)}
    assert applied[str(yaml_path)].rotations_applied == 1
    new_text = yaml_path.read_text(encoding="utf-8")
    assert "fp=aaaa" not in new_text
    assert "fp=bbbb" in new_text
    # Comments preserved (surgical replacement, not yaml.safe_dump)
    assert "# preserved comment" in new_text
    assert "# inline note preserved" in new_text


def test_apply_plan_removes_stale_entries_and_preserves_comments(tmp_path: Path) -> None:
    """Drop-in compatibility check: the prior tool removed stale entries by default."""
    yaml_path = tmp_path / "web.yaml"
    yaml_path.write_text(
        """# top-level rationale comment
allow_hits:
- key: a.py:R1:keep:fp=1111
  owner: team
  reason: still in use
- key: a.py:R1:gone:fp=2222
  owner: team
  reason: code was deleted
- key: b.py:R1:also_keep:fp=3333
  owner: team
  # inline comment on a kept entry
  reason: still in use
""",
        encoding="utf-8",
    )
    stale = StaleEntryForTest(
        key="a.py:R1:gone:fp=2222",
        source_file=str(yaml_path),
        owner="team",
        reason="code was deleted",
    )
    plan = _empty_plan_with(rotations=(), stale_entries=(stale,))
    applied = apply_plan(plan)
    assert applied[str(yaml_path)].stale_entries_removed == 1
    assert applied[str(yaml_path)].rotations_applied == 0
    new_text = yaml_path.read_text(encoding="utf-8")
    # Stale block fully gone
    assert "a.py:R1:gone" not in new_text
    assert "code was deleted" not in new_text
    # Other entries and their comments preserved byte-for-byte
    assert "# top-level rationale comment" in new_text
    assert "a.py:R1:keep:fp=1111" in new_text
    assert "b.py:R1:also_keep:fp=3333" in new_text
    assert "# inline comment on a kept entry" in new_text


def test_apply_plan_remove_stale_false_keeps_entries(tmp_path: Path) -> None:
    """Opt-out path: --no-remove-stale leaves stale entries in place."""
    yaml_path = tmp_path / "web.yaml"
    yaml_path.write_text(
        "allow_hits:\n- key: a.py:R1:gone:fp=2222\n  owner: team\n  reason: removed\n",
        encoding="utf-8",
    )
    stale = StaleEntryForTest(
        key="a.py:R1:gone:fp=2222",
        source_file=str(yaml_path),
        owner="team",
        reason="removed",
    )
    plan = _empty_plan_with(rotations=(), stale_entries=(stale,))
    apply_plan(plan, remove_stale=False)
    new_text = yaml_path.read_text(encoding="utf-8")
    assert "a.py:R1:gone:fp=2222" in new_text


def test_apply_plan_raises_when_old_key_missing(tmp_path: Path) -> None:
    yaml_path = tmp_path / "web.yaml"
    yaml_path.write_text("allow_hits: []\n", encoding="utf-8")
    rotation = Rotation(
        old_key="a.py:R1:foo:fp=missing",
        new_key="a.py:R1:foo:fp=new",
        entry_source_file=str(yaml_path),
    )
    plan = _empty_plan_with(rotations=(rotation,))
    with pytest.raises(RuntimeError, match="not found"):
        apply_plan(plan)


def test_apply_plan_raises_when_old_key_duplicated(tmp_path: Path) -> None:
    yaml_path = tmp_path / "web.yaml"
    duplicated = "a.py:R1:foo:fp=aaaa"
    yaml_path.write_text(
        f"# duplicated key fixture\n- key: {duplicated}\n- key: {duplicated}\n",
        encoding="utf-8",
    )
    rotation = Rotation(
        old_key=duplicated,
        new_key="a.py:R1:foo:fp=bbbb",
        entry_source_file=str(yaml_path),
    )
    plan = _empty_plan_with(rotations=(rotation,))
    with pytest.raises(RuntimeError, match="occurs 2x"):
        apply_plan(plan)


def _empty_plan_with(
    *,
    rotations: tuple[Rotation, ...] = (),
    stale_entries: tuple[StaleEntryForTest, ...] = (),
):
    from elspeth_lints.rules.trust_tier.tier_model.rotate import RotationPlan

    return RotationPlan(
        rotations=rotations,
        ambiguous=(),
        stale_entries=stale_entries,
        todo_entries=(),
        new_findings=(),
        unchanged_count=0,
    )


# =============================================================================
# C8-3: rotate refuses to silently re-bind judge-gated entries
# =============================================================================
#
# Rotation rebinds an entry's canonical key (and its embedded fingerprint)
# to a new AST signature. For a judge-gated entry the persisted
# file_fingerprint + ast_path were bound to the EXACT bytes and AST node the
# judge inspected; auto-rotating the key would leave the binding fields
# describing the original code while the key pointed at the new code. The
# audit-honest response is to refuse: the operator deletes the stale entry,
# re-justifies against the rotated location, and the new quartet records
# what the judge actually said about the new location.
# =============================================================================


@pytest.mark.parametrize(
    "verdict_name",
    ["ACCEPTED", "OVERRIDDEN_BY_OPERATOR"],
    ids=["judge_accepted", "operator_overridden"],
)
def test_rotate_refuses_judge_gated_entry_one_to_one(verdict_name: str) -> None:
    """A 1:1 rotation candidate carrying ANY judge_verdict raises before producing a Rotation.

    Parametrised over both terminal verdict shapes that gate the
    refusal: ``ACCEPTED`` (the judge agreed the entry can stand) and
    ``OVERRIDDEN_BY_OPERATOR`` (the operator chose to bypass the
    judge's verdict). The refusal is gated on
    ``entry.judge_verdict is not None`` so both arms must trigger; T9
    review verified ``OVERRIDDEN_BY_OPERATOR`` empirically but a
    future ``JudgeVerdict`` enum addition could silently break the
    guard without this parametrisation pinning both shapes.
    """
    from datetime import UTC, datetime

    from elspeth_lints.core.allowlist import JudgeVerdict

    verdict = JudgeVerdict[verdict_name]

    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="bbbb")
    entry = _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa", source_file="web.yaml")
    # Convert to judge-gated.
    entry.judge_verdict = verdict
    entry.judge_recorded_at = datetime(2026, 5, 1, tzinfo=UTC)
    entry.judge_model = "anthropic/claude-opus-4"
    entry.judge_rationale = f"verdict={verdict_name}"
    entry.file_fingerprint = "0" * 64
    entry.ast_path = "body[0]"

    with pytest.raises(RuntimeError, match=r"refusing to rotate judge-gated.*web\.yaml") as excinfo:
        plan_rotations(findings=[finding], allowlist_entries=[entry])
    # Message must name the actual verdict so the operator can
    # distinguish a judge refusal from an operator-override refusal
    # in CI logs.
    assert verdict_name in str(excinfo.value), (
        f"refusal message must name the gating verdict so the operator "
        f"knows whether to delete the entry and re-justify (ACCEPTED) "
        f"or re-evaluate the override (OVERRIDDEN_BY_OPERATOR). Got: "
        f"{excinfo.value!s}"
    )


def test_rotate_allows_pre_judge_entry_to_rotate_normally() -> None:
    """Pre-judge entries (no judge_verdict) rotate without obstruction.

    Pins the backward-compatible behaviour: the ~700-entry historical
    corpus has no judge metadata, so the refusal doesn't affect any
    existing rotation workflow. Refusal is gated exclusively on
    ``judge_verdict is not None``.
    """
    finding = _make_finding(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="bbbb")
    entry = _make_entry(file_path="a.py", rule_id="R1", symbol=("foo",), fingerprint="aaaa", source_file="web.yaml")
    # No judge metadata — the historical-corpus shape.
    plan = plan_rotations(findings=[finding], allowlist_entries=[entry])
    assert len(plan.rotations) == 1

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
            judge_model_verdict: BLOCKED  # fabricated divergence on a non-override entry
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
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
def test_pre_judge_entry_with_stray_field_crashes(
    tmp_path: Path, stray_field: str, stray_value: str
) -> None:
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
    """).strip()
    path = tmp_path / "al.yaml"
    path.write_text(yaml)
    al = load_allowlist(path, valid_rule_ids=set())
    assert len(al.entries) == 1
    entry = al.entries[0]
    assert entry.judge_verdict is JudgeVerdict.ACCEPTED
    assert entry.judge_model_verdict is None  # absence is the signal: no divergence

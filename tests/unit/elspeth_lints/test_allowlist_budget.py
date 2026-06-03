"""Tests for the AllowlistBudgetViolation machinery promoted into core (Plan A Task 4)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.allowlist import (
    Allowlist,
    AllowlistBudgetViolation,
    load_allowlist,
)


def test_budget_violation_is_frozen_slots_dataclass() -> None:
    v = AllowlistBudgetViolation(category="allow_hits", current=5, max_allowed=3)
    with pytest.raises(AttributeError):
        v.current = 6  # type: ignore[misc]  # frozen


def test_default_allowlist_has_permissive_budget() -> None:
    """A freshly-constructed Allowlist with no ceilings reports no violations."""
    al = Allowlist(entries=[], per_file_rules=[])
    assert al.max_allow_hits is None
    assert al.get_budget_violations() == []


def test_allow_hits_ceiling_violated(tmp_path: Path) -> None:
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget:
            max_allow_hits: 1
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: qa
            reason: ok
            safety: x
          - key: "a:b:c:fp=2"
            owner: qa
            reason: ok
            safety: x
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    al = load_allowlist(p, valid_rule_ids=set())
    violations = al.get_budget_violations()
    assert len(violations) == 1
    assert violations[0] == AllowlistBudgetViolation(category="allow_hits", current=2, max_allowed=1)


def test_permanent_distinguished_from_total(tmp_path: Path) -> None:
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget:
            max_permanent_allow_hits: 0
        allow_hits:
          - key: "a:b:c:fp=1"
            owner: qa
            reason: ok
            safety: x
            # no expires -> permanent
          - key: "a:b:c:fp=2"
            owner: qa
            reason: ok
            safety: x
            expires: 2030-01-01
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    al = load_allowlist(p, valid_rule_ids=set())
    violations = al.get_budget_violations()
    categories = {v.category for v in violations}
    assert "permanent_allow_hits" in categories
    assert all(v.category != "permanent_per_file_rules" for v in violations)


def test_negative_ceiling_raises_valueerror(tmp_path: Path) -> None:
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget:
            max_allow_hits: -1
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="non-negative"):
        load_allowlist(p, valid_rule_ids=set())


def test_non_integer_ceiling_raises_valueerror(tmp_path: Path) -> None:
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget:
            max_allow_hits: "lots"
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError):
        load_allowlist(p, valid_rule_ids=set())


def test_bool_ceiling_rejected(tmp_path: Path) -> None:
    """YAML ``true``/``false`` must not silently coerce to 1/0 at the Tier-3 boundary."""
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget:
            max_allow_hits: true
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="not a boolean"):
        load_allowlist(p, valid_rule_ids=set())


def test_block_level_null_budget_rejected(tmp_path: Path) -> None:
    """``allowlist_budget: null`` is ambiguous; require ``{}`` or omission."""
    yaml = textwrap.dedent("""
        defaults:
          allowlist_budget: null
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="allowlist_budget must be a mapping"):
        load_allowlist(p, valid_rule_ids=set())


def test_missing_budget_block_is_permissive(tmp_path: Path) -> None:
    yaml = textwrap.dedent("""
        defaults:
          fail_on_stale: true
        allow_hits: []
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    al = load_allowlist(p, valid_rule_ids=set())
    assert al.get_budget_violations() == []


def test_allow_hit_pattern_field_round_trips(tmp_path: Path) -> None:
    """The optional ``pattern`` field on allow_hits parses verbatim into AllowlistEntry.

    Plan A Task 7: tier_model encodes a typed pattern-tag on allow_hits entries
    (e.g. ``display-fallback``). Core stores the field as raw text — pattern-tag
    governance lives in tier_model's local validator, not in the core loader.
    """
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "src/x.py:R1:func:fp=abc"
            owner: bugfix
            reason: ok
            safety: s
            pattern: display-fallback
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    al = load_allowlist(p, valid_rule_ids={"R1"})
    assert len(al.entries) == 1
    assert al.entries[0].pattern == "display-fallback"


def test_allow_hit_pattern_field_defaults_to_none(tmp_path: Path) -> None:
    """Omitting ``pattern`` produces ``pattern=None`` (default)."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "src/x.py:R1:func:fp=abc"
            owner: qa
            reason: ok
            safety: s
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    al = load_allowlist(p, valid_rule_ids={"R1"})
    assert al.entries[0].pattern is None


def test_allow_hit_pattern_empty_string_rejected(tmp_path: Path) -> None:
    """Empty-string ``pattern`` is rejected by core's optional-string parser."""
    yaml = textwrap.dedent("""
        allow_hits:
          - key: "src/x.py:R1:func:fp=abc"
            owner: qa
            reason: ok
            safety: s
            pattern: ""
    """).strip()
    p = tmp_path / "al.yaml"
    p.write_text(yaml)
    with pytest.raises(ValueError, match="pattern"):
        load_allowlist(p, valid_rule_ids={"R1"})

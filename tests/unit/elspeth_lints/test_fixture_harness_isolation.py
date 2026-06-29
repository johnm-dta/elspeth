"""Regression tests for elspeth-lints fixture harness isolation."""

from __future__ import annotations

from pathlib import Path

from elspeth_lints.core.fixture_harness import RuleFixtureCase, assert_fixture_case
from elspeth_lints.rules.audit_evidence.guard_symmetry.rule import RULE as GUARD_SYMMETRY_RULE
from elspeth_lints.rules.immutability.freeze_guards.rule import RULE as FREEZE_GUARDS_RULE

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_SYMMETRY_FIXTURE = (
    REPO_ROOT
    / "elspeth-lints"
    / "src"
    / "elspeth_lints"
    / "rules"
    / "audit_evidence"
    / "guard_symmetry"
    / "fixtures"
    / "examples_violation"
    / "01_missing_loader_guard"
)
FREEZE_GUARDS_ALLOWLISTED_FIXTURE = (
    REPO_ROOT
    / "elspeth-lints"
    / "src"
    / "elspeth_lints"
    / "rules"
    / "immutability"
    / "freeze_guards"
    / "fixtures"
    / "examples_clean"
    / "03_allowlisted_isinstance"
)


def test_fixture_harness_ignores_ambient_parent_allowlist(tmp_path: Path, monkeypatch) -> None:
    ambient_allowlist_dir = tmp_path / "config" / "cicd" / "enforce_guard_symmetry"
    ambient_allowlist_dir.mkdir(parents=True)
    (ambient_allowlist_dir / "ambient.yaml").write_text(
        """
        per_file_rules:
          - pattern: "core/model_loaders.py"
            rules: [GS1]
            reason: ambient parent repo suppression must not affect fixtures
            expires: null
            max_hits: 99
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert_fixture_case(
        RuleFixtureCase(
            rule=GUARD_SYMMETRY_RULE,
            kind="examples_violation",
            fixture_path=GUARD_SYMMETRY_FIXTURE,
            expected_path=GUARD_SYMMETRY_FIXTURE.with_suffix(".expected.json"),
        )
    )


def test_fixture_harness_preserves_fixture_local_allowlist() -> None:
    assert_fixture_case(
        RuleFixtureCase(
            rule=FREEZE_GUARDS_RULE,
            kind="examples_clean",
            fixture_path=FREEZE_GUARDS_ALLOWLISTED_FIXTURE,
            expected_path=None,
        )
    )

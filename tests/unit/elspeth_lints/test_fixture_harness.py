"""Tests for the shared elspeth-lints fixture harness."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from elspeth_lints.core.fixture_harness import (
    discover_rule_fixture_cases,
    find_fixture_inventory_errors,
    find_rules_missing_fixtures,
)
from elspeth_lints.core.protocols import Category, Finding, RuleContext, RuleMetadata, RuleScope, Severity
from elspeth_lints.rules.meta_no_new_bespoke_cicd_enforcer import RULE


class RuleWithoutFixtures:
    id = "test.no-fixtures"
    scope = RuleScope.INCREMENTAL
    metadata = RuleMetadata(
        id=id,
        name="No fixtures",
        description="Used to prove missing fixtures fail clearly.",
        severity=Severity.ERROR,
        category=Category.MANIFEST,
        cwe=(),
        scope=scope,
        path_filter=r".*\.py$",
        examples_violation_count=1,
        examples_clean_count=1,
    )

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> Iterable[Finding]:
        return ()


def test_discovers_meta_rule_fixture_cases() -> None:
    cases = discover_rule_fixture_cases(RULE)

    assert [case.name for case in cases] == ["examples_violation/01_unmanifested_enforcer", "examples_clean/01_manifested_legacy"]


def test_missing_fixtures_are_reported() -> None:
    missing = find_rules_missing_fixtures([RuleWithoutFixtures()])

    assert missing == ["test.no-fixtures"]


def test_fixture_inventory_reports_missing_directories_and_metadata_drift() -> None:
    errors = find_fixture_inventory_errors([RuleWithoutFixtures()])

    assert errors == [
        "test.no-fixtures: missing examples_violation fixtures",
        "test.no-fixtures: missing examples_clean fixtures",
        "test.no-fixtures: metadata examples_violation_count=1 but discovered 0",
        "test.no-fixtures: metadata examples_clean_count=1 but discovered 0",
    ]

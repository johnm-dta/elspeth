"""Shared fixture harness coverage for built-in elspeth-lints rules."""

from __future__ import annotations

import sys

import pytest

from elspeth_lints.core.fixture_harness import (
    RuleFixtureCase,
    assert_fixture_case,
    discover_registry_fixture_cases,
    find_fixture_inventory_errors,
)
from elspeth_lints.core.registry import RuleRegistry

REGISTRY = RuleRegistry()
REGISTRY.load_builtin_rules()
BUILTIN_RULES = [rule for _rule_id, rule in REGISTRY.items()]
FIXTURE_CASES = discover_registry_fixture_cases(REGISTRY)


def test_all_builtin_rules_declare_fixtures() -> None:
    assert find_fixture_inventory_errors(BUILTIN_RULES) == []


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=[case.name for case in FIXTURE_CASES])
@pytest.mark.skipif(
    sys.version_info[:2] != (3, 13),
    reason="elspeth-lints fixture fingerprints are version-specific; Python 3.13 is the canonical lint runtime",
)
def test_builtin_rule_fixture(case: RuleFixtureCase) -> None:
    assert_fixture_case(case)

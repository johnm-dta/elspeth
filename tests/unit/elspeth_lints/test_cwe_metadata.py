"""Every shipped rule must have a non-empty, well-formed CWE tuple (Plan A Task 11)."""

from __future__ import annotations

import re

import pytest

from elspeth_lints.core.registry import DEFAULT_REGISTRY

DEFAULT_REGISTRY.load_builtin_rules()
_CWE_PATTERN = re.compile(r"^CWE-\d+$")


@pytest.mark.parametrize("rule_id", sorted(DEFAULT_REGISTRY.ids()))
def test_every_rule_has_non_empty_cwe_tuple(rule_id: str) -> None:
    rule = DEFAULT_REGISTRY.get(rule_id)
    assert rule.metadata.cwe, f"{rule_id} has empty cwe tuple"
    for cwe in rule.metadata.cwe:
        assert _CWE_PATTERN.match(cwe), f"{rule_id}: malformed cwe {cwe!r}"

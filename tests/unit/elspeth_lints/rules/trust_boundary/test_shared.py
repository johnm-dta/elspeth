"""Regression tests for shared trust-boundary honesty-rule helpers."""

from __future__ import annotations

import ast
import hashlib
import textwrap
from pathlib import Path
from typing import get_args

from elspeth.contracts.trust_boundary import BoundaryRule
from elspeth_lints.rules.trust_boundary.shared import iter_trust_boundary_decorators, make_decorator_finding
from elspeth_lints.rules.trust_boundary.tier.metadata import RULE_METADATA
from elspeth_lints.rules.trust_tier.tier_model.trust_boundary_suppress import _ALLOWED_BOUNDARY_RULES


def test_make_decorator_finding_uses_single_fingerprint_shape() -> None:
    tree = ast.parse(
        textwrap.dedent(
            """
            @trust_boundary(
                tier=2,
                source="x",
                source_param="data",
                suppresses=("R1",),
                invariant="y",
            )
            def handler(data):
                return data
            """
        )
    )
    _func, call = next(iter_trust_boundary_decorators(tree))

    finding = make_decorator_finding(
        metadata=RULE_METADATA,
        rule_id="TBT1",
        file_path="example.py",
        call=call,
        message="tier must be 3",
        suggestion="use tier=3",
    )

    expected = hashlib.sha256(f"TBT1|example.py|{call.lineno}|{call.col_offset}".encode()).hexdigest()[:16]
    assert finding.fingerprint == expected
    assert finding.line == call.lineno
    assert finding.column == call.col_offset
    assert finding.severity == RULE_METADATA.severity


def test_honesty_rules_do_not_reintroduce_private_make_finding_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    rule_root = repo_root / "elspeth-lints/src/elspeth_lints/rules/trust_boundary"

    for relative in ("scope/rule.py", "tier/rule.py", "tests/rule.py"):
        text = (rule_root / relative).read_text(encoding="utf-8")
        assert "def _make_finding(" not in text, relative
        assert "make_decorator_finding(" in text, relative


def test_runtime_boundary_rule_literal_matches_analyzer_allowlist() -> None:
    """The runtime decorator and analyzer must agree on suppressible rule IDs."""
    runtime_rules = frozenset(get_args(BoundaryRule.__value__))
    assert runtime_rules == _ALLOWED_BOUNDARY_RULES

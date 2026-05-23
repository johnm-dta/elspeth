"""Tests for the ``trust_boundary.tier`` honesty-gate rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.tier import RULE as TIER_RULE


def _analyze(source: str) -> list[Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return list(TIER_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))


def test_accepts_tier_3_literal() -> None:
    findings = _analyze(
        """
        from elspeth.contracts.trust_boundary import trust_boundary

        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
            test_ref="tests/test_x.py::test_y",
        )
        def foo(data):
            return data["x"]
        """
    )
    assert findings == []


def test_rejects_tier_2() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=2,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"
    assert "tier must be the literal integer 3" in findings[0].message


def test_rejects_tier_string() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier="3",
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"


def test_rejects_tier_boolean_even_though_int() -> None:
    """``True`` would pass an ``== 3`` check after coercion, so reject explicitly."""
    findings = _analyze(
        """
        @trust_boundary(
            tier=True,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"


def test_missing_tier_kwarg() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"
    assert "missing" in findings[0].message.lower()


def test_ignores_non_literal_kwargs() -> None:
    """Non-literal kwargs are reported by tier_model; this rule degrades silently."""
    findings = _analyze(
        """
        SOME_TIER = 3
        @trust_boundary(
            tier=SOME_TIER,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert findings == []


def test_ignores_undecorated_functions() -> None:
    findings = _analyze(
        """
        def foo(data):
            return data["x"]
        """
    )
    assert findings == []


def test_recognises_attribute_decorator_form() -> None:
    findings = _analyze(
        """
        import elspeth.contracts.trust_boundary as tb_mod

        @tb_mod.trust_boundary(
            tier=2,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"


def test_recognises_async_functions() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=1,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        async def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT1"

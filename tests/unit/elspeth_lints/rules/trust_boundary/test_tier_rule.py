"""Tests for the ``trust_boundary.tier`` honesty-gate rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.tier import RULE as TIER_RULE


def _analyze(source: str) -> list[Finding]:
    tree = ast.parse("from elspeth.contracts.trust_boundary import trust_boundary\n" + textwrap.dedent(source))
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


def test_non_literal_kwargs_fires_NONLITERAL() -> None:
    """Non-literal kwargs are self-enforced (TBT2), NOT silently deferred.

    Before the C6-4 honesty-gate hardening (epic elspeth-2ed3bb0f7d,
    ticket elspeth-1f4634235a) this rule continued silently when any
    kwarg wasn't a literal, deferring all reporting to
    ``trust_tier.tier_model``. That created a cross-rule bypass —
    suppressing tier_model on a file granted tier-honesty immunity.
    The rule now self-enforces; the redundant finding when tier_model
    is also active is deliberate.
    """
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
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT2"
    assert "tier" in findings[0].message
    assert "not a static literal" in findings[0].message


def test_double_star_unpacking_fires_NONLITERAL() -> None:
    """``@trust_boundary(**META)`` is also non-literal — covered by TBT2."""
    findings = _analyze(
        """
        META = {"tier": 3}
        @trust_boundary(**META)
        def foo(data):
            return data
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBT2"
    assert "unpacking" in findings[0].message


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

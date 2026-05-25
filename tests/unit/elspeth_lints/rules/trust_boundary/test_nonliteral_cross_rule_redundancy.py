"""C6-4 cross-rule redundancy verification.

The operator decision for ticket elspeth-1f4634235a (C6-4, epic
elspeth-2ed3bb0f7d) is that each ``trust_boundary.*`` honesty rule
self-enforces literal-only kwargs by emitting its own
``R_TB_NONLITERAL``-class finding (``TBS3`` / ``TBT2`` / ``TBE4``) rather
than silently deferring to ``trust_tier.tier_model``. The redundant
finding when ``tier_model`` is also active is deliberate — suppressing
``tier_model`` on a file must NOT grant honesty-gate immunity.

This test pins the redundancy: a single non-literal-kwarg fixture run
through all four rules emits five findings — one per honesty rule plus
the existing ``R_TB_NONLITERAL`` from ``trust_tier.tier_model``.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.scope import RULE as SCOPE_RULE
from elspeth_lints.rules.trust_boundary.tests import RULE as TESTS_RULE
from elspeth_lints.rules.trust_boundary.tier import RULE as TIER_RULE
from elspeth_lints.rules.trust_tier.tier_model import RULE as TIER_MODEL_RULE

_FIXTURE = textwrap.dedent(
    """
    from elspeth.contracts import trust_boundary

    SOME_TIER = 3
    @trust_boundary(
        tier=SOME_TIER,
        source="x",
        source_param="data",
        suppresses=("R1",),
        invariant="y",
        test_ref="tests/x.py::y",
    )
    def foo(data):
        return data["x"]
    """
)


def _run_rule(rule: object, fixture: str, root: Path) -> list[Finding]:
    tree = ast.parse(fixture)
    return list(
        rule.analyze(  # type: ignore[attr-defined]
            tree, Path("decorated.py"), RuleContext(root=root)
        )
    )


def test_non_literal_tier_kwarg_flagged_by_all_four_rules(tmp_path: Path) -> None:
    """``tier=SOME_TIER`` (Name reference) trips all four rule surfaces.

    The honesty gates (scope / tier / tests) each self-enforce; the
    canonical ``trust_tier.tier_model`` ALSO emits R_TB_NONLITERAL. The
    redundancy is the C6-4 cross-rule-bypass closure.
    """
    scope_findings = _run_rule(SCOPE_RULE, _FIXTURE, tmp_path)
    tier_findings = _run_rule(TIER_RULE, _FIXTURE, tmp_path)
    tests_findings = _run_rule(TESTS_RULE, _FIXTURE, tmp_path)
    tier_model_findings = _run_rule(TIER_MODEL_RULE, _FIXTURE, tmp_path)

    scope_ids = [f.rule_id for f in scope_findings]
    tier_ids = [f.rule_id for f in tier_findings]
    tests_ids = [f.rule_id for f in tests_findings]
    tier_model_ids = [f.rule_id for f in tier_model_findings]

    assert "TBS3" in scope_ids, f"Expected TBS3 from trust_boundary.scope; got {scope_ids!r}"
    assert "TBT2" in tier_ids, f"Expected TBT2 from trust_boundary.tier; got {tier_ids!r}"
    assert "TBE4" in tests_ids, f"Expected TBE4 from trust_boundary.tests; got {tests_ids!r}"
    # tier_model preserves its canonical R_TB_NONLITERAL surface — the
    # honesty-gate self-enforcement does NOT replace it.
    assert "R_TB_NONLITERAL" in tier_model_ids, f"Expected R_TB_NONLITERAL from trust_tier.tier_model; got {tier_model_ids!r}"

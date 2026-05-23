"""Tests for the ``trust_boundary.scope`` honesty-gate rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.scope import RULE as SCOPE_RULE


def _analyze(source: str) -> list[Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return list(SCOPE_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))


def test_accepts_param_present_and_read() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data["x"]
        """
    )
    assert findings == []


def test_rejects_param_not_in_signature() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="payload",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return data["x"]
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS1"
    assert "payload" in findings[0].message


def test_rejects_param_present_but_unused() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return 42
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS2"
    assert "never reads" in findings[0].message


def test_recognises_kwonly_parameter() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(*, data):
            return data["x"]
        """
    )
    assert findings == []


def test_recognises_varargs_kwargs() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="kwargs",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(**kwargs):
            return kwargs["x"]
        """
    )
    assert findings == []


def test_method_self_handling() -> None:
    """source_param='arguments' on a method is fine when the method reads it."""
    findings = _analyze(
        """
        class C:
            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="y",
            )
            def foo(self, arguments):
                return arguments["x"]
        """
    )
    assert findings == []


def test_subscript_read_counts() -> None:
    """``data["x"]`` is a Load on ``data``; should count as a read."""
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            if data["x"]:
                pass
        """
    )
    assert findings == []


def test_attribute_read_counts() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="payload",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(payload):
            return payload.value
        """
    )
    assert findings == []


def test_iteration_read_counts() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="items",
            suppresses=("R1",),
            invariant="y",
        )
        def foo(items):
            for it in items:
                pass
        """
    )
    assert findings == []


def test_async_function_handled() -> None:
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        async def foo(data):
            return data["x"]
        """
    )
    assert findings == []


def test_ignores_undecorated_function() -> None:
    findings = _analyze(
        """
        def foo(data):
            return 42  # would be DEAD but no decorator
        """
    )
    assert findings == []


def test_non_literal_source_param_skipped() -> None:
    findings = _analyze(
        """
        FIELD = "data"
        @trust_boundary(
            tier=3,
            source="x",
            source_param=FIELD,
            suppresses=("R1",),
            invariant="y",
        )
        def foo(data):
            return 42
        """
    )
    assert findings == []


# =============================================================================
# Regression: O4 — nested-function scope leak in _body_reads_name
# =============================================================================
#
# Before the fix, ``_body_reads_name`` used ``ast.walk(statement)`` for each
# top-level statement of the function body. ``ast.walk`` descends through
# nested function bodies, so a nested function that read the outer
# parameter (via Python's lexical-inheritance rules) caused the outer
# body's "did we read the parameter?" check to return True — even when
# the outer body itself never read it. The TBS2 (DEAD parameter)
# finding was silently masked.
#
# The fix replaces ``ast.walk`` with a scope-respecting iterator that
# skips nested-def bodies. The outer body's lexical scope is the only
# scope that satisfies the @trust_boundary contract; an inner read does
# not.
# =============================================================================


def test_dead_param_detection_does_not_falsely_pass_via_nested_function() -> None:
    """Outer fn takes ``data`` and never reads it; inner fn reads it.

    The outer parameter is structurally dead — the decorator on the
    outer function does not extend into the inner function, and the
    inner read happens in a different lexical scope. Before the fix,
    ``ast.walk`` swept the inner body into the outer's "did we read
    it?" check, masking the DEAD-parameter bug. After the fix, TBS2
    fires.
    """
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def outer(data):
            # The outer body never reads ``data``. The inner helper
            # references it via lexical inheritance, but that doesn't
            # count as the outer body reading it — the @trust_boundary
            # contract is satisfied by reads in the outer scope only.
            def inner():
                return data["x"]
            return inner
        """
    )
    assert len(findings) == 1, (
        f"Expected exactly one TBS2 (DEAD) finding; got {len(findings)}: "
        f"{[(f.rule_id, f.message[:80]) for f in findings]}"
    )
    assert findings[0].rule_id == "TBS2"


def test_dead_param_detection_does_not_falsely_pass_via_lambda() -> None:
    """Same shape using a lambda. The lambda inherits ``data`` lexically
    but its body is a separate scope. TBS2 must still fire.
    """
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def outer(data):
            _f = lambda key: data[key]
            return _f
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS2"


def test_outer_body_read_via_assignment_to_inner_does_not_falsely_clear_dead() -> None:
    """A pure inner-scope read does not satisfy the outer's contract.

    Contrast with the case where the outer body explicitly captures or
    transforms ``data`` before calling the inner — that should NOT be
    DEAD because the outer body does read ``data``. This test pins the
    boundary the other way: when the only read is inside the inner
    scope, TBS2 fires; the outer body must satisfy the @trust_boundary
    contract on its own.
    """
    # Sanity-check that the inverse (outer body genuinely reads data)
    # does not fire TBS2 — that's the existing test_accepts_param_present_and_read
    # case, so we don't repeat it here.
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant="y",
        )
        def outer(data):
            def inner(extra):
                # Inner reads data via closure; outer body does not.
                return data.get(extra)
            return inner
        """
    )
    assert [f.rule_id for f in findings] == ["TBS2"]

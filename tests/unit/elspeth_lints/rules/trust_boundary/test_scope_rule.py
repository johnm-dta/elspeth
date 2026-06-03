"""Tests for the ``trust_boundary.scope`` honesty-gate rule."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import Finding, RuleContext
from elspeth_lints.rules.trust_boundary.scope import RULE as SCOPE_RULE


def _analyze(source: str) -> list[Finding]:
    tree = ast.parse("from elspeth.contracts.trust_boundary import trust_boundary\n" + textwrap.dedent(source))
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


def test_non_literal_source_param_fires_NONLITERAL() -> None:
    """Non-literal kwargs are self-enforced (TBS3), NOT silently deferred.

    Before the C6-4 honesty-gate hardening (epic elspeth-2ed3bb0f7d,
    ticket elspeth-1f4634235a) this rule continued silently when any
    kwarg wasn't a literal, deferring all reporting to
    ``trust_tier.tier_model``. That created a cross-rule bypass —
    suppressing tier_model on a file granted scope-honesty immunity.
    The rule now self-enforces; the redundant finding when tier_model
    is also active is deliberate.
    """
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
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS3"
    assert "source_param" in findings[0].message
    assert "not a static literal" in findings[0].message


def test_non_literal_kwarg_via_call_fires_NONLITERAL() -> None:
    """A Call-valued kwarg (``invariant=str(...)``) also trips TBS3."""
    findings = _analyze(
        """
        @trust_boundary(
            tier=3,
            source="x",
            source_param="data",
            suppresses=("R1",),
            invariant=str(42),
        )
        def foo(data):
            return data["x"]
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS3"
    assert "invariant" in findings[0].message


def test_double_star_unpacking_fires_NONLITERAL() -> None:
    """``@trust_boundary(**META)`` is also non-literal — covered by TBS3."""
    findings = _analyze(
        """
        META = {"tier": 3, "source": "x", "source_param": "data",
                "suppresses": ("R1",), "invariant": "y"}
        @trust_boundary(**META)
        def foo(data):
            return data["x"]
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS3"
    assert "unpacking" in findings[0].message


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
        f"Expected exactly one TBS2 (DEAD) finding; got {len(findings)}: {[(f.rule_id, f.message[:80]) for f in findings]}"
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


# =============================================================================
# Regression: C6-2 — _body_reads_name must require taint-receiving context
# =============================================================================
#
# Before the C6-2 fix (epic elspeth-2ed3bb0f7d, ticket elspeth-9bbf3b66e9),
# ``_body_reads_name`` accepted ANY ``ast.Name(Load)`` reference, including
# dead reads like ``_ = source_param`` or a bare expression-statement
# ``source_param  # noqa: B018``. A single no-op line defeated the entire
# scope honesty gate.
#
# The fix requires the Name(Load) reference to be on the taint-RECEIVING
# side of an operation — it must participate in a subscript, attribute,
# call, iteration, unpacking, or assignment-to-bound-name. A bare
# expression-statement or an assignment whose only target is the
# underscore name (or any leading-underscore name) does NOT satisfy the
# read predicate.
# =============================================================================


def test_dead_assignment_to_underscore_does_not_count_as_read() -> None:
    """``_ = source_param`` is a dead read — TBS2 must still fire."""
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
            _ = data
            return 42
        """
    )
    assert len(findings) == 1, (
        f"Expected exactly one TBS2 (DEAD) finding; got {len(findings)}: {[(f.rule_id, f.message[:80]) for f in findings]}"
    )
    assert findings[0].rule_id == "TBS2"


def test_bare_expression_statement_does_not_count_as_read() -> None:
    """``source_param`` alone on a line is a dead read — TBS2 must still fire."""
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
            data  # noqa: B018
            return 42
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS2"


def test_dead_assignment_to_leading_underscore_name_does_not_count_as_read() -> None:
    """``_unused = source_param`` is conventionally dead — TBS2 must fire."""
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
            _unused_var = data
            return 42
        """
    )
    assert len(findings) == 1
    assert findings[0].rule_id == "TBS2"


def test_subscript_read_satisfies_predicate() -> None:
    """``value = source_param["key"]`` is taint-receiving — accepted."""
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
            value = data["key"]
            return value
        """
    )
    assert findings == []


def test_iteration_satisfies_predicate() -> None:
    """``for item in source_param:`` is taint-receiving (iter side) — accepted."""
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
            for item in items:
                pass
            return 42
        """
    )
    assert findings == []


def test_call_argument_satisfies_predicate() -> None:
    """``helper(source_param)`` is taint-receiving (call arg) — accepted."""
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
            return helper(data)
        """
    )
    assert findings == []


def test_unpacking_assignment_satisfies_predicate() -> None:
    """``a, b = source_param`` is taint-receiving (tuple-target unpacks) — accepted."""
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
            a, b = data
            return a + b
        """
    )
    assert findings == []


def test_bound_rebinding_satisfies_predicate() -> None:
    """``result = source_param`` (non-underscore target) is taint-receiving — accepted.

    A real rebinding to a bound name propagates the value to a place the
    rest of the function can observe; this distinguishes it from
    ``_ = source_param`` (dead).
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
        def foo(data):
            result = data
            return result["k"]
        """
    )
    assert findings == []


# =============================================================================
# Regression: C5-1 — class-body scope leak in iter_own_scope
# =============================================================================
#
# Symmetric to the nested-function / lambda cases above. Before the fix,
# ``iter_own_scope`` (the per-statement scope-respecting walker used by
# ``_body_reads_name``) short-circuited only at FunctionDef /
# AsyncFunctionDef / Lambda. A nested ``class`` body that *referenced*
# the outer parameter (e.g. ``class Helper: raw = data["x"]``) was
# walked through, the ``ast.Load`` of ``data`` was seen as if it
# satisfied the outer's body-read contract, and the DEAD-parameter
# finding (TBS2) was silently masked. After the fix, ClassDef short-
# circuits the walk and TBS2 fires.
#
# Symmetric coverage in tier_model's compute_derived_names lives in
# tests/unit/elspeth_lints/test_tier_model_decorator_suppression.py
# (TestClassBodyScopeLeakRegression). Both walkers consume the shared
# _NESTED_SCOPE_TYPES tuple in ast_walker.py, so the fix lands in one
# place and is exercised at both sites.
# =============================================================================


def test_dead_param_detection_does_not_falsely_pass_via_class_body() -> None:
    """Outer fn takes ``data`` and never reads it; nested class body reads it.

    Class bodies execute in a fresh namespace. A class-attribute
    assignment that references ``data`` is NOT a read in the outer
    function's body — it's a read inside the class's own scope, just
    like a nested function read is in the function's own scope. TBS2
    must fire.

    Before the fix: ``iter_own_scope`` descended into the class body,
    saw ``ast.Name(id='data', ctx=Load())`` inside the subscript, and
    returned True from ``_body_reads_name``. The TBS2 finding never
    fired. After the fix: the class body is never visited, so
    ``_body_reads_name`` returns False and TBS2 fires.
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
            class Helper:
                # Class-body read of ``data``. Python evaluates the
                # class body when the class statement executes, with
                # ``data`` lexically inherited from the outer scope —
                # but the assignment binds ``raw`` as a class
                # attribute, in the class's own namespace, not in
                # ``outer``'s locals. From the outer function body's
                # perspective, ``data`` was never read.
                raw = data["x"]
            return Helper
        """
    )
    assert len(findings) == 1, (
        f"Expected exactly one TBS2 (DEAD) finding; got {len(findings)}: {[(f.rule_id, f.message[:80]) for f in findings]}"
    )
    assert findings[0].rule_id == "TBS2"

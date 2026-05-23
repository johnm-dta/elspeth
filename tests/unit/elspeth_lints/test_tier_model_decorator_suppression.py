"""Tests for ``@trust_boundary`` decorator-aware suppression in tier_model.

The rule under test (``trust_tier.tier_model``) drops findings inside a
``@trust_boundary``-decorated function when:

1. the finding's ``rule_id`` is listed in the decorator's ``suppresses``
   tuple, and
2. the finding's AST subject is rooted at the decorator's ``source_param``
   parameter (or at a name derived from it through subscript, attribute
   access, ``.get(...)``, iteration, unpacking, or walrus).

Findings that don't satisfy both conditions remain visible — the decorator
is not a whole-function exemption cloak. Malformed decorators (non-literal
kwargs, wrong-shaped values) emit their own ``R_TB_NONLITERAL`` /
``R_TB_MALFORMED`` finding and are treated as inert.
"""

from __future__ import annotations

import ast
from textwrap import dedent

from elspeth_lints.rules.trust_tier.tier_model.rule import Finding, TierModelVisitor


def _findings(source: str, filename: str = "test_module.py") -> list[Finding]:
    """Run the tier-model visitor on ``source`` and return findings."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    visitor = TierModelVisitor(filename, source_lines)
    visitor.visit(tree)
    return visitor.findings


def _findings_by_rule(findings: list[Finding], rule_id: str) -> list[Finding]:
    return [f for f in findings if f.rule_id == rule_id]


# =============================================================================
# Positive cases: decorator suppresses qualifying findings
# =============================================================================


class TestSuppressionPositive:
    """The decorator suppresses R1 / R5 inside the function body when rooted."""

    def test_suppresses_isinstance_on_source_param(self) -> None:
        """``isinstance(arguments.get("x"), list)`` should be suppressed."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="LLM tool args",
                source_param="arguments",
                suppresses=("R1", "R5"),
                invariant="raises on shape mismatch",
            )
            def handler(arguments):
                if not isinstance(arguments.get("nodes"), list):
                    raise ValueError("nodes must be a list")
                return None
        """)
        findings = _findings(source)
        # R1 (arguments.get) and R5 (isinstance on arguments.get) should both
        # be suppressed.
        assert _findings_by_rule(findings, "R1") == []
        assert _findings_by_rule(findings, "R5") == []

    def test_suppresses_get_on_loop_variable(self) -> None:
        """``for raw in arguments["nodes"]: raw.get("id")`` — both suppressed."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                for raw in arguments["nodes"]:
                    raw.get("id")
                return None
        """)
        findings = _findings(source)
        # Both ``raw.get`` and any other arguments-rooted ``.get`` should be
        # suppressed.
        assert _findings_by_rule(findings, "R1") == []

    def test_suppresses_dataflow_propagation_through_assignment(self) -> None:
        """``raw = arguments["x"]; raw.get("y")`` — propagation through assign."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                raw = arguments["x"]
                raw.get("y")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []

    def test_suppresses_inside_comprehension(self) -> None:
        """``[x.get("y") for x in arguments]`` — comprehension target derived."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                return [x.get("y") for x in arguments]
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []

    def test_suppresses_walrus_then_method_call(self) -> None:
        """``if (x := arguments.get("k")): x.get("v")`` — both suppressed."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                if (x := arguments.get("k")):
                    x.get("v")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []


# =============================================================================
# Negative cases: decorator does NOT suppress
# =============================================================================


class TestSuppressionNegative:
    """Findings outside the decorator's scope remain visible."""

    def test_non_rooted_access_still_reported(self) -> None:
        """``self._cache.get(node_id)`` is NOT rooted at ``arguments``."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            class Foo:
                @trust_boundary(
                    tier=3,
                    source="x",
                    source_param="arguments",
                    suppresses=("R1",),
                    invariant="x",
                )
                def handler(self, arguments):
                    arguments.get("ok")  # suppressed
                    self._cache.get("not-ok")  # NOT suppressed
                    return None
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        # Exactly one R1: the ``self._cache.get`` call.
        assert len(r1) == 1
        assert "self._cache" in r1[0].code_snippet

    def test_rule_outside_suppresses_still_reported(self) -> None:
        """``suppresses=("R1",)`` does NOT cover R5."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")  # R1, suppressed
                isinstance(arguments, dict)  # R5, NOT suppressed
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []
        r5 = _findings_by_rule(findings, "R5")
        assert len(r5) == 1

    def test_no_decorator_no_suppression(self) -> None:
        """Plain function with the boundary pattern is fully reported."""
        source = dedent("""
            def handler(arguments):
                for raw in arguments["nodes"]:
                    raw.get("id")
                return None
        """)
        findings = _findings(source)
        # arguments["nodes"] subscript: no R1 (not a .get). raw.get is R1.
        # No isinstance, so no R5 either.
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) >= 1


# =============================================================================
# Decorator-shape diagnostics
# =============================================================================


class TestDecoratorDiagnostics:
    """Malformed decorators emit their own findings and do not suppress."""

    def test_non_literal_suppresses_emits_diagnostic(self) -> None:
        """``suppresses=ALLOWED`` (a Name) is not literal-evaluatable."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            ALLOWED = ("R1", "R5")

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=ALLOWED,
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        nonliteral = _findings_by_rule(findings, "R_TB_NONLITERAL")
        assert len(nonliteral) == 1
        # And the inner R1 should NOT be suppressed (the decorator is inert).
        assert len(_findings_by_rule(findings, "R1")) == 1

    def test_string_suppresses_emits_malformed(self) -> None:
        """``suppresses="R1"`` (a string instead of a tuple) is malformed."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses="R1",
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        malformed = _findings_by_rule(findings, "R_TB_MALFORMED")
        assert len(malformed) == 1
        # And the inner R1 should NOT be suppressed.
        assert len(_findings_by_rule(findings, "R1")) == 1

    def test_source_param_not_a_real_parameter_emits_malformed(self) -> None:
        """``source_param`` that isn't on the signature → R_TB_MALFORMED."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="missing",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        assert len(_findings_by_rule(findings, "R_TB_MALFORMED")) == 1
        # Inner R1 NOT suppressed.
        assert len(_findings_by_rule(findings, "R1")) == 1


# =============================================================================
# Decorator stack ordering
# =============================================================================


class TestDecoratorStackOrdering:
    """``@trust_boundary`` is recognised wherever it appears in the stack."""

    def test_recognised_above_other_decorators(self) -> None:
        """``@trust_boundary(...) @other`` — recognised."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            def some_other_decorator(fn):
                return fn

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            @some_other_decorator
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []

    def test_recognised_below_other_decorators(self) -> None:
        """``@some_other_decorator @trust_boundary(...)`` — recognised."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            def some_other_decorator(fn):
                return fn

            @some_other_decorator
            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []

    def test_attribute_access_form_recognised(self) -> None:
        """``@elspeth.contracts.trust_boundary(...)`` form is recognised."""
        source = dedent("""
            import elspeth.contracts

            @elspeth.contracts.trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []


# =============================================================================
# Async functions
# =============================================================================


class TestAsyncFunction:
    """Async functions get the same treatment."""

    def test_async_function_suppression(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            async def handler(arguments):
                arguments.get("k")
                return None
        """)
        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []


# =============================================================================
# Regression: B1 — nested-function scope leak in compute_derived_names
# =============================================================================
#
# Before the fix, ``compute_derived_names`` used ``ast.walk(func_node)`` with
# a ``sub is not func_node`` guard. ``ast.walk`` had already yielded every
# descendant of any nested function *before* the guard fired — only the
# inner FunctionDef AST node itself was skipped, not its body. Names
# bound inside inner-scope assignments from a name colliding with an
# outer-scope variable falsely tainted the outer ``derived`` set, and
# R1/R5 findings on the unrelated outer-scope variable were silently
# suppressed.
#
# The fix replaces the walk with ``walk_function_own_scope`` which
# short-circuits at nested-scope AST boundaries — the inner function's
# body is not visited at all by the outer function's taint walk.
# =============================================================================


class TestNestedScopeLeakRegression:
    """Pinning the B1 fix: inner-scope assignments must not taint outer names."""

    def test_nested_function_scope_does_not_leak_to_outer(self) -> None:
        """The reviewer's exact snippet. An inner ``raw = arguments['x']``
        must not taint the outer scope's unrelated ``raw`` variable.

        Before the fix: ``ast.walk`` descended into ``inner``'s body,
        saw ``raw = arguments["x"]`` (which contained a derived-name
        reference), and added ``raw`` to the outer function's
        ``derived`` set. The outer-scope ``raw.get("k")`` then matched
        as "rooted at a derived name" and the R1 finding was suppressed.

        After the fix: the inner body is never visited by the outer
        walk, so ``raw`` is bound only as an inner-scope name. The
        outer-scope ``raw = {"k": "v"}`` is an ordinary literal-dict
        assignment; ``raw.get("k")`` on the outer ``raw`` IS NOT rooted
        at ``arguments`` and the R1 finding fires.
        """
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def outer(arguments):
                def inner():
                    raw = arguments["x"]   # taints 'raw' inside inner only
                    return raw
                raw = {"k": "v"}            # outer 'raw' is unrelated
                return raw.get("k")          # FALSELY suppressed before fix
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        # The outer raw.get("k") must fire R1; the inner subscript is not
        # an R1 violation (subscript access != .get with default), so the
        # only expected R1 is the outer call.
        assert len(r1) == 1, (
            f"Expected exactly one R1 finding on the outer ``raw.get('k')``; "
            f"got {len(r1)}: {[(f.rule_id, f.line, f.message[:60]) for f in r1]}"
        )

    def test_lambda_body_does_not_leak_to_outer(self) -> None:
        """Same shape using a lambda instead of a nested def.

        A lambda introduces its own scope. Its body referencing
        ``arguments`` must not taint outer-scope name bindings that
        happen to share a name with the lambda's parameter list or
        with names the lambda's expression mentions.
        """
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
            )
            def outer(arguments):
                # The lambda's body references arguments but binds nothing
                # in the outer scope. The taint must stay inside the lambda.
                _f = lambda raw: arguments.get(raw)
                raw = {"k": "v"}
                return raw.get("k")
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        # The outer raw.get("k") must fire R1. The lambda's body uses
        # arguments.get(raw) which IS rooted at arguments (suppressed),
        # so the only R1 we expect is the outer call.
        outer_calls = [f for f in r1 if "raw.get" in (f.message or "")]
        assert outer_calls, (
            f"Expected R1 on outer ``raw.get('k')``; got R1 findings: "
            f"{[(f.line, f.message[:80]) for f in r1]}"
        )


# =============================================================================
# Regression: M4 — unauthorised rule IDs in suppresses tuple
# =============================================================================
#
# The decorator's runtime signature constrains ``suppresses`` to
# ``tuple[Literal["R1", "R5"], ...]``. mypy enforces that at the call site,
# but the analyzer's parse path reads kwargs from a static AST and would
# otherwise honour any string the author typed. The fix adds a closed-set
# membership check inside ``extract_boundary_metadata``: unauthorised
# entries fire R_TB_MALFORMED and the decorator becomes inert.
# =============================================================================


class TestUnauthorisedSuppressRules:
    """The closed set ``{R1, R5}`` is enforced; out-of-set entries are malformed."""

    def test_decorator_with_unauthorised_rule_in_suppresses_emits_malformed_and_does_not_suppress(self) -> None:
        """``suppresses=("R2", "R8")`` is unauthorised; treat decorator as inert.

        The function below contains an R1 violation rooted at
        ``arguments``. Before the fix the decorator's ``suppresses``
        tuple was accepted as-is, no R_TB_MALFORMED was emitted, but
        because ``"R1"`` wasn't in the (unauthorised) tuple the R1
        finding fired anyway. The visible behaviour was correct by
        accident. After the fix, the unauthorised tuple fires
        R_TB_MALFORMED on the decorator AND the metadata is inert (so
        any R1/R5 violation rooted at source_param fires too).
        """
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R2", "R8"),
                invariant="x",
            )
            def handler(arguments):
                # R1 rooted at arguments — would be suppressed by a valid
                # ('R1',) decorator. With the unauthorised tuple the
                # decorator is inert, so R1 fires.
                return arguments.get("k")
        """)
        findings = _findings(source)
        malformed = _findings_by_rule(findings, "R_TB_MALFORMED")
        assert len(malformed) == 1, (
            f"Expected exactly one R_TB_MALFORMED on the decorator; got {len(malformed)}: "
            f"{[(f.rule_id, f.line, f.message[:80]) for f in malformed]}"
        )
        # Message should name the offending rule ids so the operator can
        # locate them without re-reading the source.
        assert "R2" in malformed[0].message and "R8" in malformed[0].message, (
            f"R_TB_MALFORMED message must name the unauthorised rule IDs; got: "
            f"{malformed[0].message}"
        )
        # And the decorator is inert: R1 on arguments.get fires.
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1, (
            f"Expected R1 to fire (decorator is inert under M4); got R1: "
            f"{[(f.line, f.message[:80]) for f in r1]}"
        )

    def test_decorator_with_mixed_authorised_and_unauthorised_is_inert(self) -> None:
        """``suppresses=("R1", "R3")`` — even a partial match is rejected.

        The closed-set check is all-or-nothing: if ANY entry in the
        tuple is unauthorised, the entire decorator is malformed. We
        cannot silently honour the authorised half because the author's
        intent is structurally unclear (did they mean to write "R5"?
        was "R3" a typo for "R1"?). A blanket malformed-and-inert
        verdict is the conservative call: emit the diagnostic, let the
        author fix the source, and surface every R1/R5 finding the
        valid kwargs would have suppressed.
        """
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1", "R3"),
                invariant="x",
            )
            def handler(arguments):
                return arguments.get("k")
        """)
        findings = _findings(source)
        malformed = _findings_by_rule(findings, "R_TB_MALFORMED")
        assert len(malformed) == 1
        assert "R3" in malformed[0].message
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1, "R1 must fire — decorator is inert"

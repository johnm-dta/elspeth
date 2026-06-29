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
import hashlib
import json
from pathlib import Path
from textwrap import dedent

from elspeth_lints.core.cli import main
from elspeth_lints.rules.trust_tier.tier_model.rule import Finding, TierModelVisitor
from elspeth_lints.rules.trust_tier.tier_model.trust_boundary_suppress import (
    extract_boundary_metadata,
)


def _findings(source: str, filename: str = "test_module.py") -> list[Finding]:
    """Run the tier-model visitor on ``source`` and return findings."""
    return _visitor(source, filename=filename).findings


def _visitor(source: str, filename: str = "test_module.py") -> TierModelVisitor:
    """Run the tier-model visitor on ``source`` and return the visitor."""
    tree = ast.parse(source, filename=filename)
    source_lines = source.splitlines()
    file_fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()
    visitor = TierModelVisitor(filename, source_lines, file_fingerprint)
    visitor.visit(tree)
    return visitor


def _findings_by_rule(findings: list[Finding], rule_id: str) -> list[Finding]:
    return [f for f in findings if f.rule_id == rule_id]


def _first_function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(dedent(source))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    raise AssertionError("fixture must contain a top-level function")


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

    def test_suppression_records_non_failing_observation(self) -> None:
        """Suppressed R1/R5 findings remain auditable as observation records."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="LLM tool args",
                source_param="arguments",
                suppresses=("R1", "R5"),
                invariant="raises on shape mismatch",
                test_ref="tests/test_handler.py::test_rejects_bad_args",
                test_fingerprint="abc123",
            )
            def handler(arguments):
                if not isinstance(arguments.get("nodes"), list):
                    raise ValueError("nodes must be a list")
                return None
        """)
        visitor = _visitor(source)
        expected_file_fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()

        assert _findings_by_rule(visitor.findings, "R1") == []
        assert _findings_by_rule(visitor.findings, "R5") == []
        suppressed = _findings_by_rule(visitor.suppressed_findings, "R_TB_SUPPRESSED")
        assert sorted(item.message for item in suppressed) == [
            (
                "@trust_boundary suppressed R1 under source_param='arguments'; "
                "source='LLM tool args'; test_ref='tests/test_handler.py::test_rejects_bad_args'; "
                "suppresses=('R1', 'R5')"
            ),
            (
                "@trust_boundary suppressed R5 under source_param='arguments'; "
                "source='LLM tool args'; test_ref='tests/test_handler.py::test_rejects_bad_args'; "
                "suppresses=('R1', 'R5')"
            ),
        ]
        assert {item.file_fingerprint for item in suppressed} == {expected_file_fingerprint}

    def test_suppresses_with_fully_qualified_decorator_after_sibling_import(self) -> None:
        """A later ``elspeth.*`` import must not hide the FQ decorator spelling."""
        source = dedent("""
            import elspeth.contracts.trust_boundary
            import elspeth.web

            @elspeth.contracts.trust_boundary(
                tier=3,
                source="LLM tool args",
                source_param="arguments",
                suppresses=("R1",),
                invariant="raises on shape mismatch",
            )
            def handler(arguments):
                return arguments.get("nodes")
        """)

        findings = _findings(source)
        assert _findings_by_rule(findings, "R1") == []

    def test_core_cli_emits_suppression_observation_without_failing(self, tmp_path: Path, capsys) -> None:
        """The CI-facing CLI surfaces suppression observations at note severity."""
        allowlist_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
        allowlist_dir.mkdir(parents=True)
        (tmp_path / "handler.py").write_text(
            dedent("""
                from elspeth.contracts import trust_boundary

                @trust_boundary(
                    tier=3,
                    source="LLM tool args",
                    source_param="arguments",
                    suppresses=("R1",),
                    invariant="raises on shape mismatch",
                    test_ref="tests/test_handler.py::test_rejects_bad_args",
                    test_fingerprint="abc123",
                )
                def handler(arguments):
                    return arguments.get("nodes")
            """),
            encoding="utf-8",
        )

        exit_code = main(
            [
                "check",
                "--rules",
                "trust_tier.tier_model",
                "--root",
                str(tmp_path),
                "--format",
                "json",
            ]
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert [(item["rule_id"], item["severity"]) for item in payload] == [
            ("R_TB_SUPPRESSED", "note"),
        ]
        assert "suppressed R1" in payload[0]["message"]

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


class TestLiveDerivedNameSuppression:
    """Derived-name coverage through the production visitor path."""

    def test_annotation_augassign_with_and_namedexpr_suppress_live_findings(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="arguments",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(arguments, total):
                annotated: object = arguments["annotated"]
                total += arguments["delta"]
                with arguments["ctx"] as ctx:
                    pass
                if (chosen := arguments["maybe"]):
                    pass
                annotated.get("a")
                total.get("b")
                ctx.get("c")
                chosen.get("d")
        """)

        assert _findings_by_rule(_findings(source), "R1") == []

    def test_async_for_async_with_and_comprehension_suppress_live_findings(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="arguments",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            async def handler(arguments):
                async for item in arguments["items"]:
                    item.get("id")
                async with arguments["ctx"] as ctx:
                    ctx.get("id")
                [row.get("id") for row in arguments["rows"]]
                {tag.get("id") for tag in arguments["tags"]}
                {key.get("id"): value.get("id") for key, value in arguments["pairs"]}
                (part.get("id") for part in arguments["parts"])
        """)

        assert _findings_by_rule(_findings(source), "R1") == []


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

    def test_later_source_assignment_does_not_suppress_earlier_safe_local(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload):
                raw = {}
                raw.get("safe")
                raw = payload["raw"]
                return raw["id"]
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1
        assert 'raw.get("safe")' in r1[0].code_snippet

    def test_safe_reassignment_clears_previously_derived_local(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload):
                raw = payload["raw"]
                raw.get("external")
                raw = {}
                raw.get("safe")
                return raw
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1
        assert 'raw.get("safe")' in r1[0].code_snippet

    def test_if_branch_derived_name_only_on_one_path_does_not_suppress_after_join(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload, use_external):
                if use_external:
                    raw = {}
                else:
                    raw = payload["raw"]
                return raw.get("id")
        """)

        r1 = _findings_by_rule(_findings(source), "R1")
        assert len(r1) == 1
        assert 'raw.get("id")' in r1[0].code_snippet

    def test_if_branch_derived_name_on_every_path_suppresses_after_join(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload, choose_a):
                if choose_a:
                    raw = payload["a"]
                else:
                    raw = payload["b"]
                return raw.get("id")
        """)

        assert _findings_by_rule(_findings(source), "R1") == []

    def test_try_branch_derived_name_only_on_handler_path_does_not_suppress_after_join(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload):
                try:
                    raw = {}
                except ValueError:
                    raw = payload["raw"]
                return raw.get("id")
        """)

        r1 = _findings_by_rule(_findings(source), "R1")
        assert len(r1) == 1
        assert 'raw.get("id")' in r1[0].code_snippet

    def test_while_body_derived_name_does_not_suppress_after_zero_iteration_join(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="external payload",
                source_param="payload",
                suppresses=("R1",),
                invariant="raises ValueError on malformed payload",
            )
            def handler(payload, keep_going):
                raw = {}
                while keep_going:
                    raw = payload["raw"]
                    break
                return raw.get("id")
        """)

        r1 = _findings_by_rule(_findings(source), "R1")
        assert len(r1) == 1
        assert 'raw.get("id")' in r1[0].code_snippet


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

    def test_non_string_source_and_invariant_are_malformed_and_inert(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source=42,
                source_param="arguments",
                suppresses=("R1",),
                invariant=42,
                test_ref="tests/test_handler.py::test_rejects_bad_args",
            )
            def handler(arguments):
                return arguments.get("k")
        """)
        findings = _findings(source)
        malformed = _findings_by_rule(findings, "R_TB_MALFORMED")
        assert len(malformed) == 1
        assert "'source' must be a string" in malformed[0].message
        assert "'invariant' must be a string" in malformed[0].message
        assert len(_findings_by_rule(findings, "R1")) == 1

    def test_stacked_trust_boundary_decorators_emit_stacked_and_do_not_suppress(self) -> None:
        """Multiple boundary decorators are ambiguous and must be inert."""
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="outer feed",
                source_param="arguments",
                suppresses=("R1",),
                invariant="outer invariant",
            )
            @trust_boundary(
                tier=3,
                source="inner feed",
                source_param="arguments",
                suppresses=("R5",),
                invariant="inner invariant",
            )
            def handler(arguments):
                return arguments.get("k")
        """)
        findings = _findings(source)
        stacked = _findings_by_rule(findings, "R_TB_STACKED")
        assert len(stacked) == 1
        assert "multiple @trust_boundary" in stacked[0].message
        assert len(_findings_by_rule(findings, "R1")) == 1


class TestExtractBoundaryMetadata:
    """Direct branch coverage for decorator metadata extraction."""

    def test_no_decorator_returns_empty_result(self) -> None:
        func = _first_function(
            """
            def handler(arguments):
                return arguments
            """
        )

        metadata, diagnostics = extract_boundary_metadata(func)

        assert metadata is None
        assert diagnostics == []

    def test_positional_argument_is_malformed(self) -> None:
        func = _first_function(
            """
            @trust_boundary(3, "x", "arguments", ("R1",), "x")
            def handler(arguments):
                return arguments
            """
        )

        metadata, diagnostics = extract_boundary_metadata(func)

        assert metadata is None
        assert [item.rule_id for item in diagnostics] == ["R_TB_MALFORMED"]
        assert "positional arguments" in diagnostics[0].message

    def test_kwargs_unpacking_is_nonliteral(self) -> None:
        func = _first_function(
            """
            @trust_boundary(**metadata)
            def handler(arguments):
                return arguments
            """
        )

        metadata, diagnostics = extract_boundary_metadata(func)

        assert metadata is None
        assert [item.rule_id for item in diagnostics] == ["R_TB_NONLITERAL"]
        assert "**-unpacking" in diagnostics[0].message

    def test_missing_suppresses_is_malformed(self) -> None:
        func = _first_function(
            """
            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                invariant="x",
            )
            def handler(arguments):
                return arguments
            """
        )

        metadata, diagnostics = extract_boundary_metadata(func)

        assert metadata is None
        assert diagnostics[0].rule_id == "R_TB_MALFORMED"
        assert "missing kwarg 'suppresses'" in diagnostics[0].message

    def test_non_string_suppresses_item_is_malformed(self) -> None:
        func = _first_function(
            """
            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1", 5),
                invariant="x",
            )
            def handler(arguments):
                return arguments
            """
        )

        metadata, diagnostics = extract_boundary_metadata(func)

        assert metadata is None
        assert diagnostics[0].rule_id == "R_TB_MALFORMED"
        assert "non-string" in diagnostics[0].message

    def test_source_param_wrong_type_and_empty_string_are_malformed(self) -> None:
        for source_param, expected in (("5", "must be a string"), ("''", "empty string")):
            func = _first_function(
                f"""
                @trust_boundary(
                    tier=3,
                    source="x",
                    source_param={source_param},
                    suppresses=("R1",),
                    invariant="x",
                )
                def handler(arguments):
                    return arguments
                """
            )

            metadata, diagnostics = extract_boundary_metadata(func)

            assert metadata is None
            assert diagnostics[0].rule_id == "R_TB_MALFORMED"
            assert expected in diagnostics[0].message


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

    def test_elspeth_contracts_alias_attribute_form_recognised(self) -> None:
        """``import elspeth.contracts as contracts`` remains a valid spelling."""
        source = dedent("""
            import elspeth.contracts as contracts

            @contracts.trust_boundary(
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

    def test_non_elspeth_attribute_named_trust_boundary_does_not_suppress(self) -> None:
        """An unrelated ``foo.trust_boundary`` decorator is not Elspeth's boundary."""
        source = dedent("""
            import foo

            @foo.trust_boundary(
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
        visitor = _visitor(source)

        assert len(_findings_by_rule(visitor.findings, "R1")) == 1
        assert _findings_by_rule(visitor.suppressed_findings, "R_TB_SUPPRESSED") == []

    def test_shadowed_elspeth_name_does_not_suppress(self) -> None:
        """An alias named ``elspeth`` is not enough; it must resolve to Elspeth."""
        source = dedent("""
            import foo as elspeth

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
        visitor = _visitor(source)

        assert len(_findings_by_rule(visitor.findings, "R1")) == 1
        assert _findings_by_rule(visitor.suppressed_findings, "R_TB_SUPPRESSED") == []

    def test_non_elspeth_bare_import_named_trust_boundary_does_not_suppress(self) -> None:
        """``from foo import trust_boundary`` must not masquerade as Elspeth's decorator."""
        source = dedent("""
            from foo import trust_boundary

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
        visitor = _visitor(source)

        assert len(_findings_by_rule(visitor.findings, "R1")) == 1
        assert _findings_by_rule(visitor.suppressed_findings, "R_TB_SUPPRESSED") == []


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
# Regression: B1 — nested-function scope leak in live derived-name suppression
# =============================================================================
#
# The visitor must not let nested scopes inherit an outer function's active
# ``@trust_boundary`` suppression state. A finding in a nested function or
# lambda is only suppressible by that nested callable's own decorator.
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
        assert outer_calls, f"Expected R1 on outer ``raw.get('k')``; got R1 findings: {[(f.line, f.message[:80]) for f in r1]}"


# =============================================================================
# Regression: C5-1 — class-body scope does not inherit boundary suppression
# =============================================================================
#
# Python class bodies execute in a fresh namespace: assignments become class
# attributes, not bindings in the enclosing function's locals. The live visitor
# therefore clears inherited boundary state while visiting nested classes.
# =============================================================================


class TestClassBodyScopeLeakRegression:
    """Pinning the C5-1 fix: class-body assignments must not taint outer names."""

    def test_nested_class_assignment_does_not_leak_to_outer(self) -> None:
        """Outer fn defines a nested class that assigns from ``arguments``;
        the outer ``raw`` must NOT inherit that taint.

        The class-body ``raw`` is a class attribute, not an outer local. The
        outer ``raw.get("k")`` must therefore remain visible as an R1 finding.
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
                class Helper:
                    # Class-body assignment. ``raw`` here is a class
                    # attribute on ``Helper``, NOT a binding in
                    # ``outer``'s locals. The outer function's
                    # ``derived`` set must NOT pick this up.
                    raw = arguments["x"]
                raw = {"k": "v"}             # outer 'raw' is unrelated
                return raw.get("k")           # FALSELY suppressed before fix
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        # The outer raw.get("k") must fire R1. The class-body subscript
        # is not an R1 violation (subscript access != .get with
        # default), so the only expected R1 is the outer call.
        assert len(r1) == 1, (
            f"Expected exactly one R1 finding on the outer ``raw.get('k')``; "
            f"got {len(r1)}: {[(f.rule_id, f.line, f.message[:60]) for f in r1]}"
        )


class TestBoundaryDoesNotInheritIntoNestedScopes:
    """Outer trust-boundary metadata must not suppress nested-scope findings."""

    def test_nested_function_free_variable_get_is_not_suppressed(self) -> None:
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
                    return arguments.get("k")
                return inner()
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1

    def test_nested_function_shadowed_source_param_get_is_not_suppressed(self) -> None:
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
                def inner(arguments):
                    return arguments.get("k")
                return inner({"k": "v"})
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1

    def test_lambda_body_get_is_not_suppressed_by_outer_boundary(self) -> None:
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
                inner = lambda arguments: arguments.get("k")
                return inner({"k": "v"})
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1

    def test_nested_class_body_get_is_not_suppressed_by_outer_boundary(self) -> None:
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
                class Inner:
                    value = arguments.get("k")
                return Inner.value
        """)
        findings = _findings(source)
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1


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
            f"R_TB_MALFORMED message must name the unauthorised rule IDs; got: {malformed[0].message}"
        )
        # And the decorator is inert: R1 on arguments.get fires.
        r1 = _findings_by_rule(findings, "R1")
        assert len(r1) == 1, f"Expected R1 to fire (decorator is inert under M4); got R1: {[(f.line, f.message[:80]) for f in r1]}"

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


class TestUnknownTrustBoundaryKwargs:
    """Unknown kwargs are malformed, not ignored documentation."""

    def test_unknown_kwarg_emits_diagnostic_and_makes_decorator_inert(self) -> None:
        source = dedent("""
            from elspeth.contracts import trust_boundary

            @trust_boundary(
                tier=3,
                source="x",
                source_param="arguments",
                suppresses=("R1",),
                invariant="x",
                invarant="typo",
            )
            def handler(arguments):
                return arguments.get("k")
        """)
        findings = _findings(source)
        unknown = _findings_by_rule(findings, "R_TB_UNKNOWN_KWARG")
        assert len(unknown) == 1
        assert "invarant" in unknown[0].message
        assert len(_findings_by_rule(findings, "R1")) == 1

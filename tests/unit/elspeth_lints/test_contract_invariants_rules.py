"""Tests for contract-invariant elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap

from elspeth_lints.core.protocols import Finding
from elspeth_lints.rules.contract_invariants.validation_theatre.rule import analyze_tree


def test_validation_theatre_reports_deferred_success_factory() -> None:
    findings = _analyze_validation_theatre(
        """
        from contracts import OutputValidationResult

        def validate_output_target() -> OutputValidationResult:
            # Source field resolution is unavailable; this will be checked later.
            return OutputValidationResult.success()
        """
    )

    assert [finding.rule_id for finding in findings] == ["contract_invariants.validation_theatre"]
    assert findings[0].line == 6


def test_validation_theatre_reports_deferred_true_return() -> None:
    findings = _analyze_validation_theatre(
        """
        def check_schema_contract() -> bool:
            if schema is None:
                # Skip until runtime schema observation is available.
                return True
            return schema.is_valid()
        """
    )

    assert [finding.rule_id for finding in findings] == ["contract_invariants.validation_theatre"]


def test_validation_theatre_accepts_success_without_deferred_comment() -> None:
    findings = _analyze_validation_theatre(
        """
        from contracts import OutputValidationResult

        def validate_output_target(path) -> OutputValidationResult:
            if not path.exists():
                return OutputValidationResult.success()
            return OutputValidationResult.failure("already exists")
        """
    )

    assert findings == []


def test_validation_theatre_accepts_deferred_failure() -> None:
    findings = _analyze_validation_theatre(
        """
        from contracts import OutputValidationResult

        def validate_output_target() -> OutputValidationResult:
            # Field resolution is not available yet; this will be checked later.
            return OutputValidationResult.failure("field resolution is required before validation")
        """
    )

    assert findings == []


def test_validation_theatre_ignores_non_validation_functions() -> None:
    findings = _analyze_validation_theatre(
        """
        def build_status() -> bool:
            # Skip until the next stage.
            return True
        """
    )

    assert findings == []


def _analyze_validation_theatre(source: str) -> list[Finding]:
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)
    return analyze_tree(tree, "example.py", dedented.splitlines())

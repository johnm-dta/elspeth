"""Pre/post migration parity (ADR-010 §Decision 2) plus AuditEvidenceBase widening."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_all_four_pre_migration_members_remain_tier_1() -> None:
    from elspeth.contracts import errors as errors_mod
    from elspeth.contracts.errors import (
        AuditIntegrityError,
        FrameworkBugError,
        OrchestrationInvariantError,
        PassThroughContractViolation,
    )

    for cls in (AuditIntegrityError, FrameworkBugError, OrchestrationInvariantError, PassThroughContractViolation):
        assert cls in errors_mod.TIER_1_ERRORS, f"{cls.__name__} missing from TIER_1_ERRORS after migration"


def test_plugin_contract_violation_is_NOT_tier_1() -> None:
    """Base class stays out (ADR-008: plugin bug != framework corruption)."""
    from elspeth.contracts import errors as errors_mod
    from elspeth.contracts.errors import PluginContractViolation

    assert PluginContractViolation not in errors_mod.TIER_1_ERRORS


def test_plugin_contract_violation_is_audit_evidence() -> None:
    """But PluginContractViolation DOES inherit AuditEvidenceBase after Task 4."""
    from elspeth.contracts.audit_evidence import AuditEvidenceBase
    from elspeth.contracts.errors import PluginContractViolation

    assert issubclass(PluginContractViolation, AuditEvidenceBase)


def test_errors_module_tier_1_errors_is_live_view_not_snapshot(reset_tier_registry: None) -> None:
    """Regression test for reviewer B8.

    Before v1: from elspeth.contracts.errors import TIER_1_ERRORS captured a
    snapshot; late registrations never reached errors-module callers. After
    v1: errors.TIER_1_ERRORS is a module __getattr__ returning the live tuple.
    """
    from elspeth.contracts import errors as errors_mod
    from elspeth.contracts import tier_registry

    before = errors_mod.TIER_1_ERRORS

    class _TempViolation(Exception):
        pass

    tier_registry._register_with_module_prefix(
        cls=_TempViolation,
        reason="regression test for live view",
        caller_module="elspeth.contracts.test_only",
    )
    after = errors_mod.TIER_1_ERRORS
    assert _TempViolation in after, "errors.TIER_1_ERRORS did not reflect late registration"
    assert _TempViolation not in before, "Test setup error — class pre-existed"


def test_contracts_package_root_does_not_expose_tier_1_errors() -> None:
    import elspeth.contracts as contracts_mod

    assert "TIER_1_ERRORS" not in contracts_mod.__all__
    with pytest.raises(AttributeError):
        _ = contracts_mod.TIER_1_ERRORS  # type: ignore[attr-defined]
    with pytest.raises(ImportError, match="TIER_1_ERRORS"):
        exec("from elspeth.contracts import TIER_1_ERRORS", {})


def test_tier_registry_does_not_export_except_clause_view() -> None:
    """The registry live view is private so callers cannot use it in ``except``."""
    import elspeth.contracts.tier_registry as registry_mod

    assert not hasattr(registry_mod, "TIER_1_ERRORS")
    with pytest.raises(ImportError, match="TIER_1_ERRORS"):
        exec("from elspeth.contracts.tier_registry import TIER_1_ERRORS", {})


def test_tier_1_reason_is_non_empty_for_all_members() -> None:
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_reason

    for cls in _TIER_1_ERRORS_VIEW:
        reason = tier_1_reason(cls)
        assert reason and reason.strip(), f"{cls.__name__} registered without meaningful reason"


def test_repo_uses_live_tier_1_error_attribute_access_not_from_import_snapshots() -> None:
    """Repo code must not snapshot ``TIER_1_ERRORS`` via ``from ... import``."""
    forbidden_imports: list[str] = []
    src_root = Path("src/elspeth")

    for py_file in sorted(src_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module not in {"elspeth.contracts", "elspeth.contracts.errors", "elspeth.contracts.tier_registry"}:
                continue
            if not any(alias.name == "TIER_1_ERRORS" for alias in node.names):
                continue
            forbidden_imports.append(f"{py_file}:{node.lineno}: from {node.module} import TIER_1_ERRORS")

    assert not forbidden_imports, (
        "TIER_1_ERRORS must be accessed via module attribute (for example, "
        "`errors.TIER_1_ERRORS`) so callers see the live registry.\n" + "\n".join(forbidden_imports)
    )

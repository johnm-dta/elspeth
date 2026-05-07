"""Tests for scripts/cicd/enforce_tier_1_decoration.py — TDE2 rule.

TDE2 (ADR-010 M8, issue elspeth-3af772b9e3): every ``tier_1_error(...)``
Call must pass ``caller_module=__name__`` where the value is the literal
Python name ``__name__``. This closes the frame-offset-spoofing vector
the prior ``inspect.stack()[2]`` implementation exposed.

These tests focus on the new rule. TDE1 (class-level decoration coverage)
is exercised by the live-repo invocation through pre-commit hooks.
"""

from __future__ import annotations

from pathlib import Path

from scripts.cicd.enforce_tier_1_decoration import (
    CallerModuleFinding,
    _check_caller_module_literal,
    _is_tier_1_error_call,
    scan_file,
)


def _write(tmp_path: Path, body: str) -> Path:
    """Write a Python source file and return its path."""
    path = tmp_path / "sample.py"
    path.write_text(body)
    return path


def _parse_first_call(source: str):
    """Parse a source string and return the first ast.Call node — helper for unit-level checks."""
    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return node
    raise AssertionError("no Call node in source")


# =============================================================================
# Unit tests for the helper predicates
# =============================================================================


class TestIsTier1ErrorCall:
    def test_accepts_bare_name_call(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module=__name__)")
        assert _is_tier_1_error_call(call)

    def test_accepts_qualified_attribute_call(self) -> None:
        call = _parse_first_call("pkg.tier_1_error(reason='x', caller_module=__name__)")
        assert _is_tier_1_error_call(call)

    def test_rejects_unrelated_call(self) -> None:
        call = _parse_first_call("print('hello')")
        assert not _is_tier_1_error_call(call)


class TestCheckCallerModuleLiteral:
    def test_literal_name_accepted(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module=__name__)")
        assert _check_caller_module_literal(call) is None

    def test_missing_kwarg_rejected(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x')")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "missing caller_module" in detail

    def test_string_literal_rejected(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module='elspeth.fake')")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "must be the __name__ literal" in detail

    def test_variable_rejected(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module=my_module_var)")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "must be the __name__ literal" in detail

    def test_attribute_access_rejected(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module=obj.name)")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "must be the __name__ literal" in detail

    def test_fstring_rejected(self) -> None:
        call = _parse_first_call("tier_1_error(reason='x', caller_module=f'{prefix}')")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "must be the __name__ literal" in detail

    def test_name_other_than_dunder_rejected(self) -> None:
        """Even the bare name __file__ or mod_name is rejected — only __name__ accepted."""
        call = _parse_first_call("tier_1_error(reason='x', caller_module=__file__)")
        detail = _check_caller_module_literal(call)
        assert detail is not None and "must be the __name__ literal" in detail


# =============================================================================
# Integration: scan_file returns (tde1, tde2) tuple
# =============================================================================


class TestScanFileTDE2:
    def test_compliant_file_produces_no_tde2(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
from elspeth.contracts.tier_registry import tier_1_error

@tier_1_error(reason="ok", caller_module=__name__)
class SomeError(Exception):
    pass
""",
        )
        _, tde2 = scan_file(path, relative_path="sample.py")
        assert tde2 == []

    def test_missing_caller_module_produces_tde2(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
from elspeth.contracts.tier_registry import tier_1_error

@tier_1_error(reason="missing caller_module")
class SomeError(Exception):
    pass
""",
        )
        _, tde2 = scan_file(path, relative_path="sample.py")
        assert len(tde2) == 1
        assert isinstance(tde2[0], CallerModuleFinding)
        assert "missing caller_module" in tde2[0].detail

    def test_wrong_caller_module_shape_produces_tde2(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path,
            """
from elspeth.contracts.tier_registry import tier_1_error

@tier_1_error(reason="variable", caller_module=__file__)
class SomeError(Exception):
    pass
""",
        )
        _, tde2 = scan_file(path, relative_path="sample.py")
        assert len(tde2) == 1
        assert "must be the __name__ literal" in tde2[0].detail

    def test_function_call_form_also_checked(self, tmp_path: Path) -> None:
        """``Foo = tier_1_error(reason=..., caller_module=__name__)(Foo)`` form is also scanned."""
        path = _write(
            tmp_path,
            """
from elspeth.contracts.tier_registry import tier_1_error


class _Foo(Exception):
    pass


# Bad — missing caller_module.
Foo = tier_1_error(reason="function-call form")(_Foo)
""",
        )
        _, tde2 = scan_file(path, relative_path="sample.py")
        assert len(tde2) == 1
        assert "missing caller_module" in tde2[0].detail

    def test_canonical_key_uses_line_not_class_name(self, tmp_path: Path) -> None:
        """TDE2 findings key by file:TDE2:line — callers may not have an enclosing class."""
        path = _write(
            tmp_path,
            """
from elspeth.contracts.tier_registry import tier_1_error

@tier_1_error(reason="x")
class SomeError(Exception):
    pass
""",
        )
        _, tde2 = scan_file(path, relative_path="sample.py")
        assert len(tde2) == 1
        assert tde2[0].canonical_key == f"sample.py:TDE2:{tde2[0].line}"


def test_repo_errors_file_has_no_tde1_gap_for_zero_emission_success_contract_violation() -> None:
    """Regression test for the live repo: Tier-2 plugin violations need an explicit
    ``# TIER-2:`` marker immediately above the class so TDE1 can verify the tier
    mechanically rather than inferring it from prose in the docstring.
    """

    path = Path("src/elspeth/contracts/errors.py")
    tde1, _tde2 = scan_file(path, relative_path=str(path))
    flagged = {finding.class_name for finding in tde1}
    assert "ZeroEmissionSuccessContractViolation" not in flagged

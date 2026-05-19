"""Tests for audit-evidence elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.audit_evidence.audit_evidence_nominal import RULE as AUDIT_EVIDENCE_NOMINAL_RULE
from elspeth_lints.rules.audit_evidence.guard_symmetry import RULE as GUARD_SYMMETRY_RULE
from elspeth_lints.rules.audit_evidence.gve_attribution import RULE as GVE_ATTRIBUTION_RULE
from elspeth_lints.rules.audit_evidence.tier_1_decoration import RULE as TIER_1_DECORATION_RULE


def test_audit_evidence_nominal_reports_to_audit_dict_without_base() -> None:
    findings = list(
        AUDIT_EVIDENCE_NOMINAL_RULE.analyze(
            _tree("""
            class Mimic(RuntimeError):
                def to_audit_dict(self):
                    return {}
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["AEN1"]
    assert findings[0].file_path == "example.py"
    assert "Mimic" in findings[0].message


def test_tier_1_decoration_reports_missing_tier_marker() -> None:
    findings = list(
        TIER_1_DECORATION_RULE.analyze(
            _tree("""
            class WidgetError(Exception):
                pass
            """),
            Path("errors.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["TDE1"]
    assert "WidgetError" in findings[0].message


def test_tier_1_decoration_reports_missing_caller_module() -> None:
    findings = list(
        TIER_1_DECORATION_RULE.analyze(
            _tree("""
            @tier_1_error(reason="registered")
            class WidgetError(Exception):
                pass
            """),
            Path("errors.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["TDE2"]
    assert "caller_module" in findings[0].message


def test_guard_symmetry_reports_loader_without_read_guard(tmp_path: Path) -> None:
    _write(
        tmp_path / "contracts" / "audit.py",
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Widget:
            size: int

            def __post_init__(self):
                if self.size < 0:
                    raise ValueError("bad size")
        """,
    )
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                return Widget(size=row.size)
        """,
    )

    findings = list(
        GUARD_SYMMETRY_RULE.analyze(
            ast.Module(body=[], type_ignores=[]),
            tmp_path,
            RuleContext(root=tmp_path),
        )
    )

    assert [finding.rule_id for finding in findings] == ["GS1"]
    assert "WidgetLoader" in findings[0].message


def test_gve_attribution_reports_missing_component_id() -> None:
    findings = list(
        GVE_ATTRIBUTION_RULE.analyze(
            _tree("""
            def validate():
                raise GraphValidationError("bad")
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["GA1"]
    assert "component_id" in findings[0].message


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")

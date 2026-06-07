"""Tests for audit-evidence elspeth-lints rules."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.protocols import Finding, Rule, RuleContext
from elspeth_lints.rules.audit_evidence.audit_evidence_nominal import RULE as AUDIT_EVIDENCE_NOMINAL_RULE
from elspeth_lints.rules.audit_evidence.guard_symmetry import RULE as GUARD_SYMMETRY_RULE
from elspeth_lints.rules.audit_evidence.gve_attribution import RULE as GVE_ATTRIBUTION_RULE
from elspeth_lints.rules.audit_evidence.shared import load_class_allowlist
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


def test_audit_evidence_nominal_accepts_direct_base() -> None:
    findings = list(
        AUDIT_EVIDENCE_NOMINAL_RULE.analyze(
            _tree("""
            class Compliant(AuditEvidenceBase, RuntimeError):
                def to_audit_dict(self):
                    return {}
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert findings == []


def test_audit_evidence_nominal_reports_async_and_lambda_forms() -> None:
    findings = list(
        AUDIT_EVIDENCE_NOMINAL_RULE.analyze(
            _tree("""
            class AsyncMimic(RuntimeError):
                async def to_audit_dict(self):
                    return {}

            class LambdaMimic(RuntimeError):
                to_audit_dict = lambda self: {}
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["AEN1", "AEN1"]
    assert {finding.message.split(" ", 1)[0] for finding in findings} == {"AsyncMimic", "LambdaMimic"}


def test_audit_evidence_nominal_uses_directory_allowlist(tmp_path: Path) -> None:
    _write(
        tmp_path / "bad.py",
        """
        class Mimic(RuntimeError):
            def to_audit_dict(self):
                return {}
        """,
    )
    allowlist = tmp_path / "config" / "cicd" / "enforce_audit_evidence_nominal"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
allow_classes:
  - key: bad.py:AEN1:Mimic
    owner: tests
    reason: synthetic fixture
    task: tests
""",
        encoding="utf-8",
    )

    assert _root_findings(AUDIT_EVIDENCE_NOMINAL_RULE, tmp_path) == []


def test_load_class_allowlist_rejects_malformed_expiry(tmp_path: Path) -> None:
    """A typoed ``expires`` must fail closed, not silently drop the time bound.

    Regression for elspeth-2d73b966c5 / elspeth-44d771caad: the shared
    ``load_class_allowlist`` parser swallowed malformed dates and returned
    ``expires=None``, so ``fail_on_expired`` could never enforce a typoed
    expiry and a one-character diff silently disabled the bound.
    """
    allowlist = tmp_path / "config" / "cicd" / "enforce_audit_evidence_nominal"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
allow_classes:
  - key: bad.py:AEN1:Mimic
    owner: tests
    reason: synthetic fixture
    task: tests
    expires: not-a-date
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expires"):
        load_class_allowlist(allowlist)


def test_audit_evidence_nominal_fails_closed_on_malformed_expiry(tmp_path: Path) -> None:
    """End-to-end: a malformed expiry propagates out of ``analyze`` as a hard error."""
    _write(
        tmp_path / "bad.py",
        """
        class Mimic(RuntimeError):
            def to_audit_dict(self):
                return {}
        """,
    )
    allowlist = tmp_path / "config" / "cicd" / "enforce_audit_evidence_nominal"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
allow_classes:
  - key: bad.py:AEN1:Mimic
    owner: tests
    reason: synthetic fixture
    task: tests
    expires: not-a-date
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expires"):
        _root_findings(AUDIT_EVIDENCE_NOMINAL_RULE, tmp_path)


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


def test_tier_1_decoration_accepts_caller_module_name(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        from elspeth.contracts.tier_registry import tier_1_error

        @tier_1_error(reason="ok", caller_module=__name__)
        class WidgetError(Exception):
            pass
        """,
    )

    assert findings == []


def test_tier_1_decoration_accepts_qualified_decorator(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        import elspeth.contracts.tier_registry as reg

        @reg.tier_1_error(reason="ok", caller_module=__name__)
        class QualifiedError(Exception):
            pass
        """,
    )

    assert findings == []


def test_tier_1_decoration_accepts_tier_2_comment(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        # TIER-2: Plugin-facing validation error with row-controlled data.
        class PluginInputError(Exception):
            pass
        """,
    )

    assert findings == []


def test_tier_1_decoration_rejects_tier_2_comment_without_justification(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        # TIER-2:
        class EmptyJustificationError(Exception):
            pass
        """,
    )

    assert [finding.rule_id for finding in findings] == ["TDE1"]
    assert "EmptyJustificationError" in findings[0].message


def test_tier_1_decoration_checks_violation_suffix(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        class WeirdViolation(Exception):
            pass
        """,
    )

    assert [finding.rule_id for finding in findings] == ["TDE1"]
    assert "WeirdViolation" in findings[0].message


def test_tier_1_decoration_reports_bad_caller_module_shapes(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        from elspeth.contracts.tier_registry import tier_1_error

        @tier_1_error(reason="string", caller_module="elspeth.fake")
        class StringCallerError(Exception):
            pass

        @tier_1_error(reason="file", caller_module=__file__)
        class FileCallerError(Exception):
            pass
        """,
    )

    assert [finding.rule_id for finding in findings] == ["TDE2", "TDE2"]
    assert all("must be the __name__ literal" in finding.message for finding in findings)


def test_tier_1_decoration_checks_function_call_form(tmp_path: Path) -> None:
    findings = _tier_1_findings_from_file(
        tmp_path,
        """
        from elspeth.contracts.tier_registry import tier_1_error

        # TIER-2: Helper wrapped by a later decorator call.
        class _WrappedError(Exception):
            pass

        WrappedError = tier_1_error(reason="function-call form")(_WrappedError)
        """,
    )

    assert [finding.rule_id for finding in findings] == ["TDE2"]
    assert "missing caller_module" in findings[0].message


def test_guard_symmetry_reports_loader_without_read_guard(tmp_path: Path) -> None:
    _write_validated_widget(tmp_path)
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                return Widget(size=row.size)
        """,
    )

    findings = _root_findings(GUARD_SYMMETRY_RULE, tmp_path)

    assert [finding.rule_id for finding in findings] == ["GS1"]
    assert "WidgetLoader" in findings[0].message


def test_guard_symmetry_accepts_loader_with_audit_integrity_error(tmp_path: Path) -> None:
    _write_validated_widget(tmp_path)
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                if row.size < 0:
                    raise AuditIntegrityError("bad size")
                return Widget(size=row.size)
        """,
    )

    assert _root_findings(GUARD_SYMMETRY_RULE, tmp_path) == []


def test_guard_symmetry_ignores_freeze_only_post_init(tmp_path: Path) -> None:
    _write(
        tmp_path / "contracts" / "audit.py",
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Widget:
            metadata: dict

            def __post_init__(self):
                freeze_fields(self, "metadata")
        """,
    )
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                return Widget(metadata=row.metadata)
        """,
    )

    assert _root_findings(GUARD_SYMMETRY_RULE, tmp_path) == []


def test_guard_symmetry_detects_validation_helper_calls(tmp_path: Path) -> None:
    _write(
        tmp_path / "contracts" / "audit.py",
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Row:
            row_index: int

            def __post_init__(self):
                require_int(self.row_index, "row_index", min_value=0)

        @dataclass(frozen=True)
        class TokenOutcome:
            outcome: str

            def __post_init__(self):
                _validate_enum(self.outcome, TerminalOutcome, "outcome")
        """,
    )
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class RowLoader:
            def load(self, row):
                return Row(row_index=row.row_index)

        class TokenOutcomeLoader:
            def load(self, row):
                return TokenOutcome(outcome=row.outcome)
        """,
    )

    findings = _root_findings(GUARD_SYMMETRY_RULE, tmp_path)

    assert [finding.rule_id for finding in findings] == ["GS1", "GS1"]
    assert {finding.message.split(" has ", 1)[0] for finding in findings} == {"Row", "TokenOutcome"}


def test_guard_symmetry_skips_abstract_loader_bodies(tmp_path: Path) -> None:
    _write_validated_widget(tmp_path)
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                raise NotImplementedError
        """,
    )

    assert _root_findings(GUARD_SYMMETRY_RULE, tmp_path) == []


def test_guard_symmetry_node_state_variants_use_shared_loader_override(tmp_path: Path) -> None:
    _write(
        tmp_path / "contracts" / "audit.py",
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class NodeStateOpen:
            attempt: int

            def __post_init__(self):
                if self.attempt < 0:
                    raise ValueError("bad attempt")
        """,
    )
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class NodeStateLoader:
            def load(self, row):
                return NodeStateOpen(attempt=row.attempt)
        """,
    )

    findings = _root_findings(GUARD_SYMMETRY_RULE, tmp_path)

    assert [finding.rule_id for finding in findings] == ["GS1"]
    assert "NodeStateLoader" in findings[0].message


def test_guard_symmetry_uses_directory_allowlist(tmp_path: Path) -> None:
    _write_validated_widget(tmp_path)
    _write(
        tmp_path / "core" / "model_loaders.py",
        """
        class WidgetLoader:
            def load(self, row):
                return Widget(size=row.size)
        """,
    )
    allowlist = tmp_path / "config" / "cicd" / "enforce_guard_symmetry"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
per_file_rules:
  - pattern: core/model_loaders.py
    rules: [GS1]
    reason: synthetic fixture
    max_hits: 1
""",
        encoding="utf-8",
    )

    assert _root_findings(GUARD_SYMMETRY_RULE, tmp_path) == []


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


def test_gve_attribution_accepts_component_id() -> None:
    findings = list(
        GVE_ATTRIBUTION_RULE.analyze(
            _tree("""
            def validate():
                raise GraphValidationError("ok", component_id="node_1")
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert findings == []


def test_gve_attribution_reports_component_type_without_component_id() -> None:
    findings = list(
        GVE_ATTRIBUTION_RULE.analyze(
            _tree("""
            def validate():
                raise GraphValidationError("partial", component_type="coalesce")
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["GA1"]


def test_gve_attribution_detects_qualified_call_and_mixed_good_bad() -> None:
    findings = list(
        GVE_ATTRIBUTION_RULE.analyze(
            _tree("""
            def bad():
                raise models.GraphValidationError("missing")

            def good():
                raise models.GraphValidationError("ok", component_id="node_1")
            """),
            Path("example.py"),
            RuleContext(root=Path(".")),
        )
    )

    assert [finding.rule_id for finding in findings] == ["GA1"]
    assert "line 3" in findings[0].message


def test_gve_attribution_reports_relative_path_and_skips_syntax_errors(tmp_path: Path) -> None:
    _write(tmp_path / "bad_syntax.py", "def broken(:\n    pass\n")
    _write(
        tmp_path / "subdir" / "inner.py",
        """
        def validate():
            raise GraphValidationError("nested")
        """,
    )

    findings = _root_findings(GVE_ATTRIBUTION_RULE, tmp_path)

    assert [finding.rule_id for finding in findings] == ["GA1"]
    assert findings[0].file_path == "subdir/inner.py"


def test_gve_attribution_uses_directory_allowlist(tmp_path: Path) -> None:
    _write(
        tmp_path / "structural.py",
        """
        def validate():
            raise GraphValidationError("cycle")
        """,
    )
    allowlist = tmp_path / "config" / "cicd" / "enforce_gve_attribution"
    allowlist.mkdir(parents=True)
    (allowlist / "rules.yaml").write_text(
        """
per_file_rules:
  - pattern: structural.py
    rules: [GA1]
    reason: synthetic structural graph error
    max_hits: 1
""",
        encoding="utf-8",
    )

    assert _root_findings(GVE_ATTRIBUTION_RULE, tmp_path) == []


def test_audit_evidence_json_mode_succeeds_on_current_codebase(
    elspeth_lints_subprocess_env: dict[str, str],
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "audit_evidence.nominal_base,audit_evidence.tier_1_decoration,audit_evidence.guard_symmetry,audit_evidence.gve_attribution",
            "--root",
            "src/elspeth",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[3],
        env=elspeth_lints_subprocess_env,
    )

    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert json.loads(result.stdout) == []


def _tree(source: str) -> ast.Module:
    return ast.parse(textwrap.dedent(source))


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")


def _root_findings(rule: Rule, root: Path) -> list[Finding]:
    return list(
        rule.analyze(
            ast.Module(body=[], type_ignores=[]),
            root,
            RuleContext(root=root),
        )
    )


def _tier_1_findings_from_file(tmp_path: Path, source: str) -> list[Finding]:
    path = tmp_path / "errors.py"
    _write(path, source)
    return list(TIER_1_DECORATION_RULE.analyze(_tree(source), path, RuleContext(root=tmp_path)))


def _write_validated_widget(tmp_path: Path) -> None:
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

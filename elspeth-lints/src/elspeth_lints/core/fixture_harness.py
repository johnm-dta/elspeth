"""Shared fixture harness for elspeth-lints rules."""

from __future__ import annotations

import ast
import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from elspeth_lints.core.ast_walker import ParsedPythonFile, PythonSyntaxError, parse_python_file, walk_python_files
from elspeth_lints.core.emitters.json import render_json
from elspeth_lints.core.protocols import Finding, Rule, RuleContext, RuleScope
from elspeth_lints.core.registry import RuleRegistry

FixtureKind = Literal["examples_violation", "examples_clean"]


@dataclass(frozen=True, slots=True)
class RuleFixtureCase:
    """One rule fixture case."""

    rule: Rule
    kind: FixtureKind
    fixture_path: Path
    expected_path: Path | None

    @property
    def name(self) -> str:
        """Return a stable pytest id for the fixture case."""
        return f"{self.kind}/{self.fixture_path.name}"


def discover_registry_fixture_cases(registry: RuleRegistry) -> list[RuleFixtureCase]:
    """Discover fixture cases for every rule in a registry."""
    cases: list[RuleFixtureCase] = []
    for _rule_id, rule in registry.items():
        cases.extend(discover_rule_fixture_cases(rule))
    return cases


def discover_rule_fixture_cases(rule: Rule) -> list[RuleFixtureCase]:
    """Discover all fixture cases for one rule."""
    fixture_root = _fixture_root(rule)
    if not fixture_root.exists():
        return []
    cases: list[RuleFixtureCase] = []
    cases.extend(_discover_cases(rule, fixture_root / "examples_violation", kind="examples_violation"))
    cases.extend(_discover_cases(rule, fixture_root / "examples_clean", kind="examples_clean"))
    return cases


def find_rules_missing_fixtures(rules: Iterable[Rule]) -> list[str]:
    """Return rule ids whose fixture directories are missing or empty."""
    missing: list[str] = []
    for rule in rules:
        fixture_root = _fixture_root(rule)
        violation_cases = _case_items(fixture_root / "examples_violation")
        clean_cases = _case_items(fixture_root / "examples_clean")
        if not violation_cases or not clean_cases:
            missing.append(rule.id)
    return missing


def find_fixture_inventory_errors(rules: Iterable[Rule]) -> list[str]:
    """Return human-readable fixture inventory errors for rule metadata drift."""
    errors: list[str] = []
    for rule in rules:
        fixture_root = _fixture_root(rule)
        violation_count = len(_case_items(fixture_root / "examples_violation"))
        clean_count = len(_case_items(fixture_root / "examples_clean"))

        if violation_count == 0:
            errors.append(f"{rule.id}: missing examples_violation fixtures")
        if clean_count == 0:
            errors.append(f"{rule.id}: missing examples_clean fixtures")
        if violation_count != rule.metadata.examples_violation_count:
            errors.append(
                f"{rule.id}: metadata examples_violation_count={rule.metadata.examples_violation_count} but discovered {violation_count}"
            )
        if clean_count != rule.metadata.examples_clean_count:
            errors.append(f"{rule.id}: metadata examples_clean_count={rule.metadata.examples_clean_count} but discovered {clean_count}")
    return errors


def assert_fixture_case(case: RuleFixtureCase) -> None:
    """Run a fixture case and assert it matches its expected findings."""
    actual_findings = _findings_as_jsonable(_run_fixture_case(case))
    expected_findings = _load_expected_findings(case)
    if actual_findings != expected_findings:
        diff = {
            "case": case.name,
            "expected": expected_findings,
            "actual": actual_findings,
        }
        raise AssertionError(json.dumps(diff, indent=2, sort_keys=True))


def _discover_cases(rule: Rule, directory: Path, *, kind: FixtureKind) -> list[RuleFixtureCase]:
    cases: list[RuleFixtureCase] = []
    for item in _case_items(directory):
        expected_path = item.with_suffix(".expected.json") if kind == "examples_violation" else None
        cases.append(RuleFixtureCase(rule=rule, kind=kind, fixture_path=item, expected_path=expected_path))
    return cases


def _case_items(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(sorted(item for item in directory.iterdir() if _is_fixture_item(item)))


def _is_fixture_item(path: Path) -> bool:
    if path.name.endswith(".expected.json"):
        return False
    return path.is_dir() or path.suffix == ".py"


def _run_fixture_case(case: RuleFixtureCase) -> list[Finding]:
    root = case.fixture_path if case.fixture_path.is_dir() else case.fixture_path.parent
    context = RuleContext(root=root)
    if case.rule.scope == RuleScope.WHOLE_REPO:
        return list(case.rule.analyze(ast.Module(body=[], type_ignores=[]), root, context))

    findings: list[Finding] = []
    if case.fixture_path.is_file():
        parsed = parse_python_file(case.fixture_path)
        if isinstance(parsed, PythonSyntaxError):
            raise AssertionError(f"{case.name}: fixture has syntax error: {parsed.message}")
        findings.extend(case.rule.analyze(parsed.tree, parsed.path, context))
        return findings

    for parsed in walk_python_files(case.fixture_path):
        if isinstance(parsed, PythonSyntaxError):
            raise AssertionError(f"{case.name}: fixture has syntax error: {parsed.message}")
        findings.extend(_run_incremental_rule(case.rule, parsed, context))
    return findings


def _run_incremental_rule(rule: Rule, parsed: ParsedPythonFile, context: RuleContext) -> Iterable[Finding]:
    return rule.analyze(parsed.tree, parsed.path, context)


def _load_expected_findings(case: RuleFixtureCase) -> list[dict[str, object]]:
    if case.kind == "examples_clean":
        return []
    if case.expected_path is None or not case.expected_path.exists():
        raise AssertionError(f"{case.name}: missing expected findings file")
    raw = json.loads(case.expected_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise AssertionError(f"{case.expected_path}: expected findings must be a JSON list")
    expected: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AssertionError(f"{case.expected_path}: entry {index} must be a JSON object")
        expected.append(item)
    return _sort_finding_dicts(expected)


def _findings_as_jsonable(findings: list[Finding]) -> list[dict[str, object]]:
    raw = json.loads(render_json(_sort_findings(findings)))
    if not isinstance(raw, list):
        raise AssertionError("render_json returned a non-list payload")
    jsonable: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise AssertionError(f"render_json entry {index} is not an object")
        jsonable.append(item)
    return _sort_finding_dicts(jsonable)


def _sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda finding: (finding.file_path, finding.line, finding.column, finding.rule_id, finding.fingerprint))


def _sort_finding_dicts(findings: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(findings, key=_finding_dict_sort_key)


def _finding_dict_sort_key(finding: dict[str, object]) -> tuple[str, int, int, str, str]:
    return (
        _required_string_field(finding, "file_path"),
        _required_int_field(finding, "line"),
        _required_int_field(finding, "column"),
        _required_string_field(finding, "rule_id"),
        _required_string_field(finding, "fingerprint"),
    )


def _required_string_field(finding: dict[str, object], key: str) -> str:
    value = finding[key]
    if not isinstance(value, str):
        raise AssertionError(f"finding field {key!r} must be a string")
    return value


def _required_int_field(finding: dict[str, object], key: str) -> int:
    value = finding[key]
    if type(value) is not int:
        raise AssertionError(f"finding field {key!r} must be an integer")
    return value


def _fixture_root(rule: Rule) -> Path:
    module = importlib.import_module(rule.__class__.__module__)
    module_file_raw = module.__file__
    if module_file_raw is None:
        raise ValueError(f"{rule.id}: rule module has no __file__")
    module_file = Path(module_file_raw)
    if module_file.name in {"__init__.py", "rule.py", "metadata.py"}:
        return module_file.parent / "fixtures"
    return module_file.with_suffix("") / "fixtures"

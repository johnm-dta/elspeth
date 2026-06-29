"""Shared fixture harness for elspeth-lints rules."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    parse_python_file,
    walk_python_files,
)
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
    rule = _rule_for_case(case)
    root = case.fixture_path if case.fixture_path.is_dir() else case.fixture_path.parent
    fixture_allowlist_dir = _fixture_allowlist_dir(root)
    if fixture_allowlist_dir is not None:
        context = RuleContext(root=root, allowlist_dir_override=fixture_allowlist_dir)
        return _run_fixture_case_with_context(case, rule, root, context)

    with tempfile.TemporaryDirectory(prefix="elspeth-lints-empty-allowlist-") as empty_allowlist:
        context = RuleContext(root=root, allowlist_dir_override=Path(empty_allowlist))
        return _run_fixture_case_with_context(case, rule, root, context)


def _run_fixture_case_with_context(case: RuleFixtureCase, rule: Rule, root: Path, context: RuleContext) -> list[Finding]:
    if rule.scope == RuleScope.WHOLE_REPO:
        return list(rule.analyze(ast.Module(body=[], type_ignores=[]), root, context))

    findings: list[Finding] = []
    if case.fixture_path.is_file():
        parsed = parse_python_file(case.fixture_path)
        if isinstance(parsed, PythonSyntaxError):
            raise AssertionError(f"{case.name}: fixture has syntax error: {parsed.message}")
        if isinstance(parsed, PythonFileReadError):
            # A fixture should always be readable; if it isn't, the test
            # setup is broken — fail loudly so the test author fixes it
            # rather than silently exercising the empty case.
            raise AssertionError(f"{case.name}: fixture I/O error ({parsed.error_type}): {parsed.message}")
        findings.extend(rule.analyze(parsed.tree, parsed.path, context))
        return findings

    for parsed in walk_python_files(case.fixture_path):
        if isinstance(parsed, PythonSyntaxError):
            raise AssertionError(f"{case.name}: fixture has syntax error: {parsed.message}")
        if isinstance(parsed, PythonFileReadError):
            raise AssertionError(f"{case.name}: fixture I/O error ({parsed.error_type}): {parsed.message}")
        findings.extend(_run_incremental_rule(rule, parsed, context))
    return findings


def _fixture_allowlist_dir(root: Path) -> Path | None:
    """Return the fixture-local allowlist directory when one is unambiguous."""
    cicd_dir = root / "config" / "cicd"
    if not cicd_dir.is_dir():
        return None
    allowlist_dirs = tuple(sorted(item for item in cicd_dir.iterdir() if item.is_dir()))
    if len(allowlist_dirs) == 1:
        return allowlist_dirs[0]
    return None


def _rule_for_case(case: RuleFixtureCase) -> Rule:
    if not case.fixture_path.is_dir():
        return case.rule
    fixture_rule_path = case.fixture_path / "_fixture_rule.py"
    if not fixture_rule_path.exists():
        return case.rule
    return _load_fixture_rule(fixture_rule_path)


def _load_fixture_rule(path: Path) -> Rule:
    module_name = f"_elspeth_lints_fixture_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"{path}: could not load fixture rule module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    rule = getattr(module, "RULE", None)
    if not isinstance(rule, Rule):
        raise AssertionError(f"{path}: fixture module must expose a Rule-compatible RULE object")
    return rule


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

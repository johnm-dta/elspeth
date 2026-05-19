"""ADR-019 tests-to-source mapping inventory rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from scripts.cicd import adr019_test_inventory as legacy

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope, Severity
from elspeth_lints.rules.manifest.test_to_source_mapping.metadata import RULE_ID, RULE_METADATA


@dataclass(frozen=True, slots=True)
class TestToSourceMappingRule:
    """Run the ADR-019 tests-tree inventory through elspeth-lints."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the tests-tree inventory scan."""
        del tree, file_path
        return scan_root(context.root)


def scan_root(root: Path) -> list[Finding]:
    """Scan tests and apply the legacy ADR-019 test inventory allowlist."""
    tests_root, project_root = tests_scan_roots(root)
    findings = legacy.scan_tree(tests_root, project_root)
    active = legacy.filter_findings(findings, allowlist_path_for_root(root, "adr019_test_inventory"))
    return [to_lint_finding(finding) for finding in active]


def tests_scan_roots(root: Path) -> tuple[Path, Path]:
    """Return the tests scan root and display root."""
    root = root.resolve()
    nested_tests = root / "tests"
    if nested_tests.is_dir():
        return nested_tests, root
    if root.name == "tests":
        return root, root.parent
    return root, root


def to_lint_finding(finding: legacy.Finding) -> Finding:
    """Convert a legacy inventory finding to the elspeth-lints protocol."""
    return Finding(
        rule_id=finding.kind.value,
        file_path=finding.path,
        line=finding.line,
        column=finding.col,
        message=legacy.finding_message(finding),
        fingerprint=legacy.finding_fingerprint(finding),
        severity=Severity.ERROR,
        suggestion=str(legacy.finding_payload(finding)["suggestion"]),
    )


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


RULE = TestToSourceMappingRule()

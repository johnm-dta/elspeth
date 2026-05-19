"""ADR-019 source-symbol inventory rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from scripts.cicd import adr019_symbol_inventory as legacy

from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope, Severity
from elspeth_lints.rules.manifest.symbol_inventory.metadata import RULE_ID, RULE_METADATA


@dataclass(frozen=True, slots=True)
class SymbolInventoryRule:
    """Run the ADR-019 source-symbol inventory through elspeth-lints."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the source inventory scan."""
        del tree, file_path
        return scan_root(context.root)


def scan_root(root: Path) -> list[Finding]:
    """Scan source files and apply the legacy ADR-019 symbol allowlist."""
    source_root, project_root = source_scan_roots(root)
    findings = legacy.scan_tree(source_root, project_root)
    active = legacy.filter_findings(findings, allowlist_path_for_root(root, "adr019_symbol_inventory"))
    return [to_lint_finding(finding) for finding in active]


def source_scan_roots(root: Path) -> tuple[Path, Path]:
    """Return the source scan root and display root."""
    root = root.resolve()
    nested_source = root / "src" / "elspeth"
    if nested_source.is_dir():
        return nested_source, root
    if root.name == "elspeth" and root.parent.name == "src":
        return root, root.parent.parent
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


RULE = SymbolInventoryRule()

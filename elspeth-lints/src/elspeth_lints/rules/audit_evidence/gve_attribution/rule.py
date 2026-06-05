"""GraphValidationError attribution rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings_for_root
from elspeth_lints.core.ast_walker import PythonFileReadError, PythonSyntaxError, walk_python_files
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.gve_attribution.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION
from elspeth_lints.rules.audit_evidence.shared import (
    allowlist_path_for_root,
    display_path,
    enclosing_names,
    graph_validation_error_call,
    parent_map,
)

_ALL_RULE_IDS = frozenset({LEGACY_RULE_ID})


@dataclass(frozen=True, slots=True)
class GveAttributionRule:
    """Detect GraphValidationError raise sites missing component_id."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one syntax tree for tests, or scan a whole repository root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return scan_tree(tree, display_path(file_path, context.root), _source_lines(file_path))
        return scan_root(
            context.root,
            allowlist_dir_override=context.allowlist_dir_override,
            governance_emitted_dirs=context.allowlist_governance_emitted_dirs,
            emit_allowlist_governance=context.emit_allowlist_governance,
        )


def scan_root(
    root: Path,
    *,
    allowlist_dir_override: Path | None = None,
    governance_emitted_dirs: set[str] | None = None,
    emit_allowlist_governance: bool = True,
) -> list[Finding]:
    """Scan a root and apply the legacy per-file allowlist."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_gve_attribution")
    )
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_ALL_RULE_IDS)
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(scan_tree(item.tree, display_path(item.path, root), item.source.splitlines()))
    active = [finding for finding in findings if _allowlist_match(allowlist, finding) is None]
    return [
        *active,
        *allowlist_governance_findings_for_root(
            allowlist,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
            emitted_dirs=governance_emitted_dirs,
            enabled=emit_allowlist_governance,
        ),
    ]


def scan_tree(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[Finding]:
    """Return GA1 findings for one parsed syntax tree."""
    parents = parent_map(tree)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        if not isinstance(node.exc, ast.Call):
            continue
        if not graph_validation_error_call(node.exc):
            continue
        if any(keyword.arg == "component_id" for keyword in node.exc.keywords):
            continue
        context = enclosing_names(node, parents)
        fingerprint_payload = f"{LEGACY_RULE_ID}|{file_path}|{node.lineno}|{'::'.join(context)}"
        findings.append(
            Finding(
                rule_id=LEGACY_RULE_ID,
                file_path=file_path,
                line=node.lineno,
                column=node.col_offset,
                message=f"raise GraphValidationError(...) without component_id= at line {node.lineno}",
                fingerprint=hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16],
                severity=RULE_METADATA.severity,
                suggestion=SUGGESTION,
            )
        )
    return findings


def _allowlist_match(allowlist: Allowlist, finding: Finding) -> object | None:
    return allowlist.match(
        FindingKey(
            file_path=finding.file_path,
            rule_id=finding.rule_id,
            symbol_context=(),
            fingerprint=finding.fingerprint,
        )
    )


def _source_lines(file_path: Path) -> list[str]:
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


RULE = GveAttributionRule()

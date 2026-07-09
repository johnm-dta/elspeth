"""AuditEvidenceBase nominal inheritance rule implementation."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    parse_python_file,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.audit_evidence_nominal.metadata import (
    LEGACY_RULE_ID,
    RULE_ID,
    RULE_METADATA,
    SUGGESTION,
)
from elspeth_lints.rules.audit_evidence.shared import (
    allowlist_path_for_root,
    class_allowlist_governance_findings_for_root,
    display_path,
    iter_python_paths,
    load_class_allowlist,
)

# ADR-010 requires nominal inheritance from THIS class specifically; a base merely
# named ``AuditEvidenceBase`` (a local or wrongly-imported class) does not satisfy it.
_CANONICAL_AUDIT_EVIDENCE_MODULE = "elspeth.contracts.audit_evidence"
_AUDIT_EVIDENCE_BASE_NAME = "AuditEvidenceBase"
# The canonical module defines the base locally; a subclass there inherits it
# without an import, and that local reference IS the real base (cannot be spoofed
# — an attacker cannot make their file be the canonical module).
_CANONICAL_MODULE_SUFFIX = "contracts/audit_evidence.py"


@dataclass(frozen=True, slots=True)
class AuditEvidenceNominalRule:
    """Detect classes defining to_audit_dict without directly inheriting AuditEvidenceBase."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one syntax tree for tests, or scan a whole repository root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return scan_tree(tree, display_path(file_path, context.root))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan a root for AEN1 findings and apply the legacy allowlist."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_audit_evidence_nominal")
    )
    allowlist = load_class_allowlist(allowlist_dir)
    findings: list[Finding] = []
    for path in iter_python_paths(root):
        parsed = parse_python_file(path)
        if isinstance(parsed, PythonSyntaxError):
            raise SyntaxError(f"{parsed.path}:{parsed.line}:{parsed.column}: {parsed.message}")
        if isinstance(parsed, PythonFileReadError):
            # Mirror the syntax-error policy above: this scanner already
            # filtered candidates, so a read error indicates a race
            # between enumeration and parse — be loud, don't paper over.
            raise OSError(f"{parsed.path}: {parsed.message}")
        findings.extend(_scan_parsed(parsed, root))
    active = [finding for finding in findings if allowlist.match_key(finding.fingerprint) is None]
    return [
        *active,
        *class_allowlist_governance_findings_for_root(
            allowlist,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
        ),
    ]


def scan_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return AEN1 findings for one parsed syntax tree."""
    bindings = _audit_evidence_base_bindings(tree)
    in_canonical_module = file_path.endswith(_CANONICAL_MODULE_SUFFIX)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _class_defines_to_audit_dict(node):
            continue
        if _bases_include_audit_evidence_base(node.bases, bindings, in_canonical_module=in_canonical_module):
            continue
        findings.append(_finding(file_path, node))
    return findings


def _audit_evidence_base_bindings(tree: ast.AST) -> tuple[frozenset[str], frozenset[str]]:
    """Collect the local names / module aliases that resolve to the canonical base.

    Returns ``(local_names, module_aliases)`` where local_names are bound by
    ``from elspeth.contracts.audit_evidence import AuditEvidenceBase [as X]`` and
    module_aliases by ``import elspeth.contracts.audit_evidence as M``.
    """
    local_names: set[str] = set()
    module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == _CANONICAL_AUDIT_EVIDENCE_MODULE:
            for alias in node.names:
                if alias.name == _AUDIT_EVIDENCE_BASE_NAME:
                    local_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _CANONICAL_AUDIT_EVIDENCE_MODULE and alias.asname:
                    module_aliases.add(alias.asname)
    return frozenset(local_names), frozenset(module_aliases)


def _scan_parsed(parsed: ParsedPythonFile, root: Path) -> list[Finding]:
    return scan_tree(parsed.tree, display_path(parsed.path, root))


def _bases_include_audit_evidence_base(
    bases: list[ast.expr],
    bindings: tuple[frozenset[str], frozenset[str]],
    *,
    in_canonical_module: bool,
) -> bool:
    """True only when a base resolves to the canonical AuditEvidenceBase.

    A base merely named ``AuditEvidenceBase`` does NOT count unless it was
    imported from the canonical module (or is the local definition inside the
    canonical module itself). This closes the spoofing bypass where a local
    ``class AuditEvidenceBase`` satisfied the nominal-inheritance gate.
    """
    local_names, module_aliases = bindings
    for base in bases:
        if isinstance(base, ast.Name):
            if base.id in local_names or (in_canonical_module and base.id == _AUDIT_EVIDENCE_BASE_NAME):
                return True
        elif (
            isinstance(base, ast.Attribute)
            and base.attr == _AUDIT_EVIDENCE_BASE_NAME
            and isinstance(base.value, ast.Name)
            and base.value.id in module_aliases
        ):
            return True
    return False


def _class_defines_to_audit_dict(class_node: ast.ClassDef) -> bool:
    for node in class_node.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "to_audit_dict":
            return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "to_audit_dict":
                    return True
        # Annotated assignment WITH a value defines a real descriptor at runtime
        # (`to_audit_dict: object = lambda ...`); a bare annotation defines nothing.
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "to_audit_dict"
            and node.value is not None
        ):
            return True
    return False


def _finding(file_path: str, node: ast.ClassDef) -> Finding:
    key = f"{file_path}:{LEGACY_RULE_ID}:{node.name}"
    return Finding(
        rule_id=LEGACY_RULE_ID,
        file_path=file_path,
        line=node.lineno,
        column=node.col_offset,
        message=f"{node.name} defines to_audit_dict without inheriting AuditEvidenceBase",
        fingerprint=key,
        severity=RULE_METADATA.severity,
        suggestion=SUGGESTION,
    )


RULE = AuditEvidenceNominalRule()

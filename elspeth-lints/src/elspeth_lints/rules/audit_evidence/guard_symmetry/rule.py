"""Guard-symmetry rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    walk_python_files,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.audit_evidence.guard_symmetry.metadata import LEGACY_RULE_ID, RULE_ID, RULE_METADATA, SUGGESTION
from elspeth_lints.rules.audit_evidence.shared import allowlist_path_for_root, display_path

_ALL_RULE_IDS = frozenset({LEGACY_RULE_ID})
_VALIDATION_FUNCTIONS = frozenset({"require_int", "_validate_enum"})
_LOADER_OVERRIDES = {
    "NodeStateOpen": "NodeStateLoader",
    "NodeStatePending": "NodeStateLoader",
    "NodeStateCompleted": "NodeStateLoader",
    "NodeStateFailed": "NodeStateLoader",
    "TransformErrorRecord": "TransformErrorLoader",
    "ValidationErrorRecord": "ValidationErrorLoader",
}


@dataclass(frozen=True, slots=True)
class DataclassInfo:
    """A dataclass with post-init validation."""

    name: str
    file_path: str
    line: int


@dataclass(frozen=True, slots=True)
class LoaderInfo:
    """A concrete loader class and whether its load() body raises AuditIntegrityError."""

    name: str
    file_path: str
    line: int
    has_audit_integrity_error: bool


@dataclass(frozen=True, slots=True)
class GuardSymmetryRule:
    """Detect validated dataclass loaders without read-side integrity guards."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run the repository-scoped guard-symmetry rule."""
        del tree, file_path
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


class GuardSymmetryVisitor(ast.NodeVisitor):
    """Collect dataclasses and concrete loaders in one file."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.dataclasses: list[DataclassInfo] = []
        self.loaders: list[LoaderInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _is_dataclass(node):
            post_init = _find_method(node, "__post_init__")
            if post_init is not None and _post_init_has_validation(post_init):
                self.dataclasses.append(DataclassInfo(name=node.name, file_path=self.file_path, line=node.lineno))

        if node.name.endswith("Loader") and len(node.name) > len("Loader"):
            load_method = _find_method(node, "load")
            if load_method is not None and not _is_abstract_method(load_method):
                self.loaders.append(
                    LoaderInfo(
                        name=node.name,
                        file_path=self.file_path,
                        line=node.lineno,
                        has_audit_integrity_error=_method_raises_audit_integrity_error(load_method),
                    )
                )

        self.generic_visit(node)


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan a root and apply the legacy per-file allowlist."""
    allowlist_dir = (
        allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_guard_symmetry")
    )
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_ALL_RULE_IDS)
    dataclasses: list[DataclassInfo] = []
    loaders: list[LoaderInfo] = []
    for item in walk_python_files(root):
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        visitor = _scan_parsed(item, root)
        dataclasses.extend(visitor.dataclasses)
        loaders.extend(visitor.loaders)
    findings = find_unguarded_pairs(dataclasses, loaders)
    return [finding for finding in findings if _allowlist_match(allowlist, finding) is None]


def find_unguarded_pairs(dataclasses: list[DataclassInfo], loaders: list[LoaderInfo]) -> list[Finding]:
    """Return GS1 findings for dataclass/loader pairs missing read-side guards."""
    loader_by_name = {loader.name: loader for loader in loaders}
    findings: list[Finding] = []
    for dataclass_info in dataclasses:
        loader_name = _expected_loader_name(dataclass_info.name)
        loader = loader_by_name.get(loader_name)
        if loader is None or loader.has_audit_integrity_error:
            continue
        fingerprint_payload = f"{LEGACY_RULE_ID}|{loader.file_path}|{loader.name}|{dataclass_info.name}"
        fingerprint = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()[:16]
        findings.append(
            Finding(
                rule_id=LEGACY_RULE_ID,
                file_path=loader.file_path,
                line=loader.line,
                column=0,
                message=(
                    f"{dataclass_info.name} has __post_init__ validation "
                    f"(at {dataclass_info.file_path}:{dataclass_info.line}) but "
                    f"{loader.name}.load() has no AuditIntegrityError guards"
                ),
                fingerprint=fingerprint,
                severity=RULE_METADATA.severity,
                suggestion=SUGGESTION,
            )
        )
    return findings


def _scan_parsed(item: ParsedPythonFile, root: Path) -> GuardSymmetryVisitor:
    visitor = GuardSymmetryVisitor(display_path(item.path, root))
    visitor.visit(item.tree)
    return visitor


def _expected_loader_name(class_name: str) -> str:
    return _LOADER_OVERRIDES.get(class_name, f"{class_name}Loader")


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
            return True
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "dataclass":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "dataclass":
                return True
    return False


def _find_method(node: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _post_init_has_validation(method: ast.FunctionDef) -> bool:
    for node in ast.walk(method):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _VALIDATION_FUNCTIONS:
            return True
    return False


def _is_abstract_method(method: ast.FunctionDef) -> bool:
    statements = [
        statement
        for statement in method.body
        if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str))
    ]
    if len(statements) != 1:
        return False
    statement = statements[0]
    if isinstance(statement, ast.Pass):
        return True
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is ...:
        return True
    if isinstance(statement, ast.Raise) and statement.exc is not None:
        exc = statement.exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
        return isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError"
    return False


def _method_raises_audit_integrity_error(method: ast.FunctionDef) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        if not isinstance(exc, ast.Call):
            continue
        func = exc.func
        if isinstance(func, ast.Name) and func.id == "AuditIntegrityError":
            return True
        if isinstance(func, ast.Attribute) and func.attr == "AuditIntegrityError":
            return True
    return False


def _allowlist_match(allowlist: Allowlist, finding: Finding) -> object | None:
    return allowlist.match(
        FindingKey(
            file_path=finding.file_path,
            rule_id=finding.rule_id,
            symbol_context=(),
            fingerprint=finding.fingerprint,
        )
    )


RULE = GuardSymmetryRule()

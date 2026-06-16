"""Validation-theatre rule implementation."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import walk_function_own_scope
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.contract_invariants.validation_theatre.metadata import RULE_ID, RULE_METADATA, SUGGESTION

_VALIDATION_NAME_RE = re.compile(r"^(validate|check|verify)(_|[A-Z]|$)")
_SUCCESS_FACTORIES = frozenset({"success", "ready"})
_DEFERRED_TERMS = (
    "skip",
    "skipped",
    "defer",
    "deferred",
    "not yet",
    "later",
    "will be checked",
    "will be validated",
    "cannot validate",
    "can't validate",
)


@dataclass(frozen=True, slots=True)
class ValidationTheatreRule:
    """Detect success returns from validation branches documented as skipped/deferred."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.INCREMENTAL
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one Python file for validation-theatre findings."""
        return analyze_tree(tree, _display_path(file_path, context.root), _source_lines(file_path))


class ValidationTheatreVisitor(ast.NodeVisitor):
    """AST visitor for validation-like functions."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.symbol_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.symbol_stack.append(node.name)
        if _is_validation_like_name(node.name):
            self._scan_validation_function(node)
        self.generic_visit(node)
        self.symbol_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _scan_validation_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for child in walk_function_own_scope(node):
            if not isinstance(child, ast.Return):
                continue
            if _is_success_return(child.value) and self._has_deferred_context(child):
                self._add_finding(child, node.name)

    def _has_deferred_context(self, node: ast.Return) -> bool:
        if not self.source_lines:
            return False
        start = max(1, node.lineno - 4)
        end = min(len(self.source_lines), (getattr(node, "end_lineno", None) or node.lineno) + 1)
        context = "\n".join(self.source_lines[start - 1 : end]).lower()
        return any(term in context for term in _DEFERRED_TERMS)

    def _add_finding(self, node: ast.Return, function_name: str) -> None:
        self.findings.append(
            Finding(
                rule_id=RULE_ID,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                message=f"{function_name} returns success from a branch whose nearby comment says validation was skipped or deferred",
                fingerprint=self._fingerprint(node),
                severity=RULE_METADATA.severity,
                suggestion=SUGGESTION,
                symbol_context=tuple(self.symbol_stack),
            )
        )

    def _fingerprint(self, node: ast.Return) -> str:
        context = ":".join(self.symbol_stack) if self.symbol_stack else "_module_"
        payload = f"{RULE_ID}|{self.file_path}|{context}|{node.lineno}|{ast.dump(node, include_attributes=False)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def analyze_tree(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[Finding]:
    """Analyze one AST for validation-theatre findings."""
    visitor = ValidationTheatreVisitor(file_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


def _is_validation_like_name(name: str) -> bool:
    return _VALIDATION_NAME_RE.search(name) is not None


def _is_success_return(value: ast.expr | None) -> bool:
    if isinstance(value, ast.Constant) and value.value is True:
        return True
    if not isinstance(value, ast.Call):
        return False
    called = _called_name(value.func)
    return called in _SUCCESS_FACTORIES


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _source_lines(file_path: Path) -> list[str]:
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _display_path(file_path: Path, root: Path) -> str:
    try:
        return str(file_path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(file_path)


RULE = ValidationTheatreRule()

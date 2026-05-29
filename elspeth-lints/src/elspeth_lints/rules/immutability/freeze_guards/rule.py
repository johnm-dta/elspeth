"""Freeze-guard immutability rule implementation."""

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
from elspeth_lints.rules.immutability.freeze_guards.metadata import RULE_ID, RULE_METADATA, RULES
from elspeth_lints.rules.immutability.shared import (
    allowlist_path_for_root,
    display_path,
    is_frozen_dataclass,
    source_line,
)

_ALL_RULE_IDS = frozenset(RULES)
_FREEZE_GUARD_TYPES = {"dict", "tuple", "MappingProxyType", "frozenset", "Mapping"}
_CONTAINER_TYPES = frozenset(
    {
        "dict",
        "list",
        "set",
        "Dict",
        "List",
        "Set",
        "Mapping",
        "MutableMapping",
        "Sequence",
        "MutableSequence",
    }
)


@dataclass(frozen=True, slots=True)
class FreezeGuardsRule:
    """Detect shallow or missing freeze guards in frozen dataclass post-init methods."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Run a whole-repository scan, or a direct tree scan for focused tests."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(tree, display_path(file_path, context.root), _source_lines(file_path))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


class FreezeGuardVisitor(ast.NodeVisitor):
    """AST visitor that detects forbidden freeze patterns in __post_init__ methods."""

    def __init__(self, file_path: str, source_lines: list[str]) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.findings: list[Finding] = []
        self.symbol_stack: list[str] = []
        self._scope_is_class: list[bool] = []
        self._in_post_init = False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_stack.append(node.name)
        self._scope_is_class.append(True)
        was_in_post_init = self._in_post_init
        self._in_post_init = False

        if is_frozen_dataclass(node):
            container_fields = self._get_container_fields(node)
            if container_fields:
                post_init = self._find_post_init(node)
                if post_init is None:
                    self._add_finding(
                        "FG3",
                        node,
                        f"Frozen dataclass '{node.name}' has container fields {container_fields} but no __post_init__",
                    )
                elif not self._post_init_has_freeze_calls(post_init):
                    self._add_finding(
                        "FG3",
                        post_init,
                        f"Frozen dataclass '{node.name}' has container fields {container_fields} but __post_init__ lacks freeze_fields/deep_freeze",
                    )

        self.generic_visit(node)
        self._in_post_init = was_in_post_init
        self._scope_is_class.pop()
        self.symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.symbol_stack.append(node.name)
        self._scope_is_class.append(False)
        was_in_post_init = self._in_post_init
        self._in_post_init = node.name == "__post_init__" and self._parent_is_class()
        self.generic_visit(node)
        self._in_post_init = was_in_post_init
        self._scope_is_class.pop()
        self.symbol_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if self._in_post_init:
            if self._is_mapping_proxy_call(node):
                self._add_finding(
                    "FG1",
                    node,
                    f"Bare MappingProxyType wrap in __post_init__: {source_line(self.source_lines, node.lineno)}",
                )

            guard_types = self._isinstance_has_freeze_guard_types(node)
            if guard_types:
                self._add_finding(
                    "FG2",
                    node,
                    f"isinstance freeze guard ({', '.join(guard_types)}) in __post_init__: {source_line(self.source_lines, node.lineno)}",
                )

        self.generic_visit(node)

    def _parent_is_class(self) -> bool:
        return len(self._scope_is_class) >= 2 and self._scope_is_class[-2]

    def _annotation_contains_container(self, annotation: ast.expr | None) -> bool:
        if annotation is None:
            return False
        if isinstance(annotation, ast.Name):
            return annotation.id in _CONTAINER_TYPES
        if isinstance(annotation, ast.Subscript):
            if isinstance(annotation.value, ast.Name):
                return annotation.value.id in _CONTAINER_TYPES
            if isinstance(annotation.value, ast.Attribute):
                return annotation.value.attr in _CONTAINER_TYPES
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return self._annotation_contains_container(annotation.left) or self._annotation_contains_container(annotation.right)
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return any(container_type in annotation.value for container_type in _CONTAINER_TYPES)
        return False

    def _get_container_fields(self, node: ast.ClassDef) -> list[str]:
        return [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and self._annotation_contains_container(item.annotation)
        ]

    def _find_post_init(self, node: ast.ClassDef) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__post_init__":
                return item
        return None

    def _post_init_has_freeze_calls(self, post_init: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        for child in ast.walk(post_init):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name) and func.id in ("freeze_fields", "deep_freeze", "MappingProxyType"):
                    return True
                if isinstance(func, ast.Attribute) and func.attr in ("freeze_fields", "deep_freeze", "MappingProxyType"):
                    return True
        return False

    def _is_mapping_proxy_call(self, node: ast.Call) -> bool:
        func = node.func
        return (isinstance(func, ast.Name) and func.id == "MappingProxyType") or (
            isinstance(func, ast.Attribute) and func.attr == "MappingProxyType"
        )

    def _isinstance_has_freeze_guard_types(self, node: ast.Call) -> list[str]:
        if not (isinstance(node.func, ast.Name) and node.func.id == "isinstance"):
            return []
        if len(node.args) < 2:
            return []
        first = node.args[0]
        if not (isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name) and first.value.id == "self"):
            return []
        second = node.args[1]
        if isinstance(second, ast.Name) and second.id in _FREEZE_GUARD_TYPES:
            return [second.id]
        if isinstance(second, ast.Tuple):
            return [item.id for item in second.elts if isinstance(item, ast.Name) and item.id in _FREEZE_GUARD_TYPES]
        return []

    def _add_finding(self, rule_id: str, node: ast.expr | ast.stmt, message: str) -> None:
        rule = RULES[rule_id]
        self.findings.append(
            Finding(
                rule_id=rule_id,
                file_path=self.file_path,
                line=node.lineno,
                column=node.col_offset,
                message=message,
                fingerprint=self._fingerprint(rule_id, node),
                severity=RULE_METADATA.severity,
                suggestion=rule["remediation"],
            )
        )

    def _fingerprint(self, rule_id: str, node: ast.expr | ast.stmt) -> str:
        node_dump = ast.dump(node, include_attributes=False, annotate_fields=True)
        context = ":".join(self.symbol_stack) if self.symbol_stack else "_module_"
        payload = f"{rule_id}|{self.file_path}|{context}|{node_dump}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def analyze_tree(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[Finding]:
    """Analyze one AST for freeze-guard findings."""
    visitor = FreezeGuardVisitor(file_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Scan a root and apply the legacy per-file allowlist."""
    allowlist_dir = allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_freeze_guards")
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_ALL_RULE_IDS)
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(_analyze_parsed_file(item, root))
    return [finding for finding in findings if _allowlist_match(allowlist, finding) is None]


def _analyze_parsed_file(item: ParsedPythonFile, root: Path) -> list[Finding]:
    return analyze_tree(item.tree, display_path(item.path, root), item.source.splitlines())


def _allowlist_match(allowlist: Allowlist, finding: Finding) -> object | None:
    return allowlist.match(
        FindingKey(
            file_path=finding.file_path,
            rule_id=finding.rule_id,
            symbol_context=_symbol_context_from_key(finding),
            fingerprint=finding.fingerprint,
        )
    )


def _symbol_context_from_key(finding: Finding) -> tuple[str, ...]:
    # Fingerprints already include context; exact allow_hits are not used by
    # the freeze guard allowlist, but keep a stable context for compatibility.
    return ()


def _source_lines(file_path: Path) -> list[str]:
    try:
        return file_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


RULE = FreezeGuardsRule()

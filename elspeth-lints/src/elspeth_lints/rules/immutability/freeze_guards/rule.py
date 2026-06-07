"""Freeze-guard immutability rule implementation."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings_for_root
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
# Types whose isinstance(self.x, T) inside __post_init__ is a banned conditional
# freeze guard. list/set are included because isinstance(self.x, list) gates a
# deep_freeze the same way the wrapper types do.
_FREEZE_GUARD_TYPES = {"dict", "list", "set", "tuple", "MappingProxyType", "frozenset", "Mapping"}
# Callables whose presence in an object.__setattr__(self, "field", <expr>) RHS
# (or a freeze_fields call) means a field is frozen. tuple/frozenset are shallow
# immutable constructors; the rule already accepts shallow MappingProxyType wraps,
# so shallow tuple/frozenset coverage is consistent.
_FREEZE_PRODUCERS = frozenset({"freeze_fields", "deep_freeze", "MappingProxyType", "tuple", "frozenset"})
# Immutable carriers that STORE their subscript args as elements — recursion into
# their type parameters can reach nested mutable containers (tuple[dict, ...]).
# Deliberately excludes Callable/ClassVar/etc. whose subscripts are signatures,
# not stored containers.
_NESTED_CARRIERS = frozenset({"tuple", "frozenset", "Tuple", "FrozenSet"})
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
        return scan_root(
            context.root,
            allowlist_dir_override=context.allowlist_dir_override,
            governance_emitted_dirs=context.allowlist_governance_emitted_dirs,
            emit_allowlist_governance=context.emit_allowlist_governance,
        )


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
                else:
                    covered = self._post_init_covered_fields(post_init)
                    uncovered = [] if covered is None else [field_name for field_name in container_fields if field_name not in covered]
                    if uncovered:
                        self._add_finding(
                            "FG3",
                            post_init,
                            f"Frozen dataclass '{node.name}' has container fields {uncovered} not frozen in __post_init__ "
                            "(freeze_fields / deep_freeze / object.__setattr__ with tuple|frozenset|MappingProxyType|deep_freeze)",
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
        # Strictly additive over the original outer-name detection: every shape
        # the old detector matched still matches, PLUS recursion into subscript
        # type args so a mutable container nested inside an immutable carrier
        # (e.g. tuple[dict[str, object], ...], frozenset[Mapping[...]]) is caught.
        if annotation is None:
            return False
        if isinstance(annotation, ast.Name):
            return annotation.id in _CONTAINER_TYPES
        if isinstance(annotation, ast.Attribute):
            return annotation.attr in _CONTAINER_TYPES
        if isinstance(annotation, ast.Subscript):
            outer = annotation.value
            outer_name = outer.id if isinstance(outer, ast.Name) else (outer.attr if isinstance(outer, ast.Attribute) else "")
            if outer_name in _CONTAINER_TYPES:
                return True
            if outer_name not in _NESTED_CARRIERS:
                return False
            sub = annotation.slice
            elements = sub.elts if isinstance(sub, ast.Tuple) else [sub]
            return any(self._annotation_contains_container(element) for element in elements)
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

    def _post_init_covered_fields(self, post_init: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
        """Return the field names a __post_init__ actually freezes.

        A field is covered when it is named in a ``freeze_fields(self, "x", ...)``
        call OR reassigned via ``object.__setattr__(self, "x", <expr>)`` where the
        RHS produces a frozen value (tuple/frozenset/MappingProxyType/deep_freeze).
        Per-field, not "any freeze call exists" — partial coverage leaves the
        remaining container fields reachable and mutable.

        Returns ``None`` ("covers all, undeterminable") when freeze_fields is
        called with a non-literal/starred argument — e.g. the dynamic
        ``freeze_fields(self, *fields_to_freeze)`` pattern, where the frozen field
        set cannot be resolved statically and assuming partial coverage would be
        a false positive.
        """
        covered: set[str] = set()
        for child in ast.walk(post_init):
            if not isinstance(child, ast.Call):
                continue
            name = _called_name(child.func)
            if name == "freeze_fields":
                for arg in child.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        covered.add(arg.value)
                    elif not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                        # Starred or otherwise non-literal field arg: cannot
                        # resolve which fields are frozen — treat as covers-all.
                        return None
            elif name == "__setattr__" and len(child.args) >= 3:
                target, field_node, value_node = child.args[0], child.args[1], child.args[2]
                if (
                    isinstance(target, ast.Name)
                    and target.id == "self"
                    and isinstance(field_node, ast.Constant)
                    and isinstance(field_node.value, str)
                    and self._expr_is_freeze_producing(value_node)
                ):
                    covered.add(field_node.value)
        return covered

    def _expr_is_freeze_producing(self, expr: ast.expr) -> bool:
        return any(isinstance(node, ast.Call) and _called_name(node.func) in _FREEZE_PRODUCERS for node in ast.walk(expr))

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
        candidates = second.elts if isinstance(second, ast.Tuple) else [second]
        return [name for name in (_guard_type_name(candidate) for candidate in candidates) if name is not None]

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


def _called_name(func: ast.expr) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _guard_type_name(node: ast.expr) -> str | None:
    """Return the freeze-guard type name for a Name or qualified Attribute form."""
    if isinstance(node, ast.Name) and node.id in _FREEZE_GUARD_TYPES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in _FREEZE_GUARD_TYPES:
        return node.attr
    return None


def analyze_tree(tree: ast.AST, file_path: str, source_lines: list[str]) -> list[Finding]:
    """Analyze one AST for freeze-guard findings."""
    visitor = FreezeGuardVisitor(file_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


def scan_root(
    root: Path,
    *,
    allowlist_dir_override: Path | None = None,
    governance_emitted_dirs: set[str] | None = None,
    emit_allowlist_governance: bool = True,
) -> list[Finding]:
    """Scan a root and apply the legacy per-file allowlist."""
    allowlist_dir = allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "enforce_freeze_guards")
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=_ALL_RULE_IDS)
    findings: list[Finding] = []
    for item in walk_python_files(root):
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(_analyze_parsed_file(item, root))
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

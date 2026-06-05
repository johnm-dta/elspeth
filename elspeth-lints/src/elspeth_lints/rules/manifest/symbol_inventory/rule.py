"""ADR-019 source-symbol inventory rule implementation."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, FindingKey, load_allowlist
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings_for_root
from elspeth_lints.core.protocols import Finding as LintFinding
from elspeth_lints.core.protocols import RuleContext, RuleMetadata, RuleScope, Severity
from elspeth_lints.rules.manifest.symbol_inventory.metadata import RULE_ID, RULE_METADATA


class FindingKind(StrEnum):
    """ADR-019 source-tree inventory finding kinds."""

    IS_TERMINAL_ANNOTATION = "is_terminal_annotation"
    IS_TERMINAL_ATTRIBUTE = "is_terminal_attribute"
    IS_TERMINAL_KEYWORD = "is_terminal_keyword"
    IS_TERMINAL_DICT_KEY = "is_terminal_dict_key"
    ROW_OUTCOME_STRING_COMPARE = "row_outcome_string_compare"
    TERMINAL_OUTCOME_STRING_COMPARE = "terminal_outcome_string_compare"
    TERMINAL_PATH_STRING_COMPARE = "terminal_path_string_compare"
    ROW_OUTCOME_STRING_MEMBERSHIP = "row_outcome_string_membership"
    TERMINAL_OUTCOME_STRING_MEMBERSHIP = "terminal_outcome_string_membership"
    TERMINAL_PATH_STRING_MEMBERSHIP = "terminal_path_string_membership"


ROW_OUTCOME_VALUES = frozenset(
    {
        "completed",
        "routed",
        "routed_on_error",
        "forked",
        "failed",
        "quarantined",
        "diverted",
        "consumed_in_batch",
        "dropped_by_filter",
        "coalesced",
        "expanded",
        "buffered",
    }
)
TERMINAL_OUTCOME_VALUES = frozenset({"success", "failure", "transient"})
TERMINAL_PATH_VALUES = frozenset(
    {
        "default_flow",
        "gate_routed",
        "on_error_routed",
        "filter_dropped",
        "coalesced",
        "unrouted",
        "quarantined_at_source",
        "sink_fallback_to_failsink",
        "sink_discarded",
        "fork_parent",
        "expand_parent",
        "batch_consumed",
        "buffered",
    }
)


@dataclass(frozen=True, slots=True)
class InventoryFinding:
    """One ADR-019 source-tree inventory finding."""

    kind: FindingKind
    path: str
    line: int
    col: int
    symbol: str
    context: str

    def to_json_payload(self) -> dict[str, object]:
        """Return the legacy JSON-lines payload shape."""
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True, slots=True)
class SymbolInventoryRule:
    """Run the ADR-019 source-symbol inventory through elspeth-lints."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[LintFinding]:
        """Run the source inventory scan."""
        del tree, file_path
        return scan_root(
            context.root,
            allowlist_dir_override=context.allowlist_dir_override,
            emit_allowlist_governance=context.emit_allowlist_governance,
        )


class ADR019Visitor(ast.NodeVisitor):
    """AST visitor for ADR-019 single-axis leftovers and brittle string checks."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.findings: list[InventoryFinding] = []

    def _add(self, kind: FindingKind, node: ast.AST, symbol: str) -> None:
        line, col = _node_location(node)
        self.findings.append(
            InventoryFinding(
                kind=kind,
                path=self.path,
                line=line,
                col=col,
                symbol=symbol,
                context=_context(node),
            )
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        target = node.target
        if (isinstance(target, ast.Name) and target.id == "is_terminal") or (
            isinstance(target, ast.Attribute) and target.attr == "is_terminal"
        ):
            self._add(FindingKind.IS_TERMINAL_ANNOTATION, node, "is_terminal")
        self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg == "is_terminal" and node.annotation is not None:
            self._add(FindingKind.IS_TERMINAL_ANNOTATION, node, "is_terminal")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "is_terminal":
            self._add(FindingKind.IS_TERMINAL_ATTRIBUTE, node, "is_terminal")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for keyword in node.keywords:
            if keyword.arg == "is_terminal":
                self._add(FindingKind.IS_TERMINAL_KEYWORD, keyword.value, "is_terminal")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key in node.keys:
            if key is None:
                continue
            if _string_constant(key) == "is_terminal":
                self._add(FindingKind.IS_TERMINAL_DICT_KEY, key, "is_terminal")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if all(isinstance(op, (ast.Eq, ast.NotEq)) for op in node.ops):
            self._visit_equality_compare(node)
        if all(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            self._visit_membership_compare(node)
        self.generic_visit(node)

    def _visit_equality_compare(self, node: ast.Compare) -> None:
        sides = [node.left, *node.comparators]
        for left, right in pairwise(sides):
            self._maybe_add_string_compare(node, left, right)
            self._maybe_add_string_compare(node, right, left)

    def _maybe_add_string_compare(self, node: ast.Compare, symbol_side: ast.AST, value_side: ast.AST) -> None:
        value = _string_constant(value_side)
        if value is None:
            return
        symbol_name = _name_for_compare_side(symbol_side) or "<literal>"
        if value in TERMINAL_PATH_VALUES and symbol_name.endswith("path"):
            self._add(FindingKind.TERMINAL_PATH_STRING_COMPARE, node, value)
        elif value in TERMINAL_OUTCOME_VALUES and symbol_name.endswith("outcome"):
            self._add(FindingKind.TERMINAL_OUTCOME_STRING_COMPARE, node, value)
        elif value in ROW_OUTCOME_VALUES and symbol_name.endswith("outcome"):
            self._add(FindingKind.ROW_OUTCOME_STRING_COMPARE, node, value)

    def _visit_membership_compare(self, node: ast.Compare) -> None:
        sides = [node.left, *node.comparators]
        for symbol_side, collection_side in pairwise(sides):
            self._maybe_add_string_membership(node, symbol_side, collection_side)

    def _maybe_add_string_membership(self, node: ast.Compare, symbol_side: ast.AST, collection_side: ast.AST) -> None:
        symbol_name = _name_for_compare_side(symbol_side) or "<literal>"
        values = _literal_string_values(collection_side)
        if not values:
            return
        if symbol_name.endswith("path") and values & TERMINAL_PATH_VALUES:
            self._add(FindingKind.TERMINAL_PATH_STRING_MEMBERSHIP, node, ",".join(sorted(values & TERMINAL_PATH_VALUES)))
        if symbol_name.endswith("outcome") and values & TERMINAL_OUTCOME_VALUES:
            self._add(
                FindingKind.TERMINAL_OUTCOME_STRING_MEMBERSHIP,
                node,
                ",".join(sorted(values & TERMINAL_OUTCOME_VALUES)),
            )
        if symbol_name.endswith("outcome") and values & ROW_OUTCOME_VALUES:
            self._add(FindingKind.ROW_OUTCOME_STRING_MEMBERSHIP, node, ",".join(sorted(values & ROW_OUTCOME_VALUES)))


def scan_root(
    root: Path,
    *,
    allowlist_dir_override: Path | None = None,
    emit_allowlist_governance: bool = True,
) -> list[LintFinding]:
    """Scan source files and apply the ADR-019 symbol allowlist."""
    source_root, project_root = source_scan_roots(root)
    findings = scan_tree(source_root, project_root)
    allowlist_dir = allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root, "symbol_inventory")
    loaded = load_allowlist(allowlist_dir, valid_rule_ids={RULE_ID}) if allowlist_dir.exists() else None
    active = _filter_loaded_findings(findings, loaded)
    lint_findings = [to_lint_finding(finding) for finding in active]
    if loaded is None:
        return lint_findings
    return [
        *lint_findings,
        *allowlist_governance_findings_for_root(
            loaded,
            allowlist_dir,
            root=root,
            allowlist_dir_override=allowlist_dir_override,
            enabled=emit_allowlist_governance,
        ),
    ]


def source_scan_roots(root: Path) -> tuple[Path, Path]:
    """Return the source scan root and display root."""
    root = root.resolve()
    nested_source = root / "src" / "elspeth"
    if nested_source.is_dir():
        return nested_source, root
    if root.name == "elspeth" and root.parent.name == "src":
        return root, root.parent.parent
    return root, root


def scan_tree(root: Path, project_root: Path | None = None) -> list[InventoryFinding]:
    """Return ADR-019 source inventory findings under root."""
    project_root = project_root or root
    findings: list[InventoryFinding] = []
    for path in _iter_python_files(root):
        findings.extend(scan_file(path, project_root))
    return findings


def scan_file(path: Path, project_root: Path | None = None) -> list[InventoryFinding]:
    """Return ADR-019 source inventory findings for one file."""
    project_root = project_root or path.parent
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    visitor = ADR019Visitor(_display_path(path, project_root))
    visitor.visit(tree)
    return visitor.findings


def filter_findings(findings: Iterable[InventoryFinding], allowlist: Path | None) -> list[InventoryFinding]:
    """Filter source inventory findings via the unified core allowlist loader."""
    if allowlist is None or not allowlist.exists():
        return list(findings)
    loaded = load_allowlist(allowlist, valid_rule_ids={RULE_ID})
    return _filter_loaded_findings(findings, loaded)


def _filter_loaded_findings(findings: Iterable[InventoryFinding], loaded: Allowlist | None) -> list[InventoryFinding]:
    if loaded is None:
        return list(findings)
    return [
        finding
        for finding in findings
        if loaded.match(
            FindingKey(
                file_path=finding.path,
                rule_id=RULE_ID,
                symbol_context=(),
                fingerprint="",
            )
        )
        is None
    ]


def to_lint_finding(finding: InventoryFinding) -> LintFinding:
    """Convert an inventory finding to the elspeth-lints protocol."""
    return LintFinding(
        rule_id=finding.kind.value,
        file_path=finding.path,
        line=finding.line,
        column=finding.col,
        message=finding_message(finding),
        fingerprint=finding_fingerprint(finding),
        severity=Severity.ERROR,
        suggestion=str(finding_payload(finding)["suggestion"]),
    )


def finding_fingerprint(finding: InventoryFinding) -> str:
    """Return the stable parity fingerprint for one inventory finding."""
    return f"{finding.path}:{finding.kind.value}:{finding.symbol}:{finding.line}:{finding.col}"


def finding_message(finding: InventoryFinding) -> str:
    """Return the parity message for one inventory finding."""
    return f"{finding.kind.value}: {finding.symbol} in {finding.context}"


def finding_payload(finding: InventoryFinding) -> dict[str, object]:
    """Return an elspeth-lints parity-schema payload."""
    return {
        "rule_id": finding.kind.value,
        "file_path": finding.path,
        "line": finding.line,
        "column": finding.col,
        "message": finding_message(finding),
        "fingerprint": finding_fingerprint(finding),
        "severity": "error",
        "suggestion": "Replace ADR-019 migration-sensitive symbols and brittle strings with the two-axis outcome/path contract.",
    }


def allowlist_path_for_root(root: Path, directory_name: str) -> Path:
    """Find a CI allowlist directory from a scan root or repository cwd."""
    relative = Path("config") / "cicd" / directory_name
    candidates = [root, Path.cwd(), *root.parents]
    for candidate in candidates:
        path = candidate / relative
        if path.exists():
            return path
    return root / relative


def _context(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def _node_location(node: ast.AST) -> tuple[int, int]:
    return getattr(node, "lineno", 0), getattr(node, "col_offset", 0)


def _name_for_compare_side(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_string_values(node: ast.AST) -> frozenset[str]:
    if not isinstance(node, ast.List | ast.Tuple | ast.Set):
        return frozenset()
    values = {_string_constant(element) for element in node.elts}
    return frozenset(value for value in values if value is not None)


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _iter_python_files(root: Path) -> Iterable[Path]:
    excluded = {"__pycache__", "node_modules", "build", "dist"}
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in excluded for part in rel_parts):
            continue
        yield path


RULE = SymbolInventoryRule()

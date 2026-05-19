#!/usr/bin/env python3
"""Inventory ADR-019 migration-sensitive symbols and brittle string checks."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path

import yaml


class FindingKind(StrEnum):
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
class Finding:
    kind: FindingKind
    path: str
    line: int
    col: int
    symbol: str
    context: str

    def to_json(self) -> str:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return json.dumps(payload, sort_keys=True)


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


class ADR019Visitor(ast.NodeVisitor):
    """AST visitor for ADR-019 single-axis leftovers and brittle string checks."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.findings: list[Finding] = []

    def _add(self, kind: FindingKind, node: ast.AST, symbol: str) -> None:
        line, col = _node_location(node)
        self.findings.append(
            Finding(
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


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def scan_file(path: Path, project_root: Path | None = None) -> list[Finding]:
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


def _iter_python_files(root: Path) -> Iterable[Path]:
    excluded = {"__pycache__", "node_modules", "build", "dist"}
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in excluded for part in rel_parts):
            continue
        yield path


def scan_tree(root: Path, project_root: Path | None = None) -> list[Finding]:
    project_root = project_root or root
    findings: list[Finding] = []
    for path in _iter_python_files(root):
        findings.extend(scan_file(path, project_root))
    return findings


def _load_allowlist(allowlist: Path | None) -> list[str]:
    if allowlist is None:
        return []
    if not allowlist.exists():
        print(f"Error: allowlist path does not exist: {allowlist}", file=sys.stderr)
        sys.exit(2)
    yaml_files = sorted(allowlist.glob("*.yaml")) if allowlist.is_dir() else [allowlist]
    patterns: list[str] = []
    for yaml_file in yaml_files:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8")) or {}
        for item in data.get("allowed", []):
            file_value = str(item.get("file", "")).strip()
            justification = str(item.get("justification", "")).strip()
            if not file_value or not justification:
                print(
                    f"Error: allowlist entry in {yaml_file} must include file and justification: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(2)
            patterns.append(file_value)
    return patterns


def _is_allowed(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if path.startswith(pattern):
                return True
        elif path == pattern:
            return True
    return False


def filter_findings(findings: Iterable[Finding], allowlist: Path | None) -> list[Finding]:
    patterns = _load_allowlist(allowlist)
    return [finding for finding in findings if not _is_allowed(finding.path, patterns)]


def finding_fingerprint(finding: Finding) -> str:
    """Return the stable parity fingerprint for one inventory finding."""
    return f"{finding.path}:{finding.kind.value}:{finding.symbol}:{finding.line}:{finding.col}"


def finding_message(finding: Finding) -> str:
    """Return the parity message for one inventory finding."""
    return f"{finding.kind.value}: {finding.symbol} in {finding.context}"


def finding_payload(finding: Finding) -> dict[str, object]:
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


def render_json_findings(findings: Iterable[Finding]) -> str:
    """Render findings as a JSON list for the parity harness."""
    return json.dumps([finding_payload(finding) for finding in findings], sort_keys=True) + "\n"


def run_check(root: Path, allowlist: Path | None) -> list[Finding]:
    project_root = Path.cwd()
    try:
        root.resolve().relative_to(project_root.resolve())
    except ValueError:
        project_root = root
    return filter_findings(scan_tree(root, project_root), allowlist)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="Scan a Python tree")
    check.add_argument("--root", type=Path, default=Path("src/elspeth"))
    check.add_argument("--allowlist", type=Path, default=None)
    check.add_argument("--format", choices=("jsonl", "json"), default="jsonl")
    args = parser.parse_args(argv)

    root = args.root
    if not root.is_dir():
        print(f"Error: --root is not a directory: {root}", file=sys.stderr)
        return 2

    findings = run_check(root, args.allowlist)
    if args.format == "json":
        print(render_json_findings(findings), end="")
        return 1 if findings else 0
    for finding in findings:
        print(finding.to_json())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

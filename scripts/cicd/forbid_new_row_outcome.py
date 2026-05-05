#!/usr/bin/env python3
"""Forbid new ``RowOutcome`` references during the ADR-019 migration window.

The ADR-019 two-axis terminal-model migration replaces the single-axis
``RowOutcome`` enum with paired ``(TerminalOutcome, TerminalPath)`` fields
across approximately 779 call sites in ~80 files (per the round-4 recount
captured in the Stage 1 PR). The migration runs as a sequenced 5-stage
rollout; during the migration window, new code outside the migration set
must not introduce ``RowOutcome.X`` references — those would inflate Stage 4's
mechanical-edit volume and create the "Success-to-the-Successful" archetype
(plan gap 11): existing references make using ``RowOutcome`` look like the
sanctioned pattern, so new contributors copy it, and the migration window
never closes.

The allowlist enumerates every file currently using ``RowOutcome.X`` at
Stage 1 PR time, plus the entire ``tests/`` tree (Stage 4 flips test
assertions). Any file outside the allowlist that contains a
``RowOutcome.X`` reference fails the check.

Detection uses AST traversal (not regex). FNR1 matches ``ast.Attribute`` nodes
where the value is ``ast.Name(id="RowOutcome")`` — i.e., ``RowOutcome.X``
attribute accesses. FNR2 matches hardcoded legacy ``RowOutcome`` value strings
compared against an ``outcome`` symbol under ``src/elspeth``.

Stage 5 (post-migration) deletes both ``RowOutcome`` itself and this script.

Usage (production):
    python scripts/cicd/forbid_new_row_outcome.py check \\
        --root . \\
        --allowlist config/cicd/forbid_new_row_outcome

Usage (smoke test, with a fake tree under tmp_path):
    python scripts/cicd/forbid_new_row_outcome.py check \\
        --root /tmp/pytest-.../test_name
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

RULE_ID = "FNR1"
RULE_NAME = "no-new-row-outcome-reference"
RULE_DESCRIPTION = (
    "RowOutcome.X reference outside the ADR-019 migration set — use "
    "(TerminalOutcome, TerminalPath) pairs in new code, or add this file "
    "to the migration allowlist with a justification if it must be retyped "
    "in Stage 2/3/4."
)

RULE_ID_2 = "FNR2"
RULE_NAME_2 = "no-hardcoded-row-outcome-value-string"
RULE_DESCRIPTION_2 = (
    "String-literal comparison against a known RowOutcome value — use "
    "(TerminalOutcome, TerminalPath) pairs in new code, or add this file "
    "to the migration allowlist with a justification."
)

_ROW_OUTCOME_VALUE_STRINGS: frozenset[str] = frozenset(
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


@dataclass(frozen=True)
class Finding:
    rule_id: str
    file_path: str
    lineno: int
    symbol: str

    def render(self) -> str:
        if self.rule_id == RULE_ID:
            return (
                f"[{RULE_ID}] {self.file_path}:{self.lineno} — "
                f"RowOutcome.{self.symbol} reference outside the ADR-019 migration "
                "allowlist. Use (TerminalOutcome, TerminalPath) pairs per the ADR-019 "
                "mapping table, or add this file to the allowlist if it is a legitimate "
                "migration site."
            )
        return (
            f"[{RULE_ID_2}] {self.file_path}:{self.lineno} — "
            f"{RULE_NAME_2}: hardcoded RowOutcome value string {self.symbol!r} "
            "compared against an outcome symbol outside the ADR-019 migration "
            "allowlist. Use typed (TerminalOutcome, TerminalPath) pairs."
        )


def _is_row_outcome_attribute(node: ast.AST) -> str | None:
    """Return the member name if ``node`` is a ``RowOutcome.X`` attribute access.

    Matches ``RowOutcome.COMPLETED``, ``RowOutcome.ROUTED``, etc. Does NOT match:
    - ``self.RowOutcome.COMPLETED`` (chained attribute, the value is itself an
      Attribute, not a Name) — not a real pattern but worth being explicit.
    - ``"RowOutcome.COMPLETED"`` (string literal) — has no semantic effect.
    - ``from x import RowOutcome`` (Import node, not Attribute).
    """
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    if node.value.id != "RowOutcome":
        return None
    return node.attr


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_string_values(node: ast.AST) -> frozenset[str]:
    if not isinstance(node, ast.List | ast.Tuple | ast.Set):
        return frozenset()
    values = {_string_constant(element) for element in node.elts}
    return frozenset(value for value in values if value is not None)


def _is_outcome_symbol(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "outcome"
    if isinstance(node, ast.Attribute):
        return node.attr == "outcome"
    return False


class _HardcodedValueVisitor(ast.NodeVisitor):
    """Find hardcoded RowOutcome value strings compared to outcome symbols."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.findings: list[Finding] = []

    def visit_Compare(self, node: ast.Compare) -> None:
        sides = [node.left, *node.comparators]
        for left, op, right in zip(sides[:-1], node.ops, sides[1:], strict=True):
            if isinstance(op, ast.Eq | ast.NotEq):
                self._visit_equality(node, left, right)
                self._visit_equality(node, right, left)
            elif isinstance(op, ast.In | ast.NotIn):
                self._visit_membership(node, left, right)
        self.generic_visit(node)

    def _add(self, node: ast.AST, symbol: str) -> None:
        self.findings.append(
            Finding(
                rule_id=RULE_ID_2,
                file_path=self.file_path,
                lineno=getattr(node, "lineno", 0),
                symbol=symbol,
            )
        )

    def _visit_equality(self, node: ast.Compare, symbol_side: ast.AST, value_side: ast.AST) -> None:
        value = _string_constant(value_side)
        if value is None:
            return
        if _is_outcome_symbol(symbol_side) and value in _ROW_OUTCOME_VALUE_STRINGS:
            self._add(node, value)

    def _visit_membership(self, node: ast.Compare, symbol_side: ast.AST, values_side: ast.AST) -> None:
        if not _is_outcome_symbol(symbol_side):
            return
        matched = sorted(_literal_string_values(values_side) & _ROW_OUTCOME_VALUE_STRINGS)
        if matched:
            self._add(node, ",".join(matched))


def scan_file(path: Path, root: Path, *, include_hardcoded_values: bool = False) -> list[Finding]:
    """AST-scan a single Python file for ``RowOutcome.X`` attribute accesses.

    The ``file_path`` in returned findings is the path relative to ``root``;
    consumers compare against allowlist entries that are also relative to
    the repo root.

    Files that are not valid UTF-8 or that contain syntax errors are silently
    skipped — a ``.py`` extension on a non-Python file (e.g. a vendored binary
    payload or a malformed test fixture) cannot contain a ``RowOutcome.X``
    Python reference by definition. We do not want a stray binary to crash
    the lint guard for the whole tree.
    """
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        member = _is_row_outcome_attribute(node)
        if member is None:
            continue
        findings.append(
            Finding(
                rule_id=RULE_ID,
                file_path=rel,
                lineno=node.lineno,
                symbol=member,
            )
        )
    if include_hardcoded_values:
        visitor = _HardcodedValueVisitor(rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


def _is_within_src_root(path: Path, src_root: Path) -> bool:
    try:
        path.resolve().relative_to(src_root.resolve())
        return True
    except ValueError:
        return False


def _load_allowlist(allowlist_dir: Path | None) -> list[str]:
    """Load file/directory allowlist entries from YAML files.

    Returns a list of path patterns (relative to repo root). Each pattern is
    either:
    - An exact relative path to a file (e.g. ``src/elspeth/contracts/enums.py``).
    - A directory prefix ending in ``/`` (e.g. ``tests/``); any file under
      that directory tree is allowed.

    Every YAML entry must include a non-empty ``justification`` field — this
    enforces that allowlist additions are documented at PR time, mirroring
    the convention in ``enforce_composer_exception_channel.py``.
    """
    if allowlist_dir is None:
        return []
    if not allowlist_dir.exists():
        print(
            f"Error: allowlist path {allowlist_dir} does not exist. Fail-closed: "
            "refusing to treat a typo as an empty allowlist (which would fail "
            "every file in the tree on day one).",
            file=sys.stderr,
        )
        sys.exit(1)
    patterns: list[str] = []
    for yml in sorted(allowlist_dir.glob("*.yaml")):
        data = yaml.safe_load(yml.read_text()) or {}
        for item in data.get("allowed", []):
            if "file" not in item:
                print(
                    f"Error: allowlist entry in {yml} missing required 'file' key: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if "justification" not in item or not str(item["justification"]).strip():
                print(
                    f"Error: allowlist entry in {yml} missing non-empty 'justification' for file {item['file']!r}: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            patterns.append(str(item["file"]))
    return patterns


def _is_allowed(rel_path: str, patterns: list[str]) -> bool:
    """Return True if ``rel_path`` matches any allowlist pattern.

    A pattern ending in ``/`` is a directory prefix match; anything else is
    an exact path match. Forward-slash separators are used uniformly because
    the scanner produces POSIX-style paths via ``Path.as_posix()``.
    """
    for pattern in patterns:
        if pattern.endswith("/"):
            if rel_path.startswith(pattern):
                return True
        elif rel_path == pattern:
            return True
    return False


def _iter_python_files(root: Path) -> list[Path]:
    """Walk ``root`` and yield every .py file, excluding noise directories.

    Excludes:
    - Any path component starting with ``.`` (hidden dirs: ``.venv``, ``.git``,
      ``.uv-cache``, ``.pytest_cache``, ``.mypy_cache``, ``.ruff_cache``,
      ``.filigree``, etc.) — these hold caches, build artefacts, and tooling
      state, never first-party source.
    - ``__pycache__`` (compiled bytecode mirroring real source).
    - ``node_modules``, ``build``, ``dist`` (frontend / packaging artefacts).

    The exclusion is intentionally aggressive because the scan walks the
    whole repo from ``--root .``; a stray non-UTF-8 ``.py`` shim under any
    of these dirs would otherwise inflate scan time and could surface as a
    confusing failure even with the read-text guard.
    """
    EXCLUDE_NAMES = {"__pycache__", "node_modules", "build", "dist"}
    files: list[Path] = []
    for py in root.rglob("*.py"):
        # Path.parts of the path relative to root, so that an explicit
        # --root deeper than a hidden dir still scans cleanly.
        try:
            rel_parts = py.relative_to(root).parts
        except ValueError:
            rel_parts = py.parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in EXCLUDE_NAMES for part in rel_parts):
            continue
        files.append(py)
    return sorted(files)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check", help="Scan tree for RowOutcome.X references")
    check.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Repo root (or directory subtree to scan)",
    )
    check.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Directory containing allowlist YAML files",
    )
    check.add_argument(
        "--src-root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source root for FNR2 hardcoded RowOutcome value-string checks",
    )
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Error: --root {root} is not a directory.", file=sys.stderr)
        return 2

    patterns = _load_allowlist(args.allowlist)
    src_root = args.src_root if args.src_root.is_absolute() else root / args.src_root

    findings: list[Finding] = []
    for py in _iter_python_files(root):
        for finding in scan_file(py, root, include_hardcoded_values=_is_within_src_root(py, src_root)):
            if _is_allowed(finding.file_path, patterns):
                continue
            findings.append(finding)

    if findings:
        print(f"\n{'=' * 70}")
        print(f"FORBIDDEN RowOutcome MIGRATION REFERENCES: {len(findings)} (rules {RULE_ID}: {RULE_NAME}, {RULE_ID_2}: {RULE_NAME_2})")
        print(f"{'=' * 70}\n")
        for f in findings:
            print(f.render())
            print()
        print(f"{'=' * 70}")
        print(
            "ADR-019 migration policy: new code uses (TerminalOutcome, TerminalPath) "
            "pairs. See docs/architecture/adr/019-two-axis-terminal-model.md."
        )
        print(f"{'=' * 70}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

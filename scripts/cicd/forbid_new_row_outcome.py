#!/usr/bin/env python3
"""Forbid new ``RowOutcome.X`` references during the ADR-019 migration window.

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

Detection uses AST traversal (not regex), matching ``ast.Attribute`` nodes
where the value is ``ast.Name(id="RowOutcome")`` — i.e., ``RowOutcome.X``
attribute accesses. String literals containing ``"RowOutcome."`` are NOT
matched (they have no semantic effect on the migration).

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


@dataclass(frozen=True)
class Finding:
    file_path: str
    lineno: int
    member_name: str

    def render(self) -> str:
        return (
            f"[{RULE_ID}] {self.file_path}:{self.lineno} — "
            f"RowOutcome.{self.member_name} reference outside the ADR-019 migration "
            "allowlist. Use (TerminalOutcome, TerminalPath) pairs per the ADR-019 "
            "mapping table, or add this file to the allowlist if it is a legitimate "
            "migration site."
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


def scan_file(path: Path, root: Path) -> list[Finding]:
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
                file_path=rel,
                lineno=node.lineno,
                member_name=member,
            )
        )
    return findings


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
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"Error: --root {root} is not a directory.", file=sys.stderr)
        return 2

    patterns = _load_allowlist(args.allowlist)

    findings: list[Finding] = []
    for py in _iter_python_files(root):
        for finding in scan_file(py, root):
            if _is_allowed(finding.file_path, patterns):
                continue
            findings.append(finding)

    if findings:
        print(f"\n{'=' * 70}")
        print(f"FORBIDDEN RowOutcome.X REFERENCES: {len(findings)} (rule {RULE_ID}: {RULE_NAME})")
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

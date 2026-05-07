#!/usr/bin/env python3
"""Enforce exception-channel discipline in composer tool-handler files.

Tool handlers at the Tier-3 boundary MUST signal LLM-argument failures
via ToolArgumentError (from protocol.py), not bare TypeError/ValueError/
UnicodeError. A bare raise of those classes would be laundered through
the compose-loop catch as if it were an LLM error, masking plugin bugs.

Rules:
- CEC1: bare `raise TypeError/ValueError/UnicodeError(...)` under the
  scanned root. Fix: raise ToolArgumentError (for Tier-3 boundary errors),
  or catch locally and return _failure_result (for handler-internal
  recovery that produces a clean diagnostic).

Usage (production):
    python scripts/cicd/enforce_composer_exception_channel.py check \\
        --root src/elspeth \\
        --allowlist config/cicd/enforce_composer_exception_channel

Usage (test, with a fake tree under tmp_path):
    python scripts/cicd/enforce_composer_exception_channel.py check \\
        --root /tmp/pytest-.../test_name
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

RULES = {
    "CEC1": {
        "name": "bare-exception-raise-in-composer-tools",
        "description": "Bare raise of TypeError/ValueError/UnicodeError in composer tools — use ToolArgumentError",
        "remediation": "raise ToolArgumentError(argument=..., expected=..., actual_type=...) from exc, or catch locally and return _failure_result",
    }
}

_BANNED = frozenset({"TypeError", "ValueError", "UnicodeError", "UnicodeDecodeError", "UnicodeEncodeError"})

# The canonical file this gate protects. Expressed relative to --root.
# Kept as a single literal so a file move surfaces as a fail-closed error in main(),
# not a silent pass of an empty scan.
_CANONICAL_TARGET_REL = Path("web/composer/tools.py")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    file_path: str
    lineno: int
    message: str


def _scan_file(path: Path, root: Path) -> list[Finding]:
    """Scan a single Python file for banned raises.

    `path` must be under `root`; `file_path` in findings is the path
    relative to `root` (matches enforce_freeze_guards.py convention).
    """
    rel = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
            name = node.exc.func.id
        elif isinstance(node.exc, ast.Name):
            name = node.exc.id
        else:
            continue
        if name in _BANNED:
            findings.append(
                Finding(
                    rule_id="CEC1",
                    file_path=rel,
                    lineno=node.lineno,
                    message=f"raise {name}(...) at {rel}:{node.lineno} — use ToolArgumentError",
                )
            )
    return findings


def _load_allowlist(path: Path | None) -> set[tuple[str, int]]:
    if path is None:
        return set()
    if not path.exists():
        print(
            f"Error: allowlist path {path} does not exist. Fail-closed: refusing to treat a typo as an empty allowlist.",
            file=sys.stderr,
        )
        sys.exit(1)
    entries: set[tuple[str, int]] = set()
    for yml in path.glob("*.yaml"):
        data = yaml.safe_load(yml.read_text()) or {}
        for item in data.get("allowed", []):
            if "justification" not in item or not str(item["justification"]).strip():
                print(
                    f"Error: allowlist entry in {yml} missing non-empty 'justification': {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if "file" not in item:
                print(
                    f"Error: allowlist entry in {yml} missing required 'file' key: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if "line" not in item:
                print(
                    f"Error: allowlist entry in {yml} missing required 'line' key: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                line_no = int(item["line"])
            except (TypeError, ValueError):
                print(
                    f"Error: allowlist entry in {yml} has non-integer 'line' value {item['line']!r}: {item!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            entries.add((str(item["file"]), line_no))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    check = sub.add_parser("check")
    check.add_argument("--root", required=True, type=Path)
    check.add_argument("--allowlist", type=Path, default=None)
    check.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    root: Path = args.root.resolve()
    canonical_target = (root / _CANONICAL_TARGET_REL).resolve()
    if not canonical_target.is_file():
        print(
            f"Error: canonical target {_CANONICAL_TARGET_REL.as_posix()!r} "
            f"not found under --root {root}. The enforcer fails closed on a "
            "missing target so file moves surface immediately instead of "
            "silently skipping.",
            file=sys.stderr,
        )
        return 2

    allowlist = _load_allowlist(args.allowlist)
    if args.files:
        targets = [p.resolve() for p in args.files]
    else:
        targets = [canonical_target]

    findings: list[Finding] = []
    for target in targets:
        if not target.is_file():
            continue
        findings.extend(_scan_file(target, root))

    active = [f for f in findings if (f.file_path, f.lineno) not in allowlist]
    for f in active:
        print(f"[{f.rule_id}] {f.message}")
    return 1 if active else 0


if __name__ == "__main__":
    sys.exit(main())

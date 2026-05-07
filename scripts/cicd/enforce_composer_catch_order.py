#!/usr/bin/env python3
"""Enforce catch-order discipline for ComposerServiceError subclasses.

``ComposerPluginCrashError`` inherits from ``ComposerServiceError``. Python
evaluates ``except`` clauses top-to-bottom, so a handler for a superclass
that precedes a handler for its subclass in the same ``try`` block makes
the subclass handler **unreachable**:

    try:
        ...
    except ComposerServiceError as exc:        # WRONG — catches the subclass
        ...
    except ComposerPluginCrashError as crash:  # never reached
        ...

In the composer route handlers (``web/sessions/routes.py``), the
``ComposerPluginCrashError`` path performs the plugin-crash-specific
partial-state persistence + response redaction, while the generic
``ComposerServiceError`` path returns a 502 composer_error response.
Inverting the order silently regresses plugin crashes into 502s — the
"silent laundering" behaviour the narrowed catch was introduced to
eliminate. The existing source comment documents the invariant, but
nothing mechanically prevents regression.

Rule CCO1 flags any ``try`` block in the scanned scope where a handler
for a ``ComposerServiceError`` supertype appears before a handler for
one of its subclasses. The canonical file protected is
``web/sessions/routes.py``; the scan walks the full ``web/`` subtree so
the same invariant is enforced if a future module picks up the pair.

Hierarchy is declared explicitly in ``_SUBCLASS_TO_SUPERCLASSES``. The
enforcer itself scans AST text and does NOT walk runtime MRO — it only
fires on the pairs declared in that dict. Transitive coverage (a
second-level subclass shadowed by ``except ComposerServiceError``) is
guaranteed by the companion test file
(``tests/unit/scripts/cicd/test_enforce_composer_catch_order.py``),
which walks ``__subclasses__()`` transitively and asserts that every
declared subclass lists **every** composer-family ancestor in its
declared supertype set. A grandchild class added without updating the
dict — or an entry that lists only the immediate parent — fails CI
loudly until the enforcer's declared map matches real MRO.

Usage (production):
    python scripts/cicd/enforce_composer_catch_order.py check \\
        --root src/elspeth \\
        --allowlist config/cicd/enforce_composer_catch_order

Usage (test, with a fake tree under tmp_path):
    python scripts/cicd/enforce_composer_catch_order.py check \\
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
    "CCO1": {
        "name": "composer-catch-order",
        "description": (
            "A handler for a ComposerServiceError supertype precedes a handler "
            "for one of its subclasses in the same try block — the subclass "
            "handler is unreachable."
        ),
        "remediation": (
            "Move the subclass except-handler above the supertype handler. "
            "Python evaluates except clauses top-to-bottom; catch narrow first."
        ),
    }
}

# Declared hierarchy. The test file imports the known composer modules that
# define ``ComposerServiceError`` subclasses and cross-checks this map against
# the real MRO so drift fails CI rather than silently bypassing the gate.
_SUBCLASS_TO_SUPERCLASSES: dict[str, frozenset[str]] = {
    "ComposerPluginCrashError": frozenset({"ComposerServiceError"}),
    "ComposerConvergenceError": frozenset({"ComposerServiceError"}),
    "ComposerRuntimePreflightError": frozenset({"ComposerServiceError"}),
    "_BadRequestLLMError": frozenset({"ComposerServiceError"}),
    "_MalformedLLMResponseError": frozenset({"ComposerServiceError"}),
}

# Canonical file this gate protects — fail-closed if it is missing under
# --root so a rename surfaces as an explicit error instead of an empty scan.
_CANONICAL_TARGET_REL = Path("web/sessions/routes.py")

# Directories under --root to scan. The rule only fires when both a
# subclass and a supertype handler coexist in one try block, so widening
# the scan is free of false positives and defends against the same
# invariant being re-introduced in a sibling module.
_SCAN_SUBDIR = Path("web")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    file_path: str
    lineno: int
    message: str


def _handler_class_names(handler: ast.ExceptHandler) -> list[str]:
    """Extract exception class names from an except-handler's type node.

    Handles the forms we expect in our codebase:
    - ``except Foo:``            -> ["Foo"]
    - ``except mod.Foo:``        -> ["Foo"]   (last attribute segment)
    - ``except (Foo, mod.Bar):`` -> ["Foo", "Bar"]
    - ``except:``                -> []        (bare except, not our concern)

    Any other shape (e.g. call expressions, subscripts) is skipped — the
    rule is intentionally scope-narrow and only reasons about textual
    class names that match our known hierarchy.
    """
    if handler.type is None:
        return []
    nodes: list[ast.expr]
    if isinstance(handler.type, ast.Tuple):
        nodes = list(handler.type.elts)
    else:
        nodes = [handler.type]
    names: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
    return names


def _scan_try(try_node: ast.Try, rel: str) -> list[Finding]:
    """Flag CCO1 violations within a single try block.

    For each ordered handler, collect the class names caught. A violation
    occurs when a handler catches a class ``S`` that is the declared
    supertype of a class ``C`` caught in a **later** handler — because
    Python's top-down evaluation makes the later subclass handler
    unreachable.
    """
    handlers = [(h, _handler_class_names(h)) for h in try_node.handlers]
    findings: list[Finding] = []
    for i, (handler_i, names_i) in enumerate(handlers):
        for j in range(i + 1, len(handlers)):
            handler_j, names_j = handlers[j]
            for sub_name in names_j:
                supertypes = _SUBCLASS_TO_SUPERCLASSES.get(sub_name)
                if not supertypes:
                    continue
                for super_name in names_i:
                    if super_name in supertypes:
                        findings.append(
                            Finding(
                                rule_id="CCO1",
                                file_path=rel,
                                lineno=handler_j.lineno,
                                message=(
                                    f"{rel}:{handler_j.lineno}: except {sub_name} "
                                    f"is unreachable — preceding except {super_name} "
                                    f"at line {handler_i.lineno} catches the subclass. "
                                    "Move the narrower handler above the broader one."
                                ),
                            )
                        )
    return findings


def _scan_file(path: Path, root: Path) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            findings.extend(_scan_try(node, rel))
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


def _discover_targets(root: Path) -> list[Path]:
    scan_root = root / _SCAN_SUBDIR
    if not scan_root.is_dir():
        return []
    return sorted(p for p in scan_root.rglob("*.py") if p.is_file())


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
        targets = _discover_targets(root)

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

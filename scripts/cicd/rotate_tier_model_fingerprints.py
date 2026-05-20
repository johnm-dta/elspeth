#!/usr/bin/env python3
"""One-shot rotator for the tier-model allowlist after AST-shifted fingerprints.

When a refactor inserts an ``ImportFrom`` (or otherwise shifts the AST
body indices for a file), every fingerprint downstream rotates. The
``trust_tier.tier_model`` rule then reports the formerly-allowlisted
violations as both:

* Stale allowlist entries (the old ``fp=<hash>`` no longer matches any
  finding in the file).
* Fresh violations (the same lines of code, with new fingerprints, not
  yet allowlisted).

This script reconciles the two by pairing them on
``(file, rule_id, symbol_context)`` — extracted from the stale entry's
key and from the new violation's source position respectively — and
rewriting the YAML so each rotation preserves its original metadata
(owner, reason, safety, expires).

Drives the rule via the JSON CLI, parses the live output, edits
``config/cicd/enforce_tier_model/*.yaml`` in place. Idempotent: a second
run finds no stale entries and is a no-op.

Usage:
    .venv/bin/python scripts/cicd/rotate_tier_model_fingerprints.py
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST_DIR = REPO_ROOT / "config" / "cicd" / "enforce_tier_model"
SRC_ROOT = REPO_ROOT / "src" / "elspeth"
ROTATABLE_RULE_IDS = frozenset({"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "TC", "L1"})
KEY_RE = re.compile(
    r"^Stale tier-model allowlist entry: "
    r"(?P<file>[^:]+):(?P<rule>R\d+|TC|L1):(?P<symbol>.+?):fp=(?P<fp>[0-9a-f]+)$"
)


def run_tier_model() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "elspeth_lints.core.cli",
            "check",
            "--rules",
            "trust_tier.tier_model",
            "--root",
            "src/elspeth",
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(REPO_ROOT / "elspeth-lints" / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if not proc.stdout.strip():
        sys.stderr.write(f"tier-model produced no JSON output. stderr:\n{proc.stderr}\n")
        sys.exit(2)
    findings: list[dict[str, Any]] = json.loads(proc.stdout)
    return findings


def find_enclosing_symbol(source_path: Path, target_line: int) -> str:
    """Return the tier-model symbol context enclosing ``target_line``."""
    tree = ast.parse(source_path.read_text())
    enclosing_symbols: list[tuple[int, int, str]] = []

    def _contains_target(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        end_line = node.end_lineno if node.end_lineno is not None else node.lineno
        return node.lineno <= target_line <= end_line

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if _contains_target(child):
                    enclosing_symbols.append((child.lineno, child.col_offset, child.name))
                    _walk(child)
            else:
                _walk(child)

    _walk(tree)
    enclosing_symbols.sort(key=lambda item: (item[0], item[1]))
    return ":".join(symbol for _, _, symbol in enclosing_symbols) if enclosing_symbols else "_module_"


def split_findings(
    findings: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (stale_entries, new_violations)."""
    stale: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    for f in findings:
        if f["rule_id"] == "trust_tier.tier_model" and f["message"].startswith("Stale tier-model allowlist entry:"):
            stale.append(f)
        elif f["rule_id"] in ROTATABLE_RULE_IDS and f["severity"] == "error":
            new.append(f)
    return stale, new


def parse_stale_key(message: str) -> tuple[str, str, str, str] | None:
    m = KEY_RE.match(message)
    if m is None:
        return None
    return m.group("file"), m.group("rule"), m.group("symbol"), m.group("fp")


def yaml_for_file(rel_file: str) -> Path:
    """Map src-relative file path to its allowlist YAML."""
    # Convention: top-level package directory under src/elspeth/ → YAML basename.
    head = rel_file.split("/", 1)[0]
    candidate = ALLOWLIST_DIR / f"{head}.yaml"
    if candidate.exists():
        return candidate
    raise RuntimeError(f"no allowlist YAML for top-level dir {head!r} (file: {rel_file})")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    # Preserve the original style: block format, no flow, no aliases.
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text)


def main() -> int:
    findings = run_tier_model()
    stale, new = split_findings(findings)
    if not stale and not new:
        print("No tier-model rotations or new violations — nothing to do.")
        return 0

    print(f"Found {len(stale)} stale allowlist entries and {len(new)} new violations.")

    # Index stale entries by (file, rule, symbol) → entry dict from yaml.
    yaml_caches: dict[Path, dict[str, Any]] = {}

    def get_yaml(path: Path) -> dict[str, Any]:
        if path not in yaml_caches:
            yaml_caches[path] = load_yaml(path)
        return yaml_caches[path]

    # Build a lookup of (file, rule, symbol) → metadata copied from stale
    # entries. A symbol can legitimately have multiple hits for the same rule;
    # keep a queue so each rotated finding preserves the matching entry's
    # owner/reason/safety/expiry instead of inheriting the last stale hit.
    stale_metadata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    stale_keys_to_remove: dict[Path, set[str]] = {}
    for f in stale:
        parsed = parse_stale_key(f["message"])
        if parsed is None:
            sys.stderr.write(f"unparseable stale message: {f['message']!r}\n")
            continue
        rel_file, rule, symbol, fp = parsed
        ypath = yaml_for_file(rel_file)
        data = get_yaml(ypath)
        full_key = f"{rel_file}:{rule}:{symbol}:fp={fp}"
        for entry in data.get("allow_hits", []):
            if entry["key"] == full_key:
                # Strip the fp from the symbol triple in the index — we want
                # to match by (file, rule, symbol_context) regardless of fp.
                stale_metadata.setdefault((rel_file, rule, symbol), []).append({k: v for k, v in entry.items() if k != "key"})
                stale_keys_to_remove.setdefault(ypath, set()).add(full_key)
                break

    # For each new violation, compute its canonical key and emit a rotated entry
    # only when it pairs with stale metadata. Genuinely new findings must stay
    # review-visible; manufacturing TODO allowlist entries would weaken the gate.
    new_entries: dict[Path, list[dict[str, Any]]] = {}
    unmatched_new: list[str] = []
    for f in new:
        rel_file = f["file_path"]
        rule = f["rule_id"]
        # File paths in tier_model output are relative to src/elspeth/.
        symbol = find_enclosing_symbol(SRC_ROOT / rel_file, f["line"])
        fp = f["fingerprint"]
        new_key = f"{rel_file}:{rule}:{symbol}:fp={fp}"
        ypath = yaml_for_file(rel_file)
        metadata_queue = stale_metadata.get((rel_file, rule, symbol))
        if metadata_queue:
            metadata = metadata_queue.pop(0)
        else:
            unmatched_new.append(new_key)
            continue
        entry = {"key": new_key, **metadata}
        new_entries.setdefault(ypath, []).append(entry)

    # Now rewrite each affected yaml.
    affected_paths = set(yaml_caches) | set(new_entries)
    for ypath in sorted(affected_paths):
        data = get_yaml(ypath)
        before = len(data.get("allow_hits", []))
        # Remove stale entries.
        keys_to_remove = stale_keys_to_remove.get(ypath, set())
        data["allow_hits"] = [e for e in data.get("allow_hits", []) if e["key"] not in keys_to_remove]
        # Append new entries.
        data.setdefault("allow_hits", []).extend(new_entries.get(ypath, []))
        after = len(data["allow_hits"])
        write_yaml(ypath, data)
        print(
            f"  {ypath.relative_to(REPO_ROOT)}: "
            f"{before} → {after} entries "
            f"(removed {len(keys_to_remove)} stale, added {len(new_entries.get(ypath, []))})"
        )

    if unmatched_new:
        sys.stderr.write(
            "Refusing to auto-allow genuinely new tier-model violation(s); review or fix explicitly:\n"
            + "\n".join(f"  {key}" for key in unmatched_new)
            + "\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

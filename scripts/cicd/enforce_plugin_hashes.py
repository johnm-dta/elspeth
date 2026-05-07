#!/usr/bin/env python3
"""CI enforcement: plugin source_file_hash declarations must match computed values.

Usage:
    python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth
    python scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth --fix

check:       Verify all plugins have correct source_file_hash (CI mode).
check --fix: Auto-update stale hashes in-place (developer mode).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.cicd.plugin_hash import (
    compute_source_file_hash,
    extract_plugin_attributes,
    fix_source_file_hash,
)

PLUGIN_DIRS = [
    "plugins/sources",
    "plugins/sinks",
    "plugins/transforms",
    "plugins/transforms/azure",
    "plugins/transforms/llm",
    "plugins/transforms/rag",
]

# Files in plugin directories that are NOT plugin entry points.
_EXCLUDED_FILES = frozenset(
    {
        "__init__.py",
        "base.py",
        "config.py",
        "validation.py",
        "templates.py",
        "langfuse.py",
        "tracing.py",
        "multi_query.py",
        "capacity_errors.py",
        "provider.py",
    }
)

EXPECTED_PLUGIN_COUNT = 37


def _discover_plugin_files(root: Path) -> list[Path]:
    """Find all plugin entry-point files under root.

    Scans PLUGIN_DIRS for .py files, excludes files prefixed with ``_``,
    ``__init__.py``, and known helper files in ``_EXCLUDED_FILES``.
    Uses ``extract_plugin_attributes()`` to confirm files contain actual
    plugin classes (must have a ``name`` attribute).
    """
    files: list[Path] = []
    for rel_dir in PLUGIN_DIRS:
        d = root / rel_dir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_") or f.name in _EXCLUDED_FILES:
                continue
            files.append(f)
    return files


def run_check(root: Path, *, fix: bool = False, min_plugins: int = EXPECTED_PLUGIN_COUNT) -> int:
    """Run the enforcement check. Returns 0 on success, 1 on failure."""
    files = _discover_plugin_files(root)

    # Count guard: fail loudly if discovery finds fewer plugins than expected.
    # Guards against wrong --root, missing directories, or scan regressions.
    plugin_count = sum(len(extract_plugin_attributes(f)) for f in files)
    if plugin_count < min_plugins:
        print(f"DISCOVERY ERROR: found {plugin_count} plugins, expected at least {min_plugins}. Check --root path and PLUGIN_DIRS.")
        return 1

    violations: list[str] = []
    fixed: list[str] = []

    for file_path in files:
        attrs_list = extract_plugin_attributes(file_path)
        if not attrs_list:
            continue

        computed = compute_source_file_hash(file_path)
        rel = file_path.relative_to(root)

        for attrs in attrs_list:
            # Check plugin_version
            if attrs.plugin_version is None or attrs.plugin_version == "0.0.0":
                violations.append(f"{rel} ({attrs.class_name}): no version declaration (plugin_version is {attrs.plugin_version!r})")

            # Check source_file_hash
            if attrs.source_file_hash is None:
                violations.append(f"{rel} ({attrs.class_name}): no source_file_hash declaration")
            elif attrs.source_file_hash != computed:
                if fix:
                    fix_source_file_hash(file_path, attrs.class_name, computed)
                    fixed.append(f"{rel} ({attrs.class_name}): updated to {computed}")
                    # Recompute after fix (file content changed)
                    computed = compute_source_file_hash(file_path)
                else:
                    violations.append(
                        f"{rel} ({attrs.class_name}): stale source_file_hash\n  declared: {attrs.source_file_hash}\n  expected: {computed}"
                    )

    if fixed:
        print(f"FIXED {len(fixed)} hash(es):")
        for msg in fixed:
            print(f"  {msg}")
        print()

    if violations:
        print(f"{'=' * 60}")
        print(f"VIOLATIONS FOUND: {len(violations)}")
        print(f"{'=' * 60}")
        print()
        for v in violations:
            print(v)
            print()
        print(f"{'=' * 60}")
        print("CHECK FAILED")
        print(f"{'=' * 60}")
        return 1

    if not fixed:
        print("All plugin hashes verified. Check passed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce plugin source_file_hash declarations.")
    sub = parser.add_subparsers(dest="command")
    check_parser = sub.add_parser("check", help="Verify plugin hashes")
    check_parser.add_argument("--root", type=Path, required=True)
    check_parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-update stale hashes (developer mode, not for CI)",
    )
    check_parser.add_argument(
        "--min-plugins",
        type=int,
        default=EXPECTED_PLUGIN_COUNT,
        help=(f"Minimum expected plugin count (default: {EXPECTED_PLUGIN_COUNT}). Fail if fewer are discovered."),
    )

    args = parser.parse_args()
    if args.command != "check":
        parser.print_help()
        sys.exit(1)

    sys.exit(run_check(args.root, fix=args.fix, min_plugins=args.min_plugins))


if __name__ == "__main__":
    main()

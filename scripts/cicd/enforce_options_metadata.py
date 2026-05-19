#!/usr/bin/env python3
"""CI lint: enforce title and description on plugin configuration fields.

Run from the project root:
    .venv/bin/python scripts/cicd/enforce_options_metadata.py

Allowlist entries live at config/cicd/enforce_options_metadata/allowlist.yaml.
Each entry has the form:
    {id: "<kind>/<plugin_name>:<field_name>", reason: "<why>"}

Discriminated variants use:
    <kind>/<plugin_name>[<variant>]:<field_name>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

ROOT = Path(__file__).resolve().parents[2]
ELSPETH_LINTS_SRC = ROOT / "elspeth-lints" / "src"

__all__ = ["iter_metadata_models", "load_allowlist", "main", "run_metadata_lint"]


def iter_metadata_models(kind: str, plugin_cls: object) -> object:
    """Yield the metadata-bearing config models for one plugin class."""
    _ensure_elspeth_lints_importable()
    from elspeth_lints.rules.plugin_contract.options_metadata.rule import iter_metadata_models as _iter_metadata_models

    return _iter_metadata_models(kind, plugin_cls)


def run_metadata_lint(*, plugin_manager: object, allowlist: set[str]) -> list[str]:
    """Return metadata failures for every plugin config field not allowlisted."""
    _ensure_elspeth_lints_importable()
    from elspeth_lints.rules.plugin_contract.options_metadata.rule import collect_metadata_findings

    return [finding.message for finding in collect_metadata_findings(plugin_manager=plugin_manager, allowlist=allowlist, root=None)]


def load_allowlist(path: Path) -> set[str]:
    """Load allowlisted identifiers and require reasons for every entry."""
    _ensure_elspeth_lints_importable()
    from elspeth_lints.rules.plugin_contract.options_metadata.rule import load_options_metadata_allowlist

    return load_options_metadata_allowlist(path)


def main(argv: list[str] | None = None) -> int:
    _ensure_elspeth_lints_importable()
    from elspeth_lints.core.emitters.json import render_json
    from elspeth_lints.rules.plugin_contract.options_metadata.rule import collect_metadata_findings

    parser = argparse.ArgumentParser(description="Enforce plugin config title and description metadata.")
    parser.add_argument("command", nargs="?", choices=("check",), default="check")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    allowlist_path = root / "config" / "cicd" / "enforce_options_metadata" / "allowlist.yaml"
    try:
        allowlist = load_allowlist(allowlist_path)
    except OSError as exc:
        print(f"Plugin config metadata lint could not read allowlist: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Plugin config metadata lint has invalid allowlist: {exc}", file=sys.stderr)
        return 2

    findings = collect_metadata_findings(plugin_manager=get_shared_plugin_manager(), allowlist=allowlist, root=root)
    if args.format == "json":
        sys.stdout.write(render_json(findings))
        return 1 if findings else 0

    if findings:
        print("Plugin config metadata lint failed:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding.message}", file=sys.stderr)
        print(
            "\nFix by adding title= and description= to each Field(...). "
            "Use the allowlist only for explicitly justified temporary exceptions.",
            file=sys.stderr,
        )
        return 1
    return 0


def _ensure_elspeth_lints_importable() -> None:
    if ELSPETH_LINTS_SRC.exists() and str(ELSPETH_LINTS_SRC) not in sys.path:
        sys.path.insert(0, str(ELSPETH_LINTS_SRC))


if __name__ == "__main__":
    sys.exit(main())

"""Cross-language consistency check: Python SlotType Literal vs TypeScript mirror.

Usage:
    python scripts/cicd/check_slot_type_cross_language.py

Exit codes:
    0 — both sets are equal (prints OK + canonical set)
    1 — sets diverge (prints both sets and the diff)

Design:
    SlotType is a Literal in recipes.py, not a StrEnum, so extraction is done
    by importing the module and using ``typing.get_args`` rather than walking
    enum members.  This is simpler than AST parsing and exercises the real
    runtime value.

    The TypeScript union is extracted by regex: we locate the ``slot_type``
    field in the RecipeSlotInput interface in guided.ts and parse the
    string-literal union members.  This is deliberately lightweight — no
    TypeScript parser dependency required.

    Both sets are sorted before comparison and before output so the script
    is byte-stable across runs (deterministic for citation purposes).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, get_args, get_type_hints

# Canonical paths relative to the repository root (where this script is run from)
_PYTHON_SOURCE = Path("src/elspeth/web/composer/recipes.py")
_TS_SOURCE = Path("src/elspeth/web/frontend/src/types/guided.ts")

# ---------------------------------------------------------------------------
# Python side: import SlotType and extract Literal members via get_args
# ---------------------------------------------------------------------------


def _extract_python_members() -> set[str]:
    """Return the set of SlotType string members from recipes.py.

    Imports the module directly and uses typing.get_args on the Literal type.
    This is authoritative: it exercises the same value the runtime uses.
    """
    # Ensure the src tree is importable from the CWD (repo root)
    sys.path.insert(0, "src")
    try:
        from elspeth.web.composer.recipes import SlotType  # type: ignore[attr-defined]

        args = get_args(SlotType)
        if not args:
            # SlotType might be imported as a re-export; try get_type_hints fallback
            import elspeth.web.composer.recipes as _mod  # type: ignore[import-untyped]

            hints: dict[str, Any] = get_type_hints(_mod)
            args = get_args(hints.get("SlotType", SlotType))
        return {str(a) for a in args}
    finally:
        if "src" in sys.path:
            sys.path.remove("src")


# ---------------------------------------------------------------------------
# TypeScript side: regex extraction from guided.ts
# ---------------------------------------------------------------------------

# Pattern targets the slot_type field in RecipeSlotInput:
#   slot_type: "blob_id" | "str" | "float" | "int" | "str_list";
#
# The interface may span multiple lines.  Keep this deliberately linear:
# capture everything up to the field terminator, then extract string literal
# members from that slice.  A nested union-token regex here can backtrack
# exponentially on malformed input before the semicolon.
_TS_FIELD_RE = re.compile(r"slot_type\s*:\s*([^;]+);", re.MULTILINE)

_TS_MEMBER_RE = re.compile(r'"([^"]+)"')


def _extract_ts_members(ts_path: Path) -> set[str]:
    """Return the set of ``slot_type`` union members from guided.ts.

    Parses the string-literal union with a regex; does not require a
    TypeScript toolchain.  Tolerant of whitespace, newlines, and trailing
    commas in the union.
    """
    content = ts_path.read_text(encoding="utf-8")
    match = _TS_FIELD_RE.search(content)
    if not match:
        raise ValueError(
            f"Could not locate `slot_type: ...;` field in {ts_path}. "
            "Check that the RecipeSlotInput interface is present and "
            "the field uses a string-literal union type."
        )
    union_text = match.group(1)
    members = _TS_MEMBER_RE.findall(union_text)
    if not members:
        raise ValueError(f"Regex matched slot_type field in {ts_path} but extracted no string literals from: {union_text!r}")
    return set(members)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the cross-language consistency check.  Returns 0 on success, 1 on drift."""
    # Verify expected source files exist from CWD
    for path in (_PYTHON_SOURCE, _TS_SOURCE):
        if not path.exists():
            print(
                f"ERROR: expected source file not found: {path}\nRun this script from the repository root.",
                file=sys.stderr,
            )
            return 1

    py_members = _extract_python_members()
    ts_members = _extract_ts_members(_TS_SOURCE)

    if py_members == ts_members:
        canonical = sorted(py_members)
        print(f"OK — SlotType members are in sync: {canonical}")
        return 0

    # Drift detected — report both sets and the diff
    only_python = sorted(py_members - ts_members)
    only_ts = sorted(ts_members - py_members)
    print("FAIL — SlotType / guided.ts mirror has drifted.", file=sys.stderr)
    print(f"  Python members ({_PYTHON_SOURCE}): {sorted(py_members)}", file=sys.stderr)
    print(f"  TypeScript members ({_TS_SOURCE}): {sorted(ts_members)}", file=sys.stderr)
    if only_python:
        print(f"  In Python only (add to TypeScript): {only_python}", file=sys.stderr)
    if only_ts:
        print(f"  In TypeScript only (add to Python or remove from TS): {only_ts}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

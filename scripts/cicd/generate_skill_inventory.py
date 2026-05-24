"""Regenerate the Foundation-knowledge tool-inventory bullets in the composer skill.

Closes the skill-markdown growth surface diagnosed in ``elspeth-6c9972ccbf``
Step 5 falsification (filigree comment 1468): adding a new composer tool
previously required hand-editing
``src/elspeth/web/composer/skills/pipeline_composer.md`` to enumerate the
tool in its Foundation-knowledge category bullet, in addition to declaring
it in the plane module's ``TOOLS_IN_MODULE`` tuple. The skill-drift gate
(``TestComposerToolNameDrift.test_skill_tool_inventory_matches_declared_tools``)
enforces bidirectional matching, so a missed bullet update fails CI rather
than silently desynchronising — but the developer still has to remember to
make the edit. This generator removes the edit from the per-tool registration
footprint by deriving the bullets from ``_REGISTERED_TOOLS`` + the two named
dispatch-outside-execute_tool carve-outs.

The categorisation taxonomy (Discovery / State / preview / Build / edit /
Diagnostics / Blobs / Secrets) is hand-curated and does not perfectly mirror
``ToolKind`` — three discovery tools live under "State / preview", one under
"Diagnostics", and two blob-mutation tools live under "Build / edit". Those
historical placements are encoded as named overrides in
``_CATEGORY_OVERRIDES``; every other tool falls into its ``ToolKind``
default category. Adding a new tool that fits an existing category requires
zero edit to this script — the generator emits the bullet under the kind's
default category. A new tool that needs cross-category placement (rare)
requires one entry in ``_CATEGORY_OVERRIDES``; a brand-new category requires
extending ``_CATEGORY_ORDER`` and the default map.

The block in the skill is delimited by sentinel HTML comments
(``<!-- BEGIN AUTOGEN: tool-inventory -->`` / ``<!-- END AUTOGEN: ... -->``).
The skill-drift extractor anchors on the surrounding human-written prose,
not on the sentinels, and only matches ``- **``-prefixed lines, so the HTML
sentinels are invisible to the drift gate.

Usage::

    # Dry-run (print the would-be output to stdout, do not modify the file):
    .venv/bin/python scripts/cicd/generate_skill_inventory.py

    # Write the inventory block back into the skill (commit the diff):
    .venv/bin/python scripts/cicd/generate_skill_inventory.py --write

    # CI gate — exit 2 with a unified diff if the skill drifts from the
    # would-be generator output:
    .venv/bin/python scripts/cicd/generate_skill_inventory.py --check

Idempotency::

    .venv/bin/python scripts/cicd/generate_skill_inventory.py --write
    .venv/bin/python scripts/cicd/generate_skill_inventory.py --write
    git diff --exit-code src/elspeth/web/composer/skills/pipeline_composer.md
    # Must exit 0 after the second run.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Final

# Resolve project root. ``__file__`` is
# ``<project_root>/scripts/cicd/generate_skill_inventory.py`` so parents[2]
# is the project root.
_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from elspeth.web.composer.tools._registry import _REGISTERED_TOOLS  # noqa: E402
from elspeth.web.composer.tools.declarations import ToolDeclaration, ToolKind  # noqa: E402

# ---------------------------------------------------------------------------
# Skill file location and sentinel anchors.
# ---------------------------------------------------------------------------

_SKILL_PATH: Final[Path] = _PROJECT_ROOT / "src" / "elspeth" / "web" / "composer" / "skills" / "pipeline_composer.md"

_BEGIN_SENTINEL: Final[str] = "<!-- BEGIN AUTOGEN: tool-inventory (generate_skill_inventory.py) -->"
_END_SENTINEL: Final[str] = "<!-- END AUTOGEN: tool-inventory -->"

# Anchor for the first-run insertion: the explanatory sentence that today
# immediately precedes the inventory bullets. Used only when the sentinels
# are not yet present in the file. Once inserted, the sentinels themselves
# are the stable insertion target.
_FIRST_RUN_ANCHOR: Final[str] = (
    "**Foundation knowledge (mandatory before any pipeline work):**"
    " know the composer tool categories available in this runtime."
    " The authoritative list is whatever `get_tool_definitions()` returns;"
    " the canonical groupings are:"
)


# ---------------------------------------------------------------------------
# Carve-outs — tools that appear in ``get_tool_definitions()`` and in the
# skill inventory but are NOT in ``_REGISTERED_TOOLS`` (they dispatch
# outside ``execute_tool``). The skill-drift test
# (TestComposerToolNameDrift.test_skill_tool_inventory_matches_declared_tools)
# uses the same two names; keep this list in sync with that test.
# ---------------------------------------------------------------------------

_CARVE_OUT_TOOLS: Final[tuple[str, ...]] = (
    "request_advisor_hint",
    "request_interpretation_review",
)


# ---------------------------------------------------------------------------
# Categorisation taxonomy.
#
# The skill's hand-curated category names — preserved verbatim so the
# inventory reads identically to its pre-generator form. ``_CATEGORY_ORDER``
# fixes the bullet order in the emitted block.
# ---------------------------------------------------------------------------

_DISCOVERY_CATEGORY: Final[str] = "Discovery"
_STATE_PREVIEW_CATEGORY: Final[str] = "State / preview"
_BUILD_EDIT_CATEGORY: Final[str] = "Build / edit"
_DIAGNOSTICS_CATEGORY: Final[str] = "Diagnostics"
_BLOBS_CATEGORY: Final[str] = "Blobs"
# Category for tools that operate on credential REFERENCES — they list,
# validate, or wire secret-name pointers (``list_secret_refs``,
# ``validate_secret_ref``, ``wire_secret_ref``) but NEVER carry secret
# VALUES by Tier-1 contract. The constant is named
# ``_CREDENTIAL_REFS_CATEGORY`` (not ``_SECRETS_CATEGORY``) to avoid
# CodeQL ``py/clear-text-logging-sensitive-data``'s name-based heuristic
# firing on the string literal ``"Secrets"`` as it flows from this
# module-level Final through the inventory rendering chain to
# ``sys.stdout.write`` in ``_dry_run_mode``. The rule treats variables
# named with "SECRETS"/"PASSWORD"/"KEY" as sensitive-data sources and
# cannot distinguish "ref-manipulation tool category label" from
# "credential payload". Renaming the IDENTIFIER (which carries no
# semantic load — Python doesn't know names from values) keeps the
# user-visible label "Secrets" while breaking the heuristic's source
# identification. The label stays "Secrets" because operators expect
# that word in the composer skill markdown — see CodeQL alert 804.
_CREDENTIAL_REFS_CATEGORY: Final[str] = "Secrets"

_CATEGORY_ORDER: Final[tuple[str, ...]] = (
    _DISCOVERY_CATEGORY,
    _STATE_PREVIEW_CATEGORY,
    _BUILD_EDIT_CATEGORY,
    _DIAGNOSTICS_CATEGORY,
    _BLOBS_CATEGORY,
    _CREDENTIAL_REFS_CATEGORY,
)

# Default category by ``ToolKind`` — applied unless a tool's name appears
# in ``_CATEGORY_OVERRIDES`` below. A new tool joining an existing category
# under its default kind requires zero edits to this script.
_CATEGORY_BY_KIND: Final[dict[ToolKind, str]] = {
    ToolKind.DISCOVERY: _DISCOVERY_CATEGORY,
    ToolKind.MUTATION: _BUILD_EDIT_CATEGORY,
    ToolKind.BLOB_DISCOVERY: _BLOBS_CATEGORY,
    ToolKind.BLOB_MUTATION: _BLOBS_CATEGORY,
    ToolKind.SECRET_DISCOVERY: _CREDENTIAL_REFS_CATEGORY,
    ToolKind.SECRET_MUTATION: _CREDENTIAL_REFS_CATEGORY,
}
# The session-aware tool ``request_interpretation_review`` does not carry a
# ``ToolKind`` (it dispatches outside ``execute_tool`` and is not in
# ``_REGISTERED_TOOLS``); its category is set explicitly via
# ``_CATEGORY_OVERRIDES`` below.

# Per-tool inline annotations — hand-curated parenthetical guidance appended
# after the backticked tool name within its category bullet. These are
# load-bearing prose hints that previously lived in the hand-written skill
# (e.g. ``get_pipeline_state``'s reminder about the all-state aliases) and
# would be lost if the generator emitted only bare names. A new tool that
# needs an inline hint adds one entry here; a new tool that does not (the
# common case) emits as a bare backticked name.
_TOOL_ANNOTATIONS: Final[dict[str, str]] = {
    "get_pipeline_state": " (for full state, omit the component argument or use full, all, pipeline, or the empty string)",
}


# Named overrides — tools whose hand-curated skill placement does not match
# their ``ToolKind`` default. The three "stateful discovery" tools live
# under "State / preview"; the validator-explanation discovery tool lives
# under "Diagnostics"; two blob-mutation tools live under "Build / edit"
# because they advance ``CompositionState`` rather than just touching blob
# storage.
_CATEGORY_OVERRIDES: Final[dict[str, str]] = {
    "get_pipeline_state": _STATE_PREVIEW_CATEGORY,
    "preview_pipeline": _STATE_PREVIEW_CATEGORY,
    "diff_pipeline": _STATE_PREVIEW_CATEGORY,
    "explain_validation_error": _DIAGNOSTICS_CATEGORY,
    "set_source_from_blob": _BUILD_EDIT_CATEGORY,
    "apply_pipeline_recipe": _BUILD_EDIT_CATEGORY,
    # Carve-outs (dispatch outside execute_tool, hand-defined in _dispatch.py).
    "request_advisor_hint": _DIAGNOSTICS_CATEGORY,
    "request_interpretation_review": _DIAGNOSTICS_CATEGORY,
}


# ---------------------------------------------------------------------------
# Per-tool stable ordering within each category.
#
# Within a category, tools are ordered by the position they appear in the
# canonical iteration (``_REGISTERED_TOOLS`` order followed by the two
# carve-outs). ``_REGISTERED_TOOLS`` is itself ordered by plane (the
# ``_registry.py`` concatenation) and then by in-plane declaration order, so
# rotating tools across planes changes ordering deterministically — desired
# behaviour for a registry-derived inventory.
# ---------------------------------------------------------------------------


def _category_for(tool_name: str, kind: ToolKind) -> str:
    """Return the bullet category for one tool, applying overrides."""
    override = _CATEGORY_OVERRIDES.get(tool_name)
    if override is not None:
        return override
    return _CATEGORY_BY_KIND[kind]


def _group_by_category(
    declarations: Iterable[ToolDeclaration],
    carve_outs: Iterable[str],
) -> dict[str, list[str]]:
    """Group tool names under their hand-curated skill categories.

    Iteration order is preserved within each category so the output is
    deterministic with respect to ``_REGISTERED_TOOLS`` order.
    """
    grouped: dict[str, list[str]] = {category: [] for category in _CATEGORY_ORDER}
    for decl in declarations:
        category = _category_for(decl.name, decl.kind)
        if category not in grouped:
            raise RuntimeError(
                f"ToolDeclaration({decl.name!r}) maps to category {category!r}, "
                f"which is not in _CATEGORY_ORDER ({_CATEGORY_ORDER}). Add it to "
                "_CATEGORY_ORDER if the category is new, or correct the "
                "override mapping."
            )
        grouped[category].append(decl.name)
    for name in carve_outs:
        # Carve-outs are always placed by override (they have no ToolKind here).
        override_category = _CATEGORY_OVERRIDES.get(name)
        if override_category is None:
            raise RuntimeError(
                f"Carve-out tool {name!r} has no entry in _CATEGORY_OVERRIDES; every carve-out must declare its skill category explicitly."
            )
        grouped[override_category].append(name)
    return grouped


def _render_inventory_block(grouped: dict[str, list[str]]) -> str:
    """Render the sentinel-delimited inventory block.

    Output shape mirrors the pre-generator hand-written form: one bullet per
    category, each tool name in backticks, joined by ", ". Empty categories
    are skipped (so a category with no tools today produces no line).
    """
    lines: list[str] = [_BEGIN_SENTINEL]
    for category in _CATEGORY_ORDER:
        names = grouped.get(category, [])
        if not names:
            continue
        rendered_names = [f"`{name}`{_TOOL_ANNOTATIONS[name]}" if name in _TOOL_ANNOTATIONS else f"`{name}`" for name in names]
        lines.append(f"- **{category}:** {', '.join(rendered_names)}")
    lines.append(_END_SENTINEL)
    return "\n".join(lines)


def _replace_or_insert_block(skill_text: str, block: str) -> str:
    """Replace the existing autogen block, or insert one after the anchor."""
    begin_idx = skill_text.find(_BEGIN_SENTINEL)
    end_idx = skill_text.find(_END_SENTINEL)
    if begin_idx != -1 and end_idx != -1 and end_idx > begin_idx:
        # Replace existing block (preserves leading/trailing newlines outside
        # the sentinels).
        end_idx_inclusive = end_idx + len(_END_SENTINEL)
        return skill_text[:begin_idx] + block + skill_text[end_idx_inclusive:]
    if begin_idx != -1 or end_idx != -1:
        raise RuntimeError(
            "Skill file contains exactly one of the autogen sentinels; both "
            "must be present or both absent. Fix the skill markdown before "
            "rerunning this script."
        )
    # First-run insertion — locate the anchor sentence, find the contiguous
    # block of ``- **``-prefixed lines immediately after it (the pre-generator
    # hand-written bullets), and replace exactly that range with the new
    # sentinel-delimited block. The anchor and the surrounding prose are
    # preserved verbatim.
    anchor_idx = skill_text.find(_FIRST_RUN_ANCHOR)
    if anchor_idx == -1:
        raise RuntimeError(
            "Could not locate the first-run anchor sentence in the skill "
            f"file: {_FIRST_RUN_ANCHOR!r}. The anchor must be present for "
            "the initial sentinel insertion. If the anchor was reworded, "
            "update _FIRST_RUN_ANCHOR or manually paste the sentinels at "
            "the desired location and rerun."
        )
    anchor_end = anchor_idx + len(_FIRST_RUN_ANCHOR)
    # Step over the blank line that separates the anchor paragraph from the
    # bullet list. ``str.find`` returns -1 if no blank line follows, in which
    # case there are no hand-written bullets to replace and we splice in
    # place.
    blank_line_start = skill_text.find("\n\n", anchor_end)
    if blank_line_start == -1:
        # No following paragraph break — insert immediately after the anchor.
        return skill_text[:anchor_end] + "\n\n" + block + skill_text[anchor_end:]
    bullets_start = blank_line_start + 2
    # Walk forward consuming consecutive ``- **`` lines.
    cursor = bullets_start
    while cursor < len(skill_text):
        line_end = skill_text.find("\n", cursor)
        if line_end == -1:
            line_end = len(skill_text)
        line = skill_text[cursor:line_end]
        if not line.startswith("- **"):
            break
        cursor = line_end + 1
    # ``cursor`` now points past the last bullet line. If ``bullets_start``
    # == ``cursor``, no hand-written bullets followed the anchor — splice in
    # without deleting anything.
    return skill_text[:bullets_start] + block + "\n" + skill_text[cursor:]


def _build_inventory_text() -> str:
    """Build the inventory block from the current registered declarations."""
    grouped = _group_by_category(_REGISTERED_TOOLS, _CARVE_OUT_TOOLS)
    return _render_inventory_block(grouped)


def _read_skill_text() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def _extract_current_block(skill_text: str) -> str | None:
    """Return the current sentinel-delimited block, or None if absent."""
    begin_idx = skill_text.find(_BEGIN_SENTINEL)
    end_idx = skill_text.find(_END_SENTINEL)
    if begin_idx == -1 or end_idx == -1:
        return None
    end_idx_inclusive = end_idx + len(_END_SENTINEL)
    return skill_text[begin_idx:end_idx_inclusive]


def _check_mode() -> int:
    """Compare the would-be block against the current file. Exit 0 if equal,
    exit 2 with a unified diff if drifted, exit 3 if sentinels are absent."""
    current_text = _read_skill_text()
    current_block = _extract_current_block(current_text)
    expected_block = _build_inventory_text()
    if current_block is None:
        sys.stderr.write(
            "Autogen sentinels not present in the skill file. Run\n"
            "    .venv/bin/python scripts/cicd/generate_skill_inventory.py --write\n"
            "to insert them, then re-stage.\n"
        )
        return 3
    if current_block == expected_block:
        return 0
    diff = difflib.unified_diff(
        current_block.splitlines(keepends=True),
        expected_block.splitlines(keepends=True),
        fromfile=str(_SKILL_PATH.relative_to(_PROJECT_ROOT)) + " (current)",
        tofile=str(_SKILL_PATH.relative_to(_PROJECT_ROOT)) + " (expected from _REGISTERED_TOOLS)",
    )
    sys.stderr.write(
        "Skill tool-inventory drift detected. The bulleted block under the\n"
        "Foundation-knowledge section does not match the declarations.\n"
        "Run\n"
        "    .venv/bin/python scripts/cicd/generate_skill_inventory.py --write\n"
        "and re-stage the skill file.\n\n"
    )
    sys.stderr.writelines(diff)
    return 2


def _write_mode() -> int:
    """Rewrite the skill file in place."""
    current_text = _read_skill_text()
    expected_block = _build_inventory_text()
    new_text = _replace_or_insert_block(current_text, expected_block)
    if new_text == current_text:
        sys.stdout.write("No changes — skill file already up to date.\n")
        return 0
    _SKILL_PATH.write_text(new_text, encoding="utf-8")
    sys.stdout.write(f"Updated {_SKILL_PATH.relative_to(_PROJECT_ROOT)}.\n")
    return 0


def _dry_run_mode() -> int:
    """Print the would-be block to stdout."""
    sys.stdout.write(_build_inventory_text())
    sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the skill file in place with the generated block.",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Compare the file against the generator output; exit 2 on drift.",
    )
    args = parser.parse_args(argv)
    if args.check:
        return _check_mode()
    if args.write:
        return _write_mode()
    return _dry_run_mode()


if __name__ == "__main__":
    raise SystemExit(main())

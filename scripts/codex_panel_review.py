#!/usr/bin/env python3
"""Pre-1.0 sandblasting SME review fleet (foundation: single-file, serial lenses).

See docs/superpowers/specs/2026-06-28-codex-panel-review-design.md.

Helpers from ``codex_audit_common`` are imported lazily at their first use-site
(the runner in :func:`run_file_lenses` and the CLI in :func:`main`) via a
dual-import shim, so this module runs both as a script (``scripts/`` on
``sys.path``) and as the package ``scripts.codex_panel_review`` under pytest.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LENSES_DIR = Path(__file__).resolve().parent / "codex_lenses"
PANEL_FINDING_SCHEMA = LENSES_DIR / "panel_finding.schema.json"
# Per-lens output goes in the *-raw sibling, NOT the counted `findings-panel/`
# tree. iter_report_files (codex_audit_common.py:792) only skips `by-priority/`
# + the metadata filenames — it does NOT skip an arbitrary subdir — so per-lens
# `.md` files written under the counted tree would be double-counted by Plan 2's
# generate_summary. Keeping raw detail in its own sibling from day one matches
# the spec's output layout and avoids seeding that double-count.
DEFAULT_OUTPUT_DIR = "docs/quality-audit/findings-panel-raw"

STRICT_CATEGORIES = frozenset({"bug", "correctness", "security", "smell"})
RELAXED_CATEGORIES = frozenset({"improvement", "efficiency"})
ANCHOR_REQUIRED_CATEGORIES = STRICT_CATEGORIES | frozenset({"easy-win"})


def build_layered_prompt(
    *,
    context: str,
    file_source: str,
    file_path: str,
    persona: str,
    lens: str,
) -> str:
    """Layer the prompt most-shared -> least-shared to ENABLE the codex prompt
    cache to key on [context][source] across every lens of a file (necessary, not
    sufficient — actual cross-process reuse is verified empirically in Task 9). The
    per-call tail (file_path, lens) is the only variable content and comes last."""
    return (
        f"{context}\n\n"
        "=== TARGET FILE SOURCE (inlined; you MAY read any other repo file via the "
        "read-only sandbox for investigation) ===\n"
        f"```\n{file_source}\n```\n\n"
        "=== REVIEW LENS ===\n"
        f"{persona}\n\n"
        "Output: a JSON object matching the provided schema — a non-empty "
        "`markdown_report` narration string and a `findings` array. Every finding "
        f'MUST set `lens` to "{lens}" and `priority` to one of P0|P1|P2|P3. For '
        "categories bug/correctness/security/smell/easy-win, `evidence` MUST cite a "
        "real path and integer line. The schema is STRICT: every finding MUST "
        "include ALL fields — for any field you have no value for (confidence, "
        "effort, impact, suggested_fix, target_file, gate_note, or an evidence "
        "`line`), emit JSON `null`; do NOT omit the key. If you find nothing, "
        "return an empty findings array and say so in markdown_report.\n"
        f"--- review target: {file_path} · lens: {lens} ---\n"
    )


# Plan 1 roster (file-predicate pairs). Full routing table is Plan 2.
_EVERY_FILE = ("solution-architect", "security-architect")


def load_persona(lens: str, *, lenses_dir: Path = LENSES_DIR) -> str:
    path = lenses_dir / f"{lens}.md"
    if not path.exists():
        raise FileNotFoundError(f"persona prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def route_lenses(file_path: Path, *, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    lenses = list(_EVERY_FILE)
    if file_path.suffix == ".py" and (LENSES_DIR / "python-engineer.md").exists():
        lenses.append("python-engineer")
    return lenses

#!/usr/bin/env python3
"""Pre-1.0 sandblasting SME review fleet (foundation: single-file, serial lenses).

See docs/superpowers/specs/2026-06-28-codex-panel-review-design.md.

Helpers from ``codex_audit_common`` are imported lazily at their first use-site
(the runner in :func:`run_file_lenses` and the CLI in :func:`main`) via a
dual-import shim, so this module runs both as a script (``scripts/`` on
``sys.path``) and as the package ``scripts.codex_panel_review`` under pytest.
"""

from __future__ import annotations

import json
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


def _has_line_anchor(finding: dict) -> bool:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return False
    return any(isinstance(e, dict) and isinstance(e.get("line"), int) and bool(e.get("path")) for e in evidence)


def _has_impact(finding: dict) -> bool:
    # `impact` is nullable in the strict schema, so finding.get("impact") may be
    # None. Coerce with `or ""` BEFORE strip() — str(None) == "None" reads truthy,
    # which would silently defeat the relaxed/fail-closed downgrade.
    return bool(str(finding.get("impact") or "").strip())


def apply_panel_evidence_gate(sidecar_path: Path, *, lens: str) -> int:
    """Category-aware, fail-CLOSED structured gate. Downgrades anchor-required
    findings lacking a path+line, and any other priority-bearing finding lacking
    an `impact` rationale (relaxed AND uncovered categories like `design` or a
    typo'd/future enum value — nothing rides through ungated). Stamps each
    finding's `lens` from the known-correct caller value (never trusts the model's
    self-reported lens). Always rewrites the sidecar so its mtime exceeds the
    per-lens .md (defeats the staleness guard at codex_audit_common.py:820).
    Returns the downgrade count."""
    raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"panel sidecar must be a JSON object: {sidecar_path}")
    findings = raw.get("findings")
    if not isinstance(findings, list):
        raise RuntimeError(f"panel sidecar missing findings array: {sidecar_path}")

    downgraded = 0
    for finding in findings:
        if not isinstance(finding, dict):
            raise RuntimeError(f"panel finding must be an object: {sidecar_path}")
        finding["lens"] = lens  # deterministic stamp; do not trust the model's value
        category = finding.get("category")
        if category in ANCHOR_REQUIRED_CATEGORIES and not _has_line_anchor(finding):
            finding["priority"] = "P3"
            finding["confidence"] = "low"
            finding["gate_note"] = "needs verification: no file:line anchor"
            downgraded += 1
        elif category in RELAXED_CATEGORIES and not _has_impact(finding):
            finding["priority"] = "P3"
            finding["gate_note"] = "needs rationale: empty impact"
            downgraded += 1
        elif category not in ANCHOR_REQUIRED_CATEGORIES and category not in RELAXED_CATEGORIES and not _has_impact(finding):
            # Fail-closed: `design` and any unknown/typo/future category need at
            # least an `impact` rationale, or they are downgraded — no silent
            # pass-through of an unsubstantiated high-priority claim.
            finding["priority"] = "P3"
            finding["gate_note"] = f"uncovered category {category!r}: needs impact rationale"
            downgraded += 1

    # Unconditional rewrite (last) — see docstring.
    sidecar_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return downgraded

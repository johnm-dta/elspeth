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

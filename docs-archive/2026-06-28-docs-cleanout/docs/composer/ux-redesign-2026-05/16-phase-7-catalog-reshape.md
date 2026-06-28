# Phase 7 — Catalog Reshape (Reference, Not Toolkit)

**Status:** 2026-05-15. Companion to [08-catalog-reshape.md](08-catalog-reshape.md)
(the design spec) and [00-implementation-roadmap.md](00-implementation-roadmap.md)
(the phase map).

This phase is **split into three plans** because the surface-area diff is
large enough that a single plan would exceed the 1500-line budget:

- **[16a-phase-7a-backend.md](16a-phase-7a-backend.md)** — Plugin metadata
  schema extension (class-attribute fields on `BaseSource` / `BaseTransform` /
  `BaseSink`), catalog API surface extension (new optional fields on
  `PluginSummary`), one canonical filled-in example
  (`csv_source.py`) so future authors have a pattern to copy. **No prose
  authoring for other plugins** — that's a per-plugin documentation task that
  proceeds incrementally outside this phase.
- **[16b-phase-7b-frontend.md](16b-phase-7b-frontend.md)** — Frontend
  primitives: extended `PluginSummary` types, the centralised
  `auditCharacteristics` metadata table, `AuditCharacteristicIcon`,
  and the `PluginCard.tsx` rewrite (removing the "Use in pipeline"
  toolkit affordance).
- **[16c-phase-7c-frontend-integration.md](16c-phase-7c-frontend-integration.md)**
  — Drawer integration: `FilterChipStrip` (capability +
  audit-characteristic filter chips), synthetic `InlineChatSourceEntry`
  at the top of the Sources tab, `CatalogDrawer.tsx` wire-up with
  extended `scorePlugin` across prose + tags, and the `ShortcutsHelp`
  regroup under "Actions" / "Reference."

This file is the **index**. The actual TDD-shaped task lists live in 16a,
16b, and 16c.

Mergeability:

- 16a is a pure backend schema extension that adds optional fields and one
  filled-in example; the existing frontend safely ignores them.
- 16b is the frontend primitives and `PluginCard` rewrite. Cards "come
  alive" with reference content once 16a ships; before that, every
  plugin's card renders the "see the technical description" fallback.
- 16c is the drawer integration. It imports from 16b (filter chips use
  `lookupAuditCharacteristic`; the synthetic entry uses 16b's
  `PREFILL_CHAT_INPUT_EVENT` re-export). 16c **must not ship before 16b**.

Recommended ship order: 16a → 16b → 16c. 16a + 16b can run in parallel
(different surfaces); 16c depends on 16b.

## Implementation worktree (batched)

All four plans (16, 16a, 16b, 16c) ship as a single batched implementation
from one worktree. Create it once at the start of the batch:

```bash
git -C /home/john/elspeth worktree add .worktrees/phase-7-catalog -b feat/phase-7-catalog
cd /home/john/elspeth/.worktrees/phase-7-catalog
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cd src/elspeth/web/frontend && npm install
```

**Execution order inside the worktree:** 16a (backend) → 16b (frontend
primitives) → 16c (drawer integration). 16c's Task 1 Step 0 preflight
hard-aborts if 16b primitives are absent; 16b's vitest tests against the
extended `PluginSummary` will fail if 16a's API surface isn't shipped
first. 16a and 16b touch different surfaces (Python backend vs. React
frontend) so once 16a is on a green commit, 16b can proceed in parallel
with the 16a regression sweep; 16c must follow 16b.

**Worktree discipline (operator-known gotchas):**

- **Activate `.venv` before any `uv pip install`** — installing from the
  worktree without `--python` finds main's `.venv` and clobbers it.
- **Match Python version with main (3.13)** — `scripts/cicd/enforce_tier_model.py`
  reports ~300 spurious violations under version drift.
- **Subagents inherit parent CWD** — if any task delegates to a subagent,
  prefix the prompt with absolute paths and CWD discipline; subagents
  silently misread when the worktree path is stated only in prose.
  Preferred alternative: run subagents from the main checkout against
  files in the worktree by absolute path.
- **`filigree` CLI rejects realpath-escaping DBs from worktrees** — prefer
  MCP tools (`mcp__filigree__*`); for CLI fall back to
  `(cd "$(git rev-parse --git-common-dir)/.." && filigree …)`.

**On completion of all four plans:** single PR from
`feat/phase-7-catalog` → `RC5.2` covering 7A + 7B + 7C. The PR
description should pull the rev-history bullets from each plan as the
change log. Single PR review, single merge, single revert lever if any
post-merge issue surfaces.

---

## Phase 7 in the roadmap

Per [00-implementation-roadmap.md §C](00-implementation-roadmap.md), Phase 7
is **independent** of Phases 4–6 and can ship **any time after Phase 2**.
This independence is structural: the catalog is reference; it has no
data-flow dependency on Phases 3 (IA cleanup), 4 (tutorial), 5a/5b (chat as
data entry / interpretation surface), or 6 (completion gestures).

The only upstream coupling is the architectural pattern itself — Phase 7
follows the same backend/frontend split as Phase 1 (plans 12 / 13). When
reading 16a or 16b, the sibling-pattern reference is plan 12 and plan 13;
the design reference is [08-catalog-reshape.md](08-catalog-reshape.md).

## Trust-tier scope (rescinded 2026-05-18)

The original Phase 7A scope included a per-plugin `data_trust_tier`
field rendered as a tier badge on each card. **That scope was
rescinded on 2026-05-18** after operator review concluded the field
failed the "each tag must represent a meaningful per-plugin decision"
test: every Source surfaces Tier 3 data because the kind requires it
(not a per-plugin choice); every external-call Transform is Tier 3
because the author's `Determinism.EXTERNAL_CALL` declaration already
says so; pure Transforms are Tier 2 because their kind requires it.

Boundary classification for the audit-readiness panel is derived from
`(kind, determinism)` directly — see `_build_plugin_trust_row` in
`src/elspeth/web/audit_readiness/service.py`. The catalog drawer
surfaces author-declared `audit_characteristics` plus determinism-
derived flags that **vary across plugins of the same kind**; kind-
default determinism does not emit a flag.

## What 7a is *not* doing

- 7a does **not** write the "when you'd use this" / "when you wouldn't" /
  "example use" prose for every plugin. That is a per-plugin documentation
  task; the design doc 08-§Risks calls it out as an incremental author
  responsibility. 7a ships the **schema** for the prose plus **one** filled
  example (`csv_source.py`) as a pattern to copy.
- 7a does **not** generate prose via LLM. Open question D3 (per
  [00-implementation-roadmap.md §A](00-implementation-roadmap.md)) admits
  (c) LLM-drafted-to-bootstrap as a possible future path, but the plan
  explicitly excludes it — the schema must exist *first* and authors must
  see a hand-written canonical example before any LLM-bootstrap pass
  proceeds, because the canonical example sets the voice and tone.
- 7a does **not** infer prose from docstrings. Existing plugin docstrings
  are technical descriptions; the new "when you'd use this" framing is
  intentionally different (persona-facing, not implementation-facing).
  Reusing docstrings would produce technical prose with a "when you'd use
  this" header, which is worse than leaving the field `None` and letting
  the frontend fall back to "see the technical description."
- 7a does **not** create a backend "Inline data from chat" plugin. Per
  reconnaissance (see scope-boundaries in 16b), there is no such plugin
  today and design doc 08 frames it as "explicitly an option, not a
  plugin." 16b adds it as a synthetic frontend-only catalog entry.

## What 7b / 7c are *not* doing

- 7b/7c do **not** ship per-plugin prose. If 7a's canonical example is
  the only plugin with prose by the time the frontend ships, the new
  card layout will show "see the technical description" fallbacks for
  every other plugin — that's the expected staging behaviour. Per
  design doc 08-§Risks: "Empty entries fall back to a generic 'see the
  technical description' message rather than blocking display."
- 7b/7c do **not** add "Add to pipeline" / "Select" buttons. Per design
  doc 08, these are the load-bearing removal; the catalog stops being a
  toolkit.
- 7b/7c do **not** add "in use" highlighting, drag handles, "recently
  used," or "favorites." Per design doc 08, these would imply a
  workflow link back to composition and reintroduce the toolkit framing.
- 7b/7c do **not** add schema-field-name search (a stretch goal from
  design doc 08-§Search). Schema is lazy-fetched; extending fuzzy
  search across schema field names would require a preload pass and
  conflicts with the existing lazy-load design. Flagged in 16b's "Out
  of scope" section.

## Independence from Phase 1

Phase 1 (plans 12 / 13) added the `user_preferences` table; Phase 7 does
not touch it. There is no preference for "show / hide audit-characteristic
icons" — they're always shown, because the design's whole point is that
audit characteristics are not a power-user toggle, they're the central
reference content. If a future phase wants to add a "show technical
details" preference, that's a Phase 8 polish concern, not a Phase 7
concern.

## Memory references

- `feedback_catalog_is_reference_not_toolkit` — the design call this implements.
- `project_composer_personas` — each persona's catalog use case (08-§"Plugin
  card content design" maps the four persona reads).
- `project_composer_dynamic_source_from_chat` — the inline-data entry that
  16b adds as a synthetic frontend-only catalog row.
- `project_db_migration_policy` — informs 16a's no-Alembic posture (no DB
  migration is needed; the schema extension is class-attribute level on
  Python plugin code).

## Sequencing recommendation

The recommended sequence is **16a → 16b → 16c**:

- **16a first** (small, surgical schema extension + one canonical
  example). Lets 16b's TDD steps assert against real backend responses
  in the integration-test layer.
- **16b next** (frontend primitives + `PluginCard` rewrite). At this
  point each card renders the new reference layout; the CSV card shows
  authored prose, others show the "see the technical description"
  fallback.
- **16c last** (drawer integration, filter chips, synthetic entry,
  shortcuts regroup). 16c imports from 16b's primitives and must not
  ship before 16b.

16a and 16b can run in parallel — different surfaces, no shared files.
16c depends on 16b's primitives existing. If 16b and 16c ship as a
stacked PR, the sequence between them is 16b merge → 16c rebase →
16c merge.

If 16b/16c ship before 16a, the frontend renders all-fallback content
for every plugin, then "comes alive" for the canonical CSV example when
16a merges. That order is acceptable but provides less verification
value.

---

## Review history

- 2026-05-18 Worktree batch protocol added: All four Phase 7 plans (16, 16a, 16b, 16c) now share a single worktree at `.worktrees/phase-7-catalog` on branch `feat/phase-7-catalog`, with a single PR to RC5.2 covering 7A + 7B + 7C. Added an `## Implementation worktree (batched)` section with worktree-creation commands, execution order (16a → 16b → 16c with parallelism notes), and operator-known gotchas (venv leak, Python 3.13 version match, subagent CWD discipline, filigree CLI realpath workaround). Sibling plans 16a/16b/16c each got a customized variant of the section that links back here for the canonical batch protocol.

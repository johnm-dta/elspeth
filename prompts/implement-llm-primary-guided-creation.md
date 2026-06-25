# Implementation Handover — LLM-primary guided pipeline creation

You are the implementation controller for a reviewed, ready-to-build plan series in
the **ELSPETH** repo. Your job is to execute it task-by-task to green, on the
existing branch, without crossing the operator-owned boundaries below.

## What you are building

The guided/tutorial composer is being reworked so the **LLM transform (plain
English → pipeline config) is the PRIMARY way to build every phase**, not an
opt-in helper. The structured form becomes the editable *result* of an LLM
proposal; users switch freely between LLM-driven and manual on the same panel; the
tutorial becomes a passive worked example over synthetic scrape pages. Full intent
and rationale: the design spec (below). You do **not** need to re-derive the
design — it is settled. Build what the plans specify.

## Read these first (the source of truth — in this order)

1. `docs/superpowers/plans/2026-06-25-llm-primary-guided-creation/00-overview.md`
   — the series map: the four plans, the **dependency/execution order**, the
   shared contract summary, the Global Constraints, and execution guidance.
2. The design spec: `docs/superpowers/specs/2026-06-25-llm-primary-guided-creation-design.md`
   (authority for *what* and *why*; folds in the synthetic-scrape spec
   `2026-06-25-tutorial-synthetic-scrape-page-design.md`).
3. The four plan files (your task lists), in dependency order:
   - `…/p1-backend-drivers.md` — per-phase LLM drivers + the `/guided/chat` apply contract (load-bearing)
   - `…/p3-prompt-shield-3state.md` — global always-on 3-state prompt-shield review
   - `…/p2-frontend-intent-surface.md` — intent-primary ChatPanel + remove the tutorial freeform dead-end
   - `…/p4-tutorial-synthetic-scrape.md` — the synthetic-scrape passive worked example

These plans were authored from a decided cross-plan contract, then **per-task
deep-reviewed** (reality + cross-plan + domain) and **technical-writer reviewed**
(all four APPROVE); findings are already applied. Treat the plans as correct, but
verify anchors live (next section).

## How to execute

- **REQUIRED SKILL:** use `superpowers:subagent-driven-development` — a fresh
  implementer subagent per task, a task review (spec compliance + code quality)
  after each, fix loop, then a broad whole-branch review at the end. Track
  progress in the durable ledger (`.superpowers/sdd/progress.md`) so a compaction
  can't lose your place.
- **Execution order (from the overview):** p1 and p3 are independent and can start
  first; **p2 consumes p1's apply contract**, so do p2 after p1's contract lands;
  **p4 depends on p1 + p2 + p3**, so do p4 last. Within a plan, follow the task
  order.
- **Concurrency / rate limits — IMPORTANT:** this environment rate-limits wide
  fan-out. Keep concurrent subagents **low (≈2 at a time)**; do not blast many
  agents at once. If you orchestrate with a workflow, throttle it; prefer
  small verified batches over large parallel waves.
- **Anchors may have drifted** since authoring. The plans pin most symbols with a
  "confirm by grep" note — honor it: re-grep the symbol/line before editing rather
  than trusting a literal `:NNN`. Loomweave MCP (`entity_find`/`entity_at`) is the
  fast way to resolve a symbol.

## The load-bearing contract (do NOT violate)

- **Apply-in-place via the same `handle_step_*` commit seam the manual form uses.**
  A `POST /guided/chat` submit *attempts* to drive the current phase; it mutates
  **only** when it produced an actionable config; non-actionable input (a question,
  ambiguity, LLM failure/timeout/malformed) falls back to **advisory prose with NO
  mutation**. The step pointer stays unchanged (no auto-advance — p1 removes the
  existing STEP_1 auto-advance). Preserve the unknown-step **400**, terminal
  **409**, and step-mismatch **409** guards.
- **No schema-epoch bump.** `SESSION_SCHEMA_EPOCH` (=24) and
  `GUIDED_SESSION_SCHEMA_VERSION` (=6) do **not** change in this series (apply-in-place
  writes only already-serialized `GuidedSession` fields). A bump would force a
  delete-the-DB migration and fail-close the live boot — if a task seems to require
  one, **stop and ask**.
- **Advisory-never-blocks.** Prompt-shield and interpretation reviews are advisory;
  they must never hard-block advancement. Reuse the `pipeline_decision` kind +
  `user_term=prompt_injection_shield_recommendation`; do **not** add a new
  `InterpretationKind`.

## Global constraints (operational — the plans' Global Constraints sections are authoritative; this is the summary)

- **Branch:** work lands on **`release/0.7.0`** (the current branch). Do **not**
  create a feature branch, and do **not** start on `main`.
- **Agent signs nothing; operator pushes.** You do not hold the HMAC signing key
  and you do not push. The tier-model and plugin-hash gates are **operator-owed**:
  surface what a re-sign would need, never sign it. The tier-model rotation tool is
  `python -m elspeth_lints.core.cli rotate` (the old `scripts/cicd/rotate_tier_model_fingerprints.py`
  was removed); the plugin `source_file_hash` gate is refreshed via
  `scripts/cicd/plugin_hash.py`'s library functions. RUN these to *report* the owed
  set; do not commit signatures.
- **Commit discipline:** stage **explicit paths only** — never `git add -A` / `git
  add .`. For code commits use
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit` (keeps the
  secret-scan and other protective hooks running); never a blanket `--no-verify`.
  Commit frequently (per task). Never commit conflict markers.
- **Existing tests that assert about-to-change behavior must be UPDATED, not
  reverted.** A wave of failures after a structural change is the change landing
  visibly. The plans already list these tests per task.
- **Coupled tutorial constants:** editing `CANONICAL_TUTORIAL_PROMPT` couples the
  backend constant (`web/preferences/tutorial_cache.py`) + a byte-identical frontend
  mirror + `composer_skill_hash` + a live-prompt restart. Change them in lockstep.
- **`entry_seed` is server-side only** — never put it on the wire.
- **SSRF:** the tutorial `web_scrape.allowed_hosts` must stay tight (`public_only`
  for a public base, a loopback CIDR for dev — never `allow_private`).

## Verification (recorded-baseline discipline)

This repo carries **operator-owed pre-existing red gates** (tier-model HMAC,
freeze_guards, a `cli.py` mypy item, etc.). Do **not** treat those as your
regressions: record the baseline reds before you start, and attribute any *new*
failure by diff against that baseline. Run the `ci.yaml` Static-analysis set
**locally before declaring done** (CI cycles are slow). For tests, use the default
selection — plain `pytest tests/` (do **not** `pytest -o addopts=""`, which
force-runs slow/stress suites and produces phantom failures). Run `wardline scan .
--fail-on ERROR` on any code that touches external input (the `web_scrape` fetched
content boundary; the LLM-driven apply path consuming Tier-3 free text).

## Known decisions already baked into the plans (so you don't second-guess them)

- **Revise mode is a distinct path** (resolved fork): p1 adds `build_revise_addendum`
  + a `revise_context` param on `solve_chain`; a revise instruction is framed as a
  change request, **not** as "your proposal failed validation." Keep `repair_context`
  for genuine validation-repair. Implement both; don't collapse them.
- **The sink driver returns `tuple[SinkResolved, str] | None`** (the `str` carries
  the assistant message, for STEP_1/STEP_2 parity) — this intentionally supersedes
  the contract's `SinkResolved | None`. It is intra-p1; no other plan consumes it.
- **The tutorial `allowed_hosts` seam is `handle_step_2_5_recipe_apply`**
  (`src/elspeth/web/composer/guided/steps.py`, ~line 219 — confirm by grep), NOT a
  `_helpers.py`/`edited_values` path (that does not exist).
- The MVP single-output sink cap and similar validation belong at the **parse
  boundary** (Tier-3 LLM-originated input), server-side, not only in the tool JSON
  schema.

## Stop and ask the operator (do not guess, do not self-authorize)

- A plan task contradicts another task, the contract, or the spec — present both
  and ask which governs.
- Any **operator-owed** action: signing an HMAC-gated gate, pushing, deleting a
  DB, restarting a service, or anything irreversible/outward-facing.
- A task appears to require a schema-epoch bump or a new external-input boundary
  (new tier-model allowlist entry) — surface it; that's a heavier gate.
- A genuine blocker you cannot resolve from the plan + codebase.

## When the series is complete

Run the broad whole-branch code review (most-capable model), fix its
Critical/Important findings, then use `superpowers:finishing-a-development-branch`.
Leave the merge and push to the operator (release/X.Y → PR → main; no direct-to-main).

---
*Series committed on `release/0.7.0` (unpushed): spec `7d7bfaffd` → plans through
`55cf0398f`. This handover is a pointer; the plan files are the source of truth —
read them.*

# Tutorial Staged Recut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-pass "big-bang" first-run tutorial (one canonical
sentence → one inference wires source + transforms + sink) with a **staged
wizard** that walks the learner through source → sink → transform → wire. The
tutorial becomes a specific *instance* of the existing `composer/guided/` staged
state machine, parameterised by a server-owned `WorkflowProfile`; live guided mode
is the same engine with the empty (canonical-default) profile. This fixes the four
named failure modes (fragile below top-tier models; doesn't teach source→transform→sink→wiring;
magic-box opacity; high single-inference latency) by decomposing both the rule/prompt
blocks (per-stage skill files) and the flow/UX (a staged stepper that shows the logic),
and by adding a **web_scrape recipe** so the canonical pipeline composes deterministically
(zero LLM calls at compose time). Target version **0.7.0**, pre-release, no feature flag,
no backward-compat.

**Architecture:** One engine, two profiles. `composer/guided/` (today's audited
source → sink → recipe-match → transforms wizard) becomes *the* staged workflow engine.
It gains a **fifth, global stage** `STEP_4_WIRE` (appended after transforms — the stage SET
is a global engine property, the frozen-total `GuidedStep` enum, **not** a `WorkflowProfile`
field, D14). The `WorkflowProfile` (internal plumbing; persisted on `GuidedSession` in the
`composition_states.composer_meta` JSON column) gates *behaviour* — entry seed, per-stage
coaching copy, advisor checkpoints, recipe-match, welcome/graduation bookends — but never the
stage set. The tutorial constructs the one concrete `TUTORIAL_PROFILE` at the new
session-scoped `POST /api/sessions/{session_id}/guided/start` entry endpoint; the
empty profile is a canonical value (not `None`) that
reproduces live-guided behaviour modulo the new wire stage. A new **web_scrape `RecipeSpec`**
(+ predicate keyed on the materialized URL-row source: `json`/`csv` with
`source.options["blob_ref"]`; `inline_blob` is only the authoring alias, and
`web_scrape` is a transform) lets the
canonical `inline_blob → web_scrape → llm_rate → field_mapper → jsonl` pipeline compose with
zero LLM calls and reach the wire stage via the deterministic recipe-apply path. Per-stage
interpretation review reuses the freeform `interpretation_events` store/UI (no new backend
guided `TurnType`, D12). The advisor END sign-off is a **pre-terminal gate inside the
`STEP_4_WIRE` branch**, gated on the server-owned `profile.advisor_checkpoints` (D13). The
session DB is **purged** (not migrated) via a `GUIDED_SESSION_SCHEMA_VERSION` 5→6 +
`SESSION_SCHEMA_EPOCH` 23→24 bump (D15; the epoch was already advanced to 23 by a prior fix — see the corrected Schema version constants below).

**Tech Stack:** Python 3.12 (FastAPI web service under `src/elspeth/web/`, SQLite session
DB, Pydantic strict response models, frozen-dataclass domain state, LiteLLM advisor path);
TypeScript/React frontend under `src/elspeth/web/frontend/` (Vitest unit, Playwright E2E);
elspeth-lints trust-tier static analysis + wardline taint gate; pytest + ruff + mypy.

## Global Constraints

- Target version 0.7.0 (pre-release); **no feature flag**, **no backward-compat** — in-place migration (D10), remove the big-bang components.
- **Verify ALL code against the current checkout `/home/john/elspeth/src/elspeth` (branch `release/0.7.0`).** ⚠️ This REVERSES the plan's original instruction. The plan was authored against the worktree `/home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth`, but the composer-routes decomposition has since **merged into `release/0.7.0`**, making the worktree the STALE pre-refactor tree. Execute against main `src/`; use the worktree only as a `git diff` reference to map old→new symbol locations.
- **Codebase note — composer-routes decomposition (already merged, `0e754a67e`):** the monolith `sessions/routes/composer.py` was split into the package `sessions/routes/composer/` (`__init__.py`, `compose.py`, `guided.py`, `proposals.py`, `state.py`). The split was **PARTIAL**: the HTTP route handlers + the GET-rebuild (`_build_get_guided_turn`) moved into `routes/composer/guided.py`; the dispatcher logic (`_dispatch_guided_respond`, the accept-commit seams) STAYED in `sessions/routes/_helpers.py`, and the completion seams stayed in `composer/guided/steps.py`. Inside the package, relative imports are `from .._helpers import …` (two dots). Any task that edited `routes/composer.py` now targets `routes/composer/guided.py`.
- **Line numbers throughout this plan are ADVISORY** — the source drifted after authoring (uniform per-file shifts). Navigate by grepping the named symbol, never by the cited line.
- **Route names:** the canonical guided-start API path is
  `POST /api/sessions/{session_id}/guided/start`. If a task uses shorthand
  `guided/start`, it is prose shorthand only; mocks, client helpers, and tests
  must target the full session-scoped API path and must never post to bare
  `/guided/start`.
- **Composer eval harness reality:** `evals/composer-harness/` is the live harness.
  `evals/2026-05-03-composer/` is frozen historical evidence; do not point new
  harness work or assertions at the frozen tree.
- **Tutorial prompt duplication:** the canonical tutorial prompt text is duplicated
  across backend cache (`preferences/tutorial_cache.py`), frontend tutorial strings
  (`components/tutorial/tutorialMachine.ts` / `copy.ts`), and tests. Any prompt edit
  must update all copies and the drift tests in the same slice.
- The canonical pipeline is exactly `inline_blob → web_scrape → llm_rate → field_mapper(select_only, raw-HTML cleanup) → jsonl` at the authoring level; once materialized, the recipe predicate keys the URL-row `json`/`csv` source with `source.options["blob_ref"]`, never `web_scrape` and never the bare `inline_blob` alias.
- **Shield advisory stays LIVE:** the web_scrape recipe omits the *unbuildable* `azure_prompt_shield` hard node, but the existing medium-severity prompt-shield advisory warning (`prompt_shield_recommendation_warning_pairs`) MUST remain present in the wire validation payload. Tests pin the advisory's **presence** (+ absence of the hard node), not its absence; comment-reference `elspeth-abb2cb0931`. Do not let the flagship example hide the signal.
- The advisor terminal END sign-off is **profile-gated** on the server-owned `profile.advisor_checkpoints` (closed-enum, server-constructed — a client cannot flip it). The empty/live-guided profile gets the wire stage but no mandatory terminal advisor call (so D14's global wire stage is not a blocking-advisor regression for live guided).
- The stage SET is a **global** engine property (frozen-total `GuidedStep` enum), NOT a `WorkflowProfile` field (D14). A future profile must add NO new state-machine dispatch branch (the additivity acceptance test).
- Gate commands (§9.2) — run ALL before claiming completion:
  - `uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/`
  - `uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/`
  - `uv run mypy src/ elspeth-lints/src/`
  - `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth`
  - `uv run python scripts/cicd/check_slot_type_cross_language.py` (SlotType / guided.ts mirror gate — this work edits `guided.ts`)
  - targeted `pytest` over the new guided/tutorial/recipe test files
  - frontend (run from `src/elspeth/web/frontend`): `npm run typecheck`, `npm test -- --run`, `npm run build`, `npm run test:e2e` (+ `test:e2e:staging`)
  - `wardline scan . --fail-on ERROR` (exit 0 clean / 1 gate tripped / 2 tool error) — B1/B3/recipe touch externally-fed advisor/user-text trust boundaries; fix at the boundary, not the sink. See `.agents/skills/wardline-gate/SKILL.md`.
- elspeth-lints trust-tier + trust_boundary honesty gate apply: B1/B3 add/move Tier-3 advisor/user-text handling; new `@trust_boundary` decorations must pass `trust_boundary.tests,scope,tier`. Operator-triggered advisor escapes must NOT route through Tier-3 `_validate_advisor_arguments`; backend checkpoints must not consume unvalidated user text.

---

## File Structure

Each phase file owns its executable file list. Use this overview as the shared
map only:

- Backend guided engine: `src/elspeth/web/composer/guided/{protocol.py,state_machine.py,steps.py,emitters.py,prompts.py,profile.py,skills/}`.
- Session route layer: `src/elspeth/web/sessions/routes/_helpers.py` for dispatcher
  and accept seams; `src/elspeth/web/sessions/routes/composer/guided.py` for
  HTTP guided routes and GET rebuilds.
- Frontend guided surface: `src/elspeth/web/frontend/src/types/guided.ts`,
  `components/chat/ChatPanel.tsx`, and `components/chat/guided/*`.
- Tutorial/cache surface: `src/elspeth/web/preferences/tutorial_cache.py`,
  `src/elspeth/web/frontend/src/components/tutorial/*`, and
  `tests/{unit,integration}/web/*tutorial*`.
- Verification/evidence: live harness under `evals/composer-harness/`; frozen
  evidence under `evals/2026-05-03-composer/`.

## Shared Interfaces (canonical names — use verbatim)

The following names are the cross-phase contract. Phase files may add narrower
details, but they must not rename these symbols or change their value domains.

### New `TurnType` member
- `TurnType.CONFIRM_WIRING = "confirm_wiring"` — the single new turn type for the wire stage (also carries the advisor-revise re-emit; an attached `advisor_findings` payload key distinguishes the revise re-emit from the initial confirm). Register it in `_LEGAL_TURN_MATRIX[STEP_4_WIRE]`, `_REQUIRED_KEYS` (both TOTAL over `TurnType` — omission crashes at import), and `_NESTED_SHAPES`. Mirror as `"confirm_wiring"` in the `guided.ts` `TurnType` union.

### New `GuidedStep` member (global, appended)
- `GuidedStep.STEP_4_WIRE = "step_4_wire"` — appended LAST (append-only; mid-insert renumbers the wire protocol and is forbidden). Add to: `_LEGAL_TURN_MATRIX`, `prompts._STEP_FILE_NAMES` + `_STEP_PLAYBOOK_ORDER` (+ create `skills/step_4_wire.md` — wiring CONSTRAINTS only, no UX copy), the TWO duplicated `_ORDER` tuples (`emitters.py` and `sessions/routes/_helpers.py`), the `guided.ts` `GuidedStep` union, and the `step_advance` branch dispatch.

### `run_signoff_checkpoint` (new public protocol method)
```python
async def run_signoff_checkpoint(
    self,
    *,
    state: CompositionState,
    session_id: str | None,
    recorder: BufferingRecorder | None,
    progress: ComposerProgressSink | None = None,
) -> AdvisorCheckpointVerdict: ...
```
Add to the `ComposerService` Protocol; `ComposerServiceImpl` delegates to the existing private `_run_advisor_checkpoint(phase="end", ...)`. `AdvisorCheckpointVerdict` gains a `failure_class: Literal["none","unavailable","malformed"] = "none"` field (added in P5.3): the existing `_run_advisor_checkpoint` collapses EVERY exception — timeout/auth/transport AND malformed/parse — to `ok=False` (service.py:4210-4230), so `(ok, blocking)` ALONE cannot separate a malformed response (must fail-closed) from a transport outage (may take the audited escape). Verdict classes for D13: CLEAN = `ok and not blocking`; FLAGGED = `ok and blocking` (fail-closed); MALFORMED = `not ok and failure_class=="malformed"` (**fail-closed, no escape**); UNAVAILABLE = `not ok and failure_class=="unavailable"` (escape only at budget-exhaustion).

### Schema version constants
- `GUIDED_SESSION_SCHEMA_VERSION` (`composer/guided/state_machine.py:41`): **5 → 6**.
- `SESSION_SCHEMA_EPOCH` (`sessions/models.py:120`): **23 → 24** (boot fail-close via `_assert_schema_sentinels`). The prior bug-fix already advanced the epoch to **23** (committed value in main), so P0.6 targets 23→24. The epoch-23 work (constant + the two pinning-test assertions) is now COMMITTED at `5e46c226c` (operator-confirmed 2026-06-23) — 23 is the settled prior epoch.

---


## Phases (execute P0 → P7 in order)

This plan was split from a single 10,841-line file into one file per phase.
Task numbering (`PX.Y`) is preserved verbatim, so every cross-reference in the
prose still resolves by search. Read this overview first (Global Constraints +
the "use VERBATIM" Shared Interfaces above), then work each phase file in
order — each ends in a gate-sweep + commit.

1. [P0 — Schema & profile foundation](./p0-schema-profile.md)
2. [P1 — Wire stage skeleton (`STEP_4_WIRE`) + terminal-stamp move](./p1-wire-skeleton.md)
3. [P2 — Wire stage data model](./p2-wire-data-model.md)
4. [P3 — B1 interpretation surfacing (all 5 kinds) + frontend projection](./p3-interp-surfacing.md)
5. [P4 — `web_scrape` recipe (D11) — re-polarized shield](./p4-web-scrape-recipe.md)
6. [P5 — Advisor sign-off gate (B3/D13) — profile-gated + UNAVAILABLE escape](./p5-advisor-signoff.md)
7. [P6 — Entry protocol + profile lifecycle + concurrency](./p6-entry-lifecycle.md)
8. [P7 — Cache (C2) + `TutorialGuidedShell` + migration](./p7-cache-shell-migration.md)

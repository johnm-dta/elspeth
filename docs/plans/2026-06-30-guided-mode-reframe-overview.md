# Guided-Mode Reframe — Slices A/B/C Implementation Plan (Overview)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement each slice plan task-by-task. Adapt the pytest-TDD template to React/vitest where the work is frontend; do NOT force red-green on pure component-mount/layout reuse.

**Goal:** Land the first three independently-shippable slices of the guided-mode reframe (epic `elspeth-e7757e5c58`) on `release/0.7.0`: surface cleanup (A), the wire-stage advisor dead-end fix (B), and the live-graph verification surface (C). Slices D/E/F are planned separately and are out of this pass.

**Source of truth:** `docs/superpowers/specs/2026-06-30-guided-mode-reframe-design.md` (committed `7c83594dd`, corrected `46826e44f`). Locked decisions D1–D3 are not reopened.

**Architecture:** Predominantly reuse + reroute + copy against data the system already produces. A removes two genuinely-dead branches and adds an existing indicator. B renders an advisor payload the backend already emits and gates the affordances on the sign-off outcome. C mounts an already-store-coupled graph component in the one column shared by live-guided and the tutorial. No new subsystems.

**Tech stack:** React 18 + TypeScript + Zustand (`useSessionStore`/`useExecutionStore`) + `@xyflow/react`/`@dagrejs/dagre` (frontend); FastAPI + Python guided state machine (`src/elspeth/web/composer/guided/`; the guided HTTP route file is `src/elspeth/web/sessions/routes/composer/guided.py`, the helpers `src/elspeth/web/sessions/routes/_helpers.py`). Tests: `vitest` (frontend), `pytest` (backend). The a11y suite at `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx` is a FIXED enumerated list — new/changed surfaces (B's wire-stage buttons, C's gloss + validation summary) must be ADDED explicitly or they go uncovered.

---

## Sequencing — STRICT, sequential by slice

Slices A/B/C touch overlapping frontend files (`types/guided.ts`, `components/chat/guided/*`, `ChatPanel.tsx`). **Implement them sequentially**, never in parallel worktrees (overlapping edits would conflict). Parallelism is allowed only *within* a slice (e.g., independent test files). Order: **A → B → C** (low-blast-radius first; A and B are independent, C is additive).

Each slice ends at a green gate before the next begins:

```bash
# Frontend gate (run from src/elspeth/web/frontend)
npm run test        # vitest
npm run lint        # eslint
npm run lint:css    # stylelint
npm run build       # production build (tsc + vite)

# Backend gate (run from repo root, main .venv active)
# NOTE: tests/unit/web/sessions/routes is REQUIRED — the wire-stage escape/gate
# invariant suites (test_request_advisor_escape, test_wire_stage_signoff_gate,
# test_wire_signoff_audit_and_blocked) live there, NOT under composer/guided.
pytest tests/unit/web/composer/guided tests/integration/web/composer/guided tests/unit/web/sessions/routes -q
```

> Verify the exact npm script names against `src/elspeth/web/frontend/package.json` at slice start — substitute if they differ. Do not run `pytest -o addopts=""` (forces slow/stress → phantom fails); plain `pytest <paths>` is CI-equivalent.

## Baseline (captured 2026-06-30 on a clean tree at `87e75e54d` — the reference frame for every gate run)

- **Frontend gate: FULLY GREEN** (`test`, `lint`, `lint:css`, `build`). ⇒ ANY frontend red after a slice is THAT slice's regression — fix it before the slice lands.
- **Backend gate: `539 passed, 5 PRE-EXISTING failures`** — owed 0.7.0 baseline reds, NOT caused by this work. **Do NOT fix them in A/B/C** (separate debt), and **do NOT let them mask a new regression.** The five:
  1. `test_auto_drop.py::TestRepairSucceeds::test_first_fails_repair_succeeds_returns_confirm_wiring_then_completed`
  2. `test_step_3_e2e.py::TestStep3ChainAccept::test_csv_to_json_step_3_accept_returns_confirm_wiring_then_completes_session`
  3. `test_step_chat_sink_driver.py::test_sink_driver_revise_threads_current_sink`
  4. `test_step_chat_source_driver.py::test_source_driver_includes_current_source_in_prompt`
  5. `test_wire_dispatch.py::test_confirm_wiring_stamps_completed_terminal`

**"Backend gate green" for a slice means:** exactly these 5 fail, everything else passes. A 6th failure = a regression to fix before landing. Note #2 is in `test_step_3_e2e.py` — Slice A2 edits the *reject* test in that file, a DIFFERENT test from this failing *accept* one; do not conflate. Note #5 is in the wire area Slice B works in (the advisor-unavailable→`blocked_unavailable` completion path) — pre-existing; Slice B (frontend render + `passes_remaining` at `:3703`/`:3858`) does not touch it.

## Commit discipline

- One atomic commit per task (per the implementation-planning skill). Conventional messages; end each with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Commit only; do NOT push** unless the operator explicitly asks. (Push, when authorized, is as `johnm-dta` — `gh auth switch --user johnm-dta` before, switch back after.)
- Doc-only commits use `--no-verify`. Code commits run the hooks; if the ruff PostToolUse autofix strips an import added before its use, edit the use-site first then re-add the import; SIM117 blocks nested `with`.

## Cross-cutting constraints (honor in every slice)

| Constraint | Where it bites | Rule |
|---|---|---|
| **Tier-3 egress** | B | Do NOT widen the "commit failed" HTTP detail at `_helpers.py:2925/3045/3231` (STEP_1/STEP_2 only — not the wire path). Slice B must not touch them; verify by diff. |
| **Persistence safety** | A | `recipe_offer`/`STEP_2_5`, `coaching`/`recipe_match`, `inspect_and_confirm` are RETAINED (persisted/structural). Removing them is a fail-closed migration, not cleanup. No sessions-DB wipe in Slice A. |
| **Mirror hook** | A | `scripts/cicd/check_slot_type_cross_language.py` validates only `RecipeSlotInput.slot_type` — a no-op gate for the `TurnType` edit, but it re-runs on any `guided.ts` change; keep `slot_type` intact. |
| **D1 (zero rows)** | C | The graph builds from `compositionState` + `validationResult` (store) only — zero source rows read. Explicit DoD check, not a manual afterthought. |
| **Crown jewel** | all | The chat→build path is a fixed contract; none of A/B/C changes the guided solver/state-machine spine behaviourally. |
| **Tutorial parity** | A/B/C | Tutorial must not be *easier* than live. C closes a real gap (tutorial had no graph). B/F keep the advisor gate honest; do not re-hide affordances in the tutorial. |
| **Tutorial advisor-off** | B | The tutorial runs advisor-*off* → `passes_remaining`/`signoff_outcome` ABSENT. Cost copy MUST gate on `passes_remaining !== undefined` (not outcome-absent), else the tutorial shows a false "uses 1 of N." |
| **No duplicate graph surface** | C | `GraphModal` at `App.tsx:408` is UNCONDITIONAL (serves tutorial + live) — do NOT add a second. `GraphMiniView` already renders in the live-guided rail — the column thumbnail is tutorial-only; gloss + validation summary go in-column for both. |

## Tracker linkage

| Slice | Issue | Type | Status → working | Notes |
|---|---|---|---|---|
| A | `elspeth-b30e59bfa3` | task | `proposed`→`building` (advance) | |
| B | `elspeth-7b0f75e90e` | bug | `triage`→`confirmed`→`fixing` | needs `severity` field set before `confirmed` |
| C | `elspeth-aabb519a49` | feature | `proposed`→`building` (advance) | blocks D |
| epic | `elspeth-e7757e5c58` | epic | `open`→`in_progress` | |

Use the atomic `work_start --advance` verb; for B set `severity` first. Close each issue at its slice's green gate with the commit SHA.

## Per-slice plan files

- `2026-06-30-guided-mode-reframe-slice-a.md` — surface cleanup + silent-compute indicator.
- `2026-06-30-guided-mode-reframe-slice-b.md` — wire-stage advisor dead-end.
- `2026-06-30-guided-mode-reframe-slice-c.md` — live graph as verification surface.

Each was kept to a single slice deliberately, so the plan-review reality lens does not stall on a >2k-line combined plan.

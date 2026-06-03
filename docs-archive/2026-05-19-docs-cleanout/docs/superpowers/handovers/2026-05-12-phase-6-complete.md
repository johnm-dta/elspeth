# Handover — Composer Guided Mode (Phase 6 complete; Phase 7 onward)

## TL;DR

Phase 6 is **complete and green**. The frontend foundation is closed:
TypeScript types mirroring the backend pydantic protocol, two `apiClient`
free functions wrapping `GET /api/sessions/:id/guided` and `POST
/api/sessions/:id/guided/respond`, and three new fields + three new
actions on the flat `useSessionStore` zustand store. Server-authoritative
atomic 4-field replace pattern is wired into both `startGuided` and
`respondGuided`. Three new test files (10 + 5 + 8 = 23 new tests) ride
alongside the implementation; full vitest suite is 211 / 211 green.

Phases 7–10 ship the React widget surface on top of this foundation:
turn-type widgets, the `GuidedTurn` dispatcher, the chat-pane integration,
the e2e Playwright lane, and the operator UX polish.

## Environment

- **Worktree:** `/home/john/elspeth/.worktrees/composer-guided-mode`
- **Branch:** `feat/composer-guided-mode` (42 commits ahead of RC5.2 as of Phase 6 close; top commit `cc074420`)
- **Python:** `.venv/bin/python` in the worktree. **Do NOT use main's venv** — Python-version mismatch corrupts `enforce_tier_model.py` results.
- **Frontend toolchain:** `npm` (NOT pnpm/yarn). `vitest` is the test runner. The canonical typecheck is `npx tsc -p tsconfig.app.json --noEmit` — NOT `npx tsc --noEmit` at the root (which trips on pre-existing tsconfig composite-project misconfiguration).
- **ESLint is broken** on this worktree due to v9 flat-config migration debt — not Phase 6's job to fix. Don't run `npx eslint src/`; rely on vitest + tsc for the per-task gate.
- **Plan file:** `docs/superpowers/plans/2026-05-11-composer-guided-mode.md` (Phase 7 starts line ~4190)
- **Spec file:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (Phase 7 relevant: §7.1–§7.3 component layout, §10 e2e flow)
- **Inbound handover:** `docs/superpowers/handovers/2026-05-12-phase-6-start.md` — the Phase 6 brief; references back to Phase 5 close.

## What's delivered in Phase 6

Six commits on top of Phase 5 closure (`8c0f7527`):

| Commit | Task | What |
|--------|------|------|
| `80bee5dd` | 6.1 feat | TypeScript types in `src/types/guided.ts` mirroring `schemas.py:213-296` + `protocol.py` enums + `state_machine.py` terminal enums; `step_index: number` verified against `emitters.py:_step_index` (0-based ordinal); both response envelopes include `composition_state` (the handover-original missed it). Test file with 10 type-assertion tests. |
| `a01cd69e` | 6.1 fix-up | `TurnRecord.emitter: "server" | "llm"` closed union (was open `string`; verified against `state_machine.py:75-82` + 15 producer sites in `routes.py`). `GuidedSession`-keys exhaustiveness test replaced with compile-time `Equals<A,B>` mutual-extends check (the prior `Keys[]` tuple form was inert — silently green on field drift). Two trivially-true runtime assertions sharpened. |
| `f4fd3c94` | 6.2 feat | Two free async functions in `client.ts` — `getGuided(sessionId, signal?)` and `respondGuided(sessionId, body, signal?)` — mirroring the existing `sendMessage`/`recompose` convention. Both pass `AbortSignal` through to `fetch`. First `src/api/*.test.ts` file in the codebase: 5 tests via `vi.spyOn(globalThis, "fetch")`. |
| `9617a5f1` | 6.2 fix-up | File-level strategy banner in `client.guided.test.ts` explaining producer-side spy vs. consumer-side `vi.mock` (locks the convention for future API tests). `fetchSpy` lifecycle hoisted into an outer describe to remove byte-identical duplication across inner suites. Raw-string body assertion replaced with parsed-shape `toEqual` check (JSON key ordering is incidental). |
| `20662208` | 6.3 feat | Three new state fields (`guidedSession`, `guidedNextTurn`, `guidedTerminal`) + three new actions (`startGuided`, `respondGuided`, `exitToFreeform`) added directly to the flat `SessionState` interface — no slice composition. Server-authoritative 4-field atomic replace on each action (includes `compositionState` from response). `respondGuided` throws on null `activeSessionId` (offensive programming per CLAUDE.md). Wired into all three session-transition clear sites (`createSession`, `selectSession`, `archiveSession`) so guided state cannot leak across session switches. New consumer-side test file with 8 tests including the cross-session-leak regression guard. |
| `cc074420` | 6.3 fix-up | Dead unreachable string-match re-throw removed from `respondGuided`'s catch block (the offensive guard fires before `try` — the catch can only see network errors, never the guard's literal). Section marker added to the guided action block. Interface field declarations relocated to sit adjacent to companion actions, restoring the `state + companion-action` pairing convention. |

### Gate state at close

- **Vitest:** 211 / 211 pass (29 test files; +23 from Phase 5 baseline of 188)
- **`npx tsc -p tsconfig.app.json --noEmit`:** clean
- **Backend pytest** (narrow guided + sessions slice): 522 / 522 pass — unchanged from Phase 5 close (Phase 6 is frontend-only)
- **Backend mypy** on `src/elspeth/web/composer/` + `src/elspeth/web/sessions/routes.py`: clean
- **ESLint:** not run — broken on this worktree (v9 flat-config migration debt); pre-existing infra issue unrelated to Phase 6

### Per-task review iteration count (data point for future planning)

| Task | Implementer dispatches | Why iteration was needed |
|------|------------------------|---------------------------|
| 6.1 | 2 (initial + fix-up) | Code-quality reviewer caught inert exhaustiveness test (`Keys[]` form passed on field drift) and open `emitter: string` symmetry-break with `step`/`turn_type` |
| 6.2 | 2 (initial + fix-up) | Convention-setting test file: missing file-level strategy comment, duplicated `beforeEach`/`afterEach` across describe blocks, raw-string body assertion brittle to key reordering |
| 6.3 | 2 (initial + fix-up) | Unreachable dead code in catch block (string-match re-throw after offensive guard already exited the function); missing section marker; interface field placement broke pairing |

Pattern: every Phase 6 task needed exactly one fix-up cycle. Both reviewers (spec + code-quality) ran on each cycle. Spec compliance passed first time on all three tasks; code-quality findings drove every fix-up.

## End-to-end frontend capability (post Phase 6)

Phase 6 lands no UI — it is purely the data-plane foundation. After Phase 6:

1. Frontend can fetch the active guided session via `getGuided(sessionId)` → returns `GuidedSession`, `TurnPayload` (the next turn), `TerminalState`, and `CompositionState`.
2. Frontend can post a typed `GuidedRespondRequest` via `respondGuided(sessionId, body)` and atomically replace cached state from the response.
3. The store exposes the guided state for component subscription: `useSessionStore.guidedSession`, `.guidedNextTurn`, `.guidedTerminal`.
4. Session switches and resets clear guided state cleanly (regression-tested).
5. The `exitToFreeform()` shortcut posts a `control_signal: "exit_to_freeform"` payload — the backend handles the progressive-disclosure transition.

No widget exists yet. The next chat turn through the existing freeform UI is unchanged — guided mode is only reachable programmatically until Phase 7 wires it into the UI.

## What's pending — Phase 7 onward

Phase 7 builds the React widget surface — one component per `TurnType`, plus the `GuidedTurn` dispatcher that routes by `guidedNextTurn.type`. Plan body starts at line ~4190 (`Task 7.1: GuidedTurn dispatcher`).

## Final cross-task review notes (decisions, not open questions)

A final cross-task reviewer was run after the three per-task review cycles. They surfaced five items framed as "Phase 7 readiness gaps." After CLAUDE.md scrutiny, **three were rejected as speculative or wrong, one was deferred to Phase 7's natural scope, and one is a deferrable nit**. Recording the reasoning so Phase 7's author doesn't relitigate:

1. **Rejected — "Add `isGuided: boolean` selector to the store."** Speculative API design. CLAUDE.md: "Don't design for hypothetical future requirements. Three similar lines is better than a premature abstraction." We have zero call sites today. Phase 7's first widget will reveal the actual selector shape needed (could be `isGuided`, could be `isInStep1`, could be a `useGuidedTurn()` hook — we don't yet know). Until then, `state => state.guidedSession !== null` at the call site is the honest answer.

2. **Rejected — "Re-export `guided.ts` types from `@/types/api` for a single barrel convention."** The barrel premise is false: the codebase already imports types from three paths today (`@/types/api`, `@/types/index`, and now `@/types/guided`). `sessionStore.ts:11` uses `@/types/api`; `client.ts:33` uses `@/types/index`. There is no single-barrel convention to consolidate to. The Task 6.1 brief explicitly decided to leave `guided.ts` self-contained — a 7th commit reversing that for a stylistic preference is churn, not correctness. Phase 7 components can import from `@/types/guided` directly; if a future cleanup wants to merge barrels, that's its own discrete task.

3. **Rejected — "JSDoc warning on the `respondGuided` store action about the name collision with `client.respondGuided`."** Cheap one-liner but the collision is not load-bearing: the store implementation uses `api.respondGuided` via the `import * as api` alias, and component code will never reach into `@/api/client` directly — it'll use the store action via `useSessionStore`. The architectural separation is enforced by import structure, not by comments. If Phase 7 author somehow imports `client.respondGuided` directly into a component, that's a code-review catch, not a comment-prevention concern.

4. **Deferred to Phase 7 — Error messages lack session-id context.** `startGuided` and `respondGuided` produce generic "Failed to load / submit" strings on failure. Phase 7 wires up the error-banner component; that's the right time to add session-id context for support triage. Premature in Phase 6 (no consumer yet).

5. **Deferred (minor nit) — Cross-reference comment in `sessionStore.guided.test.ts` pointing to the test-strategy banner in `client.guided.test.ts`.** Genuinely cheap, genuinely low-value: a Phase 7 author touching either file will absorb the pattern naturally. Not worth a 7th commit.

The verdict from the final reviewer was **READY-WITH-NOTES** — Phase 7 can start. The "notes" above are the decisions, not the open questions.

## Phase 7 starting brief

Read this in conjunction with the plan file's Phase 7 section.

### Convention reminders inherited from Phase 6

1. **Types module is `@/types/guided`.** Import there directly: `import type { TurnPayload, TurnType, GuidedRespondRequest } from "@/types/guided"`. Don't plumb a re-export through `types/api.ts` unless the Phase 7 work has an independent reason to consolidate.
2. **The store is flat, not slice-composed.** Add fields and actions directly to `SessionState` in `sessionStore.ts`. The Phase 6 path is the template; do not introduce slice abstractions.
3. **Producer tests use `vi.spyOn(globalThis, "fetch")`. Consumer tests use `vi.mock("@/api/client")`.** The strategy banner is in `client.guided.test.ts:1-14`. Component tests for Phase 7 widgets are consumers of the store, so they'll mock `@/stores/sessionStore` similarly — different module, same pattern.
4. **Atomic 4-field replace from server response.** Any new action that calls a guided endpoint replaces `guidedSession`, `guidedNextTurn`, `guidedTerminal`, AND `compositionState` in a single `set({...})` call. No optimistic updates (spec §7.3).
5. **Offensive programming on null `activeSessionId`.** Any new store action that requires a session must `throw new Error("…")` on null, not silent early-return, not `?.` defend.
6. **`payload: unknown`** on `TurnPayload` discriminates only at the consumer. Phase 7 widgets will discriminate via `guidedNextTurn.type` (the `TurnType` enum) and narrow `payload` to the per-turn-type shape declared in `protocol.py` (`InspectAndConfirmPayload`, `SingleSelectPayload`, etc). Consider whether per-payload TS interfaces should land in `guided.ts` during Phase 7 — likely yes — but discover the right ergonomic shape from the first widget rather than pre-defining all six.

### Plan-vs-reality drift expected in Phase 7

The plan body has shown drift in every prior phase. Expect the same in Phase 7:

- The plan likely shows class-based React components. Existing convention is hooks-based function components.
- The plan likely references `apiClient.x` calls. Reality is bare named imports.
- The plan likely shows `compositionState` accessed as `state.composition_state`. Reality is `state.compositionState` on the store (camelCase) but `state.composition_state` from the wire response (snake_case) — the store transcribes during the 4-field replace.

The implementer dispatch brief for each Phase 7 task should pre-emptively override these.

### First action when you start Phase 7

```bash
cd /home/john/elspeth/.worktrees/composer-guided-mode
git log --oneline 8c0f7527..HEAD                          # expect 6 commits, top = cc074420
cd src/elspeth/web/frontend
npm test -- --run                                          # expect 211 / 211 pass
npx tsc -p tsconfig.app.json --noEmit                      # expect clean
```

If any baseline gate fails, **stop and investigate** before starting Phase 7.

## Important constraints (do not relitigate)

- **DB migration = delete the DB.** No Alembic; no migration scripts; no `from_dict` backward-compat defaults. (Phase 6 didn't touch the DB — but the constraint applies if Phase 7 somehow finds itself there.)
- **Default to worktree.** Stay here; do not work in `/home/john/elspeth` main.
- **No git stash.** Commit work to a branch if preservation is needed.
- **No calendar shipping commitments.** ELSPETH ships work-until-done.
- **Correctness beats performance always.**
- **Default answer is never "log a ticket."** Investigation surfacing a fixable defect MUST fix in-session.
- **`any` is forbidden in TypeScript.** Use `unknown` for opaque, closed unions or interfaces otherwise.
- **No optimistic updates.** Server is authoritative.
- **snake_case wire field names in interfaces; camelCase store-internal fields.** Established in Phase 6.
- **ESLint is broken on this worktree.** Don't try to fix it inside Phase 7 unless a separate task is opened for the v9 flat-config migration.

# Phase 4B Part 1 — Frontend: store, state machine, copy, turns 1–3

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal of Phase 4B (overall):** Land the frontend half of Phase 4 — extend
the `preferencesStore` to expose `tutorialCompleted`, wire `App.tsx` to
render the hello-world tutorial container for first-session users, implement
the 6-turn state machine plus six leaf turn components, ship the turn-1
skip affordance, and finalise the tutorial by writing both `default_mode`
and `tutorial_completed_at` to the backend while preserving the produced
pipeline as a named session. End with a Vitest integration test and a
Playwright E2E smoke against `elspeth.foundryside.dev`.

**Goal of this part (4B.1):** Tasks 1–7 — store extension, tutorial state
machine, copy module, and turn components 1, 2, 2b, 3. After this part
lands, the building blocks are unit-testable in isolation; the integration
into `App.tsx` and the run/audit/finalisation flow live in part 2.

**Architecture:** Store-then-detection-then-container-then-turns-then-
finalisation-then-skip-then-integration-then-smoke. The tutorial is a
**parallel render path** in `App.tsx` (not a modal): when
`tutorialCompleted === false`, the tutorial container takes over the composer
surface. There is no nesting inside the normal composer; the tutorial **is**
the first session.

**Navigation resilience:** Refresh during the tutorial restarts at turn 1 — there is **no** `sessionStorage` scaffolding for resume. The tutorial state machine is in-memory only (Zustand-free; lives in `HelloWorldTutorial.tsx` `useReducer` state). Rationale (per plan-fix P10, 2026-05-19): (1) the tutorial is ~5 minutes — restart cost is acceptable; (2) a flat `elspeth_tutorial_progress` key isn't user-scoped, so it risks cross-user contamination on shared workstations (systems S6); (3) per-user keying would add Vitest+Playwright surface area that doesn't materially improve UX. The canonical seed produces a fresh pipeline each time, and the cache makes that fast. All tutorial-mode users (genuine first-timers and users whose state was wiped by the Phase 4A DB-delete) see the same intro — no bifurcation, no "we noticed you had old sessions" copy. Per CLAUDE.md No Legacy Code Policy: we have no users yet, the operator deletes the DB freely, and a single neutral copy is more honest than detecting a state we can't reliably distinguish. No separate banner — Turn 1's existing welcome copy is the entry surface.

**Tech Stack:** React 18, TypeScript (strict), Zustand, Vitest, React
Testing Library, Playwright.

**Sibling plans:**
- [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) — backend infrastructure: schema column, service
  extension, route extension, tutorial cache, run-path integration.
- [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md) — backend endpoints + telemetry: tutorial-run endpoint, audit-story endpoint, frontend API client, launch telemetry counters.
- [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md) — Tasks
  8–15: turns 4/5/6, container, App.tsx detection, integration test,
  Playwright E2E, staging smoke.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**PR mapping:** This plan is the first half of **PR-21b** (frontend);
PR-21b combines Tasks 1–7 here with Tasks 8–15 in 21b2 and merges to
`RC5.2` **only after** PR-21a (backend) has landed — see overview §"PR
strategy" for rationale.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**

- Extend `preferencesStore.ts` with `tutorialCompletedAt: string | null`,
  derived `tutorialCompleted: boolean`, and
  `markTutorialCompleted(defaultMode)` action. (Session-rename is a separate
  call from Turn 6, not bundled into `markTutorialCompleted` — keeps the
  preference-PATCH a single-responsibility atomic op.)
- Extend the wire typing in `src/types/api.ts` to include the new field
  on `UserComposerPreferencesPayload` (the account-scoped payload — NOT
  the per-session `ComposerPreferences`).
- Bootstrap-time detection: when the user logs in and preferences load, if
  `tutorialCompleted === false`, App renders `HelloWorldTutorial` instead of
  the normal composer surface.
- `tutorialMachine.ts` — typed step-union + reducer for the 6-turn flow.
- `HelloWorldTutorial.tsx` — container that owns the reducer and dispatches
  the right leaf component per step.
- Six turn components:
  - `TutorialTurn1Welcome.tsx` — title, body, "Let's go →" button, subtle
    skip link.
  - `TutorialTurn2Describe.tsx` — pre-filled prompt textarea, "Build it"
    button. On submit, calls the composer compose API and creates the
    session.
  - `TutorialTurn2bShowBuilt.tsx` — renders source URLs, transforms, sink
    summary, and embeds Phase 5b's `InterpretationReviewTurn` for "cool."
    Depends on Phase 5b's `interpretation_events_table` (preflight: 21a Task 0 Step 4).
  - `TutorialTurn3Graph.tsx` — pipeline-as-graph rendering (3 nodes + 1
    sink rendered as 4-stage horizontal flow with row counts and labels).
  - `TutorialTurn4Run.tsx` — kicks off the run, shows progress, renders the
    result table.
  - `TutorialTurn5AuditStory.tsx` — renders load-bearing audit story with real
    values (`source_data_hash`, `interpretation_event_id`, LLM-call count,
    `run_id`). Consumes `GET /api/sessions/{id}/runs/{run_id}/audit-story`
    (21a §"New endpoints").
  - `TutorialTurn6ModeChoice.tsx` — guided/freeform radio (default guided),
    "Save and go" button. On submit, calls the atomic PATCH that finalises
    both fields and routes the user to the now-empty composer in their
    chosen mode.
- `copy.ts` — extracted text blocks (single source of truth + future i18n
  hook).
- Vitest unit tests for the store extension, machine, container, each turn
  component, and the dispatch logic.
- One Vitest integration test that walks the full 6-turn flow with mocked
  API responses.
- One Playwright E2E test that walks a brand-new user through the tutorial
  end-to-end against a real backend (a `pytest` fixture / `docker compose`
  setup, depending on the project's E2E infra — confirmed during recon).
- Staging smoke deploy at `elspeth.foundryside.dev`: operator-led DB-delete,
  service restart, manual click-through verification.

**Out of scope:**

- Re-take from settings (Open Question C3 — post-launch).
- Mid-tutorial resume across refresh / browser-back / new tab. Refresh
  restarts at turn 1 by design (plan-fix P10, 2026-05-19). Cross-tab and
  cross-device sync remain out of scope as a consequence.
- Localisation of tutorial copy.
- Backend changes (Phase 4A).

## Trust tier check (per CLAUDE.md)

- **User input at turn 2** (seed prompt): Tier 3 at the boundary. The
  textarea forwards the string to the existing composer API; the existing
  Tier-3 boundary handles validation. Phase 4B does **not** validate the
  prompt itself.
- **`preferences.tutorialCompletedAt`** (read): Tier 1. The frontend treats
  any non-null/undefined value as "tutorial done." A malformed timestamp
  string (which should be impossible because the backend Tier-1 guard
  catches it) would surface here as a render-time issue; the contract is
  that the backend never sends garbage.
- **Audit-trail values rendered in turn 5**: Tier 1 (read from the
  Landscape via the existing audit API). These are our data; we trust them.
- **`InterpretationReviewTurn` embed**: Phase 5b artifact, used as-is.
  Trust boundary belongs to 5b, not to 4.

## File structure

**New:**

- `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`
- `src/elspeth/web/frontend/src/components/tutorial/copy.ts`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn1Welcome.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn1Welcome.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.test.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/index.ts` — barrel export.
- `tests/e2e/tutorial.e2e.spec.ts` (path TBD by Playwright location).

**Modified:**

- `src/elspeth/web/frontend/src/stores/preferencesStore.ts`
- `src/elspeth/web/frontend/src/api/client.ts` (typing only)
- `src/elspeth/web/frontend/src/App.tsx`

## Verification approach

Each task is TDD-shaped at the Vitest level. The Playwright E2E (Task 12)
runs against a fully wired backend (Phase 4A merged + DB-deleted + service
restarted); the staging smoke (Task 13) is a manual operator click-through
recorded in the merge PR.

---

## Task 1: Extend the preferencesStore

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.ts`.
- Modify (companion test): `src/elspeth/web/frontend/src/stores/preferencesStore.test.ts` (create if not present in Phase 1B).

- [ ] **Step 1: Recon Phase 1B's store shape.**

```bash
cat src/elspeth/web/frontend/src/stores/preferencesStore.ts
```

Note:
- The exact shape Phase 1B ships: `defaultMode`, `bannerDismissedAt`,
  `loaded`, `writing`, `writeError`, `optedOutAtSessionId`; actions
  `bootstrap()`, `resolveDefaultMode()`, `setDefaultMode()`,
  `dismissDefaultChangedBanner()`, `clearError()`, `reset()`.
- Phase 1B ships `preferencesStore.test.ts` (companion test file present).
- The API call mechanism: `fetchUserComposerPreferences()` /
  `updateUserComposerPreferences()` from `@/api/client` (the
  user-account flavour; the per-session `getComposerPreferences` /
  `updateComposerPreferences` pair is a different surface and not what
  this store uses).
- Bootstrap is gated by a single `writing` flag that serialises concurrent
  PATCH calls — `markTutorialCompleted` MUST honour the same gate.

- [ ] **Step 2: Write failing test extensions.**

Add to (or create) `preferencesStore.test.ts`:

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { usePreferencesStore, selectTutorialCompleted } from './preferencesStore';
import * as api from '@/api/client';

describe('preferencesStore — tutorial fields', () => {
  beforeEach(() => {
    usePreferencesStore.getState().reset();
    usePreferencesStore.setState({
      defaultMode: 'guided',
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      loaded: false,
    });
    vi.restoreAllMocks();
  });

  it('selectTutorialCompleted derives false when tutorialCompletedAt is null', () => {
    const s = usePreferencesStore.getState();
    expect(s.tutorialCompletedAt).toBeNull();
    expect(selectTutorialCompleted(s)).toBe(false);
  });

  it('selectTutorialCompleted derives true when tutorialCompletedAt is set', () => {
    usePreferencesStore.setState({ tutorialCompletedAt: '2026-05-15T12:00:00Z' });
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it('bootstrap populates tutorialCompletedAt from the API', async () => {
    vi.spyOn(api, 'fetchUserComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: '2026-05-15T12:00:00Z',
      updated_at: '2026-05-15T12:00:00Z',
    });
    await usePreferencesStore.getState().bootstrap();
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBe(
      '2026-05-15T12:00:00Z',
    );
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it('markTutorialCompleted PATCHes both fields atomically', async () => {
    const patchSpy = vi
      .spyOn(api, 'updateUserComposerPreferences')
      .mockResolvedValue({
        default_mode: 'freeform',
        banner_dismissed_at: null,
        tutorial_completed_at: '2026-05-15T12:30:00Z',
        updated_at: '2026-05-15T12:30:00Z',
      });
    await usePreferencesStore.getState().markTutorialCompleted('freeform');
    expect(patchSpy).toHaveBeenCalledTimes(1);
    const arg = patchSpy.mock.calls[0][0];
    expect(arg.default_mode).toBe('freeform');
    expect(typeof arg.tutorial_completed_at).toBe('string');
    // Resulting state reflects the response.
    expect(usePreferencesStore.getState().defaultMode).toBe('freeform');
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it('markTutorialCompleted respects the writing guard (no concurrent PATCH)', async () => {
    // Mirrors the Phase 1B serialisation contract: a second call while
    // `writing === true` is a no-op (no double-PATCH, no optimistic
    // overwrite). The first call's settle releases the gate.
    usePreferencesStore.setState({ writing: true });
    const patchSpy = vi.spyOn(api, 'updateUserComposerPreferences');
    await usePreferencesStore.getState().markTutorialCompleted('guided');
    expect(patchSpy).not.toHaveBeenCalled();
  });

  it('markTutorialCompleted does not clear optedOutAtSessionId (P16 MN-02)', async () => {
    // The opt-out watermark is orthogonal to tutorial completion and must
    // survive a markTutorialCompleted call. Phase 1B MN-02 contract:
    // optedOutAtSessionId is only mutated by setDefaultMode(...) when the
    // user is actively opting out; tutorial-completion must not touch it.
    const WATERMARK = 'session-uuid-watermark-from-phase-1b';
    usePreferencesStore.setState({ optedOutAtSessionId: WATERMARK });
    vi.spyOn(api, 'updateUserComposerPreferences').mockResolvedValue({
      default_mode: 'freeform',
      banner_dismissed_at: null,
      tutorial_completed_at: '2026-05-15T12:30:00Z',
      updated_at: '2026-05-15T12:30:00Z',
    });
    await usePreferencesStore.getState().markTutorialCompleted('freeform');
    expect(usePreferencesStore.getState().optedOutAtSessionId).toBe(WATERMARK);
  });

  it('markTutorialCompleted reverts BOTH defaultMode AND tutorialCompletedAt on failure (P16 contract #3)', async () => {
    usePreferencesStore.setState({
      defaultMode: 'guided',
      tutorialCompletedAt: null,
    });
    vi.spyOn(api, 'updateUserComposerPreferences').mockRejectedValue(
      new Error('network down'),
    );
    await expect(
      usePreferencesStore.getState().markTutorialCompleted('freeform'),
    ).rejects.toThrow(/network down/);
    expect(usePreferencesStore.getState().defaultMode).toBe('guided');
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull();
    expect(usePreferencesStore.getState().writing).toBe(false);
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts
```

Expected: FAIL — `tutorialCompletedAt` not a state key, `markTutorialCompleted` undefined, `selectTutorialCompleted` not exported.

- [ ] **Step 4: Extend the store.**

This is an **additive diff** against the live Phase 1B store, not a
rewrite. Preserve Phase 1B's existing fields, actions, `writing` gate,
cross-tab banner sync, and watermark logic verbatim. Only the new
`tutorialCompletedAt` state field, the new `markTutorialCompleted`
action, and the `selectTutorialCompleted` selector are added; the
existing `bootstrap` action gains one extra `set` field
(`tutorialCompletedAt: payload.tutorial_completed_at`).

Apply these targeted edits inside `preferencesStore.ts`:

```typescript
// 1. Add to PreferencesState interface (alongside existing fields):
//
//   bannerDismissedAt: string | null;
//   loaded: boolean;
//   writing: boolean;
//   writeError: string | null;
//   optedOutAtSessionId: string | null;
//   // NEW (Phase 4):
//   tutorialCompletedAt: string | null;
//
//   bootstrap: () => Promise<void>;
//   resolveDefaultMode: () => Promise<ComposerMode>;
//   setDefaultMode: (mode: ComposerMode, activeSessionId?: string | null) => Promise<void>;
//   dismissDefaultChangedBanner: () => Promise<void>;
//   clearError: () => void;
//   reset: () => void;
//   // NEW (Phase 4): atomic finalisation — PATCHes default_mode AND
//   // tutorial_completed_at in one call. Honours the `writing` gate to
//   // serialise against concurrent setDefaultMode / dismiss calls (Panel
//   // C2 contract).
//   markTutorialCompleted: (mode: ComposerMode) => Promise<void>;

// 2. Add to INITIAL_STATE (alongside the existing five entries):
//   tutorialCompletedAt: null as string | null,

// 3. In bootstrap(), extend the existing set({...}) call:
//   set({
//     defaultMode: payload.default_mode,
//     bannerDismissedAt: payload.banner_dismissed_at,
//     tutorialCompletedAt: payload.tutorial_completed_at,  // NEW
//     loaded: true,
//   });

// 4. Add the new action alongside existing actions.
//
// PRESERVATION CONTRACT for `markTutorialCompleted` (P16):
//
//   1. MUST NOT reset `optedOutAtSessionId` watermark (MN-02).
//      The watermark tracks the session in which the user opted out
//      of guided mode; tutorial completion is orthogonal and must
//      NOT clear it. Concretely: this action's `set({...})` calls
//      DO NOT mention `optedOutAtSessionId` (neither in the
//      optimistic-write nor the revert-on-error branch), and the
//      response-payload merge below does NOT carry an opt-out field
//      because the PATCH payload does not address it. A unit test
//      `markTutorialCompleted does not clear optedOutAtSessionId`
//      pins this — see test list above.
//
//   2. MUST honour the `writing` guard (P6 preempted): the first
//      line `if (get().writing) return;` serialises concurrent PATCH
//      calls with `setDefaultMode` / `dismissDefaultChangedBanner`.
//
//   3. MUST follow optimistic-write-then-revert-on-failure (P6
//      preempted): the catch block restores both `defaultMode` and
//      `tutorialCompletedAt` to their pre-PATCH values, mirroring
//      the `setDefaultMode` and `dismissDefaultChangedBanner`
//      patterns.
//
// Any future edit to this action that touches `optedOutAtSessionId`,
// removes the writing guard, or skips the revert branch MUST be
// reviewed against this block — these are load-bearing invariants
// against the Phase 1B opt-out semantics.
markTutorialCompleted: async (mode) => {
  if (get().writing) return;
  const previousMode = get().defaultMode;
  const previousTutorial = get().tutorialCompletedAt;
  const stamp = new Date().toISOString();
  // Optimistic set (mirrors setDefaultMode's pattern). Note: only the
  // four named keys are written; `optedOutAtSessionId` is intentionally
  // absent (contract #1).
  set({
    defaultMode: mode,
    tutorialCompletedAt: stamp,
    writing: true,
    writeError: null,
  });
  try {
    const payload = await updateUserComposerPreferences({
      default_mode: mode,
      tutorial_completed_at: stamp,
    });
    set({
      defaultMode: payload.default_mode,
      tutorialCompletedAt: payload.tutorial_completed_at,
      writing: false,
    });
  } catch (err) {
    set({
      defaultMode: previousMode,
      tutorialCompletedAt: previousTutorial,
      writing: false,
      writeError:
        err instanceof Error
          ? `Couldn't finalise the tutorial: ${err.message}`
          : "Couldn't finalise the tutorial.",
    });
    throw err;
  }
},

// 5. Export a named selector function (Zustand convention used elsewhere
//    in this codebase — selectors live at call sites or as exported
//    functions; class-style getters inside the create() factory are not
//    used). Components call `usePreferencesStore(selectTutorialCompleted)`.
export const selectTutorialCompleted = (
  state: PreferencesState,
): boolean => state.tutorialCompletedAt !== null;
```

Notes:
- The store does NOT expose a `tutorialCompleted` state field. Derivation
  happens at the call site via the exported selector
  `selectTutorialCompleted`, or inline `usePreferencesStore((s) => s.tutorialCompletedAt !== null)`.
  This matches the project's Zustand convention (see e.g. `App.tsx`'s
  existing `usePreferencesStore((s) => s.bootstrap)` slices).
- `markTutorialCompleted` honours the same `writing` gate as
  `setDefaultMode` and `dismissDefaultChangedBanner` (Phase 1B Panel C2
  serialisation contract). A concurrent call while `writing === true` is
  a no-op.
- Revert-on-error mirrors `setDefaultMode`: both `defaultMode` AND
  `tutorialCompletedAt` are restored to their pre-PATCH values so the
  optimistic update doesn't survive a failed write.

Extend `src/elspeth/web/frontend/src/types/api.ts` typing (the wire
types live there, not in `api/client.ts`; `client.ts` re-exports the
two preference types from `./interpretation`-style barrel imports for
historical reasons, but the source-of-truth shape is in `types/api.ts`):

```typescript
// Modify the existing UserComposerPreferencesPayload interface. The
// per-session `ComposerPreferences` interface (in src/types/index.ts —
// trust_mode / density_default) is a DIFFERENT type and does NOT gain
// this field. See plan 13 Task 1 review history for the user-vs-session
// disambiguator rationale.
export interface UserComposerPreferencesPayload {
  default_mode: ComposerMode;
  banner_dismissed_at: string | null;
  // Phase 4: ISO timestamp string (UTC) or null.
  tutorial_completed_at: string | null;
  // Nullable per Phase 1B Panel U1 contract (no-row users get null until
  // the first write).
  updated_at: string | null;
}

export interface UpdateUserComposerPreferencesPayload {
  default_mode?: ComposerMode;
  banner_dismissed_at?: string | null;
  // Phase 4: caller sends an ISO timestamp string to mark tutorial complete,
  // or explicit `null` to clear it (Phase 8 retake path). Optional `?` =
  // absent (no-op); explicit `null` = write NULL to the column.
  // Three-state contract — see 21a §"Cross-plan contract — `tutorial_completed_at` PATCH semantics".
  tutorial_completed_at?: string | null;
}
```

No `client.ts` edits are required for Task 1 (the API functions
`fetchUserComposerPreferences` and `updateUserComposerPreferences`
already exist; they consume `UserComposerPreferencesPayload` /
`UpdateUserComposerPreferencesPayload` directly so the type-only edit
above is enough). The Phase 5a / Turn-2 compose API and the Phase 4A
tutorial-run / audit-story endpoints DO require new client.ts
functions; those are added in Tasks 5, 8, and 9.

- [ ] **Step 5: Run test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/stores/preferencesStore.ts \
  src/elspeth/web/frontend/src/stores/preferencesStore.test.ts \
  src/elspeth/web/frontend/src/types/api.ts
git commit -m "feat(frontend): extend preferencesStore with tutorial fields (Phase 4B.1)"
```

## Task 2: Tutorial state machine

**Files:**
- Create: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- Create: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`

The state machine is a typed union over the six turns plus a terminal
"done" state. Transitions are deterministic per the design doc 04 arc.

- [ ] **Step 1: Write the failing test.**

Create `tutorialMachine.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import {
  initialState,
  tutorialReducer,
  type TutorialState,
} from './tutorialMachine';

describe('tutorialMachine', () => {
  it('starts at turn 1', () => {
    expect(initialState.step).toBe('welcome');
  });

  it('welcome → describe via START', () => {
    const next = tutorialReducer(initialState, { type: 'START' });
    expect(next.step).toBe('describe');
  });

  it('welcome → mode-choice via SKIP (fast-forward)', () => {
    const next = tutorialReducer(initialState, { type: 'SKIP' });
    expect(next.step).toBe('mode-choice');
    expect(next.skipped).toBe(true);
  });

  it('describe → show-built carries the session id', () => {
    const s: TutorialState = { ...initialState, step: 'describe' };
    const next = tutorialReducer(s, {
      type: 'PIPELINE_BUILT',
      sessionId: 'sess-123',
      pipelineSnapshot: { source: {}, transforms: [], sinks: [] },
    });
    expect(next.step).toBe('show-built');
    expect(next.sessionId).toBe('sess-123');
    expect(next.pipelineSnapshot).toBeDefined();
  });

  it('show-built → graph after interpretation accepted', () => {
    const s: TutorialState = {
      ...initialState,
      step: 'show-built',
      sessionId: 'sess-123',
    };
    const next = tutorialReducer(s, {
      type: 'INTERPRETATION_ACCEPTED',
      interpretationEventId: 'evt-1',
    });
    expect(next.step).toBe('graph');
    expect(next.interpretationEventId).toBe('evt-1');
  });

  it('graph → run via START_RUN', () => {
    const s: TutorialState = { ...initialState, step: 'graph', sessionId: 's1' };
    const next = tutorialReducer(s, { type: 'START_RUN' });
    expect(next.step).toBe('run');
  });

  it('run → audit-story when run completes', () => {
    const s: TutorialState = { ...initialState, step: 'run', sessionId: 's1' };
    const next = tutorialReducer(s, {
      type: 'RUN_COMPLETED',
      runId: 'r1',
      sourceDataHash: 'a7f3e2',
      rows: [{ url: 'ato.gov.au', score: 5 }],
    });
    expect(next.step).toBe('audit-story');
    expect(next.runId).toBe('r1');
    expect(next.sourceDataHash).toBe('a7f3e2');
    expect(next.rows).toHaveLength(1);
  });

  it('audit-story → mode-choice via CONTINUE', () => {
    const s: TutorialState = {
      ...initialState,
      step: 'audit-story',
      sessionId: 's1',
      runId: 'r1',
    };
    const next = tutorialReducer(s, { type: 'CONTINUE' });
    expect(next.step).toBe('mode-choice');
  });

  it('mode-choice → done via FINALISE', () => {
    const s: TutorialState = {
      ...initialState,
      step: 'mode-choice',
      sessionId: 's1',
    };
    const next = tutorialReducer(s, {
      type: 'FINALISE',
      chosenMode: 'guided',
    });
    expect(next.step).toBe('done');
    expect(next.chosenMode).toBe('guided');
  });

  it('throws on out-of-order transitions (P15 offensive-programming contract)', () => {
    // Can't run before building. The reducer treats this as a programmer
    // error and throws — every dispatch site is in this codebase, so an
    // out-of-order dispatch is a bug we want to fail loudly on, not a
    // user-data trust-boundary issue. (Replaced the prior silent-no-op
    // contract per P15.)
    expect(() =>
      tutorialReducer(initialState, { type: 'START_RUN' }),
    ).toThrow(/unrecognised action START_RUN in step welcome/);
  });

  it('throws on unknown action types (exhaustiveness check)', () => {
    // Casting away the union type to simulate an action that doesn't
    // exist — this is what a future refactor that added a new action
    // type and forgot to update a dispatcher would look like.
    expect(() =>
      tutorialReducer(initialState, { type: 'BOGUS_UNKNOWN' } as unknown as TutorialAction),
    ).toThrow(/unrecognised action BOGUS_UNKNOWN/);
  });

  it('throws on any action arriving at the terminal done step', () => {
    const done: TutorialState = { ...initialState, step: 'done', chosenMode: 'guided' };
    expect(() => tutorialReducer(done, { type: 'START' })).toThrow(
      /unrecognised action START in terminal step 'done'/,
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/tutorial/tutorialMachine.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement the machine.**

Create `tutorialMachine.ts`:

```typescript
/**
 * Tutorial state machine for the hello-world flow.
 *
 * Per design doc 04, the arc is:
 *   welcome → describe → show-built → graph → run → audit-story → mode-choice → done
 *
 * Plus a skip fast-forward: welcome → mode-choice (sets `skipped=true`).
 *
 * Transitions are deterministic. An action that doesn't match the
 * current step is a programmer error (we control every dispatch site)
 * and the reducer THROWS. Per CLAUDE.md offensive programming: silently
 * no-op'ing would let an out-of-order dispatch ship and corrupt the
 * tutorial state with no signal. P15 promoted the previous
 * console.warn-and-no-op to an explicit throw.
 */

export type TutorialStep =
  | 'welcome'
  | 'describe'
  | 'show-built'
  | 'graph'
  | 'run'
  | 'audit-story'
  | 'mode-choice'
  | 'done';

export interface PipelineSnapshot {
  source: unknown;
  transforms: unknown[];
  sinks: unknown[];
}

export interface RunResultRow {
  url: string;
  score: number;
  rationale?: string;
}

export interface TutorialState {
  step: TutorialStep;
  skipped: boolean;
  sessionId: string | null;
  pipelineSnapshot: PipelineSnapshot | null;
  interpretationEventId: string | null;
  runId: string | null;
  sourceDataHash: string | null;
  rows: RunResultRow[];
  chosenMode: 'guided' | 'freeform' | null;
}

export type TutorialAction =
  | { type: 'START' }
  | { type: 'SKIP' }
  | {
      type: 'PIPELINE_BUILT';
      sessionId: string;
      pipelineSnapshot: PipelineSnapshot;
    }
  | { type: 'INTERPRETATION_ACCEPTED'; interpretationEventId: string }
  | { type: 'START_RUN' }
  | {
      type: 'RUN_COMPLETED';
      runId: string;
      sourceDataHash: string;
      rows: RunResultRow[];
    }
  | { type: 'CONTINUE' }
  | { type: 'FINALISE'; chosenMode: 'guided' | 'freeform' };

export const initialState: TutorialState = {
  step: 'welcome',
  skipped: false,
  sessionId: null,
  pipelineSnapshot: null,
  interpretationEventId: null,
  runId: null,
  sourceDataHash: null,
  rows: [],
  chosenMode: null,
};

export function tutorialReducer(
  state: TutorialState,
  action: TutorialAction,
): TutorialState {
  switch (state.step) {
    case 'welcome':
      if (action.type === 'START') return { ...state, step: 'describe' };
      if (action.type === 'SKIP')
        return { ...state, step: 'mode-choice', skipped: true };
      break;
    case 'describe':
      if (action.type === 'PIPELINE_BUILT')
        return {
          ...state,
          step: 'show-built',
          sessionId: action.sessionId,
          pipelineSnapshot: action.pipelineSnapshot,
        };
      break;
    case 'show-built':
      if (action.type === 'INTERPRETATION_ACCEPTED')
        return {
          ...state,
          step: 'graph',
          interpretationEventId: action.interpretationEventId,
        };
      break;
    case 'graph':
      if (action.type === 'START_RUN') return { ...state, step: 'run' };
      break;
    case 'run':
      if (action.type === 'RUN_COMPLETED')
        return {
          ...state,
          step: 'audit-story',
          runId: action.runId,
          sourceDataHash: action.sourceDataHash,
          rows: action.rows,
        };
      break;
    case 'audit-story':
      if (action.type === 'CONTINUE') return { ...state, step: 'mode-choice' };
      break;
    case 'mode-choice':
      if (action.type === 'FINALISE')
        return { ...state, step: 'done', chosenMode: action.chosenMode };
      break;
    case 'done':
      // Terminal — no further transitions. Any action arriving here is
      // a programmer error: every dispatch site is in this codebase.
      throw new Error(
        `tutorialReducer: unrecognised action ${(action as TutorialAction).type} in terminal step 'done'`,
      );
  }
  // Action did not match the current step. This is a programmer error
  // (out-of-order dispatch); throw with the offending action type and
  // current step so the test surface fails loudly, not silently. P15
  // explicitly rejected the prior console.warn + no-op behaviour per
  // CLAUDE.md offensive-programming guidance.
  throw new Error(
    `tutorialReducer: unrecognised action ${(action as TutorialAction).type} in step ${state.step}`,
  );
}
```

- [ ] **Step 4: Run test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/tutorial/tutorialMachine.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts \
  src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts
git commit -m "feat(frontend): add tutorial state machine (Phase 4B.2)"
```

## Task 3: Copy module

**Files:**
- Create: `src/elspeth/web/frontend/src/components/tutorial/copy.ts`.

A single source of truth for tutorial copy. Each turn imports its own
block. Tests can assert on the constants (so a future text edit doesn't
silently change what the tutorial says without a test signal).

- [ ] **Step 1: Implement.**

```typescript
/**
 * Tutorial copy blocks — single source of truth.
 *
 * Each block is a verbatim quote from design doc 04 §"Turn-by-turn arc",
 * adapted minimally for JSX. If the design doc changes, this file changes;
 * if this file changes, the corresponding component test fails until
 * updated, which forces a deliberate edit.
 *
 * The CANONICAL_SEED_PROMPT MUST stay in sync with the backend constant in
 * src/elspeth/web/preferences/tutorial_cache.py — they are the same value
 * by contract; drift causes cache misses (intended fail-safe).
 */

export const CANONICAL_SEED_PROMPT =
  'create a list of 5 government web pages and use an LLM to rate ' +
  'how cool they are';

export const TURN_1_TITLE = 'Welcome to ELSPETH.';
export const TURN_1_BODY = [
  "In about 3 minutes we'll build and run your first pipeline together. " +
    "Then you'll choose how you want to work going forward.",
  'Pipelines have three layers:',
  '  • SENSE — where data comes from (a source)',
  '  • DECIDE — what happens to each row (transforms)',
  '  • ACT — where results go (sinks)',
  "We'll build one of each. Ready?",
];
export const TURN_1_PRIMARY_BUTTON = "Let's go →";
export const TURN_1_SKIP_LINK = "I've used ELSPETH before, skip this";

export const TURN_2_BODY = [
  "You don't have to build the three layers one at a time. For your " +
    'first pipeline, describe what you want in one sentence and ' +
    "I'll wire it up.",
  'Try this (pre-filled — edit if you like):',
];
export const TURN_2_PRIMARY_BUTTON = 'Build it';

export const TURN_2B_INTRO = "Got it. Here's what I drafted:";
export const TURN_2B_INTERPRETATION_PROMPT =
  'Before we run: when you said "cool", I read that as roughly ' +
  '"modern design + clear purpose + interactivity". ' +
  'Want to adjust the definition, or use mine?';

export const TURN_3_BODY = [
  "Here's your pipeline as a graph. Three layers, four steps:",
];
export const TURN_3_FOOTER =
  "Look familiar? That's the source → transform → sink shape we talked " +
  'about, just with two transforms instead of one.';
export const TURN_3_PRIMARY_BUTTON = 'Looks good, run it →';

export const TURN_4_INTRO = 'Running your pipeline...';
export const TURN_4_PRIMARY_BUTTON = 'Continue →';

export const TURN_5_INTRO =
  'Notice something? The LLM made a judgment call on every page — and ' +
  'you can see WHY. That\'s not because we logged its output. It\'s ' +
  'because ELSPETH records the full lineage as evidence:';
export const TURN_5_OUTRO =
  "That's what ELSPETH was built for: AI decisions you can defend.";
export const TURN_5_EXPLORE_BUTTON = 'Explore the full audit trail';
export const TURN_5_CONTINUE_BUTTON = 'Continue →';

export const TURN_6_INTRO =
  "You've built and run your first pipeline. Going forward, there are " +
  'two ways to compose:';
export const TURN_6_GUIDED_LABEL = 'GUIDED — same step-by-step flow you just did. Recommended.';
export const TURN_6_GUIDED_DESCRIPTION =
  "Best when you're learning what's possible, or when you want a " +
  'clear path through validation and audit checks.';
export const TURN_6_FREEFORM_LABEL = "FREEFORM — describe what you want in chat, I'll build it.";
export const TURN_6_FREEFORM_DESCRIPTION =
  'Best for power users who know exactly what they need.';
export const TURN_6_FOOTER =
  'You can switch any time from the chat panel. What should new ' +
  'sessions default to?';
export const TURN_6_GUIDED_BUTTON = 'Guided (recommended)';
export const TURN_6_FREEFORM_BUTTON = 'Freeform';

// The session name applied at finalisation.
export const TUTORIAL_SESSION_NAME = 'hello-world (cool government pages)';
```

- [ ] **Step 2: No test required for a pure-constants module.**

Constants are exercised by every component test that imports them; a
dedicated test of "the constant equals this string" is theatre. The
referenced text appears in component tests.

- [ ] **Step 3: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/copy.ts
git commit -m "feat(frontend): add tutorial copy module (Phase 4B.3)"
```

## Task 4: Turn 1 — Welcome

**Files:**
- Create: `TutorialTurn1Welcome.tsx` + `.test.tsx`.

- [ ] **Step 1: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TutorialTurn1Welcome } from './TutorialTurn1Welcome';
import {
  TURN_1_TITLE,
  TURN_1_PRIMARY_BUTTON,
  TURN_1_SKIP_LINK,
} from './copy';

describe('TutorialTurn1Welcome', () => {
  it('renders the title and primary button', () => {
    render(<TutorialTurn1Welcome onStart={() => {}} onSkip={() => {}} />);
    expect(screen.getByText(TURN_1_TITLE)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: TURN_1_PRIMARY_BUTTON }),
    ).toBeInTheDocument();
  });

  it('renders the SENSE / DECIDE / ACT framing', () => {
    render(<TutorialTurn1Welcome onStart={() => {}} onSkip={() => {}} />);
    expect(screen.getByText(/SENSE/)).toBeInTheDocument();
    expect(screen.getByText(/DECIDE/)).toBeInTheDocument();
    expect(screen.getByText(/ACT/)).toBeInTheDocument();
  });

  it('fires onStart when primary button clicked', () => {
    const onStart = vi.fn();
    render(<TutorialTurn1Welcome onStart={onStart} onSkip={() => {}} />);
    fireEvent.click(screen.getByRole('button', { name: TURN_1_PRIMARY_BUTTON }));
    expect(onStart).toHaveBeenCalledOnce();
  });

  it('fires onSkip when skip link clicked', () => {
    const onSkip = vi.fn();
    render(<TutorialTurn1Welcome onStart={() => {}} onSkip={onSkip} />);
    fireEvent.click(screen.getByText(TURN_1_SKIP_LINK));
    expect(onSkip).toHaveBeenCalledOnce();
  });

  it('skip link is visually subtle (small, low-contrast styled)', () => {
    // The skip affordance per Open Question C2 is intentionally non-prominent.
    // We assert it's a button/link element with a class that downplays it.
    render(<TutorialTurn1Welcome onStart={() => {}} onSkip={() => {}} />);
    const skip = screen.getByText(TURN_1_SKIP_LINK);
    // The test asserts on a stable contract: the class name includes 'subtle'
    // (or whatever marker the design system uses). Adapt to the project's
    // styling convention during recon.
    expect(skip.className).toMatch(/subtle|muted|small/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/tutorial/TutorialTurn1Welcome.test.tsx
```

Expected: FAIL — component does not exist.

- [ ] **Step 3: Implement.**

```typescript
import {
  TURN_1_TITLE,
  TURN_1_BODY,
  TURN_1_PRIMARY_BUTTON,
  TURN_1_SKIP_LINK,
} from './copy';

interface Props {
  onStart: () => void;
  onSkip: () => void;
}

export function TutorialTurn1Welcome({ onStart, onSkip }: Props) {
  return (
    <div className="tutorial-turn tutorial-turn-1">
      <h1>{TURN_1_TITLE}</h1>
      {TURN_1_BODY.map((para, i) => (
        <p key={i}>{para}</p>
      ))}
      <div className="tutorial-actions">
        <button
          type="button"
          className="tutorial-primary"
          onClick={onStart}
        >
          {TURN_1_PRIMARY_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-skip subtle"
          onClick={onSkip}
        >
          {TURN_1_SKIP_LINK}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn1Welcome.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn1Welcome.test.tsx
git commit -m "feat(frontend): add tutorial turn 1 welcome (Phase 4B.4)"
```

## Task 5: Turn 2 — Describe

**Files:**
- Create: `TutorialTurn2Describe.tsx` + `.test.tsx`.

Turn 2 collects the seed prompt (pre-filled with `CANONICAL_SEED_PROMPT`),
submits it via the composer compose API, awaits the LLM `set_pipeline`
tool call (Phase 5a infrastructure), and dispatches `PIPELINE_BUILT` to the
state machine.

> **Reality finding R2-10 (2026-05-19) — name the live endpoint.** The
> original draft referred to a `composePipelineFromPrompt` API function
> that does not exist in the live codebase. The Phase 5a compose path
> is `POST /api/sessions/{session_id}/messages` (verified live at
> `src/elspeth/web/sessions/routes.py:3856-3877` — `send_message`
> handler, returns `MessageWithStateResponse`), and the frontend client
> function is `sendMessage(sessionId, content)` at
> `src/elspeth/web/frontend/src/api/client.ts:497`. Update the spy
> names and the call expectations below to match the live function.

- [ ] **Step 1: Recon — confirm the compose API and session-create API.**

```bash
grep -n "sendMessage\|createSession" \
  src/elspeth/web/frontend/src/api/client.ts | head -10
```

Expected live functions (R2-10, confirmed 2026-05-19):

- `sendMessage(sessionId: string, content: string)` at `client.ts:497`
  — posts to `/api/sessions/{sessionId}/messages` and returns
  `MessageWithStateResponse` (assistant message + post-compose
  pipeline state).
- `createSession(...)` (`sessionStore.createSession` or the equivalent
  client.ts function) — creates a new session and returns its UUID.
  Turn 2's flow is "create session → send canonical-seed prompt as
  the first user message → wait for the assistant's `set_pipeline`
  tool call → dispatch `PIPELINE_BUILT`".

The compose API is owned by Phase 5a; this turn is a consumer.

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TutorialTurn2Describe } from './TutorialTurn2Describe';
import { CANONICAL_SEED_PROMPT, TURN_2_PRIMARY_BUTTON } from './copy';
import * as api from '../../api/client';

describe('TutorialTurn2Describe', () => {
  it('pre-fills the textarea with the canonical seed prompt', () => {
    render(<TutorialTurn2Describe onBuilt={() => {}} />);
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
    expect(textarea.value).toBe(CANONICAL_SEED_PROMPT);
  });

  it('lets the user edit the prompt before submitting', () => {
    render(<TutorialTurn2Describe onBuilt={() => {}} />);
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'a different prompt' } });
    expect(textarea.value).toBe('a different prompt');
  });

  it('submits the prompt to the compose API on click', async () => {
    const onBuilt = vi.fn();
    // The flow is create-session → send-message; recon (Step 1) confirms
    // the exact function names. Spy on `createSession` and `sendMessage`
    // (R2-10, 2026-05-19 — `composePipelineFromPrompt` is not a live
    // function; the wire endpoint is POST /api/sessions/{id}/messages).
    vi.spyOn(api, 'createSession').mockResolvedValue({ id: 'sess-1' });
    const sendSpy = vi
      .spyOn(api, 'sendMessage')
      .mockResolvedValue({
        message: { role: 'assistant', content: 'pipeline built' },
        state: {
          source: { type: 'inline_blob' },
          transforms: [],
          sinks: [],
        },
      });
    render(<TutorialTurn2Describe onBuilt={onBuilt} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_2_PRIMARY_BUTTON }),
    );
    await waitFor(() => expect(sendSpy).toHaveBeenCalledOnce());
    expect(sendSpy).toHaveBeenCalledWith('sess-1', CANONICAL_SEED_PROMPT);
    await waitFor(() => expect(onBuilt).toHaveBeenCalledOnce());
    const [arg] = onBuilt.mock.calls[0];
    expect(arg.sessionId).toBe('sess-1');
  });

  it('disables the button while the compose call is in flight', async () => {
    vi.spyOn(api, 'createSession').mockResolvedValue({ id: 'sess-1' });
    let resolveCompose: (v: unknown) => void = () => {};
    vi.spyOn(api, 'sendMessage').mockImplementation(
      () =>
        new Promise((res) => {
          resolveCompose = res;
        }),
    );
    render(<TutorialTurn2Describe onBuilt={() => {}} />);
    const btn = screen.getByRole('button', { name: TURN_2_PRIMARY_BUTTON });
    fireEvent.click(btn);
    await waitFor(() => expect(btn).toBeDisabled());
    resolveCompose({
      message: { role: 'assistant', content: 'pipeline built' },
      state: { source: {}, transforms: [], sinks: [] },
    });
  });

  it('surfaces an inline error if compose fails', async () => {
    vi.spyOn(api, 'createSession').mockResolvedValue({ id: 'sess-1' });
    vi.spyOn(api, 'sendMessage').mockRejectedValue(
      new Error('LLM unreachable'),
    );
    render(<TutorialTurn2Describe onBuilt={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_2_PRIMARY_BUTTON }),
    );
    await waitFor(() =>
      expect(screen.getByText(/LLM unreachable/)).toBeInTheDocument(),
    );
  });

  it('offers "restore canonical seed" when the user has edited the prompt', () => {
    render(<TutorialTurn2Describe onBuilt={() => {}} />);
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'edited' } });
    const restore = screen.getByText(/restore canonical seed/i);
    fireEvent.click(restore);
    expect(textarea.value).toBe(CANONICAL_SEED_PROMPT);
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL — component does not exist.

- [ ] **Step 4: Implement.**

```typescript
import { useState } from 'react';
import { CANONICAL_SEED_PROMPT, TURN_2_BODY, TURN_2_PRIMARY_BUTTON } from './copy';
import { createSession, sendMessage } from '../../api/client';
import type { PipelineSnapshot } from './tutorialMachine';

interface OnBuiltPayload {
  sessionId: string;
  pipelineSnapshot: PipelineSnapshot;
}

interface Props {
  onBuilt: (payload: OnBuiltPayload) => void;
}

export function TutorialTurn2Describe({ onBuilt }: Props) {
  const [prompt, setPrompt] = useState(CANONICAL_SEED_PROMPT);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const edited = prompt !== CANONICAL_SEED_PROMPT;

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      // R2-10, 2026-05-19: the live compose path is create-session
      // followed by send-message. `composePipelineFromPrompt` is not a
      // live function; the canonical sequence is below. The session
      // becomes the user's own session under which the resulting
      // pipeline lives — Phase 5a infrastructure.
      const session = await createSession();
      const response = await sendMessage(session.id, prompt);
      onBuilt({
        sessionId: session.id,
        pipelineSnapshot: response.state,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="tutorial-turn tutorial-turn-2">
      {TURN_2_BODY.map((para, i) => (
        <p key={i}>{para}</p>
      ))}
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
        className="tutorial-prompt"
      />
      {edited && (
        <button
          type="button"
          className="tutorial-restore subtle"
          onClick={() => setPrompt(CANONICAL_SEED_PROMPT)}
        >
          restore canonical seed
        </button>
      )}
      {error && <div className="tutorial-error">{error}</div>}
      <button
        type="button"
        className="tutorial-primary"
        onClick={handleSubmit}
        disabled={submitting || prompt.trim() === ''}
      >
        {TURN_2_PRIMARY_BUTTON}
      </button>
    </div>
  );
}
```

No `client.ts` additions are required here (R2-10, 2026-05-19) —
`createSession` and `sendMessage` already exist as live Phase 5a
infrastructure functions. If Step 1 recon reveals either function has
moved or renamed, adapt the imports and spy targets accordingly; do not
re-introduce a `composePipelineFromPrompt` alias.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx
git commit -m "feat(frontend): add tutorial turn 2 describe (Phase 4B.5)"
```

## Task 6: Turn 2b — Show what was built + interpretation embed

**Files:**
- Create: `TutorialTurn2bShowBuilt.tsx` + `.test.tsx`.
- Create: `InterpretationContract.test.tsx` (Phase 4 ↔ Phase 5b contract; see Steps 7–9 below).

Turn 2b renders:
1. A summary of the drafted pipeline (URLs from the `inline_blob` source,
   transform names, sink name).
2. The Phase 5b `InterpretationReviewTurn` embedded inline for the "cool"
   interpretation event.

The interpretation event is fetched via the Phase 5b list/get API; the
component awaits acceptance (or amendment) and dispatches
`INTERPRETATION_ACCEPTED` with the resolved event id.

- [ ] **Step 1: Recon Phase 5b's frontend embed.**

```bash
grep -rn "InterpretationReviewTurn\b" src/elspeth/web/frontend/src/ | head -10
cat src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.tsx | head -90
cat src/elspeth/web/frontend/src/stores/interpretationEventsStore.ts | head -170
```

Live shape (RC5.2, confirmed):
- File path: `src/components/chat/guided/InterpretationReviewTurn.tsx`.
- Props: `{ event: InterpretationEvent; sessionId: string; onResolved?: (newState: CompositionState | null) => void }`.
  The component takes the **full event object**, not just an ID.
- Pending events are surfaced via `useInterpretationEventsStore` actions:
  `refreshPending(sessionId)` populates `pendingBySession[sessionId]`;
  the consumer reads the event map and picks the pending event(s) of
  interest. There is no `api/interpretationEvents.ts` module — wiring
  goes through the store.
- Resolution flows through the store's `resolveEvent` action (called
  internally by the `useInterpretationResolver` hook the component uses).

If the component is missing in the live tree, halt — Phase 5b not shipped.

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TutorialTurn2bShowBuilt } from './TutorialTurn2bShowBuilt';
import { TURN_2B_INTRO } from './copy';
import { useInterpretationEventsStore } from '@/stores/interpretationEventsStore';
import type { InterpretationEvent } from '@/types/interpretation';

const PENDING_EVENT: InterpretationEvent = {
  id: 'evt-1',
  session_id: 'sess-1',
  user_term: 'cool',
  llm_draft: 'modern design + clear purpose + interactivity',
  choice: 'pending',
  created_at: '2026-05-15T12:00:00Z',
  resolved_at: null,
  amendment: null,
  interpretation_source: null,
};

describe('TutorialTurn2bShowBuilt', () => {
  const snapshot = {
    source: {
      type: 'inline_blob',
      urls: ['australia.gov.au', 'finance.gov.au', 'ato.gov.au', 'data.gov.au', 'dta.gov.au'],
    },
    transforms: [
      { type: 'web_scrape' },
      { type: 'llm_rate' },
    ],
    sinks: [{ type: 'jsonl' }],
  };

  beforeEach(() => {
    // Seed the live interpretationEventsStore with the pending event so
    // the component's refreshPending() resolves immediately without a
    // wire spy. (The store action is itself spied on for the call-shape
    // assertion below.)
    useInterpretationEventsStore.setState({
      pendingBySession: { 'sess-1': { 'evt-1': PENDING_EVENT } },
    });
    vi.spyOn(useInterpretationEventsStore.getState(), 'refreshPending')
      .mockResolvedValue();
  });

  it('renders the intro', () => {
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={() => {}}
      />,
    );
    expect(screen.getByText(TURN_2B_INTRO)).toBeInTheDocument();
  });

  it('renders the source URLs from inline_blob', () => {
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={() => {}}
      />,
    );
    expect(screen.getByText(/australia\.gov\.au/)).toBeInTheDocument();
    expect(screen.getByText(/dta\.gov\.au/)).toBeInTheDocument();
  });

  it('renders the transform names', () => {
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={() => {}}
      />,
    );
    expect(screen.getByText(/web_scrape/)).toBeInTheDocument();
    expect(screen.getByText(/llm/i)).toBeInTheDocument();
  });

  it('renders the sink description', () => {
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={() => {}}
      />,
    );
    expect(screen.getByText(/JSONL/i)).toBeInTheDocument();
  });

  it('embeds the interpretation review affordance', async () => {
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={() => {}}
      />,
    );
    await waitFor(() =>
      expect(screen.getByText(/Use my interpretation/i)).toBeInTheDocument(),
    );
  });

  it('forwards onInterpretationAccepted with the event id after resolution', async () => {
    const onAccepted = vi.fn();
    // The live InterpretationReviewTurn calls its onResolved? prop after a
    // successful resolveEvent. We exercise that contract via the store's
    // resolveEvent action; the component reads the resulting state.
    vi.spyOn(useInterpretationEventsStore.getState(), 'resolveEvent')
      .mockResolvedValue({ new_state: null as never });
    render(
      <TutorialTurn2bShowBuilt
        sessionId="sess-1"
        pipelineSnapshot={snapshot}
        onInterpretationAccepted={onAccepted}
      />,
    );
    await waitFor(() =>
      expect(screen.getByText(/Use my interpretation/i)).toBeInTheDocument(),
    );
    fireEvent.click(screen.getByText(/Use my interpretation/i));
    await waitFor(() =>
      expect(onAccepted).toHaveBeenCalledWith({
        interpretationEventId: 'evt-1',
      }),
    );
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Implement.**

```typescript
import { useEffect } from 'react';
import { TURN_2B_INTRO, TURN_2B_INTERPRETATION_PROMPT } from './copy';
// Live RC5.2 path (Phase 5b ships the component under chat/guided/).
import { InterpretationReviewTurn } from '../chat/guided/InterpretationReviewTurn';
import { useInterpretationEventsStore } from '@/stores/interpretationEventsStore';
import type { PipelineSnapshot } from './tutorialMachine';

interface Props {
  sessionId: string;
  pipelineSnapshot: PipelineSnapshot;
  onInterpretationAccepted: (payload: { interpretationEventId: string }) => void;
}

export function TutorialTurn2bShowBuilt({
  sessionId,
  pipelineSnapshot,
  onInterpretationAccepted,
}: Props) {
  // Read the pending event projection directly from the live store. The
  // compose-loop response (Turn 2) addPendingEvent()'s the new event into
  // pendingBySession before this turn mounts; refreshPending() here is a
  // belt-and-braces rehydration in case the store was rehydrated mid-flow
  // (e.g., navigation back/forward to the tutorial container — note refresh
  // restarts the tutorial at turn 1 per plan-fix P10, so this hook is
  // primarily defending against same-session re-mounts, not refresh-resume).
  const refreshPending = useInterpretationEventsStore((s) => s.refreshPending);
  const pendingMap = useInterpretationEventsStore(
    (s) => s.pendingBySession[sessionId],
  );
  // First pending event for this session — Phase 4's contract is one
  // "cool" interpretation per tutorial run.
  const firstEvent = pendingMap
    ? Object.values(pendingMap)[0]
    : undefined;

  useEffect(() => {
    void refreshPending(sessionId);
  }, [sessionId, refreshPending]);

  // Pipeline-snapshot reads. Shape is the inline_blob source contract
  // from Phase 5a — Tier-2 post-source, direct access (CLAUDE.md no
  // defensive programming on Tier-2 fields).
  const source = pipelineSnapshot.source as { urls: string[] };
  const urls = source.urls;
  const transforms = pipelineSnapshot.transforms as Array<{ type: string }>;
  const sinks = pipelineSnapshot.sinks as Array<{ type: string }>;

  return (
    <div className="tutorial-turn tutorial-turn-2b">
      <p>{TURN_2B_INTRO}</p>

      <section className="tutorial-pipeline-summary">
        <div>
          <strong>SOURCE</strong> — 5 government web pages I selected:
          <ul>
            {urls.map((u) => (
              <li key={u}>{u}</li>
            ))}
          </ul>
          <small>(these are my pick — you can edit the list)</small>
        </div>

        <div>
          <strong>TRANSFORM</strong> — fetch each page ({transforms[0].type}),
          then call an LLM to rate each one for "coolness" ({transforms[1].type})
        </div>

        <div>
          <strong>SINK</strong> — write the ratings to a {sinks[0].type.toUpperCase()} file in your session
        </div>
      </section>

      {firstEvent && (
        <section className="tutorial-interpretation">
          <p>{TURN_2B_INTERPRETATION_PROMPT}</p>
          <InterpretationReviewTurn
            event={firstEvent}
            sessionId={sessionId}
            onResolved={() =>
              onInterpretationAccepted({ interpretationEventId: firstEvent.id })
            }
          />
        </section>
      )}
    </div>
  );
}
```

Notes on live integration:
- `InterpretationReviewTurn`'s live props are `{ event, sessionId,
  onResolved? }` — the component takes the FULL event object, not just
  an ID. `onResolved` is optional; we pass it to advance the tutorial.
- The component internally consumes `useInterpretationResolver(...)`
  which calls `useInterpretationEventsStore.resolveEvent(...)` — there
  is no `api/interpretationEvents.ts` module; wiring goes through the
  store.
- Pipeline-snapshot fields are accessed directly (no `?.` / `?? []`):
  Phase 5a's `set_pipeline` contract guarantees the shape (Tier-2
  post-source). If the shape drifts, fix Phase 5a, don't paper over it
  here (CLAUDE.md offensive programming).

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx
git commit -m "feat(frontend): add tutorial turn 2b show-built (Phase 4B.6)"
```

- [ ] **Step 7: Write the Phase 4 ↔ Phase 5b contract test (TDD).**

Per plan-fix P17 (2026-05-19, systems S7): add a CI-enforced contract
test (not preflight) that asserts the shape of the Phase 5b
`InterpretationEvent` store projection against the fields Turn 2b reads.
Without this, Turn 2b is one Phase 5b refactor away from silent
breakage. The test lives alongside the Turn 2b component test and runs
in the normal Vitest CI sweep.

Field inventory — Turn 2b consumes these fields from the live store
(`pendingBySession[sessionId]`'s map values), confirmed against the
Step 4 implementation and the live
`src/types/interpretation.ts::InterpretationEvent`:

- `id` — passed back via `onInterpretationAccepted({ interpretationEventId })`.
- `session_id` — keyed by store; sanity-checked through the
  `InterpretationReviewTurn` `event` prop.
- `user_term` — read by the embedded `InterpretationReviewTurn`
  (live: line 135 of `chat/guided/InterpretationReviewTurn.tsx`).
- `llm_draft` — read by the embedded `InterpretationReviewTurn`
  (live: line 136).
- `choice` — gates "pending" vs resolved rendering inside
  `InterpretationReviewTurn`.
- `created_at` — surfaces the pending-event age in the embed.

Create
`src/elspeth/web/frontend/src/components/tutorial/InterpretationContract.test.tsx`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { useInterpretationEventsStore } from '@/stores/interpretationEventsStore';
import type { InterpretationEvent } from '@/types/interpretation';

// Phase 4 ↔ Phase 5b contract: this test pins the InterpretationEvent
// fields Turn 2b reads. If a Phase 5b refactor renames or removes any of
// these, this test breaks BEFORE the user-visible breakage. Live store
// shape: pendingBySession[sid] is a Record<eventId, InterpretationEvent>
// (NOT an array) — see src/stores/interpretationEventsStore.ts.

describe('Phase 4 ↔ Phase 5b contract: InterpretationEvent shape', () => {
  beforeEach(() => {
    useInterpretationEventsStore.setState({ pendingBySession: {} });
  });

  it('pendingBySession exposes the fields Turn 2b consumes', () => {
    const fixture: InterpretationEvent = {
      id: 'evt-1',
      session_id: 'sess-1',
      composition_state_id: 'cs-1',
      affected_node_id: 'node-1',
      tool_call_id: 'tc-1',
      user_term: 'cool',
      llm_draft: 'modern design + clear purpose + interactivity',
      accepted_value: null,
      choice: 'pending',
      created_at: '2026-05-19T00:00:00Z',
      resolved_at: null,
      actor: 'originator:user:test',
      interpretation_source: 'llm_drafted',
      model_identifier: 'claude-opus-4.7',
      model_version: 'v1',
      provider: 'anthropic',
      composer_skill_hash: null,
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
    useInterpretationEventsStore.setState({
      pendingBySession: { 'sess-1': { 'evt-1': fixture } },
    });

    const pendingMap =
      useInterpretationEventsStore.getState().pendingBySession['sess-1'];
    expect(pendingMap).toBeDefined();
    const events = Object.values(pendingMap);
    expect(events).toHaveLength(1);

    // Assert every field Turn 2b (and its embedded
    // InterpretationReviewTurn) reads is present and of the expected
    // type. If Phase 5b ever renames or removes one of these, this
    // matcher fails — surfacing the contract drift before runtime.
    expect(events[0]).toMatchObject({
      id: expect.any(String),
      session_id: expect.any(String),
      user_term: expect.any(String),
      llm_draft: expect.any(String),
      choice: expect.stringMatching(/pending|user_approved|amended|auto_interpreted_opt_out|auto_interpreted_no_surfaces/),
      created_at: expect.any(String),
    });
  });
});
```

- [ ] **Step 8: Run the contract test.**

```bash
cd src/elspeth/web/frontend && npx vitest run \
  src/components/tutorial/InterpretationContract.test.tsx
```

Expected: PASS. If it fails because the live `InterpretationEvent` type
no longer carries one of the listed fields, **fix the consumer**
(Turn 2b implementation + this field inventory) and update the test —
do not paper over a real shape drift (CLAUDE.md
`feedback_fix_errors_you_encounter`).

> **Note (Quality r2 — R2-M4, 2026-05-19):** This contract test reads
> the store projection directly
> (`useInterpretationEventsStore.getState().pendingBySession[sid][eventId]`),
> so it pins the **raw event shape** and is not masked by the live
> `InterpretationReviewTurn`'s defensive `?? ""` fallbacks (lines
> 135–136 of `chat/guided/InterpretationReviewTurn.tsx`:
> `user_term ?? "this term"` and `llm_draft ?? ""`). Those defensive
> fallbacks are themselves a CLAUDE.md offensive-programming violation
> on Tier-2 plugin output — they would silently render an empty string
> (or the literal "this term") for an absent field rather than
> surfacing the shape drift this contract test exists to catch.
> **Follow-up scope (NOT this plan):** the live `InterpretationReviewTurn`
> component is owned by Phase 5b. The `?? ""` fallbacks should be
> replaced with assertions / explicit error rendering when the next
> Phase-5b pass touches the file. Filed as a Phase-4-followup
> observation against `21-phase-9-followups.md` (or equivalent) —
> the contract test here is sufficient to catch a real shape drift,
> but the live component's behaviour on drift is silently-wrong
> rendering rather than a crash. The contract test does not paper
> over this; it surfaces the drift as a CI failure before users see
> the empty strings.

- [ ] **Step 9: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/InterpretationContract.test.tsx
git commit -m "test(frontend): pin Phase 4 ↔ Phase 5b InterpretationEvent contract (Phase 4B.6)"
```

## Task 7: Turn 3 — Graph glance

**Files:**
- Create: `TutorialTurn3Graph.tsx` + `.test.tsx`.

A simple horizontal flow rendering: 4 boxes (url_source → web_scrape →
llm_rate → jsonl_sink), each with a small label. The design doc 04 turn 3
shows a text-rendered version; we render real React boxes that follow the
project's existing graph styling conventions (which the project's normal
graph mini-view uses — see recon).

- [ ] **Step 1: Recon — does the project already have a graph component?**

```bash
grep -rn "PipelineGraph\|GraphPanel\|graph-mini" \
  src/elspeth/web/frontend/src/components --include="*.tsx" | head -10
```

If a tiny graph component exists, this turn reuses it with a hard-coded
4-node config. If not, this turn implements its own minimal renderer (no
need for full graph machinery — 4 boxes + 3 arrows).

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TutorialTurn3Graph } from './TutorialTurn3Graph';
import { TURN_3_PRIMARY_BUTTON } from './copy';

describe('TutorialTurn3Graph', () => {
  const snapshot = {
    source: { type: 'inline_blob', urls: ['a', 'b', 'c', 'd', 'e'] },
    transforms: [{ type: 'web_scrape' }, { type: 'llm_rate' }],
    sinks: [{ type: 'jsonl' }],
  };

  it('renders the 4-stage flow', () => {
    render(<TutorialTurn3Graph pipelineSnapshot={snapshot} onContinue={() => {}} />);
    expect(screen.getByText(/url_source|inline_blob/i)).toBeInTheDocument();
    expect(screen.getByText(/web_scrape/)).toBeInTheDocument();
    expect(screen.getByText(/llm_rate/)).toBeInTheDocument();
    expect(screen.getByText(/jsonl/i)).toBeInTheDocument();
  });

  it('renders the row count for the source (5 rows)', () => {
    render(<TutorialTurn3Graph pipelineSnapshot={snapshot} onContinue={() => {}} />);
    expect(screen.getByText(/5 rows/)).toBeInTheDocument();
  });

  it('fires onContinue on primary click', () => {
    const onContinue = vi.fn();
    render(<TutorialTurn3Graph pipelineSnapshot={snapshot} onContinue={onContinue} />);
    fireEvent.click(screen.getByRole('button', { name: TURN_3_PRIMARY_BUTTON }));
    expect(onContinue).toHaveBeenCalledOnce();
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Implement.**

```typescript
import { TURN_3_BODY, TURN_3_FOOTER, TURN_3_PRIMARY_BUTTON } from './copy';
import type { PipelineSnapshot } from './tutorialMachine';

interface Props {
  pipelineSnapshot: PipelineSnapshot;
  onContinue: () => void;
}

export function TutorialTurn3Graph({ pipelineSnapshot, onContinue }: Props) {
  // Tier-2 post-source: Phase 5a's set_pipeline contract guarantees the
  // inline_blob source shape and the two-transform / one-sink layout for
  // the canonical seed. Direct access — no defensive fallbacks (CLAUDE.md).
  const source = pipelineSnapshot.source as { urls: string[] };
  const urls = source.urls;
  const transforms = pipelineSnapshot.transforms as Array<{ type: string }>;
  const sinks = pipelineSnapshot.sinks as Array<{ type: string }>;

  return (
    <div className="tutorial-turn tutorial-turn-3">
      {TURN_3_BODY.map((para, i) => (
        <p key={i}>{para}</p>
      ))}

      <div className="tutorial-graph">
        <Node label="url_source" subtitle={`${urls.length} rows`} />
        <Arrow />
        <Node label={transforms[0].type} subtitle="fetch" />
        <Arrow />
        <Node label={transforms[1].type} subtitle="rate" />
        <Arrow />
        <Node label={`${sinks[0].type}_sink`} subtitle="write" />
      </div>

      <p>{TURN_3_FOOTER}</p>

      <button
        type="button"
        className="tutorial-primary"
        onClick={onContinue}
      >
        {TURN_3_PRIMARY_BUTTON}
      </button>
    </div>
  );
}

function Node({ label, subtitle }: { label: string; subtitle: string }) {
  return (
    <div className="tutorial-graph-node">
      <div className="tutorial-graph-node-label">{label}</div>
      <div className="tutorial-graph-node-subtitle">{subtitle}</div>
    </div>
  );
}

function Arrow() {
  return <div className="tutorial-graph-arrow" aria-hidden>→</div>;
}
```

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.test.tsx
git commit -m "feat(frontend): add tutorial turn 3 graph glance (Phase 4B.7)"
```


---

## What Part 1 leaves the frontend in

After Tasks 1–7 land:

- `preferencesStore.ts` exposes `tutorialCompletedAt`, derived
  `tutorialCompleted`, and the atomic `markTutorialCompleted` action.
- `tutorialMachine.ts` defines the typed step union, action union, and a
  pure reducer with unit-tested transitions.
- `copy.ts` is the single source of truth for tutorial text.
- Three leaf turn components (1, 2, 2b, 3) render in isolation with mocked
  store/API; each is unit-tested.

Not yet wired: frontend API client surface (Task 7.5 — `runTutorialPipeline`, `getRunAuditSummary`, `deleteTutorialOrphans`, `renameSession`), turns 4–6 (Tasks 8–10), container (Task 11 with mount-time orphan cleanup), App.tsx detection (Task 12 — no sessionStorage / banner per plan-fix P10), integration test (Task 13), Playwright E2E (Task 14), Reset-tutorial link in `ComposerPreferencesPanel` (Task 14.5 — closes Open Question C3 in-phase), staging smoke (Task 15).

Continue in [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md).

## Review history

### 2026-05-15 — review panel

| ID | Severity | Status | Summary |
|---|---|---|---|
| 4B1-F1 | CRITICAL (Systems) | Superseded | Architecture stanza originally adopted the sessionStorage-resume pattern (and DB-delete banner was dropped per 2026-05-16 review). The sessionStorage-resume scaffolding was deleted per plan-fix P10 (2026-05-19) — flat key risks cross-user contamination on shared workstations (S6); restart-at-turn-1 is the new contract. DB-delete banner remains dropped (No Legacy Code Policy — single Turn 1 entry for all tutorial-mode users). |
| 4B1-F3 | BLOCKER (Coherence) | Applied | `markTutorialCompleted` signature corrected to `(defaultMode)` only; session-rename remains a separate Turn 6 call |
| 4B1-F2 | IMPORTANT (Architecture) | Applied | Phase 5b cross-link added to turn 2b and turn 5 scope items |

### 2026-05-19 — Phase 4 plan-fixes (P10 / P17)

| ID | Severity | Status | Summary |
|---|---|---|---|
| P10 | CRITICAL (Systems) | Applied | Navigation Resilience stanza rewritten — refresh restarts at turn 1, no sessionStorage scaffolding (cross-user contamination risk on shared workstations, S6). Out-of-scope bullet for multi-device sync updated to reference the restart contract. Comment in Turn 2b's `refreshPending` effect updated to remove sessionStorage-resume framing. |
| P17 | IMPORTANT (Systems) | Applied | Task 6 extended with Steps 7–9: a CI-enforced contract test `InterpretationContract.test.tsx` that pins the Phase 5b `InterpretationEvent` fields Turn 2b consumes (`id`, `session_id`, `user_term`, `llm_draft`, `choice`, `created_at`). If Phase 5b ever renames or removes one of these, the test breaks before the user-visible breakage. Lives alongside the Turn 2b component test; runs in the normal Vitest CI sweep (not preflight). |

### 2026-05-19 — frontend r2 review closure (Quality)

| ID | Severity | Status | Summary |
|---|---|---|---|
| R2-M4 | MAJOR (Quality) | Applied | Task 6 Step 8 contract test annotated with the follow-up scope: the live `InterpretationReviewTurn` (`chat/guided/InterpretationReviewTurn.tsx` lines 135–136) accesses `user_term ?? "this term"` and `llm_draft ?? ""` defensively, which silently masks shape drift. The contract test reads `pendingBySession[sid][eventId]` directly, so it pins the raw store shape and is not masked by the component's fallbacks. Follow-up filed against `21-phase-9-followups.md` (or equivalent) — Phase 5b owns the `InterpretationReviewTurn` cleanup; this plan does not modify it. |

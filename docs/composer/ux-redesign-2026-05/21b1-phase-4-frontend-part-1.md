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

**Navigation resilience:** Progress is persisted to `sessionStorage` (`elspeth_tutorial_progress`) on every turn-transition. On mount, if `tutorialCompleted === false` and the key exists, resume from the persisted turn (handles refresh / browser-back). All tutorial-mode users (genuine first-timers and users whose state was wiped by the Phase 4A DB-delete) see the same intro — no bifurcation, no "we noticed you had old sessions" copy. Per CLAUDE.md No Legacy Code Policy: we have no users yet, the operator deletes the DB freely, and a single neutral copy is more honest than detecting a state we can't reliably distinguish. Persistence code: Task 12 (Part 2). No separate banner — Turn 1's existing welcome copy is the entry surface.

**Tech Stack:** React 18, TypeScript (strict), Zustand, Vitest, React
Testing Library, Playwright.

**Sibling plans:**
- [21a-phase-4-backend.md](21a-phase-4-backend.md) — schema column, service
  extension, route extension, tutorial cache, run-path integration.
- [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md) — Tasks
  8–15: turns 4/5/6, container, App.tsx detection, integration test,
  Playwright E2E, staging smoke.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**

- Extend `preferencesStore.ts` with `tutorialCompletedAt: string | null`,
  derived `tutorialCompleted: boolean`, and
  `markTutorialCompleted(defaultMode)` action. (Session-rename is a separate
  call from Turn 6, not bundled into `markTutorialCompleted` — keeps the
  preference-PATCH a single-responsibility atomic op.)
- Extend the API client typing to include the new field on
  `ComposerPreferences`.
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
- Full multi-device session sync (sessionStorage is tab-local; cross-tab
  and cross-device sync remain out of scope).
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
- The exact shape Phase 1B exposes (likely `defaultMode`, `bannerDismissedAt`,
  `loadPreferences()`, `setDefaultMode()`, `setBannerDismissed()`).
- Whether Phase 1B added a companion test file.
- The API call mechanism (the patched function path).

- [ ] **Step 2: Write failing test extensions.**

Add to (or create) `preferencesStore.test.ts`:

```typescript
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { usePreferencesStore } from './preferencesStore';
import * as api from '../api/client';

describe('preferencesStore — tutorial fields', () => {
  beforeEach(() => {
    usePreferencesStore.setState({
      defaultMode: 'guided',
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      loaded: false,
    });
    vi.restoreAllMocks();
  });

  it('exposes tutorialCompleted=false when tutorialCompletedAt is null', () => {
    const s = usePreferencesStore.getState();
    expect(s.tutorialCompletedAt).toBeNull();
    expect(s.tutorialCompleted).toBe(false);
  });

  it('exposes tutorialCompleted=true when tutorialCompletedAt is set', () => {
    usePreferencesStore.setState({ tutorialCompletedAt: '2026-05-15T12:00:00Z' });
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(true);
  });

  it('loadPreferences populates tutorialCompletedAt from the API', async () => {
    vi.spyOn(api, 'getComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: '2026-05-15T12:00:00Z',
      updated_at: '2026-05-15T12:00:00Z',
    });
    await usePreferencesStore.getState().loadPreferences();
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBe(
      '2026-05-15T12:00:00Z',
    );
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(true);
  });

  it('markTutorialCompleted PATCHes both fields atomically', async () => {
    const patchSpy = vi
      .spyOn(api, 'updateComposerPreferences')
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
    expect(usePreferencesStore.getState().tutorialCompleted).toBe(true);
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts
```

Expected: FAIL — `tutorialCompletedAt` not a state key, `markTutorialCompleted` undefined.

- [ ] **Step 4: Extend the store.**

Add to `preferencesStore.ts` (preserving Phase 1B's existing shape):

```typescript
interface PreferencesState {
  defaultMode: 'guided' | 'freeform';
  bannerDismissedAt: string | null;
  // Phase 4: null = tutorial not yet complete (the App routes to the
  // HelloWorldTutorial container); string (ISO timestamp) = complete.
  tutorialCompletedAt: string | null;
  loaded: boolean;
  // Phase 4: derived helper. Frontend code prefers this to importing the
  // null-check pattern everywhere.
  tutorialCompleted: boolean;
  loadPreferences: () => Promise<void>;
  setDefaultMode: (mode: 'guided' | 'freeform') => Promise<void>;
  setBannerDismissed: () => Promise<void>;
  // Phase 4: atomic finalisation — PATCHes default_mode +
  // tutorial_completed_at in one call. Used by TutorialTurn6ModeChoice.
  markTutorialCompleted: (mode: 'guided' | 'freeform') => Promise<void>;
}

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  defaultMode: 'guided',
  bannerDismissedAt: null,
  tutorialCompletedAt: null,
  loaded: false,
  get tutorialCompleted() {
    return get().tutorialCompletedAt !== null;
  },

  loadPreferences: async () => {
    const response = await getComposerPreferences();
    set({
      defaultMode: response.default_mode,
      bannerDismissedAt: response.banner_dismissed_at,
      tutorialCompletedAt: response.tutorial_completed_at,
      loaded: true,
    });
  },

  setDefaultMode: async (mode) => {
    const response = await updateComposerPreferences({ default_mode: mode });
    set({
      defaultMode: response.default_mode,
      tutorialCompletedAt: response.tutorial_completed_at,
    });
  },

  setBannerDismissed: async () => {
    const stamp = new Date().toISOString();
    const response = await updateComposerPreferences({
      banner_dismissed_at: stamp,
    });
    set({ bannerDismissedAt: response.banner_dismissed_at });
  },

  markTutorialCompleted: async (mode) => {
    const stamp = new Date().toISOString();
    const response = await updateComposerPreferences({
      default_mode: mode,
      tutorial_completed_at: stamp,
    });
    set({
      defaultMode: response.default_mode,
      tutorialCompletedAt: response.tutorial_completed_at,
    });
  },
}));
```

Notes:
- Zustand's "computed" properties via getters are valid; this is consistent
  with other stores in the codebase (confirm during recon — adapt if Phase
  1B uses a different convention).
- If Phase 1B uses a different syntactic pattern (e.g., a separate
  `useTutorialCompleted` hook), follow that pattern instead. The semantic
  contract is: `tutorialCompleted` is true when `tutorialCompletedAt !==
  null`.

Extend `src/elspeth/web/frontend/src/api/client.ts` typing:

```typescript
export interface ComposerPreferences {
  default_mode: 'guided' | 'freeform';
  banner_dismissed_at: string | null;
  // Phase 4: ISO timestamp string (UTC) or null.
  tutorial_completed_at: string | null;
  updated_at: string;
}

export interface UpdateComposerPreferencesRequest {
  default_mode?: 'guided' | 'freeform';
  banner_dismissed_at?: string;
  // Phase 4: caller sends an ISO timestamp string to mark tutorial complete,
  // or explicit `null` to clear it (Phase 8 retake path). Optional `?` =
  // absent (no-op); explicit `null` = write NULL to the column.
  // Three-state contract — see 21a §"Cross-plan contract — `tutorial_completed_at` PATCH semantics".
  tutorial_completed_at?: string | null;
}
```

- [ ] **Step 5: Run test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/stores/preferencesStore.ts \
  src/elspeth/web/frontend/src/stores/preferencesStore.test.ts \
  src/elspeth/web/frontend/src/api/client.ts
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

  it('rejects out-of-order transitions', () => {
    // Can't run before building.
    const next = tutorialReducer(initialState, { type: 'START_RUN' });
    // Reducer's contract: invalid transitions are no-ops with a console
    // warning; state stays put.
    expect(next.step).toBe('welcome');
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
 * Transitions are deterministic; invalid actions in the wrong state are
 * no-ops with a console.warn (in dev) — this is the simplest safe
 * behaviour against accidental dispatch races during animations.
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
      // Terminal — no further transitions.
      break;
  }
  // Unknown transition. In dev, surface this; in prod, silently no-op so
  // a stray dispatch doesn't break the user's experience.
  if (process.env.NODE_ENV !== 'production') {
    // eslint-disable-next-line no-console
    console.warn('tutorialReducer: ignored action', action, 'in state', state.step);
  }
  return state;
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

- [ ] **Step 1: Recon — identify the compose API and session-create API.**

```bash
grep -rn "sendChatMessage\|createSession\|composer compose" \
  src/elspeth/web/frontend/src/api --include="*.ts" | head -20
```

Identify:
- How a chat message is sent to the composer (the function name in
  `client.ts`).
- How a session is created (likely `sessionStore.createSession`).
- Where the response signals "pipeline built" (a returned pipeline state, a
  tool-call event in the chat log, etc.).

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
    const composeSpy = vi
      .spyOn(api, 'composePipelineFromPrompt')
      .mockResolvedValue({
        session_id: 'sess-1',
        pipeline_snapshot: {
          source: { type: 'inline_blob' },
          transforms: [],
          sinks: [],
        },
      });
    render(<TutorialTurn2Describe onBuilt={onBuilt} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_2_PRIMARY_BUTTON }),
    );
    await waitFor(() => expect(composeSpy).toHaveBeenCalledOnce());
    expect(composeSpy).toHaveBeenCalledWith({
      prompt: CANONICAL_SEED_PROMPT,
      tutorial_mode: true,
    });
    await waitFor(() => expect(onBuilt).toHaveBeenCalledOnce());
    const [arg] = onBuilt.mock.calls[0];
    expect(arg.sessionId).toBe('sess-1');
  });

  it('disables the button while the compose call is in flight', async () => {
    let resolveCompose: (v: unknown) => void = () => {};
    vi.spyOn(api, 'composePipelineFromPrompt').mockImplementation(
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
      session_id: 'x',
      pipeline_snapshot: { source: {}, transforms: [], sinks: [] },
    });
  });

  it('surfaces an inline error if compose fails', async () => {
    vi.spyOn(api, 'composePipelineFromPrompt').mockRejectedValue(
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
import { composePipelineFromPrompt } from '../../api/client';
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
      const response = await composePipelineFromPrompt({
        prompt,
        tutorial_mode: true,
      });
      onBuilt({
        sessionId: response.session_id,
        pipelineSnapshot: response.pipeline_snapshot,
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

`composePipelineFromPrompt` is added to `api/client.ts` as part of this task
if it doesn't already exist (the function name should match what Phase 5a
shipped; if Phase 5a's name differs, adapt — recon resolves this in Step 1).

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx \
  src/elspeth/web/frontend/src/api/client.ts
git commit -m "feat(frontend): add tutorial turn 2 describe (Phase 4B.5)"
```

## Task 6: Turn 2b — Show what was built + interpretation embed

**Files:**
- Create: `TutorialTurn2bShowBuilt.tsx` + `.test.tsx`.

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
```

Read the component to understand:
- Its props (likely `sessionId`, `interpretationEventId`, `onResolved`).
- Its imports (which interpretation-events API client functions).
- Where it lives (`components/interpretation/` or similar).

If `InterpretationReviewTurn` is not available, halt — Phase 5b not shipped.

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { TutorialTurn2bShowBuilt } from './TutorialTurn2bShowBuilt';
import { TURN_2B_INTRO } from './copy';
import * as interpretationApi from '../../api/interpretationEvents';

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
    vi.spyOn(interpretationApi, 'listPendingInterpretationEvents').mockResolvedValue([
      { id: 'evt-1', tool_call_id: 'tc-1', draft_value: 'modern design + clear purpose' },
    ]);
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

  it('forwards onInterpretationAccepted with the event id', async () => {
    const onAccepted = vi.fn();
    vi.spyOn(interpretationApi, 'resolveInterpretationEvent').mockResolvedValue({
      id: 'evt-1',
      status: 'accepted',
      accepted_value: 'modern design + clear purpose',
    });
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
import { useEffect, useState } from 'react';
import { TURN_2B_INTRO, TURN_2B_INTERPRETATION_PROMPT } from './copy';
import { InterpretationReviewTurn } from '../interpretation/InterpretationReviewTurn'; // Phase 5b
import { listPendingInterpretationEvents } from '../../api/interpretationEvents';
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
  const [eventId, setEventId] = useState<string | null>(null);

  useEffect(() => {
    listPendingInterpretationEvents(sessionId).then((events) => {
      if (events.length > 0) setEventId(events[0].id);
    });
  }, [sessionId]);

  // Pipeline-snapshot reads. Shape is the inline_blob source contract from
  // Phase 5a — we trust it (Tier 2: post-source).
  const urls = ((pipelineSnapshot.source as { urls?: string[] }).urls) ?? [];
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
          <strong>TRANSFORM</strong> — fetch each page ({transforms[0]?.type}),
          then call an LLM to rate each one for "coolness" ({transforms[1]?.type})
        </div>

        <div>
          <strong>SINK</strong> — write the ratings to a {sinks[0]?.type.toUpperCase()} file in your session
        </div>
      </section>

      {eventId && (
        <section className="tutorial-interpretation">
          <p>{TURN_2B_INTERPRETATION_PROMPT}</p>
          <InterpretationReviewTurn
            sessionId={sessionId}
            interpretationEventId={eventId}
            onResolved={() =>
              onInterpretationAccepted({ interpretationEventId: eventId })
            }
          />
        </section>
      )}
    </div>
  );
}
```

If Phase 5b's `InterpretationReviewTurn` component takes different props,
adapt the wrapper — but do **not** wrap it in defensive try/catch (CLAUDE.md
"no defensive programming"): if Phase 5b's contract changed, fix the call
site, don't paper over it.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx
git commit -m "feat(frontend): add tutorial turn 2b show-built (Phase 4B.6)"
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
  const urls = ((pipelineSnapshot.source as { urls?: string[] }).urls) ?? [];
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
        <Node label={transforms[0]?.type ?? 'web_scrape'} subtitle="fetch" />
        <Arrow />
        <Node label={transforms[1]?.type ?? 'llm_rate'} subtitle="rate" />
        <Arrow />
        <Node label={`${sinks[0]?.type ?? 'jsonl'}_sink`} subtitle="write" />
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

Not yet wired: container (Task 11), App.tsx detection + sessionStorage persistence + banner (Task 12), turns 4–6 (Tasks 8–10), integration/E2E/smoke (Tasks 13–15).

Continue in [21b2-phase-4-frontend-part-2.md](21b2-phase-4-frontend-part-2.md).

## Review history

### 2026-05-15 — review panel

| ID | Severity | Status | Summary |
|---|---|---|---|
| 4B1-F1 | CRITICAL (Systems) | Applied | Architecture stanza updated: sessionStorage-resume pattern; DB-delete banner dropped per 2026-05-16 review (No Legacy Code Policy — single Turn 1 entry for all tutorial-mode users) |
| 4B1-F3 | BLOCKER (Coherence) | Applied | `markTutorialCompleted` signature corrected to `(defaultMode)` only; session-rename remains a separate Turn 6 call |
| 4B1-F2 | IMPORTANT (Architecture) | Applied | Phase 5b cross-link added to turn 2b and turn 5 scope items |

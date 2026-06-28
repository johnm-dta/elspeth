# Phase 4B Part 2 — Frontend: turns 4–6, container, detection, integration, smoke

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal of this part (4B.2):** Tasks 8–15 — turn components 4 (Run), 5
(Audit story, load-bearing), and 6 (Mode choice + finalisation); the
`HelloWorldTutorial` container; App.tsx tutorial detection; a Vitest full-
flow integration test; the Playwright E2E for a brand-new user; and the
staging smoke deploy at `elspeth.foundryside.dev`.

**Prerequisite:** [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md)
must have landed first (Tasks 1–7: store, state machine, copy, turns 1/2/2b/3).

**Sibling plans:**
- [21a1-phase-4-backend-part-1.md](21a1-phase-4-backend-part-1.md) — backend infrastructure: schema column, service
  extension, route extension, tutorial cache, run-path integration.
- [21a2-phase-4-backend-part-2.md](21a2-phase-4-backend-part-2.md) — backend endpoints + telemetry: tutorial-run endpoint, audit-story endpoint, frontend API client, launch telemetry counters.
- [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md) — store,
  state machine, copy, turns 1/2/2b/3.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**PR mapping:** This plan is the second half of **PR-21b** (frontend);
PR-21b combines Tasks 1–7 in 21b1 with Tasks 8–15 here and merges to
`RC5.2` **only after** PR-21a (backend) has landed — see overview §"PR
strategy" for rationale.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope (this part):**

- Frontend API client surface (Task 7.5, relocated from 21a2 Task 7.3):
  `runTutorialPipeline`, `getRunAuditSummary`, `deleteTutorialOrphans`,
  and the `updateSessionTitle` → `renameSession` rename.
- `TutorialTurn4Run.tsx` — kicks off the run via the standard run API,
  shows progress, renders the result table (including per-row failures as
  teaching moments).
- `TutorialTurn5AuditStory.tsx` — reads real audit-trail values from the
  just-completed run's Landscape entry and renders the load-bearing audit
  story.
- `TutorialTurn6ModeChoice.tsx` — guided/freeform radio + atomic
  finalisation PATCH + session rename.
- `HelloWorldTutorial.tsx` — container that owns
  `useReducer(tutorialReducer, initialState)` and dispatches the correct
  leaf component per step. Mount-time `DELETE /api/tutorial/orphans` call
  handles refresh-during-tutorial orphan cleanup (Systems R2-S5).
- `App.tsx` — wire tutorial detection so first-session users see
  `HelloWorldTutorial` instead of the normal composer surface. Drop the
  prior `.catch((err) => console.error(...))` from the bootstrap effect —
  errors flow through the prefs `writeError` surface (Architecture
  N-R2-3 / Panel C2).
- Vitest full-flow integration test exercising all 6 turns with mocked APIs.
- Playwright E2E for a brand-new user end-to-end on staging.
- "Reset tutorial" link in `ComposerPreferencesPanel` (Task 14.5;
  closes Open Question C3 in-phase per Systems R2-S7 — no sibling-plan
  deferral).
- Staging smoke deploy + manual click-through verification.

**Out of scope:**

- Anything in Part 1 (already landed).
- Anything in 21a (backend).
- Localisation.

## Trust tier check (per CLAUDE.md)

Same surfaces as Part 1, plus:

- **Run results rendered in turn 4** (per-row data including LLM rationale
  strings, scores, errors): Tier 2 — post-pipeline. The data is what the
  run path produced; render as-is. No coercion. Per-row error fields are
  audit-aware: a failure was already recorded.
- **Audit-summary call in turn 5**: Tier 1 read. The Landscape data is our
  data. If the call fails (Landscape unreachable, run not found), surface
  the error visibly — do not fall back to canned text (CLAUDE.md
  "no defensive programming"; design doc 04 §"Implementation notes":
  "Otherwise the demonstration is theatre.").
- **Session-rename API call**: Tier 1 write. Standard session-rename
  contract; failures propagate.
- **PATCH at finalisation**: Tier 1 write. The atomic PATCH writes both
  `default_mode` and `tutorial_completed_at` — see Phase 4A Task 3.

## Verification approach

Each task remains TDD-shaped at the Vitest level. The Playwright E2E
(Task 14) runs against a fully wired backend (Phase 4A merged + DB-deleted +
service restarted). The staging smoke (Task 15) is a manual operator
click-through recorded in the merge PR.

---

## Task 7.5: Frontend API client surface — `runTutorialPipeline`, `getRunAuditSummary`, `renameSession`, `deleteTutorialOrphans`

**Files:**

- Modify: `src/elspeth/web/frontend/src/api/client.ts` — four function additions / one rename.
- Create: `src/elspeth/web/frontend/src/api/client.tutorial.test.ts` — tests for the new functions (matches the per-feature test-file convention live in the repo — see `client.preferences.test.ts`, `client.recovery.test.ts`).

This task lands the frontend API client symbols that downstream Tasks 8,
9, 10, and 11 consume. **Relocated from 21a2 Task 7.3** (Architecture
r2 M-R2-2): these are frontend `client.ts` symbols and belong with the
frontend plan, not the backend plan. Backend endpoints behind them are
all defined in PR-21a (Tasks 7.1, 7.2, 7.4 + the pre-existing
`PATCH /api/sessions/{id}`), so PR-21a must merge first per the
overview's §"PR strategy".

**Pre-existing surface (confirmed by 21a recon, 2026-05-19):**

- The backend already exposes `PATCH /api/sessions/{id}` with body
  `{title: str}`. The current `client.ts` exports `updateSessionTitle`
  (line 328 at recon time) which calls this endpoint. 21b2 Task 10 uses
  the name `renameSession` for the same wire endpoint. **No backend
  route change is needed.** Per CLAUDE.md no-legacy-code: rename
  `updateSessionTitle` → `renameSession` (single rename, all call sites
  updated in the same commit; no shim).

- `runTutorialPipeline` and `getRunAuditSummary` are new functions
  consuming the newly-defined backend routes (PR-21a Tasks 7.1 and 7.2).

- `deleteTutorialOrphans` is a new function consuming the orphan-cleanup
  endpoint (PR-21a Task 7.4 — see 21a2 §"Task 7.4: orphan-session
  cleanup endpoint" for the wire contract). The endpoint is
  `DELETE /api/tutorial/orphans`; it is user-scoped via the auth header
  and returns `{deleted_count: int}`. The frontend call fires from the
  tutorial container on mount (21b2 §"Task 11" Step 3) to clean up any
  sessions a refresh-during-tutorial event orphaned.

- [ ] **Step 1: Reconnaissance — confirm current client.ts surface and call sites.**

```bash
grep -n "updateSessionTitle\|runTutorialPipeline\|getRunAuditSummary\|renameSession\|deleteTutorialOrphans" \
  src/elspeth/web/frontend/src/ -r
```

Catalogue every call site of `updateSessionTitle` (these are the
rename-target files Task 7.5 will edit). Confirm that no other consumer
relies on the name `updateSessionTitle`.

- [ ] **Step 2: Write the failing test.**

Create `src/elspeth/web/frontend/src/api/client.tutorial.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  runTutorialPipeline,
  getRunAuditSummary,
  renameSession,
  deleteTutorialOrphans,
} from './client';

describe('client.tutorial — runTutorialPipeline', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('POSTs to /api/tutorial/run with session_id + prompt', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: 'r1',
          output: {
            rows: [{ url: 'a', score: 5 }],
            source_data_hash: 'a7f3e2',
          },
          seeded_from_cache: false,
          cache_key: null,
        }),
        { status: 200 },
      ),
    );
    const result = await runTutorialPipeline({
      session_id: 'sess-1',
      prompt: 'rate these',
    });
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/tutorial/run',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ session_id: 'sess-1', prompt: 'rate these' }),
      }),
    );
    expect(result.run_id).toBe('r1');
    expect(result.seeded_from_cache).toBe(false);
  });

  it('surfaces backend error status as a thrown error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response('{"detail":"unknown session"}', { status: 404 }),
    );
    await expect(
      runTutorialPipeline({ session_id: 'bad', prompt: 'x' }),
    ).rejects.toThrow();
  });
});

describe('client.tutorial — getRunAuditSummary', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('GETs /api/sessions/{id}/runs/{run_id}/audit-story', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({
          run_id: 'r1',
          session_id: 'sess-1',
          llm_call_count: 5,
          output_file_hash: 'cafe',
          started_at: '2026-05-15T12:00:00Z',
          plugin_versions: { web_scrape: '1.0.0' },
          seeded_from_cache: false,
          cache_key: null,
        }),
        { status: 200 },
      ),
    );
    const summary = await getRunAuditSummary('sess-1', 'r1');
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sessions/sess-1/runs/r1/audit-story',
      expect.objectContaining({ headers: expect.anything() }),
    );
    expect(summary.llm_call_count).toBe(5);
  });

  it('surfaces 500 (corrupt audit row) as a thrown error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response('{"detail":"missing llm_call_count"}', { status: 500 }),
    );
    await expect(getRunAuditSummary('s', 'r')).rejects.toThrow();
  });
});

describe('client.tutorial — renameSession', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('PATCHes /api/sessions/{id} with {title}', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(
        JSON.stringify({ id: 'sess-1', title: 'hello-world (cool government pages)' }),
        { status: 200 },
      ),
    );
    const result = await renameSession(
      'sess-1',
      'hello-world (cool government pages)',
    );
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/sessions/sess-1',
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ title: 'hello-world (cool government pages)' }),
      }),
    );
    expect(result.title).toBe('hello-world (cool government pages)');
  });
});

describe('client.tutorial — deleteTutorialOrphans', () => {
  beforeEach(() => {
    global.fetch = vi.fn();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('DELETEs /api/tutorial/orphans and returns deleted_count', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response(JSON.stringify({ deleted_count: 2 }), { status: 200 }),
    );
    const result = await deleteTutorialOrphans();
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/tutorial/orphans',
      expect.objectContaining({ method: 'DELETE' }),
    );
    expect(result.deleted_count).toBe(2);
  });

  it('surfaces backend error status as a thrown error', async () => {
    (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue(
      new Response('{"detail":"server fault"}', { status: 500 }),
    );
    await expect(deleteTutorialOrphans()).rejects.toThrow();
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

```bash
cd src/elspeth/web/frontend && npm test -- client.tutorial.test.ts
```

Expected: FAIL — `runTutorialPipeline`, `getRunAuditSummary`,
`renameSession`, `deleteTutorialOrphans` not exported.

- [ ] **Step 4: Implement.**

In `src/elspeth/web/frontend/src/api/client.ts`:

1. **Add new types** alongside the existing type declarations:

   ```typescript
   export interface TutorialRunRequest {
     session_id: string;
     prompt: string;
   }

   export interface TutorialRunOutput {
     rows: Array<Record<string, unknown>>;
     source_data_hash: string;
   }

   export interface TutorialRunResponse {
     run_id: string;
     output: TutorialRunOutput;
     seeded_from_cache: boolean;
     cache_key: string | null;
   }

   export interface RunAuditStoryResponse {
     run_id: string;
     session_id: string;
     llm_call_count: number;
     output_file_hash: string;
     started_at: string;
     plugin_versions: Record<string, string>;
     seeded_from_cache: boolean;
     cache_key: string | null;
   }

   export interface TutorialOrphanCleanupResponse {
     deleted_count: number;
   }
   ```

2. **Add `runTutorialPipeline`**:

   ```typescript
   /** Run the tutorial pipeline. Backend may serve a cached replay; the
    * returned run_id is ALWAYS owned by the current session — see 21a
    * §"New endpoints" for the cache-replay contract.
    */
   export async function runTutorialPipeline(
     body: TutorialRunRequest,
   ): Promise<TutorialRunResponse> {
     const response = await fetch('/api/tutorial/run', {
       method: 'POST',
       headers: authHeaders('application/json'),
       body: JSON.stringify(body),
     });
     return parseResponse<TutorialRunResponse>(response);
   }
   ```

3. **Add `getRunAuditSummary`**:

   ```typescript
   /** Read the audit-story for a (session_id, run_id) pair.
    *
    * All fields are real audit data — no synthesis. A 500 response means
    * the audit row is Tier-1 corrupt; surface the error rather than
    * fabricating defaults (per CLAUDE.md no-defensive-programming).
    */
   export async function getRunAuditSummary(
     sessionId: string,
     runId: string,
   ): Promise<RunAuditStoryResponse> {
     const response = await fetch(
       `/api/sessions/${sessionId}/runs/${runId}/audit-story`,
       { headers: authHeaders() },
     );
     return parseResponse<RunAuditStoryResponse>(response);
   }
   ```

4. **Rename `updateSessionTitle` → `renameSession`.** Per CLAUDE.md "No
   Legacy Code Policy", do not leave an alias. The existing function body
   (PATCH `/api/sessions/${sessionId}` with `{title}`) is structurally
   correct — just rename. Update every call site discovered in Step 1's
   `grep`. Atomic single commit.

   ```typescript
   /** Update the user-visible title for a session.
    *
    * Used by the tutorial finalisation flow (21b2 Task 10) and by any
    * other UI that lets a user rename a session. Wire endpoint:
    * PATCH /api/sessions/{id} with body {title}.
    */
   export async function renameSession(
     sessionId: string,
     title: string,
   ): Promise<Session> {
     const response = await fetch(`/api/sessions/${sessionId}`, {
       method: 'PATCH',
       headers: authHeaders('application/json'),
       body: JSON.stringify({ title }),
     });
     return parseResponse<Session>(response);
   }
   ```

5. **Add `deleteTutorialOrphans`**:

   ```typescript
   /** Clean up orphaned tutorial sessions for the authenticated user.
    *
    * Called from the tutorial container on mount (21b2 §"Task 11") to
    * handle the Systems R2-S5 refresh-during-tutorial case: refresh
    * restarts the tutorial at turn 1, but a session created during the
    * pre-refresh turn 2 would otherwise orphan in the user's session
    * list. Backend owns the orphan-identification predicate; see
    * 21a2 §"Task 7.4" for the wire contract.
    *
    * Tier-1 write (our data). A 5xx response means cleanup failed;
    * surface the error rather than swallowing — the caller in the
    * container fires-and-(implicitly)-forgets but the rejection still
    * propagates to the React error boundary if it bubbles.
    */
   export async function deleteTutorialOrphans(): Promise<TutorialOrphanCleanupResponse> {
     const response = await fetch('/api/tutorial/orphans', {
       method: 'DELETE',
       headers: authHeaders(),
     });
     return parseResponse<TutorialOrphanCleanupResponse>(response);
   }
   ```

- [ ] **Step 5: Update all call sites of the renamed function.**

```bash
grep -rn "updateSessionTitle" src/elspeth/web/frontend/src/ --include="*.ts" --include="*.tsx"
```

Replace every hit with `renameSession`. Run TypeScript compile to confirm
no stragglers:

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit
```

- [ ] **Step 6: Run test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npm test -- client.tutorial.test.ts
```

Expected: PASS.

- [ ] **Step 7: Run the full frontend test suite to catch regressions.**

```bash
cd src/elspeth/web/frontend && npm test
```

Expected: PASS — the rename's call-site updates do not break existing tests.

- [ ] **Step 8: Commit.**

```bash
git add src/elspeth/web/frontend/src/api/client.ts \
        src/elspeth/web/frontend/src/api/client.tutorial.test.ts \
        <renamed-call-sites>
git commit -m "feat(frontend): runTutorialPipeline + getRunAuditSummary + deleteTutorialOrphans + rename updateSessionTitle (Phase 4B.7.5)"
```

---

## Task 8: Turn 4 — Run

**Files:**
- Create: `TutorialTurn4Run.tsx` + `.test.tsx`.

Turn 4 kicks off the run via the standard run API (which — per Phase 4A
Task 7 — consults the tutorial cache for the canonical seed), shows
progress, and renders the result table when the run completes. The
cache-hit path is transparent: the turn doesn't know or care; it just
receives a faster-than-usual response.

- [ ] **Step 1: Recon — identify the run API and progress channel.**

```bash
grep -rn "runPipeline\|executeRun\|sessions.*runs" \
  src/elspeth/web/frontend/src/api --include="*.ts" | head -10
```

Identify whether run progress streams over WebSocket (`websocket.ts`)
or is polled. Adapt the implementation accordingly.

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TutorialTurn4Run } from './TutorialTurn4Run';
import { TURN_4_PRIMARY_BUTTON } from './copy';
import * as api from '../../api/client';

describe('TutorialTurn4Run', () => {
  it('renders the running state immediately', () => {
    render(
      <TutorialTurn4Run sessionId="sess-1" onCompleted={() => {}} />,
    );
    expect(screen.getByText(/Running your pipeline/i)).toBeInTheDocument();
  });

  it('renders the result table after run completes', async () => {
    // Mock shape mirrors the live TutorialRunResponse Pydantic model
    // (21a Task 7.1): rows + source_data_hash + llm_call_count are nested
    // under `output`; run_id, seeded_from_cache, cache_key are top-level.
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      seeded_from_cache: false,
      cache_key: null,
      output: {
        rows: [
          { url: 'australia.gov.au', score: 6, rationale: 'dated' },
          { url: 'dta.gov.au', score: 9, rationale: 'bold' },
        ],
        source_data_hash: 'a7f3e2',
        llm_call_count: 5,
      },
    });
    render(<TutorialTurn4Run sessionId="sess-1" onCompleted={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText(/australia\.gov\.au/)).toBeInTheDocument(),
    );
    // Note: runTutorialPipeline / getRunAuditSummary / renameSession are
    // NEW functions added to api/client.ts by Tasks 8/9/10. Verify each
    // is exported and typed against the response shape used in these
    // spies before writing the implementation. See 21a §"New endpoints".
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getByText('9')).toBeInTheDocument();
    expect(screen.getByText(/dated/)).toBeInTheDocument();
  });

  it('fires onCompleted with run details on continue', async () => {
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      seeded_from_cache: false,
      cache_key: null,
      output: {
        rows: [{ url: 'a', score: 5 }],
        source_data_hash: 'a7f3e2',
        llm_call_count: 1,
      },
    });
    const onCompleted = vi.fn();
    render(<TutorialTurn4Run sessionId="sess-1" onCompleted={onCompleted} />);
    await waitFor(() => screen.getByRole('button', { name: TURN_4_PRIMARY_BUTTON }));
    fireEvent.click(screen.getByRole('button', { name: TURN_4_PRIMARY_BUTTON }));
    expect(onCompleted).toHaveBeenCalledWith({
      runId: 'r1',
      sourceDataHash: 'a7f3e2',
      rows: [{ url: 'a', score: 5 }],
    });
  });

  it('shows per-row failures inline as a teaching moment', async () => {
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      seeded_from_cache: false,
      cache_key: null,
      output: {
        rows: [
          { url: 'a', score: 5 },
          { url: 'broken.gov.au', error: 'HTTP 503' },
        ],
        source_data_hash: 'a7f3e2',
        llm_call_count: 2,
      },
    });
    render(<TutorialTurn4Run sessionId="sess-1" onCompleted={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText(/HTTP 503/)).toBeInTheDocument(),
    );
    expect(
      screen.getByText(/recorded in audit/i),
    ).toBeInTheDocument();
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Implement.**

```typescript
import { useEffect, useState } from 'react';
import { TURN_4_INTRO, TURN_4_PRIMARY_BUTTON } from './copy';
import { runTutorialPipeline } from '../../api/client';
import type { RunResultRow } from './tutorialMachine';

interface RunResult {
  runId: string;
  sourceDataHash: string;
  rows: RunResultRow[];
}

interface Props {
  sessionId: string;
  onCompleted: (result: RunResult) => void;
}

export function TutorialTurn4Run({ sessionId, onCompleted }: Props) {
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Access shape mirrors the TutorialRunResponse Pydantic model
    // (21a Task 7.1): rows + source_data_hash live under `output`;
    // run_id is top-level.
    runTutorialPipeline({ session_id: sessionId })
      .then((response) =>
        setResult({
          runId: response.run_id,
          sourceDataHash: response.output.source_data_hash,
          rows: response.output.rows,
        }),
      )
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [sessionId]);

  if (error) {
    return (
      <div className="tutorial-turn tutorial-turn-4">
        <p className="tutorial-error">Run failed: {error}</p>
      </div>
    );
  }

  if (result === null) {
    return (
      <div className="tutorial-turn tutorial-turn-4">
        <p>{TURN_4_INTRO}</p>
        <Progress />
      </div>
    );
  }

  return (
    <div className="tutorial-turn tutorial-turn-4">
      <p>Here's what came back:</p>
      <table className="tutorial-results">
        <thead>
          <tr>
            <th>URL</th>
            <th>Coolness</th>
            <th>LLM rationale</th>
          </tr>
        </thead>
        <tbody>
          {result.rows.map((r) => (
            <tr key={r.url}>
              <td>{r.url}</td>
              {('error' in r && (r as { error?: string }).error) ? (
                <>
                  <td colSpan={2} className="tutorial-row-error">
                    {(r as { error: string }).error} — recorded in audit
                  </td>
                </>
              ) : (
                <>
                  <td>{r.score}</td>
                  <td>{r.rationale ?? ''}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <button
        type="button"
        className="tutorial-primary"
        onClick={() => onCompleted(result)}
      >
        {TURN_4_PRIMARY_BUTTON}
      </button>
    </div>
  );
}

function Progress() {
  return (
    <div className="tutorial-progress" aria-live="polite">
      <span>fetching pages…</span>
      <span>rating with LLM…</span>
      <span>writing output…</span>
    </div>
  );
}
```

The `runTutorialPipeline` API function in `client.ts` calls
`POST /api/tutorial/run` (specified in 21a §"New endpoints"). Add or rename
the function in `client.ts` to match that endpoint. The spy in the test above
uses the name as it appears in `client.ts` after this step.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.test.tsx
git commit -m "feat(frontend): add tutorial turn 4 run (Phase 4B.8)"
```

## Task 9: Turn 5 — Audit story (the load-bearing turn)

**Files:**
- Create: `TutorialTurn5AuditStory.tsx` + `.test.tsx`.

This is the load-bearing turn: it makes the audit-trail promise concrete
against the user's own run. It MUST render **real values**, not canned
text:

- `sourceDataHash` from turn 4's result (rendered as a short hex prefix
  with full-hash on hover/expand).
- The `interpretationEventId` from the state (cited as "your accepted
  definition of cool — recorded in interpretation events row evt-1").
- The `runId` (cited in the "Explore the full audit trail" link href).
  This is **always the current session's run_id** — never a foreign
  run_id from a cache-seeding session. On a cache hit the backend
  synthesises a new Landscape entry under the current session and
  returns its run_id; the audit-story call below targets that same
  current-session-owned run.
- The LLM-call row count (read from the run's Landscape entry via the
  existing audit-readiness API — see Phase 2 audit-readiness work).
- The `seeded_from_cache` provenance flag and `cache_key`. When the
  audit story reports `seeded_from_cache: true`, the run was a
  cache-replay: this turn briefly acknowledges that ("Your run reused
  cached LLM responses from a prior canonical run; the audit trail
  records the cache key so the original generation can be traced"). The
  narration neither hides the replay nor pretends the LLM was called for
  this run.

- [ ] **Step 1: Recon Phase 2's audit-readiness API.**

```bash
grep -rn "auditReadiness\|getAuditSummary\|audit-readiness" \
  src/elspeth/web/frontend/src/api --include="*.ts" | head -10
```

If Phase 2 shipped, the audit-readiness panel's data fetch pattern is the
template. If not (the roadmap status table shows Phase 2 is NEEDS-RECON not
SHIPPED at plan-writing time), this turn's data fetch goes through the
generic Landscape read API; halt and recon. **Phase 4 cannot proceed if
the Landscape read API has no way to surface `source_data_hash` and the
LLM-call summary for a given run.**

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TutorialTurn5AuditStory } from './TutorialTurn5AuditStory';
import { TURN_5_EXPLORE_BUTTON, TURN_5_CONTINUE_BUTTON } from './copy';
import * as api from '../../api/client';

describe('TutorialTurn5AuditStory', () => {
  const props = {
    sessionId: 'sess-1',
    runId: 'run-1',
    sourceDataHash: 'a7f3e2deadbeef0123',
    interpretationEventId: 'evt-1',
    onContinue: () => {},
  };

  beforeEach(() => {
    vi.spyOn(api, 'getRunAuditSummary').mockResolvedValue({
      llm_call_count: 5,
      output_file_hash: 'cafebabe',
      started_at: '2026-05-15T12:00:00Z',
      plugin_versions: { web_scrape: '1.0.0', llm_rate: '1.0.0' },
      seeded_from_cache: false,
      cache_key: null,
    });
  });

  it('renders the source data hash (short prefix)', async () => {
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(screen.getByText(/a7f3e2/)).toBeInTheDocument(),
    );
  });

  it('renders the interpretation event id', async () => {
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(screen.getByText(/evt-1/)).toBeInTheDocument(),
    );
  });

  it('renders the LLM call count from the audit summary', async () => {
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(screen.getByText(/5/)).toBeInTheDocument(),
    );
  });

  it('renders the output file hash', async () => {
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(screen.getByText(/cafebabe/)).toBeInTheDocument(),
    );
  });

  it('renders the explore-audit-trail link with the run id', async () => {
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() => screen.getByText(TURN_5_EXPLORE_BUTTON));
    const link = screen.getByText(TURN_5_EXPLORE_BUTTON);
    // The href targets the CURRENT session's run — same-ownership.
    expect(link.getAttribute('href') ?? '').toContain('sess-1');
    expect(link.getAttribute('href') ?? '').toContain('run-1');
  });

  it('calls audit-story under the current session id', async () => {
    // Architectural invariant: the audit-story endpoint is always a
    // same-ownership query against the props.sessionId + props.runId pair
    // returned by POST /api/tutorial/run. There is no path that emits a
    // foreign run_id.
    const spy = vi.spyOn(api, 'getRunAuditSummary');
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() => expect(spy).toHaveBeenCalledWith('sess-1', 'run-1'));
  });

  it('on cache hit: narrates the seeded_from_cache provenance', async () => {
    vi.spyOn(api, 'getRunAuditSummary').mockResolvedValue({
      llm_call_count: 0,
      output_file_hash: 'cafebabe',
      started_at: '2026-05-15T12:00:00Z',
      plugin_versions: { web_scrape: '1.0.0', llm_rate: '1.0.0' },
      seeded_from_cache: true,
      cache_key: 'a'.repeat(64),
    });
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(
        screen.getByText(/reused cached LLM responses/i),
      ).toBeInTheDocument(),
    );
    // The cache key is rendered (short prefix) so the auditor narrative
    // is concrete, not vague.
    expect(screen.getByText(/aaaaaa/)).toBeInTheDocument();
    // llm_call_count is 0 on the replay (the cache served the responses).
    expect(screen.getByText(/0 calls/)).toBeInTheDocument();
  });

  it('on cache miss: does NOT narrate seeded_from_cache', async () => {
    // Default beforeEach mock has seeded_from_cache: false.
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() => screen.getByText(/5/));
    expect(
      screen.queryByText(/reused cached LLM responses/i),
    ).not.toBeInTheDocument();
  });

  it('fires onContinue on continue click', async () => {
    const onContinue = vi.fn();
    render(<TutorialTurn5AuditStory {...props} onContinue={onContinue} />);
    await waitFor(() =>
      screen.getByRole('button', { name: TURN_5_CONTINUE_BUTTON }),
    );
    fireEvent.click(
      screen.getByRole('button', { name: TURN_5_CONTINUE_BUTTON }),
    );
    expect(onContinue).toHaveBeenCalledOnce();
  });

  it('crashes the component (surfaces error) if audit summary fails', async () => {
    vi.spyOn(api, 'getRunAuditSummary').mockRejectedValue(
      new Error('Landscape unreachable'),
    );
    render(<TutorialTurn5AuditStory {...props} />);
    await waitFor(() =>
      expect(screen.getByText(/Landscape unreachable/)).toBeInTheDocument(),
    );
    // Per CLAUDE.md no-defensive-programming: we do NOT fall back to canned
    // text. The error is visible.
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Implement.**

```typescript
import { useEffect, useState } from 'react';
import {
  TURN_5_INTRO,
  TURN_5_OUTRO,
  TURN_5_EXPLORE_BUTTON,
  TURN_5_CONTINUE_BUTTON,
} from './copy';
import { getRunAuditSummary } from '../../api/client';

interface Props {
  sessionId: string;
  runId: string;
  sourceDataHash: string;
  interpretationEventId: string | null;
  onContinue: () => void;
}

interface AuditSummary {
  llm_call_count: number;
  output_file_hash: string;
  started_at: string;  // reused from Landscape runs_table; see 21a Task 7.0
  plugin_versions: Record<string, string>;
  seeded_from_cache: boolean;
  cache_key: string | null;
}

export function TutorialTurn5AuditStory({
  sessionId,
  runId,
  sourceDataHash,
  interpretationEventId,
  onContinue,
}: Props) {
  const [summary, setSummary] = useState<AuditSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getRunAuditSummary(sessionId, runId)
      .then(setSummary)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [sessionId, runId]);

  if (error) {
    return (
      <div className="tutorial-turn tutorial-turn-5">
        <p className="tutorial-error">{error}</p>
      </div>
    );
  }

  if (summary === null) {
    return (
      <div className="tutorial-turn tutorial-turn-5">
        <p>Loading audit trail…</p>
      </div>
    );
  }

  const shortHash = sourceDataHash.substring(0, 6);
  const shortOutputHash = summary.output_file_hash.substring(0, 8);

  return (
    <div className="tutorial-turn tutorial-turn-5">
      <p>{TURN_5_INTRO}</p>
      {summary.seeded_from_cache && summary.cache_key && (
        <p className="tutorial-cache-replay">
          Your run reused cached LLM responses from a prior canonical run; the
          audit trail records the cache key{' '}
          <code>{summary.cache_key.slice(0, 6)}…</code> so the original
          generation can be traced. This replay made {summary.llm_call_count}{' '}
          calls to the LLM.
        </p>
      )}
      <ul className="tutorial-audit-bullets">
        <li>
          Every URL you started with — hash <code>{shortHash}…</code>
        </li>
        {interpretationEventId && (
          <li>
            Your accepted definition of "cool" — recorded as a prompt template
            <small> (interpretation event {interpretationEventId})</small>
          </li>
        )}
        <li>
          Every LLM call — full prompt, full response, model, version,
          timestamp <small>({summary.llm_call_count} calls)</small>
        </li>
        <li>
          The output file — SHA-256-hashed (<code>{shortOutputHash}…</code>),
          chain-of-custody recorded
        </li>
        <li>
          The run itself — when, who ran it, plugin versions in use
        </li>
      </ul>
      <p>{TURN_5_OUTRO}</p>
      <div className="tutorial-actions">
        <a
          className="tutorial-secondary"
          href={`/sessions/${sessionId}/runs/${runId}/audit`}
        >
          {TURN_5_EXPLORE_BUTTON}
        </a>
        <button
          type="button"
          className="tutorial-primary"
          onClick={onContinue}
        >
          {TURN_5_CONTINUE_BUTTON}
        </button>
      </div>
    </div>
  );
}
```

`getRunAuditSummary` in `client.ts` calls
`GET /api/sessions/{session_id}/runs/{run_id}/audit-story` (specified in
21a §"New endpoints"). Add this function to `client.ts` if not present;
confirm it matches the response shape in the test spy above.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.test.tsx \
  src/elspeth/web/frontend/src/api/client.ts
git commit -m "feat(frontend): add tutorial turn 5 audit story (Phase 4B.9)"
```

## Task 10: Turn 6 — Mode choice + finalisation

**Files:**
- Create: `TutorialTurn6ModeChoice.tsx` + `.test.tsx`.

Turn 6 renders the guided/freeform radio with guided pre-selected,
displays the two descriptions side-by-side, and offers two buttons. On
click, it:

1. Calls `preferencesStore.markTutorialCompleted(chosenMode)` — the single
   atomic PATCH that writes both `default_mode` and `tutorial_completed_at`.
2. Optionally renames the tutorial session to `TUTORIAL_SESSION_NAME` via
   the standard session-rename API.
3. Calls `onDone()` so the container can transition to "done" and the App
   re-renders without the tutorial.

The session rename happens here, not in turn 4 or 5, because:
- Renaming too early means the user might see "hello-world (cool government
  pages)" as the session title before they finished the tutorial — fine but
  imprecise.
- Renaming at finalisation makes the tutorial session into a "real artifact"
  precisely at the moment design doc 04 §"What happens after the tutorial"
  promises: *"The tutorial pipeline is preserved as a session in their
  history."*

- [ ] **Step 1: Recon the session-rename API.**

```bash
grep -rn "renameSession\|updateSessionTitle" \
  src/elspeth/web/frontend/src/api --include="*.ts" | head -10
```

At RC5.2 HEAD the live name is `updateSessionTitle(sessionId, title)`.
PR-21a (Phase 4 backend) Task 7.3 renames it to `renameSession` per
the No Legacy Code Policy; PR-21b (this plan) merges after PR-21a, so
by the time these tests run `renameSession` IS the live name. Use
`renameSession` here. If recon at task-start shows PR-21a has not yet
landed, halt and surface to the operator — do not introduce a
temporary `updateSessionTitle` call site, as PR-21a's rename will then
break this turn (CLAUDE.md No Legacy Code Policy: change all call
sites in the same commit).

- [ ] **Step 2: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TutorialTurn6ModeChoice } from './TutorialTurn6ModeChoice';
import {
  TURN_6_GUIDED_BUTTON,
  TURN_6_FREEFORM_BUTTON,
  TUTORIAL_SESSION_NAME,
} from './copy';
import { usePreferencesStore } from '../../stores/preferencesStore';
import * as sessionApi from '../../api/client';

describe('TutorialTurn6ModeChoice', () => {
  beforeEach(() => {
    usePreferencesStore.setState({
      defaultMode: 'guided',
      tutorialCompletedAt: null,
    });
  });

  it('renders both options with guided pre-selected', () => {
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    const guided = screen.getByLabelText(/Guided/i) as HTMLInputElement;
    expect(guided.checked).toBe(true);
  });

  it('marks tutorial complete with chosen mode on submit', async () => {
    // Use setState to replace the live action — matches the project's
    // existing Zustand test convention (see stores/preferencesStore.test.ts,
    // stores/authStore.test.ts). vi.spyOn(getState(), ...) spies a snapshot,
    // not the live store, so re-renders or action re-binds escape the spy.
    const markSpy = vi.fn().mockResolvedValue(undefined);
    usePreferencesStore.setState({ markTutorialCompleted: markSpy });
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() => expect(markSpy).toHaveBeenCalledWith('guided'));
  });

  it('marks tutorial complete with freeform when freeform clicked', async () => {
    const markSpy = vi.fn().mockResolvedValue(undefined);
    usePreferencesStore.setState({ markTutorialCompleted: markSpy });
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByLabelText(/Freeform/i),
    );
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_FREEFORM_BUTTON }),
    );
    await waitFor(() => expect(markSpy).toHaveBeenCalledWith('freeform'));
  });

  it('renames the session to the canonical tutorial name on finalisation', async () => {
    const renameSpy = vi
      .spyOn(sessionApi, 'renameSession')
      .mockResolvedValue({ id: 'sess-1', title: TUTORIAL_SESSION_NAME });
    usePreferencesStore.setState({
      markTutorialCompleted: vi.fn().mockResolvedValue(undefined),
    });
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() =>
      expect(renameSpy).toHaveBeenCalledWith('sess-1', TUTORIAL_SESSION_NAME),
    );
  });

  it('fires onDone after both PATCH and rename complete', async () => {
    usePreferencesStore.setState({
      markTutorialCompleted: vi.fn().mockResolvedValue(undefined),
    });
    vi.spyOn(sessionApi, 'renameSession').mockResolvedValue({
      id: 'sess-1',
      title: TUTORIAL_SESSION_NAME,
    });
    const onDone = vi.fn();
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={onDone} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() => expect(onDone).toHaveBeenCalledWith('guided'));
  });

  it('disables buttons while the finalisation calls are in flight', async () => {
    let resolveMark: () => void = () => {};
    usePreferencesStore.setState({
      markTutorialCompleted: vi.fn().mockImplementation(
        () => new Promise<void>((res) => { resolveMark = res; }),
      ),
    });
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
      ).toBeDisabled(),
    );
    resolveMark();
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Implement.**

```typescript
import { useState } from 'react';
import {
  TURN_6_INTRO,
  TURN_6_GUIDED_LABEL,
  TURN_6_GUIDED_DESCRIPTION,
  TURN_6_FREEFORM_LABEL,
  TURN_6_FREEFORM_DESCRIPTION,
  TURN_6_FOOTER,
  TURN_6_GUIDED_BUTTON,
  TURN_6_FREEFORM_BUTTON,
  TUTORIAL_SESSION_NAME,
} from './copy';
import { usePreferencesStore } from '../../stores/preferencesStore';
import { renameSession } from '../../api/client';

interface Props {
  sessionId: string | null;
  onDone: (mode: 'guided' | 'freeform') => void;
}

export function TutorialTurn6ModeChoice({ sessionId, onDone }: Props) {
  const [chosen, setChosen] = useState<'guided' | 'freeform'>('guided');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { markTutorialCompleted } = usePreferencesStore.getState();

  async function finalise(mode: 'guided' | 'freeform') {
    setSubmitting(true);
    setError(null);
    try {
      await markTutorialCompleted(mode);
      // Rename the session iff we have one (skip path doesn't).
      if (sessionId !== null) {
        await renameSession(sessionId, TUTORIAL_SESSION_NAME);
      }
      onDone(mode);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setSubmitting(false);
    }
  }

  return (
    <div className="tutorial-turn tutorial-turn-6">
      <p>{TURN_6_INTRO}</p>

      <label className="tutorial-mode-option">
        <input
          type="radio"
          name="default-mode"
          value="guided"
          checked={chosen === 'guided'}
          onChange={() => setChosen('guided')}
        />
        <strong>{TURN_6_GUIDED_LABEL}</strong>
        <small>{TURN_6_GUIDED_DESCRIPTION}</small>
      </label>

      <label className="tutorial-mode-option">
        <input
          type="radio"
          name="default-mode"
          value="freeform"
          checked={chosen === 'freeform'}
          onChange={() => setChosen('freeform')}
        />
        <strong>{TURN_6_FREEFORM_LABEL}</strong>
        <small>{TURN_6_FREEFORM_DESCRIPTION}</small>
      </label>

      <p>{TURN_6_FOOTER}</p>

      {error && <div className="tutorial-error">{error}</div>}

      <div className="tutorial-actions">
        <button
          type="button"
          className="tutorial-primary"
          disabled={submitting}
          onClick={() => finalise('guided')}
        >
          {TURN_6_GUIDED_BUTTON}
        </button>
        <button
          type="button"
          className="tutorial-secondary"
          disabled={submitting}
          onClick={() => finalise('freeform')}
        >
          {TURN_6_FREEFORM_BUTTON}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx \
  src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.test.tsx
git commit -m "feat(frontend): add tutorial turn 6 mode choice + finalisation (Phase 4B.10)"
```

## Task 11: Container + barrel export

**Files:**
- Create: `HelloWorldTutorial.tsx` + `.test.tsx`.
- Create: `index.ts` (barrel).

The container owns the `useReducer(tutorialReducer, initialState)` state
and dispatches the correct leaf component per step.

- [ ] **Step 1: Write the failing test.**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HelloWorldTutorial } from './HelloWorldTutorial';
import { TURN_1_PRIMARY_BUTTON } from './copy';
import * as api from '../../api/client';

describe('HelloWorldTutorial', () => {
  it('renders turn 1 initially', () => {
    render(<HelloWorldTutorial onDone={() => {}} />);
    expect(
      screen.getByRole('button', { name: TURN_1_PRIMARY_BUTTON }),
    ).toBeInTheDocument();
  });

  it('advances to turn 2 on Start', async () => {
    render(<HelloWorldTutorial onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_1_PRIMARY_BUTTON }),
    );
    await waitFor(() =>
      expect(screen.getByRole('textbox')).toBeInTheDocument(),
    );
  });

  it('skip from turn 1 fast-forwards to turn 6 without building or running', async () => {
    // R2-10, 2026-05-19: the live compose path is createSession + sendMessage,
    // not a fictional composePipelineFromPrompt. Spy on both so the
    // skip-path assertion ("compose-API never called") covers the real surface.
    const createSpy = vi.spyOn(api, 'createSession');
    const sendSpy = vi.spyOn(api, 'sendMessage');
    const runSpy = vi.spyOn(api, 'runTutorialPipeline');
    render(<HelloWorldTutorial onDone={() => {}} />);
    fireEvent.click(screen.getByText(/skip this/i));
    await waitFor(() =>
      expect(screen.getByLabelText(/Guided/i)).toBeInTheDocument(),
    );
    expect(createSpy).not.toHaveBeenCalled();
    expect(sendSpy).not.toHaveBeenCalled();
    expect(runSpy).not.toHaveBeenCalled();
  });

  it('fires onDone when finalised', async () => {
    // Full integration: walk through all 6 turns with mocked APIs.
    // (Detailed mock setup omitted here; see Task 12's integration test for
    // the full end-to-end variant.)
  });

  it('calls DELETE /api/tutorial/orphans on mount (Systems R2-S5)', async () => {
    const orphanSpy = vi
      .spyOn(api, 'deleteTutorialOrphans')
      .mockResolvedValue({ deleted_count: 0 });
    render(<HelloWorldTutorial onDone={() => {}} />);
    await waitFor(() => expect(orphanSpy).toHaveBeenCalledTimes(1));
  });
});
```

- [ ] **Step 2: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 3: Implement.**

```typescript
import { useEffect, useReducer } from 'react';
import { initialState, tutorialReducer } from './tutorialMachine';
import { TutorialTurn1Welcome } from './TutorialTurn1Welcome';
import { TutorialTurn2Describe } from './TutorialTurn2Describe';
import { TutorialTurn2bShowBuilt } from './TutorialTurn2bShowBuilt';
import { TutorialTurn3Graph } from './TutorialTurn3Graph';
import { TutorialTurn4Run } from './TutorialTurn4Run';
import { TutorialTurn5AuditStory } from './TutorialTurn5AuditStory';
import { TutorialTurn6ModeChoice } from './TutorialTurn6ModeChoice';
import { deleteTutorialOrphans } from '../../api/client';

interface Props {
  onDone: (mode: 'guided' | 'freeform') => void;
}

export function HelloWorldTutorial({ onDone }: Props) {
  const [state, dispatch] = useReducer(tutorialReducer, initialState);

  // Systems R2-S5: refresh during the tutorial restarts at turn 1 (per
  // plan-fix P10), but a backend session may have been created in turn 2
  // before the refresh — that session orphans (the user's session list
  // grows with abandoned tutorial sessions). On container mount (= turn 1
  // entry, either genuine first-load or post-refresh re-entry), invoke
  // the backend cleanup endpoint to delete orphaned tutorial sessions
  // for this user. Backend owns the cleanup logic (a single source of
  // truth — see 21a2 §"Task 7.4: orphan-session cleanup endpoint" for
  // the wire contract and the orphan-identification predicate). The
  // frontend just fires the DELETE on mount.
  //
  // Tier-1 write (our data, our cleanup). Failure is non-fatal — log the
  // error via the standard error-boundary path and continue with the
  // tutorial (the orphans are still ignorable on the user's side; a
  // future mount will retry). Do NOT swallow silently to console; the
  // store/error-boundary path is the channel.
  useEffect(() => {
    void deleteTutorialOrphans();
    // No dependency on session id — the endpoint is user-scoped via the
    // auth header. Fires exactly once per container mount.
  }, []);

  switch (state.step) {
    case 'welcome':
      return (
        <TutorialTurn1Welcome
          onStart={() => dispatch({ type: 'START' })}
          onSkip={() => dispatch({ type: 'SKIP' })}
        />
      );
    case 'describe':
      return (
        <TutorialTurn2Describe
          onBuilt={({ sessionId, pipelineSnapshot }) =>
            dispatch({ type: 'PIPELINE_BUILT', sessionId, pipelineSnapshot })
          }
        />
      );
    case 'show-built':
      return (
        <TutorialTurn2bShowBuilt
          sessionId={state.sessionId!}
          pipelineSnapshot={state.pipelineSnapshot!}
          onInterpretationAccepted={({ interpretationEventId }) =>
            dispatch({ type: 'INTERPRETATION_ACCEPTED', interpretationEventId })
          }
        />
      );
    case 'graph':
      return (
        <TutorialTurn3Graph
          pipelineSnapshot={state.pipelineSnapshot!}
          onContinue={() => dispatch({ type: 'START_RUN' })}
        />
      );
    case 'run':
      return (
        <TutorialTurn4Run
          sessionId={state.sessionId!}
          onCompleted={({ runId, sourceDataHash, rows }) =>
            dispatch({ type: 'RUN_COMPLETED', runId, sourceDataHash, rows })
          }
        />
      );
    case 'audit-story':
      return (
        <TutorialTurn5AuditStory
          sessionId={state.sessionId!}
          runId={state.runId!}
          sourceDataHash={state.sourceDataHash!}
          interpretationEventId={state.interpretationEventId}
          onContinue={() => dispatch({ type: 'CONTINUE' })}
        />
      );
    case 'mode-choice':
      return (
        <TutorialTurn6ModeChoice
          sessionId={state.sessionId}
          onDone={(mode) => {
            dispatch({ type: 'FINALISE', chosenMode: mode });
            onDone(mode);
          }}
        />
      );
    case 'done':
      // Should be unreachable (parent unmounts on onDone), but render
      // something benign as defence in depth.
      return null;
  }
}
```

Note the use of `!` non-null assertions: each step's required state fields
are guaranteed by the state machine's transitions. If a `!` ever crashes,
the state machine is buggy and we want the crash to surface (CLAUDE.md
offensive programming).

Create `index.ts`:

```typescript
export { HelloWorldTutorial } from './HelloWorldTutorial';
```

- [ ] **Step 4: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx \
  src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx \
  src/elspeth/web/frontend/src/components/tutorial/index.ts
git commit -m "feat(frontend): add tutorial container + dispatcher (Phase 4B.11)"
```

## Task 12: Detection — wire App.tsx to render the tutorial

**Files:**
- Modify: `src/elspeth/web/frontend/src/App.tsx`.

- [ ] **Step 1: Recon App.tsx's bootstrap pattern.**

```bash
cat src/elspeth/web/frontend/src/App.tsx | head -100
```

Identify:
- Where the user-loaded check happens.
- Where preferences are loaded. Live (`App.tsx`): `usePreferencesStore((s) => s.bootstrap)` slice + `useEffect(() => bootstrap(), [bootstrap])`. The action is `bootstrap`, not `loadPreferences`.
- The point in the render tree where the composer surface mounts (we replace
  this with the tutorial container when `tutorialCompleted === false`).

- [ ] **Step 2: Write the failing test.**

Add to `App.test.tsx` (or create a new `App.tutorial.test.tsx`):

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { App } from './App';
import { usePreferencesStore } from './stores/preferencesStore';
import * as api from './api/client';

describe('App — tutorial routing', () => {
  it('renders HelloWorldTutorial when tutorialCompleted=false', async () => {
    vi.spyOn(api, 'fetchUserComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      updated_at: '2026-05-15T12:00:00Z',
    });
    render(<App />);
    await waitFor(() =>
      expect(screen.getByText(/Welcome to ELSPETH/)).toBeInTheDocument(),
    );
  });

  it('renders the normal composer when tutorialCompleted=true', async () => {
    vi.spyOn(api, 'fetchUserComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: '2026-05-14T12:00:00Z',
      updated_at: '2026-05-15T12:00:00Z',
    });
    render(<App />);
    await waitFor(() =>
      expect(screen.queryByText(/Welcome to ELSPETH/)).not.toBeInTheDocument(),
    );
  });

  it('transitions to the normal composer when tutorial finalises', async () => {
    // Setup: tutorial not yet complete.
    vi.spyOn(api, 'fetchUserComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      updated_at: '2026-05-15T12:00:00Z',
    });
    render(<App />);
    await waitFor(() => screen.getByText(/Welcome to ELSPETH/));
    // Simulate the tutorial finalising: the preferencesStore mutation flows
    // through, App re-renders.
    usePreferencesStore.setState({
      tutorialCompletedAt: '2026-05-15T12:30:00Z',
    });
    await waitFor(() =>
      expect(screen.queryByText(/Welcome to ELSPETH/)).not.toBeInTheDocument(),
    );
  });
});
```

- [ ] **Step 3: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 4: Modify App.tsx.**

Pattern (adapt to App.tsx's actual structure). **No sessionStorage
scaffolding.** Refresh during the tutorial restarts at turn 1 — no
sessionStorage state. The tutorial is short enough (~5 minutes) that
restart cost is acceptable; sessionStorage adds cross-user contamination
risk on shared workstations (a flat `elspeth_tutorial_progress` key is
not user-scoped, and per-user keying adds Vitest+Playwright surface area
that doesn't materially improve UX). The canonical seed produces a fresh
pipeline each time, and the cache makes that fast.

**No separate DB-delete banner** — per 21b1 §"Navigation resilience", all
tutorial-mode users (genuine first-timers and DB-delete-resets) see the
same Turn 1 welcome copy. CLAUDE.md No Legacy Code Policy: we have no
users yet, the operator deletes the DB freely, and a bifurcated banner
would detect a state we can't reliably distinguish. One copy, treated
identically.

```typescript
import { HelloWorldTutorial } from './components/tutorial';
import { usePreferencesStore } from './stores/preferencesStore';

export function App() {
  // Derive tutorialCompleted at the call site via inline selector. The
  // store does NOT expose a `tutorialCompleted` field — see plan 21b1
  // Task 1 §"Notes" and the existing `selectTutorialCompleted` export.
  const tutorialCompleted = usePreferencesStore(
    (s) => s.tutorialCompletedAt !== null,
  );
  const loaded = usePreferencesStore((s) => s.loaded);
  const bootstrap = usePreferencesStore((s) => s.bootstrap);

  // Phase 1B Panel C2: bootstrap may fail (no-row 5xx, network); errors
  // are surfaced via the existing prefs `writeError` surface rather than
  // swallowed to console. The store's `bootstrap` action must capture
  // any thrown error onto `writeError` (matching the `setDefaultMode` /
  // `dismissDefaultChangedBanner` pattern in `preferencesStore.ts` lines
  // 124–142). Per CLAUDE.md no-defensive-programming: do NOT silently
  // swallow rejections with `console.error` — the role="alert" region
  // in `ComposerPreferencesForm` (lines 70–82 of
  // `ComposerPreferencesPanel.tsx`) is the user-visible channel and the
  // bootstrap path MUST feed into it. The bootstrap effect therefore
  // does not need its own `.catch` — the store has captured the error,
  // `loaded` stays false, and the "Loading…" branch below renders
  // followed by the existing `writeError` alert region once the user
  // opens the preferences panel. If bootstrap-failure UX needs a
  // dedicated banner above the composer surface, that's a separate task
  // (file at Phase 4 follow-up time, not here).
  useEffect(() => {
    // Bootstrap rejection is captured by the store onto `writeError`.
    // Do not add a `.catch` here — that would re-surface the rejection
    // to the console and break the writeError contract by suppressing
    // the rejection before the store sees it. The store action MUST
    // be implemented to set `writeError` and resolve (not reject) so
    // this effect's promise is harmless. If the store's `bootstrap`
    // still throws (it should not), the React error boundary in
    // `App.tsx` (or wherever the global boundary lives) catches it.
    void bootstrap();
  }, [bootstrap]);

  if (!loaded) return <div>Loading…</div>;

  if (!tutorialCompleted) {
    // Refresh restarts at turn 1 — no sessionStorage / persisted progress.
    // `onDone` is required by HelloWorldTutorial's signature (21b1 Task 1);
    // re-bootstrap preferences after the tutorial flushes so the post-
    // tutorial state (tutorial_completed_at populated) is reflected in the
    // store without a page reload. The HelloWorldTutorial flow PATCHes
    // tutorial_completed_at itself in turn 6 / on skip — `onDone` here is
    // the local-cache refresh, not the persistence side.
    return <HelloWorldTutorial onDone={() => bootstrap()} />;
  }
  return <NormalComposerSurface />;
}
```

No `initialStep` / `onStepChange` props — `HelloWorldTutorial` always
mounts at the reducer's initial state on every render of this branch.
`onDone` is required (the component's signature in 21b1 Task 1 declares it
as a non-optional prop), so the mount site supplies a `bootstrap()`
re-fetch callback.

**Orphan-cleanup on Turn-1 entry (Systems finding R2-S5, 2026-05-19).**
Turn 2 of the tutorial creates a backend session via `createSession()`
(21b1 Task 5 — R2-10 live compose shape). A refresh after Turn 2
restarts the tutorial at Turn 1 (per P10), but the session the user
created in their previous Turn 2 attempt is now orphaned — it lives
forever under the canonical `hello-world (...)` name with no further
edits, polluting the session list. The App must reconcile this on
Turn-1 entry.

The reconciliation strategy is **soft-delete-via-rename** (operator
decision 2026-05-19): the App lists the user's sessions, finds any
that match the canonical tutorial name prefix AND whose
`tutorial_completed_at` is still NULL, and renames them to
`abandoned-hello-world-<timestamp>` via the existing `renameSession`
function (21b2 Task 7.5). Audit history is preserved (the renamed
session still exists and its compose/run records remain queryable for
post-hoc analysis); the session-list UI filters out the `abandoned-*`
prefix so the user sees a clean list.

Wire the cleanup as a one-shot effect inside the tutorial-mode branch:

```typescript
import { listSessions, renameSession } from '../../api/client';

function useOrphanedTutorialSessionCleanup(
  tutorialCompleted: boolean,
): void {
  useEffect(() => {
    if (tutorialCompleted) return;  // user already finished — no cleanup needed
    let cancelled = false;
    (async () => {
      try {
        const sessions = await listSessions();
        if (cancelled) return;
        const orphans = sessions.filter(
          (s) => s.title.startsWith('hello-world (') && !s.title.startsWith('abandoned-'),
        );
        for (const orphan of orphans) {
          if (cancelled) return;
          await renameSession(
            orphan.id,
            `abandoned-${orphan.title}-${Date.now()}`,
          );
        }
      } catch {
        // Cleanup is best-effort. A failure here leaves the orphan in
        // place; the next Turn-1 entry retries. We deliberately do NOT
        // surface this to `writeError` because the user did not
        // initiate the cleanup and shouldn't see a banner.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tutorialCompleted]);
}
```

The hook fires once per tutorial-mode entry. Add a Vitest test under
`App.tutorial.test.tsx` asserting that when `listSessions` returns a
canonical-prefix session AND `tutorialCompleted=false`, `renameSession`
is called exactly once with an `abandoned-hello-world-…` payload.
Add a second test asserting that when `tutorialCompleted=true`,
`renameSession` is NOT called.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Audit and tighten `preferencesStore.bootstrap` to capture errors onto `writeError` (Architecture r2 / N-R2-3).**

The Step 4 bootstrap effect drops its prior `.catch((err) => console.error(...))`
to comply with Panel C2's "errors surfaced via the existing prefs
writeError surface rather than swallowed." Verify the live
`preferencesStore.bootstrap` action (`src/elspeth/web/frontend/src/stores/preferencesStore.ts`
lines 79–86 at recon time) captures any thrown error onto `writeError`
and resolves cleanly — matching `setDefaultMode`'s try/catch pattern
on lines 123–141 of that file. The current `bootstrap` action does
NOT catch; it lets the rejection escape. Add a try/catch around the
`fetchUserComposerPreferences()` call that sets `writeError` to the
chained error message and leaves `loaded` false:

```typescript
bootstrap: async () => {
  try {
    const payload = await fetchUserComposerPreferences();
    set({
      defaultMode: payload.default_mode,
      bannerDismissedAt: payload.banner_dismissed_at,
      tutorialCompletedAt: payload.tutorial_completed_at,
      loaded: true,
      writeError: null,
    });
  } catch (err) {
    set({
      writeError:
        err instanceof Error
          ? `Couldn't load preferences: ${err.message}`
          : "Couldn't load preferences.",
    });
    throw err; // Re-throw so the React error boundary catches catastrophic
               // failures (preserve current behaviour for tests asserting
               // rejection). The role="alert" region still surfaces the
               // error via the captured writeError.
  }
},
```

Add a Vitest case to `preferencesStore.test.ts`:

```typescript
it('bootstrap sets writeError when fetch throws', async () => {
  vi.spyOn(api, 'fetchUserComposerPreferences').mockRejectedValue(
    new Error('network down'),
  );
  await expect(
    usePreferencesStore.getState().bootstrap(),
  ).rejects.toThrow('network down');
  expect(usePreferencesStore.getState().writeError).toContain(
    'network down',
  );
});
```

Run:

```bash
cd src/elspeth/web/frontend && npx vitest run \
  src/stores/preferencesStore.test.ts
```

Expected: PASS.

- [ ] **Step 7: Run the full frontend test suite to catch regressions.**

```bash
cd src/elspeth/web/frontend && npx vitest run
```

Expected: PASS. If existing App.tsx tests broke, the bootstrap path
changed under them; fix the tests or the implementation depending on
intent.

- [ ] **Step 8: Commit.**

```bash
git add src/elspeth/web/frontend/src/App.tsx \
  src/elspeth/web/frontend/src/App.test.tsx \
  src/elspeth/web/frontend/src/stores/preferencesStore.ts \
  src/elspeth/web/frontend/src/stores/preferencesStore.test.ts
git commit -m "feat(frontend): route first-session users to tutorial container (Phase 4B.12)"
```

## Task 13: Integration test — Vitest full flow with mocked API

**Files:**
- Create or extend an existing integration test for the full tutorial walk.

This is a Vitest test that drives the container through all 6 turns with
mocked API responses, asserting on each transition. It catches integration
bugs at the seams that unit tests miss.

- [ ] **Step 1: Write the test.**

Append to `HelloWorldTutorial.test.tsx`:

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { HelloWorldTutorial } from './HelloWorldTutorial';
import * as api from '@/api/client';
import { useInterpretationEventsStore } from '@/stores/interpretationEventsStore';
import type { InterpretationEvent } from '@/types/interpretation';

const PENDING_EVENT: InterpretationEvent = {
  id: 'evt-1',
  session_id: 'sess-1',
  user_term: 'cool',
  llm_draft: 'modern design',
  choice: 'pending',
  created_at: '2026-05-15T12:00:00Z',
  resolved_at: null,
  amendment: null,
  interpretation_source: null,
};

describe('HelloWorldTutorial — full 6-turn integration', () => {
  it('walks from welcome to done with cached run', async () => {
    // R2-10, 2026-05-19: live compose path is createSession + sendMessage.
    vi.spyOn(api, 'createSession').mockResolvedValue({ id: 'sess-1' });
    vi.spyOn(api, 'sendMessage').mockResolvedValue({
      message: { role: 'assistant', content: 'pipeline built' },
      state: {
        source: { type: 'inline_blob', urls: ['a.gov.au'] },
        transforms: [{ type: 'web_scrape' }, { type: 'llm_rate' }],
        sinks: [{ type: 'jsonl' }],
      },
    });
    // Live: interpretation events come through the store, not a wire
    // module. Seed pendingBySession and spy on store actions.
    useInterpretationEventsStore.setState({
      pendingBySession: { 'sess-1': { 'evt-1': PENDING_EVENT } },
    });
    vi.spyOn(useInterpretationEventsStore.getState(), 'refreshPending')
      .mockResolvedValue();
    vi.spyOn(useInterpretationEventsStore.getState(), 'resolveEvent')
      .mockResolvedValue({ new_state: null as never });
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      source_data_hash: 'a7f3e2',
      rows: [{ url: 'a.gov.au', score: 7 }],
    });
    vi.spyOn(api, 'getRunAuditSummary').mockResolvedValue({
      llm_call_count: 1,
      output_file_hash: 'cafebabe',
      started_at: '2026-05-15T12:00:00Z',
      plugin_versions: {},
      seeded_from_cache: false,
      cache_key: null,
    });
    vi.spyOn(api, 'renameSession').mockResolvedValue({
      id: 'sess-1',
      title: 'hello-world (cool government pages)',
    });
    const patchSpy = vi
      .spyOn(api, 'updateUserComposerPreferences')
      .mockResolvedValue({
        default_mode: 'guided',
        banner_dismissed_at: null,
        tutorial_completed_at: '2026-05-15T12:30:00Z',
        updated_at: '2026-05-15T12:30:00Z',
      });

    const onDone = vi.fn();
    render(<HelloWorldTutorial onDone={onDone} />);

    // Turn 1 → 2.
    fireEvent.click(screen.getByRole('button', { name: /Let's go/i }));
    await waitFor(() => screen.getByRole('textbox'));

    // Turn 2 → 2b.
    fireEvent.click(screen.getByRole('button', { name: /Build it/i }));
    await waitFor(() => screen.getByText(/Got it/));

    // Turn 2b → 3.
    await waitFor(() => screen.getByText(/Use my interpretation/i));
    fireEvent.click(screen.getByText(/Use my interpretation/i));
    await waitFor(() => screen.getByText(/Three layers, four steps/));

    // Turn 3 → 4.
    fireEvent.click(screen.getByRole('button', { name: /run it/i }));
    await waitFor(() => screen.getByText(/a\.gov\.au/));

    // Turn 4 → 5.
    fireEvent.click(screen.getByRole('button', { name: /Continue/i }));
    await waitFor(() => screen.getByText(/a7f3e2/));

    // Turn 5 → 6.
    fireEvent.click(screen.getByRole('button', { name: /Continue/i }));
    await waitFor(() => screen.getByLabelText(/Guided/i));

    // Turn 6 → done.
    fireEvent.click(screen.getByRole('button', { name: /Guided \(recommended\)/i }));
    await waitFor(() => expect(onDone).toHaveBeenCalledWith('guided'));

    expect(patchSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        default_mode: 'guided',
        tutorial_completed_at: expect.any(String),
      }),
    );
  });
});
```

- [ ] **Step 2: Run test to verify it passes.**

Expected: PASS. If any seam fails, fix the seam (do not paper over with
test-level workarounds — CLAUDE.md feedback_fix_errors_you_encounter).

- [ ] **Step 3: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx
git commit -m "test(frontend): add full 6-turn integration test (Phase 4B.13)"
```

## Task 14: Playwright E2E — brand-new user end-to-end

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts`. (Path is fixed
  by the project's `playwright.config.ts`, which sets `testDir: './tests/e2e'`
  relative to the frontend package. Spec files in this project use `.spec.ts`,
  not `.e2e.spec.ts`.)

This is the highest-value test in the plan. It walks a brand-new user
through the tutorial against a real backend. The DB-delete is performed
once at the start of the test file; subsequent tests share the wiped state.

- [ ] **Step 1: Recon project Playwright setup.**

```bash
find . -name "playwright.config.*" -not -path "*/node_modules/*" 2>/dev/null | head -5
find . -path "*/e2e/*" -not -path "*/node_modules/*" 2>/dev/null | head -10
```

Identify:
- Where Playwright tests live.
- Whether they spin up a backend via docker-compose / pytest-fixture /
  external URL.
- The login flow (how a new user is provisioned for the test).

If Playwright is not yet configured for this project, that recon surface is
where to escalate to the operator: "Phase 4 needs E2E coverage; the
project has no Playwright setup. Should I add one?"

- [ ] **Step 2: Write the test.**

**Fresh-user provisioning (plan-fix P22, 2026-05-19).** Per scenario,
provision a brand-new user — `DELETE` any existing `user_preferences`
row for the user, then recreate via the standard provisioning helper.
This makes Scenario A's "tutorial_completed_at IS NULL" precondition
explicit (rather than implicit-on-DB-delete) and gives Scenario C an
independent way to set "tutorial_completed_at IS NOT NULL" without
walking the tutorial first.

Recon prerequisite: confirm whether the codebase already exposes a
test-provisioning endpoint. As of 2026-05-19 a `grep -rn "provision"
src/elspeth/web --include="*.py"` returns no such surface — the
endpoint below must be added as part of this task, gated by an env
flag (`WEBUI_TEST_PROVISION_ENABLED=true`) so it cannot mount in
production. If an equivalent surface lands before this task executes,
reuse it rather than adding a parallel endpoint.

Create the fixture file
`src/elspeth/web/frontend/tests/e2e/fixtures.ts`:

```typescript
import { test as base } from '@playwright/test';

type FreshUserFixture = {
  freshUser: { user_id: string };
};

export const test = base.extend<FreshUserFixture>({
  freshUser: async ({ request }, use) => {
    // Provision a brand-new user. The test-only endpoint
    // `POST /api/test/provision-fresh-user` is mounted ONLY when
    // WEBUI_TEST_PROVISION_ENABLED=true on the deployment. Behaviour:
    //   - DELETE FROM user_preferences WHERE user_id = :user_id
    //   - Re-insert via the standard provisioning helper with
    //     tutorial_completed_at = NULL.
    // Adding this endpoint is part of this task — see the recon note
    // above. If reuse of an existing surface is possible, prefer it.
    const user_id = `e2e-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const resp = await request.post('/api/test/provision-fresh-user', {
      data: { user_id },
    });
    if (!resp.ok()) {
      throw new Error(
        `freshUser fixture: provisioning failed (${resp.status()}). ` +
          `Confirm WEBUI_TEST_PROVISION_ENABLED=true on staging.`,
      );
    }
    await use({ user_id });
    // Teardown is optional; each test gets its own user_id, so no
    // cross-test leakage. If a regression makes teardown necessary,
    // add a DELETE to the same endpoint.
  },
});

export { expect } from '@playwright/test';
```

```typescript
// tests/e2e/tutorial.spec.ts
import { test, expect } from './fixtures';

// Helper: log in as the provisioned fresh user.
async function loginAs(page, userId: string) {
  await page.goto('https://elspeth.foundryside.dev');
  await page.fill('[data-testid="login-username"]', userId);
  await page.fill('[data-testid="login-password"]', 'redacted');
  await page.click('[data-testid="login-submit"]');
}

test.describe('Hello-world tutorial — brand-new user (P22 scenarios)', () => {
  // Scenario A: complete flow — fresh user walks all 6 turns and finishes.
  test('Scenario A — first-session user sees the tutorial and completes it', async ({ page, freshUser, request }) => {
    await loginAs(page, freshUser.user_id);

    // Tutorial turn 1 renders (tutorial_completed_at IS NULL).
    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();
    await page.click('button:has-text("Let\'s go")');

    // Turn 2 — confirm the prompt is pre-filled.
    const textarea = page.locator('textarea').first();
    await expect(textarea).toHaveValue(
      /create a list of 5 government web pages and use an LLM to rate how cool they are/,
    );
    await page.click('button:has-text("Build it")');

    // Turn 2b — the LLM's interpretation surface renders.
    await expect(page.getByText(/Got it/)).toBeVisible({ timeout: 30000 });
    await page.click('button:has-text("Use my interpretation")');

    // Turn 3 — graph renders.
    await expect(page.getByText(/Three layers, four steps/)).toBeVisible();
    await page.click('button:has-text("Looks good, run it")');

    // Turn 4 — run completes (cached → fast).
    await expect(page.locator('table')).toBeVisible({ timeout: 30000 });
    await page.click('button:has-text("Continue")');

    // Turn 5 — audit story renders with real hashes.
    await expect(page.locator('code').first()).toBeVisible();
    await page.click('button:has-text("Continue")');

    // Turn 6 — mode choice.
    await expect(page.getByText(/What should new sessions default to/)).toBeVisible();
    await page.click('button:has-text("Guided (recommended)")');

    // Tutorial is done; the normal composer renders.
    await expect(page.getByText('Welcome to ELSPETH.')).not.toBeVisible();
    await expect(page.getByText(/hello-world \(cool government pages\)/)).toBeVisible();

    // Assert the PATCH wrote tutorial_completed_at (server-side check).
    // Live route is `/api/composer-preferences` (the request is authenticated
    // as `freshUser` via the loginAs/cookie setup, so the server scopes the
    // read to that user — no `{user_id}` path segment exists).
    const prefs = await request.get(`/api/composer-preferences`);
    const body = await prefs.json();
    expect(body.tutorial_completed_at).toEqual(expect.any(String));
  });

  // Scenario B: skip flow — fresh user clicks skip in turn 1, lands on
  // turn 6 directly, PATCH writes tutorial_completed_at, no pipeline
  // session is created (skip is fast-forward only).
  test('Scenario B — skip fast-forwards to turn 6, writes tutorial_completed_at, creates no session', async ({ page, freshUser, request }) => {
    await loginAs(page, freshUser.user_id);

    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();
    await page.click('text=I\'ve used ELSPETH before, skip this');

    // Lands directly on the mode-choice screen.
    await expect(
      page.getByText(/What should new sessions default to/),
    ).toBeVisible();
    await page.click('button:has-text("Guided (recommended)")');

    // PATCH wrote tutorial_completed_at.
    const prefs = await request.get(`/api/composer-preferences`);
    const body = await prefs.json();
    expect(body.tutorial_completed_at).toEqual(expect.any(String));

    // INVARIANT (per design doc 04 — skip is fast-forward only): no
    // tutorial session was created during skip. The sessions list is
    // user-scoped server-side via the auth cookie; the live route is the
    // sessions endpoint without a `{user_id}` path segment — confirm
    // exact path during Step 1 recon (likely `/api/sessions`).
    const sessions = await request.get(`/api/sessions`);
    const sessionsBody = await sessions.json();
    const tutorialSessions = (sessionsBody.sessions ?? []).filter(
      (s: { title: string }) =>
        /hello-world \(cool government pages\)/.test(s.title),
    );
    expect(tutorialSessions).toHaveLength(0);
  });

  // Scenario C: second-login (tutorial already completed) — fresh user,
  // then explicit PATCH to set tutorial_completed_at; load composer;
  // tutorial container does NOT render.
  test('Scenario C — second-login (tutorial completed) shows normal composer, not tutorial', async ({ page, freshUser, request }) => {
    // Explicitly set tutorial_completed_at via PATCH before login.
    // Live route is `/api/composer-preferences`; the fresh-user provisioning
    // fixture is responsible for establishing an auth context under which
    // this PATCH targets `freshUser`'s preferences row (the server scopes
    // the write to the authenticated principal — there is no
    // `{user_id}` path segment on the live route).
    const patchResp = await request.patch(
      `/api/composer-preferences`,
      {
        data: { tutorial_completed_at: '2026-05-19T00:00:00Z' },
      },
    );
    expect(patchResp.ok()).toBe(true);

    await loginAs(page, freshUser.user_id);

    // Tutorial container must NOT render.
    await expect(page.getByText('Welcome to ELSPETH.')).not.toBeVisible({
      timeout: 5000,
    });
    // Normal composer surface DOES render — assert against a known
    // composer landmark (adjust selector during Step 1 recon).
    await expect(page.locator('[data-testid="composer-root"]')).toBeVisible();
  });

  // Refresh-restart scenario (P10) — fresh user, walk partway, refresh,
  // assert turn 1 and null sessionStorage.
  test('refresh mid-tutorial restarts at turn 1 (no sessionStorage)', async ({ page, freshUser }) => {
    await loginAs(page, freshUser.user_id);

    // Walk to turn 3.
    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();
    await page.click('button:has-text("Let\'s go")');
    await page.click('button:has-text("Build it")');
    await expect(page.getByText(/Got it/)).toBeVisible({ timeout: 30000 });
    await page.click('button:has-text("Use my interpretation")');
    await expect(page.getByText(/Three layers, four steps/)).toBeVisible();

    // Refresh.
    await page.reload();

    // Assert turn 1 is visible (not turn 3).
    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();
    await expect(page.getByText(/Three layers, four steps/)).not.toBeVisible();

    // Assert no sessionStorage key exists (no scaffolding remains).
    const progressKey = await page.evaluate(() =>
      sessionStorage.getItem('elspeth_tutorial_progress'),
    );
    expect(progressKey).toBeNull();
  });
});
```

- [ ] **Step 3: Run test against staging.**

```bash
cd src/elspeth/web/frontend
npx playwright test tests/e2e/tutorial.spec.ts --headed
```

Expected: PASS — all three scenarios green. If any fail, diagnose against
the real backend (the failure is real, not a test-environment artifact —
CLAUDE.md `feedback_fix_errors_you_encounter`).

- [ ] **Step 4: Add tutorial spec to the Playwright CI matrix.**

Open `playwright.config.ts` (or the GitHub Actions Playwright workflow — confirm during Step 1 recon). Add the new spec to `testMatch` or the workflow's spec list:

```typescript
testMatch: [/* existing... */ 'tests/e2e/tutorial.spec.ts']
```

- [ ] **Step 5: Commit.**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts \
  src/elspeth/web/frontend/tests/e2e/fixtures.ts \
  src/elspeth/web/frontend/playwright.config.ts
git commit -m "test(e2e): add tutorial brand-new-user end-to-end with freshUser fixture + CI matrix (Phase 4B.14)"
```

## Task 14.5: Reset-tutorial link in `ComposerPreferencesPanel.tsx`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`.
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.ts` (add `resetTutorial` action).
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.test.ts`.
- Create: `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.reset.test.tsx`
  (or extend the existing `ComposerPreferencesPanel.test.tsx` if test-file
  conventions favour append — recon decides).

This task resolves Open Question C3 **in-phase**. Per Systems R2-S7
(2026-05-19): the prior draft deferred the retake UI to a "sibling
settings-panel plan" that does not exist in the roadmap. Per the user's
correction — *no more deferrals to sibling plans that don't exist* —
the Reset link lands here, inside the existing `ComposerPreferencesPanel`
modal that Phase 1B already shipped. The wire contract
(`PATCH {tutorial_completed_at: null}`) is unchanged; this task adds the
UI that fires it.

**Wire contract recap** (already shipped by Task 1 of 21b1 and by
PR-21a):

- The preferences PATCH endpoint accepts `tutorial_completed_at: null` —
  see 21a §"Cross-plan contract — `tutorial_completed_at` PATCH semantics".
- The TypeScript request type already permits `null` (not just
  `undefined`) — 21b1 Task 1 §"Notes" enforces this.

- [ ] **Step 1: Recon — confirm the live `ComposerPreferencesPanel` shape.**

```bash
cat src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx
```

Expect:
- `ComposerPreferencesForm` (lines ~19–85 at recon time) — the inner
  form with the default-mode radio group and the `role="alert"`
  `writeError` region. This is where the Reset link mounts.
- `ComposerPreferencesPanel` — modal wrapper around the form. No
  change here.

The Reset link goes inside `ComposerPreferencesForm`, below the
`fieldset` and above (or beside) the `writeError` alert region. The
link is enabled only when the tutorial has been completed (i.e.,
`tutorialCompletedAt !== null` in the store) — for a user who hasn't
yet completed the tutorial, the link is hidden entirely (showing a
disabled "Reset tutorial" link for a user who's never taken the
tutorial is noise).

- [ ] **Step 2: Add the `resetTutorial` store action.**

In `src/elspeth/web/frontend/src/stores/preferencesStore.ts`, extend
the `PreferencesState` interface and the store implementation:

```typescript
interface PreferencesState {
  // ... existing fields ...
  tutorialCompletedAt: string | null;
  // ... existing fields ...

  resetTutorial: () => Promise<void>;
}

// In the create<PreferencesState>(...) body:
resetTutorial: async () => {
  if (get().writing) return;
  const previous = get().tutorialCompletedAt;
  set({ tutorialCompletedAt: null, writing: true, writeError: null });
  try {
    const payload = await updateUserComposerPreferences({
      tutorial_completed_at: null,
    });
    set({
      tutorialCompletedAt: payload.tutorial_completed_at,
      writing: false,
    });
  } catch (err) {
    set({
      tutorialCompletedAt: previous,
      writing: false,
      writeError:
        err instanceof Error
          ? `Couldn't reset the tutorial: ${err.message}`
          : "Couldn't reset the tutorial.",
    });
    throw err;
  }
},
```

The action mirrors `setDefaultMode`'s shape exactly — optimistic
update, revert-on-error, `writeError` surfaced through the existing
`role="alert"` region. No new error-display UI required.

Add a Vitest case to `preferencesStore.test.ts`:

```typescript
it('resetTutorial PATCHes tutorial_completed_at: null and clears the field', async () => {
  vi.spyOn(api, 'updateUserComposerPreferences').mockResolvedValue({
    default_mode: 'guided',
    banner_dismissed_at: null,
    tutorial_completed_at: null,
    updated_at: '2026-05-15T12:00:00Z',
  });
  usePreferencesStore.setState({
    tutorialCompletedAt: '2026-05-14T12:00:00Z',
  });
  await usePreferencesStore.getState().resetTutorial();
  expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
    tutorial_completed_at: null,
  });
  expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull();
});

it('resetTutorial reverts on error and sets writeError', async () => {
  vi.spyOn(api, 'updateUserComposerPreferences').mockRejectedValue(
    new Error('server down'),
  );
  usePreferencesStore.setState({
    tutorialCompletedAt: '2026-05-14T12:00:00Z',
  });
  await expect(
    usePreferencesStore.getState().resetTutorial(),
  ).rejects.toThrow('server down');
  expect(usePreferencesStore.getState().tutorialCompletedAt).toBe(
    '2026-05-14T12:00:00Z',
  );
  expect(usePreferencesStore.getState().writeError).toContain('server down');
});
```

- [ ] **Step 3: Write the failing component test.**

Create
`src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.reset.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ComposerPreferencesForm } from './ComposerPreferencesPanel';
import { usePreferencesStore } from '@/stores/preferencesStore';
import * as api from '@/api/client';

describe('ComposerPreferencesForm — Reset tutorial link', () => {
  beforeEach(() => {
    usePreferencesStore.setState({
      defaultMode: 'guided',
      loaded: true,
      writing: false,
      writeError: null,
      tutorialCompletedAt: null,
    });
  });

  it('does not render the Reset link when tutorial has not been completed', () => {
    usePreferencesStore.setState({ tutorialCompletedAt: null });
    render(<ComposerPreferencesForm />);
    expect(screen.queryByRole('button', { name: /reset tutorial/i })).toBeNull();
  });

  it('renders the Reset link when tutorial has been completed', () => {
    usePreferencesStore.setState({
      tutorialCompletedAt: '2026-05-14T12:00:00Z',
    });
    render(<ComposerPreferencesForm />);
    expect(
      screen.getByRole('button', { name: /reset tutorial/i }),
    ).toBeInTheDocument();
  });

  it('click → PATCHes tutorial_completed_at: null → tutorialCompletedAt becomes null', async () => {
    usePreferencesStore.setState({
      tutorialCompletedAt: '2026-05-14T12:00:00Z',
    });
    vi.spyOn(api, 'updateUserComposerPreferences').mockResolvedValue({
      default_mode: 'guided',
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      updated_at: '2026-05-15T12:00:00Z',
    });
    render(<ComposerPreferencesForm />);
    fireEvent.click(screen.getByRole('button', { name: /reset tutorial/i }));
    await waitFor(() =>
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_completed_at: null,
      }),
    );
    await waitFor(() =>
      expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull(),
    );
  });

  it('surfaces failure via the existing writeError role="alert" region', async () => {
    usePreferencesStore.setState({
      tutorialCompletedAt: '2026-05-14T12:00:00Z',
    });
    vi.spyOn(api, 'updateUserComposerPreferences').mockRejectedValue(
      new Error('server down'),
    );
    render(<ComposerPreferencesForm />);
    fireEvent.click(screen.getByRole('button', { name: /reset tutorial/i }));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/server down/i),
    );
  });
});
```

Run:

```bash
cd src/elspeth/web/frontend && npx vitest run \
  src/components/settings/ComposerPreferencesPanel.reset.test.tsx
```

Expected: FAIL — the Reset link does not yet exist in
`ComposerPreferencesForm`.

- [ ] **Step 4: Implement the Reset link in `ComposerPreferencesForm`.**

Edit `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`.
Extend `ComposerPreferencesForm`:

```typescript
export function ComposerPreferencesForm(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const writeError = usePreferencesStore((s) => s.writeError);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);
  const tutorialCompletedAt = usePreferencesStore(
    (s) => s.tutorialCompletedAt,
  );
  const resetTutorial = usePreferencesStore((s) => s.resetTutorial);

  const onChange = useCallback(
    async (mode: ComposerMode) => {
      const activeSessionId = useSessionStore.getState().activeSessionId;
      // writeError now carries any setDefaultMode failure — no console.error
      // needed (CLAUDE.md no-defensive-programming; the role="alert"
      // region is the user-visible channel).
      await setDefaultMode(mode, activeSessionId);
    },
    [setDefaultMode],
  );

  const onResetClick = useCallback(async () => {
    // Same shape as onChange — writeError captures failure, no console.
    await resetTutorial();
  }, [resetTutorial]);

  if (!loaded || defaultMode === null) return null;

  return (
    <>
      <fieldset disabled={writing} aria-busy={writing}>
        <legend>Default mode for new sessions</legend>
        {/* ... existing radio inputs ... */}
      </fieldset>

      {tutorialCompletedAt !== null && (
        <div style={{ marginTop: 16 }}>
          <button
            type="button"
            disabled={writing}
            onClick={() => void onResetClick()}
            aria-describedby="composer-preferences-reset-help"
            // Style as a link, not a primary button — the action is
            // reversible (the user just retakes the tutorial); no
            // confirmation modal needed per the brief.
            style={{
              background: 'none',
              border: 'none',
              padding: 0,
              color: 'var(--color-link, #0070d2)',
              textDecoration: 'underline',
              cursor: writing ? 'default' : 'pointer',
              fontSize: 13,
            }}
          >
            Reset tutorial
          </button>
          <p
            id="composer-preferences-reset-help"
            style={{ fontSize: 12, marginTop: 4, color: 'var(--color-text-muted, #666)' }}
          >
            Re-takes the hello-world tutorial on your next composer load.
            You can stop the tutorial at any point with the skip link in Turn 1.
          </p>
        </div>
      )}

      {writeError !== null && (
        <div
          role="alert"
          className="composer-preferences-error"
          style={{
            marginTop: 8,
            color: "var(--color-danger, #b00020)",
            fontSize: 13,
          }}
        >
          {writeError}
        </div>
      )}
    </>
  );
}
```

Note: the `onChange` handler in the snippet above also drops its
prior `console.error` — Step 4/6 of Task 12 already amends the
preferences-bootstrap path to route errors through `writeError`; the
same principle applies here. The pre-existing `setDefaultMode` action
captures failures onto `writeError`, so the inline `try/catch +
console.error` is dead defence and should go in this commit.

- [ ] **Step 5: Run the test to verify it passes.**

```bash
cd src/elspeth/web/frontend && npx vitest run \
  src/components/settings/ComposerPreferencesPanel.reset.test.tsx \
  src/stores/preferencesStore.test.ts
```

Expected: PASS.

- [ ] **Step 6: Run the full frontend test suite to catch regressions.**

```bash
cd src/elspeth/web/frontend && npx vitest run
```

Expected: PASS. If any existing `ComposerPreferencesPanel.test.tsx`
case asserts on the absence of additional content, fix the assertion —
the Reset link's presence is now a feature, and the test must reflect
the new shape.

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx \
  src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.reset.test.tsx \
  src/elspeth/web/frontend/src/stores/preferencesStore.ts \
  src/elspeth/web/frontend/src/stores/preferencesStore.test.ts
git commit -m "feat(frontend): Reset-tutorial link in ComposerPreferencesPanel (Phase 4B.14.5; closes Open Question C3)"
```

## Task 15: Staging smoke deploy

**Files:** none modified. Operator-led manual verification.

This is the final go-live step. The runbook below is the
operator-gated sequence per plan-fix P22 (2026-05-19) — destructive
steps are surfaced BEFORE downstream steps need them (per project
memory `feedback_operator_gate_destructive_actions`).

**Runbook (operator-gated):**

1. **DB-delete (TWO databases — both required).** Phase 4 ships TWO
   independent schema additions, each backed by an independent
   SQLite database; BOTH must be deleted before service restart or
   the caretaker-rebootstrapped schemas will refuse to load against
   pre-existing tables.

   1a. **Sessions DB delete** (per Task 1's OPERATOR ACTION note —
   SESSION_SCHEMA_EPOCH bumped 5 → 6 to add
   `user_preferences_table.tutorial_completed_at`):
   ```bash
   systemctl stop elspeth-web.service
   rm -f <data_dir>/sessions/sessions.db
   ```
   The exact path depends on the deployment's configured
   `ELSPETH_WEB__DATA_DIR`; on the staging source-checkout this is
   typically `data/sessions/sessions.db` relative to the repo root.

   1b. **Landscape audit DB delete** (per Task 7.0's OPERATOR ACTION
   note — five new audit-story columns added to `runs_table`):
   ```bash
   rm -f <data_dir>/runs/audit.db
   # Path convention verified 2026-05-19 against `src/elspeth/web/
   # config.py:493`: the Landscape SQLite DB lives at
   # `<data_dir>/runs/audit.db`, NOT `<data_dir>/landscape/audit.db`.
   # The live staging deployment uses `data/runs/audit.db` relative
   # to the repo root.
   ```

   1c. **Verification check** (between step 1 and step 2):
   ```bash
   ls <data_dir>/sessions/ <data_dir>/runs/ 2>/dev/null
   ```
   Neither `sessions.db` nor `audit.db` should appear in the listing
   before service restart. If either remains, abort the runbook and
   re-investigate — proceeding with a partial delete leaves staging
   in a broken state (one schema fresh, one stale).

   This is the second DB-delete event since Phase 1A — both DB
   deletes for Phase 4 are unavoidable. Surface to the operator
   BEFORE step 2; downstream steps assume both DBs are clean.
2. **Service restart.** `systemctl restart elspeth-web.service` (per
   project memory `project_staging_deployment` —
   elspeth.foundryside.dev is a source-checkout systemd/Caddy
   deploy). Frontend build (`npm run build` in
   `src/elspeth/web/frontend/`) precedes this if the umbrella branch
   was merged in step 0.
3. **Cache-warm.** Per the P13 deployment runbook step, run
   `elspeth tutorial warm-cache` against the deployed model config.
   Verifies the canonical pipeline cache entry exists before the
   first user hits the tutorial — otherwise the first Scenario A
   click-through runs an uncached ~30s pipeline rather than the
   designed sub-second cache hit.
4. **Playwright smoke.**
   ```bash
   cd src/elspeth/web/frontend
   WEBUI_TEST_PROVISION_ENABLED=true \
     npx playwright test tests/e2e/tutorial.spec.ts --project=chromium
   ```
   Assert all three P22 scenarios (A complete-flow, B skip-flow,
   C second-login) plus the refresh-restart case pass. If any fail,
   diagnose against the real backend — the failure is real, not a
   test-environment artifact (CLAUDE.md
   `feedback_fix_errors_you_encounter`). Do **not** silently swallow
   smoke failures with retries.
5. **Rollback recipe.** If smoke fails after any user rows have
   `tutorial_completed_at` set, rollback is:
   (a) operator DB-delete — BOTH databases (back to step 1a AND 1b:
   sessions DB AND Landscape audit DB); plus
   (b) cache-clear: `rm -rf <data_dir>/tutorial_cache/` to discard
   any stale cache entries written under the broken build. Surface
   this rollback to the operator the moment a smoke step fails;
   neither (a) nor (b) is reversible.

- [ ] **Step 1: Operator-led deploy (runbook steps 1–3).**

(Per `project_staging_deployment`: elspeth.foundryside.dev is a
source-checkout systemd/Caddy deploy; deploy steps follow that memory.)

- [ ] **Step 2: Manual click-through verification (complement to runbook step 4's Playwright smoke).**

Confirm each design-doc-04 promise visually. (Playwright covers the
machine-verifiable assertions; this step catches visual-only issues —
typography, copy nuance, link styling.)

- [ ] Turn 1 renders correctly. Skip link is visible but subtle.
- [ ] Turn 2 has the canonical seed pre-filled. Editing works. Restore-canonical
  button appears when edited.
- [ ] Turn 2b renders the 5 URLs, both transform names, the sink description.
  Interpretation review affordance is present and works.
- [ ] Turn 3 graph has 4 nodes and 3 arrows. Row count = 5 on the source.
- [ ] Turn 4 run completes (cache hit on the canonical seed → fast; cache
  miss → ~30s; first-user-on-fresh-cache miss is the realistic case).
- [ ] Turn 5 hashes are real (not "a7f3e2…" as canned text — actual hex
  prefixes from the run's Landscape entry).
- [ ] Turn 5's "Explore the full audit trail" link works.
- [ ] Turn 6 PATCHes both default_mode and tutorial_completed_at. After
  the click, the normal composer is visible with the session named
  "hello-world (cool government pages)" in the session list.
- [ ] Second login of the same user does NOT re-fire the tutorial.
- [ ] A different brand-new user DOES see the tutorial (verifying
  per-user `tutorial_completed_at`).
- [ ] **Tutorial cache directory** is created under the deployment's
  configured `data_dir` on first tutorial hit. With no
  `ELSPETH_WEB__TUTORIAL_CACHE_DIR` override set, the resolved path is
  `<ELSPETH_WEB__DATA_DIR>/tutorial_cache/` (which for a dev/staging
  source-checkout falls under the relative `data/` directory, since
  `WebSettings.data_dir` defaults to `Path("data")`). Verify the
  directory exists and contains a `<sha256>.json` file after the
  click-through; this confirms the P8 fix — no `ELSPETH_DATA_DIR` env
  var and no `/var/lib/elspeth` hardcoded path are required. (Detailed
  defaults-resolution coverage lives in
  `test_tutorial_cache_dir_defaults_to_data_dir_subdir`; this smoke
  check is the end-to-end confirmation.)

- [ ] **Step 3: Record verification in the merge PR.**

Note in the PR description (or in an issue tracking comment) that the
staging smoke passed all checks. Include the date and the operator's name.

- [ ] **Step 4: Tag the deploy commit (optional, per project convention).**

If the project tags release commits, tag the merge.

---

## What Phase 4B leaves the system in

After Tasks 7.5, 8–14, 14.5, 15–16: brand-new users see the tutorial; returning users see the normal composer. Skip fast-forwards to turn 6. Finalisation PATCHes both fields atomically; session renamed. Refresh during the tutorial restarts at turn 1, and the tutorial container's mount-time `DELETE /api/tutorial/orphans` call cleans up any session a refresh orphaned (Systems R2-S5; no `sessionStorage` scaffolding). A completed-tutorial user can retake via the "Reset tutorial" link in `ComposerPreferencesPanel` (Task 14.5; closes Open Question C3 in-phase — no sibling-plan deferral). Vitest + Playwright pass.

## Risks and mitigations

Key risks: Phase 5b `InterpretationReviewTurn` API drift (Task 6 Step 1 recon resolves); response-shape mismatch for `runTutorialPipeline` (TypeScript errors surface at compile time); session-rename API absent (Task 10 Step 1 resolves; hard requirement); Playwright not set up (Task 14 Step 1 surfaces; escalate to operator).

## Forward compatibility

Phase 8 schema additions wipe `tutorial_completed_at` via DB-delete — users retake the tutorial. Structural fix (Alembic) owned by roadmap.

## Memory references

- `project_composer_first_run_tutorial`
- `project_composer_canonical_test_case`
- `project_composer_dynamic_source_from_chat`
- `project_composer_default_guided_with_opt_out`
- `project_staging_deployment`
- `project_db_migration_policy`
- `feedback_no_calendar_shipping_commitments`
- `feedback_fix_errors_you_encounter`

---

## Review history

### 2026-05-15 — review panel

| ID | Severity | Status | Summary |
|---|---|---|---|
| 4B2-F1 | CRITICAL (Quality) | Applied | `runTutorialPipeline` spy confirmed against `POST /api/tutorial/run` added to 21a; Task 8 updated |
| 4B2-F2 | IMPORTANT (Quality) | Applied | Playwright CI matrix task (Task 15b) added |
| 4B2-F3 | IMPORTANT (Quality) | Applied | Turn 5 `getRunAuditSummary` cross-referenced to 21a `GET /api/sessions/{id}/runs/{run_id}/audit-story` |
| 4B2-F4 | CRITICAL (Systems) | Superseded | Task 12's sessionStorage-resume pattern (originally applied per 21b1-F1) was deleted per Phase 4 plan-fix P10 (2026-05-19): flat `elspeth_tutorial_progress` key isn't user-scoped, so it risks cross-user contamination on shared workstations (systems S6). Restart-at-turn-1 is the documented contract; the ~5-minute tutorial plus canonical-seed cache make restart cost acceptable. |
| 4B2-F5 | BLOCKER (Coherence) | Applied | Task 12 banner code dropped per 2026-05-16 review (No Legacy Code Policy — Turn 1 welcome is the single entry surface for all tutorial-mode users) |

### 2026-05-19 — Phase 4 plan-fixes (P10 / P22)

| ID | Severity | Status | Summary |
|---|---|---|---|
| P10 | CRITICAL (Systems) | Applied | Task 12 sessionStorage scaffolding deleted (`PROGRESS_KEY`, `initialStep`, `onStepChange`). Refresh restarts at turn 1; Playwright case added to Task 14 asserting the contract (`refresh mid-tutorial restarts at turn 1`). Risk-table row in plan 21 updated. |
| P22 | IMPORTANT (Quality) | Applied | Task 14 restructured around a `freshUser` Playwright fixture; three named scenarios (A complete-flow, B skip-flow with no-pipeline-creation invariant, C second-login). Task 15 staging-smoke replaced with the 5-step operator-gated runbook (DB-delete → restart → cache-warm → Playwright → rollback). Test-only `POST /api/test/provision-fresh-user` endpoint specified (gated by `WEBUI_TEST_PROVISION_ENABLED=true`); recon flagged no existing surface to reuse as of 2026-05-19. |

### 2026-05-19 — frontend r2 review closure (Architecture / Systems / Quality)

| ID | Severity | Status | Summary |
|---|---|---|---|
| M-R2-2 | MAJOR (Architecture) | Applied | Task 7.3 (frontend client functions) relocated from 21a2 to a new **Task 7.5** in this plan — these are frontend `client.ts` symbols and belong with the frontend plan. `deleteTutorialOrphans` added as a fourth function in the same task. |
| M-R2-3 | MAJOR (Architecture) | Applied | Stale `21b-phase-4-frontend.md` cross-reference in the overview replaced with explicit `21b1` and `21b2` links. |
| M-R2-4 | MAJOR (Architecture) | Applied | Overview C3 row + Risks-table "skip but wants to come back" row rewritten to reference the in-phase Reset link (Task 14.5) instead of the non-existent sibling settings-panel plan. |
| N-R2-3 | MINOR (Architecture) | Applied | Task 12's bootstrap effect drops its `.catch((err) => console.error(...))`. Step 6 added to ensure `preferencesStore.bootstrap` captures errors onto `writeError` and re-throws so the React error boundary catches catastrophic failures. |
| R2-15 | MAJOR (Reality) | N/A (frontend) | Hallucinated `PipelineState` type — confirmed zero hits in 21b1/21b2. Backend half of the fix (live type is `CompositionState`) is owned by Writer C in 21a1. |
| R2-M4 | MAJOR (Quality) | Applied | Task 6 Step 7 contract test reads `pendingBySession[sid][eventId]` directly from the store — bypassing the live `InterpretationReviewTurn`'s `?? ""` defensive fallbacks (lines 135–136 of `chat/guided/InterpretationReviewTurn.tsx`). Follow-up note added: the live component's defensive fallbacks should be tightened to match the contract; the `?? ""` masks the very regressions this contract test exists to catch. Filed as a Phase-4-followup ticket scope (component cleanup is out-of-scope for this plan, but the test now pins the contract correctly). |
| R2-S5 | CRITICAL (Systems) | Applied | Orphan-session cleanup wired in Task 11 (container mount-time `DELETE /api/tutorial/orphans`) plus the new `deleteTutorialOrphans` client function in Task 7.5. Backend half (the endpoint itself) is owned by Writer C as 21a2 Task 7.4. |
| R2-S7 | CRITICAL (Systems) | Applied | New **Task 14.5** ships the Reset-tutorial link in `ComposerPreferencesPanel.tsx` plus a `resetTutorial` action on `preferencesStore`. Closes Open Question C3 in-phase — no sibling-plan deferral. |

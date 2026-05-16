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
- [21a-phase-4-backend.md](21a-phase-4-backend.md) — schema column, service
  extension, route extension, tutorial cache, run-path integration.
- [21b1-phase-4-frontend-part-1.md](21b1-phase-4-frontend-part-1.md) — store,
  state machine, copy, turns 1/2/2b/3.

**Overview document:** [21-phase-4-hello-world-tutorial.md](21-phase-4-hello-world-tutorial.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope (this part):**

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
  leaf component per step.
- `App.tsx` — wire tutorial detection so first-session users see
  `HelloWorldTutorial` instead of the normal composer surface.
- Vitest full-flow integration test exercising all 6 turns with mocked APIs.
- Playwright E2E for a brand-new user end-to-end on staging.
- Staging smoke deploy + manual click-through verification.

**Out of scope:**

- Anything in Part 1 (already landed).
- Anything in 21a (backend).
- Re-take from settings (Open Question C3 — post-launch).
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
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      source_data_hash: 'a7f3e2',
      rows: [
        { url: 'australia.gov.au', score: 6, rationale: 'dated' },
        { url: 'dta.gov.au', score: 9, rationale: 'bold' },
      ],
    });
    render(<TutorialTurn4Run sessionId="sess-1" onCompleted={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText(/australia\.gov\.au/)).toBeInTheDocument(),
    );
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getByText('9')).toBeInTheDocument();
    expect(screen.getByText(/dated/)).toBeInTheDocument();
  });

  it('fires onCompleted with run details on continue', async () => {
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      source_data_hash: 'a7f3e2',
      rows: [{ url: 'a', score: 5 }],
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
      source_data_hash: 'a7f3e2',
      rows: [
        { url: 'a', score: 5 },
        { url: 'broken.gov.au', error: 'HTTP 503' },
      ],
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
    runTutorialPipeline({ session_id: sessionId })
      .then((response) =>
        setResult({
          runId: response.run_id,
          sourceDataHash: response.source_data_hash,
          rows: response.rows,
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
- The LLM-call row count (read from the run's Landscape entry via the
  existing audit-readiness API — see Phase 2 audit-readiness work).

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
      run_started_at: '2026-05-15T12:00:00Z',
      plugin_versions: { web_scrape: '1.0.0', llm_rate: '1.0.0' },
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
    expect(link.getAttribute('href') ?? '').toContain('run-1');
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
  run_started_at: string;
  plugin_versions: Record<string, string>;
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
grep -rn "renameSession\|updateSession.*title" \
  src/elspeth/web/frontend/src/api --include="*.ts" | head -10
```

Confirm the function name and signature. If renames go through
`sessionStore`, prefer the store path.

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
    const markSpy = vi.spyOn(
      usePreferencesStore.getState(),
      'markTutorialCompleted',
    );
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() => expect(markSpy).toHaveBeenCalledWith('guided'));
  });

  it('marks tutorial complete with freeform when freeform clicked', async () => {
    const markSpy = vi.spyOn(
      usePreferencesStore.getState(),
      'markTutorialCompleted',
    );
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
    vi.spyOn(usePreferencesStore.getState(), 'markTutorialCompleted')
      .mockResolvedValue();
    render(<TutorialTurn6ModeChoice sessionId="sess-1" onDone={() => {}} />);
    fireEvent.click(
      screen.getByRole('button', { name: TURN_6_GUIDED_BUTTON }),
    );
    await waitFor(() =>
      expect(renameSpy).toHaveBeenCalledWith('sess-1', TUTORIAL_SESSION_NAME),
    );
  });

  it('fires onDone after both PATCH and rename complete', async () => {
    vi.spyOn(usePreferencesStore.getState(), 'markTutorialCompleted')
      .mockResolvedValue();
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
    vi.spyOn(usePreferencesStore.getState(), 'markTutorialCompleted')
      .mockImplementation(
        () => new Promise<void>((res) => { resolveMark = res; }),
      );
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
    const composeSpy = vi.spyOn(api, 'composePipelineFromPrompt');
    const runSpy = vi.spyOn(api, 'runTutorialPipeline');
    render(<HelloWorldTutorial onDone={() => {}} />);
    fireEvent.click(screen.getByText(/skip this/i));
    await waitFor(() =>
      expect(screen.getByLabelText(/Guided/i)).toBeInTheDocument(),
    );
    expect(composeSpy).not.toHaveBeenCalled();
    expect(runSpy).not.toHaveBeenCalled();
  });

  it('fires onDone when finalised', async () => {
    // Full integration: walk through all 6 turns with mocked APIs.
    // (Detailed mock setup omitted here; see Task 12's integration test for
    // the full end-to-end variant.)
  });
});
```

- [ ] **Step 2: Run test to verify it fails.**

Expected: FAIL.

- [ ] **Step 3: Implement.**

```typescript
import { useReducer } from 'react';
import { initialState, tutorialReducer } from './tutorialMachine';
import { TutorialTurn1Welcome } from './TutorialTurn1Welcome';
import { TutorialTurn2Describe } from './TutorialTurn2Describe';
import { TutorialTurn2bShowBuilt } from './TutorialTurn2bShowBuilt';
import { TutorialTurn3Graph } from './TutorialTurn3Graph';
import { TutorialTurn4Run } from './TutorialTurn4Run';
import { TutorialTurn5AuditStory } from './TutorialTurn5AuditStory';
import { TutorialTurn6ModeChoice } from './TutorialTurn6ModeChoice';

interface Props {
  onDone: (mode: 'guided' | 'freeform') => void;
}

export function HelloWorldTutorial({ onDone }: Props) {
  const [state, dispatch] = useReducer(tutorialReducer, initialState);

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
- Where preferences are loaded (likely `useEffect(() => loadPreferences(), [])`).
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
    vi.spyOn(api, 'getComposerPreferences').mockResolvedValue({
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
    vi.spyOn(api, 'getComposerPreferences').mockResolvedValue({
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
    vi.spyOn(api, 'getComposerPreferences').mockResolvedValue({
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

Pattern (adapt to App.tsx's actual structure). Include sessionStorage-resume
(per 21b1 §"Navigation resilience"). **No separate DB-delete banner** — per
21b1 §"Navigation resilience", all tutorial-mode users (genuine first-timers
and DB-delete-resets) see the same Turn 1 welcome copy. CLAUDE.md No Legacy
Code Policy: we have no users yet, the operator deletes the DB freely, and a
bifurcated banner would detect a state we can't reliably distinguish. One
copy, treated identically.

```typescript
import { HelloWorldTutorial } from './components/tutorial';
import { usePreferencesStore } from './stores/preferencesStore';

const PROGRESS_KEY = 'elspeth_tutorial_progress';

export function App() {
  const tutorialCompleted = usePreferencesStore((s) => s.tutorialCompleted);
  const loaded = usePreferencesStore((s) => s.loaded);
  const loadPreferences = usePreferencesStore((s) => s.loadPreferences);

  useEffect(() => { loadPreferences(); }, [loadPreferences]);

  if (!loaded) return <div>Loading…</div>;

  if (!tutorialCompleted) {
    const saved = sessionStorage.getItem(PROGRESS_KEY);
    const initialStep = saved ? JSON.parse(saved) : undefined;
    return (
      <HelloWorldTutorial
        initialStep={initialStep}
        onStepChange={(s) => sessionStorage.setItem(PROGRESS_KEY, JSON.stringify(s))}
        onDone={() => sessionStorage.removeItem(PROGRESS_KEY)}
      />
    );
  }
  return <NormalComposerSurface />;
}
```

Wire `initialStep` / `onStepChange` props through `HelloWorldTutorial` to `useReducer`; call `onStepChange` after every `dispatch` in the container.

- [ ] **Step 5: Run test to verify it passes.**

Expected: PASS.

- [ ] **Step 6: Run the full frontend test suite to catch regressions.**

```bash
cd src/elspeth/web/frontend && npx vitest run
```

Expected: PASS. If existing App.tsx tests broke, the bootstrap path
changed under them; fix the tests or the implementation depending on
intent.

- [ ] **Step 7: Commit.**

```bash
git add src/elspeth/web/frontend/src/App.tsx \
  src/elspeth/web/frontend/src/App.test.tsx
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
import * as api from '../../api/client';
import * as interpretationApi from '../../api/interpretationEvents';

describe('HelloWorldTutorial — full 6-turn integration', () => {
  it('walks from welcome to done with cached run', async () => {
    vi.spyOn(api, 'composePipelineFromPrompt').mockResolvedValue({
      session_id: 'sess-1',
      pipeline_snapshot: {
        source: { type: 'inline_blob', urls: ['a.gov.au'] },
        transforms: [{ type: 'web_scrape' }, { type: 'llm_rate' }],
        sinks: [{ type: 'jsonl' }],
      },
    });
    vi.spyOn(
      interpretationApi,
      'listPendingInterpretationEvents',
    ).mockResolvedValue([
      { id: 'evt-1', tool_call_id: 'tc-1', draft_value: 'modern design' },
    ]);
    vi.spyOn(interpretationApi, 'resolveInterpretationEvent').mockResolvedValue({
      id: 'evt-1',
      status: 'accepted',
      accepted_value: 'modern design',
    });
    vi.spyOn(api, 'runTutorialPipeline').mockResolvedValue({
      run_id: 'r1',
      source_data_hash: 'a7f3e2',
      rows: [{ url: 'a.gov.au', score: 7 }],
    });
    vi.spyOn(api, 'getRunAuditSummary').mockResolvedValue({
      llm_call_count: 1,
      output_file_hash: 'cafebabe',
      run_started_at: '2026-05-15T12:00:00Z',
      plugin_versions: {},
    });
    vi.spyOn(api, 'renameSession').mockResolvedValue({
      id: 'sess-1',
      title: 'hello-world (cool government pages)',
    });
    const patchSpy = vi
      .spyOn(api, 'updateComposerPreferences')
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
- Create: `tests/e2e/tutorial.e2e.spec.ts` (or per project Playwright location).

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

```typescript
import { test, expect } from '@playwright/test';

test.describe('Hello-world tutorial — brand-new user', () => {
  test.beforeAll(async () => {
    // Delete the sessions DB and restart the service.
    // The operator's standard staging-deploy invocation for this is:
    //   systemctl stop elspeth-web.service
    //   rm /var/lib/elspeth/sessions.db
    //   systemctl start elspeth-web.service
    // The Playwright test runner orchestrates this via a pytest fixture or
    // a global setup hook — pattern confirmed during Task 14 Step 1 recon.
  });

  test('first-session user sees the tutorial and completes it', async ({ page }) => {
    // 1. Log in as a brand-new user.
    await page.goto('https://elspeth.foundryside.dev');
    await page.fill('[data-testid="login-username"]', 'tutorial-tester');
    await page.fill('[data-testid="login-password"]', 'redacted');
    await page.click('[data-testid="login-submit"]');

    // 2. Tutorial turn 1 renders.
    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();

    // 3. Click through to turn 2.
    await page.click('button:has-text("Let\'s go")');

    // 4. Confirm the prompt is pre-filled.
    const textarea = page.locator('textarea').first();
    await expect(textarea).toHaveValue(
      /create a list of 5 government web pages and use an LLM to rate how cool they are/,
    );

    // 5. Click "Build it" — this consults the cache or runs live.
    await page.click('button:has-text("Build it")');

    // 6. Turn 2b: the LLM's interpretation surface renders.
    await expect(page.getByText(/Got it/)).toBeVisible({ timeout: 30000 });

    // 7. Accept the LLM's interpretation.
    await page.click('button:has-text("Use my interpretation")');

    // 8. Turn 3: the graph renders.
    await expect(page.getByText(/Three layers, four steps/)).toBeVisible();
    await page.click('button:has-text("Looks good, run it")');

    // 9. Turn 4: run completes (cached, so fast).
    await expect(page.locator('table')).toBeVisible({ timeout: 30000 });

    // 10. Continue to turn 5.
    await page.click('button:has-text("Continue")');

    // 11. Turn 5: audit story renders with real hashes.
    await expect(page.locator('code').first()).toBeVisible();
    await page.click('button:has-text("Continue")');

    // 12. Turn 6: mode choice. Pick guided.
    await expect(page.getByText(/What should new sessions default to/)).toBeVisible();
    await page.click('button:has-text("Guided (recommended)")');

    // 13. Tutorial is done; the normal composer renders.
    await expect(page.getByText('Welcome to ELSPETH.')).not.toBeVisible();
    await expect(page.getByText(/hello-world \(cool government pages\)/)).toBeVisible();
  });

  test('second login does not re-fire the tutorial', async ({ page }) => {
    // Continuing from the previous test's state: tutorial-tester has
    // tutorial_completed_at set.
    await page.goto('https://elspeth.foundryside.dev');
    await page.fill('[data-testid="login-username"]', 'tutorial-tester');
    await page.fill('[data-testid="login-password"]', 'redacted');
    await page.click('[data-testid="login-submit"]');

    await expect(page.getByText('Welcome to ELSPETH.')).not.toBeVisible({
      timeout: 5000,
    });
  });

  test('skip path fast-forwards to turn 6 without building', async ({ page }) => {
    // A different brand-new user for this test.
    await page.goto('https://elspeth.foundryside.dev');
    await page.fill('[data-testid="login-username"]', 'skip-tester');
    await page.fill('[data-testid="login-password"]', 'redacted');
    await page.click('[data-testid="login-submit"]');

    await expect(page.getByText('Welcome to ELSPETH.')).toBeVisible();
    await page.click('text=I\'ve used ELSPETH before, skip this');

    // Should land directly on the mode-choice screen.
    await expect(
      page.getByText(/What should new sessions default to/),
    ).toBeVisible();
    // No tutorial session was created.
    await page.click('button:has-text("Guided (recommended)")');
    await expect(page.getByText(/hello-world \(cool government pages\)/)).not.toBeVisible();
  });
});
```

- [ ] **Step 3: Run test against staging.**

```bash
npx playwright test tests/e2e/tutorial.e2e.spec.ts --headed
```

Expected: PASS — all three scenarios green. If any fail, diagnose against
the real backend (the failure is real, not a test-environment artifact —
CLAUDE.md `feedback_fix_errors_you_encounter`).

- [ ] **Step 4: Add tutorial spec to the Playwright CI matrix.**

Open `playwright.config.ts` (or the GitHub Actions Playwright workflow — confirm during Step 1 recon). Add the new spec to `testMatch` or the workflow's spec list:

```typescript
testMatch: [/* existing... */ 'tests/e2e/tutorial.e2e.spec.ts']
```

- [ ] **Step 5: Commit.**

```bash
git add tests/e2e/tutorial.e2e.spec.ts playwright.config.ts
git commit -m "test(e2e): add tutorial brand-new-user end-to-end + CI matrix (Phase 4B.14)"
```

## Task 15: Staging smoke deploy

**Files:** none modified. Operator-led manual verification.

This is the final go-live step. The operator (or the implementing agent
with operator assistance, per `project_staging_deployment`):

1. Merges the umbrella branch to the deployment branch.
2. Stops `elspeth-web.service`.
3. **Deletes the sessions DB** (the second DB-delete since Phase 1A —
   plan 21 explicitly acknowledges this).
4. Runs the frontend build (`npm run build` in
   `src/elspeth/web/frontend/`).
5. Starts `elspeth-web.service`.
6. Clicks through the tutorial in a brand-new browser session.

- [ ] **Step 1: Operator-led deploy.**

(Per `project_staging_deployment`: elspeth.foundryside.dev is a
source-checkout systemd/Caddy deploy; deploy steps follow that memory.)

- [ ] **Step 2: Manual click-through verification.**

Confirm each design-doc-04 promise visually:

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

- [ ] **Step 3: Record verification in the merge PR.**

Note in the PR description (or in an issue tracking comment) that the
staging smoke passed all checks. Include the date and the operator's name.

- [ ] **Step 4: Tag the deploy commit (optional, per project convention).**

If the project tags release commits, tag the merge.

---

## What Phase 4B leaves the system in

After Tasks 1–15: brand-new users see the tutorial; returning users see the normal composer. Skip fast-forwards to turn 6. Finalisation PATCHes both fields atomically; session renamed. Tutorial progress persists across refresh via `sessionStorage`. Vitest + Playwright pass.

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
| 4B2-F4 | CRITICAL (Systems) | Applied | Task 12 updated: sessionStorage-resume pattern (from 21b1-F1); implementation lives here |
| 4B2-F5 | BLOCKER (Coherence) | Applied | Task 12 banner code dropped per 2026-05-16 review (No Legacy Code Policy — Turn 1 welcome is the single entry surface for all tutorial-mode users) |

# Composer Progress Persistence — Phase 4: Frontend Recovery Panel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the React modal that opens when a compose request fails with `partial_state` + `failed_turn` in the response body. Show a pipeline diff against the editor's current state and a tool transcript; let the user Apply (with concurrent-edit confirmation) or Discard. Wire the existing `useComposer.sendMessage` `onError` path to trigger the panel.

**Architecture:** Frontend-only. No backend changes. Phase 3's response shape (`failed_turn`, `partial_state`) drives the panel; Phase 1's `audit_access_log` table records the read when the panel fetches the transcript. Component layout follows spec §7.1.

**Tech Stack:** React, TypeScript, Vitest for unit tests, the existing project Modal pattern, and the existing `useComposer` hook.

**Spec sections:** §7 (Frontend Recovery Surface — full), §8.4 (frontend tests), §11 Phase 4 scope. Accessibility requirements: §7.7.

---

## File Structure

### Files to create

- `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx`
- `src/elspeth/web/frontend/src/components/recovery/recoveryTypes.ts`
- `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts`
- `src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryPanel.test.tsx`
- `src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts`

### Files to modify

- `src/elspeth/web/frontend/src/hooks/useComposer.ts` — extend the `onError` path to detect `partial_state + failed_turn` and open the recovery panel.

### Files NOT touched in Phase 4

- Backend code. Phase 4 is frontend-only.
- Live chat panel (`components/chat/`) — its default `include_tool_rows=false` is preserved per spec §7.8.

---

## Task 1: Recovery types module

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/recoveryTypes.ts`

- [ ] **Step 1: Define the response-body types**

Create the file:

```typescript
/**
 * Types matching the rev-4 error response body shape (spec §6.1).
 */

export interface FailedTurn {
  assistant_message_id: string;
  tool_calls_attempted: number;
  tool_responses_persisted: number;
  transcript_url: string;
}

export interface PartialState {
  // Mirrors the CompositionState shape used elsewhere in the editor.
  // Keep this aligned with the existing CompositionState type in
  // src/elspeth/web/frontend/src/types/state.ts.
  source: unknown;
  nodes: ReadonlyArray<unknown>;
  edges: ReadonlyArray<unknown>;
  version: number;
}

export interface ComposerErrorResponse {
  error: string;
  reason: string;
  headline: string;
  evidence: ReadonlyArray<string>;
  partial_state: PartialState | null;
  failed_turn: FailedTurn | null;
}

export interface ToolTranscriptRow {
  id: string;
  role: 'tool';
  tool_call_id: string;
  parent_assistant_id: string;
  sequence_no: number;
  content: string;            // JSON-string of the redacted tool response
  composition_state_id: string | null;
  created_at: string;
}

export interface AssistantTranscriptRow {
  id: string;
  role: 'assistant';
  sequence_no: number;
  content: string;
  tool_calls: ReadonlyArray<{ id: string; function: { name: string; arguments: unknown } }>;
  composition_state_id: string | null;
  created_at: string;
}

export type TranscriptRow = ToolTranscriptRow | AssistantTranscriptRow;
```

- [ ] **Step 2: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/recoveryTypes.ts
git commit -m "feat(frontend): recovery panel types (composer-progress-persistence phase 4)"
```

---

## Task 2: `useRecoveryPanel` hook — gating matrix

**Files:**
- Create: `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts`
- Create: `src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { useRecoveryPanel } from '../useRecoveryPanel';
import type { ComposerErrorResponse } from '../../components/recovery/recoveryTypes';


function makeResponse(overrides: Partial<ComposerErrorResponse> = {}): ComposerErrorResponse {
  return {
    error: 'ComposerConvergenceError',
    reason: 'convergence_wall_clock_timeout',
    headline: 'Compose request timed out',
    evidence: [],
    partial_state: { source: null, nodes: [], edges: [], version: 1 },
    failed_turn: {
      assistant_message_id: 'a_1',
      tool_calls_attempted: 4,
      tool_responses_persisted: 3,
      transcript_url: '/api/sessions/s/messages?since=u_1&include_tool_rows=true',
    },
    ...overrides,
  };
}


describe('useRecoveryPanel — gating matrix (spec §7.2)', () => {
  it('opens when both partial_state and failed_turn are present', () => {
    const { result } = renderHook(() => useRecoveryPanel());
    act(() => result.current.handleError(makeResponse()));
    expect(result.current.isOpen).toBe(true);
  });

  it('does NOT open when partial_state is missing', () => {
    const { result } = renderHook(() => useRecoveryPanel());
    act(() => result.current.handleError(makeResponse({ partial_state: null })));
    expect(result.current.isOpen).toBe(false);
  });

  it('does NOT open when failed_turn is missing', () => {
    const { result } = renderHook(() => useRecoveryPanel());
    act(() => result.current.handleError(makeResponse({ failed_turn: null })));
    expect(result.current.isOpen).toBe(false);
  });

  it('does NOT open when both are missing', () => {
    const { result } = renderHook(() => useRecoveryPanel());
    act(() => result.current.handleError(makeResponse({
      partial_state: null,
      failed_turn: null,
    })));
    expect(result.current.isOpen).toBe(false);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src/elspeth/web/frontend && npm test -- useRecoveryPanel
```
Expected: FAIL — hook does not exist.

- [ ] **Step 3: Implement the hook**

Create `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts`:

```typescript
import { useCallback, useState } from 'react';

import type {
  ComposerErrorResponse,
  PartialState,
  FailedTurn,
} from '../components/recovery/recoveryTypes';


export interface RecoveryPanelState {
  isOpen: boolean;
  partialState: PartialState | null;
  failedTurn: FailedTurn | null;
  reason: string | null;
  headline: string | null;
  evidence: ReadonlyArray<string>;
  handleError: (resp: ComposerErrorResponse) => void;
  apply: (opts: { confirmedConcurrentEdit?: boolean }) => Promise<{ applied: boolean; needsConfirmation?: boolean }>;
  discard: () => void;
}


export function useRecoveryPanel(opts: {
  onApplyState?: (state: PartialState) => void;
  hasUnsavedEditorEdits?: () => boolean;
} = {}): RecoveryPanelState {
  const [isOpen, setIsOpen] = useState(false);
  const [partialState, setPartialState] = useState<PartialState | null>(null);
  const [failedTurn, setFailedTurn] = useState<FailedTurn | null>(null);
  const [reason, setReason] = useState<string | null>(null);
  const [headline, setHeadline] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<ReadonlyArray<string>>([]);

  const handleError = useCallback((resp: ComposerErrorResponse) => {
    if (resp.partial_state == null || resp.failed_turn == null) {
      // Spec §7.2: only the (present, present) case opens the panel.
      return;
    }
    setPartialState(resp.partial_state);
    setFailedTurn(resp.failed_turn);
    setReason(resp.reason);
    setHeadline(resp.headline);
    setEvidence(resp.evidence);
    setIsOpen(true);
  }, []);

  const apply = useCallback(
    async ({ confirmedConcurrentEdit }: { confirmedConcurrentEdit?: boolean }) => {
      if (partialState == null) {
        return { applied: false };
      }
      if (
        opts.hasUnsavedEditorEdits?.() &&
        !confirmedConcurrentEdit
      ) {
        // Spec §7.6: request explicit confirmation before overwriting unsaved edits.
        return { applied: false, needsConfirmation: true };
      }
      opts.onApplyState?.(partialState);
      setIsOpen(false);
      return { applied: true };
    },
    [partialState, opts],
  );

  const discard = useCallback(() => {
    // Spec §7.6: discard is a UI choice, NOT a data-deletion command.
    // The DB record remains for audit; the panel just closes.
    setIsOpen(false);
  }, []);

  return {
    isOpen,
    partialState,
    failedTurn,
    reason,
    headline,
    evidence,
    handleError,
    apply,
    discard,
  };
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd src/elspeth/web/frontend && npm test -- useRecoveryPanel
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts
git commit -m "feat(frontend): useRecoveryPanel hook with gating matrix (composer-progress-persistence phase 4)"
```

---

## Task 3: `useRecoveryPanel` — concurrent-edit guard

**Files:**
- Test: `src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts` (extend)

- [ ] **Step 1: Write the failing tests**

Add to the test file:

```typescript
describe('useRecoveryPanel — concurrent-edit guard (spec §7.6)', () => {
  it('returns needsConfirmation=true when editor has unsaved edits', async () => {
    const onApply = vi.fn();
    const { result } = renderHook(() =>
      useRecoveryPanel({
        onApplyState: onApply,
        hasUnsavedEditorEdits: () => true,
      }),
    );
    act(() => result.current.handleError(makeResponse()));

    let outcome: { applied: boolean; needsConfirmation?: boolean } = { applied: false };
    await act(async () => {
      outcome = await result.current.apply({});
    });
    expect(outcome.applied).toBe(false);
    expect(outcome.needsConfirmation).toBe(true);
    expect(onApply).not.toHaveBeenCalled();
    expect(result.current.isOpen).toBe(true);  // panel stays open
  });

  it('applies when concurrent-edit confirmation is provided', async () => {
    const onApply = vi.fn();
    const { result } = renderHook(() =>
      useRecoveryPanel({
        onApplyState: onApply,
        hasUnsavedEditorEdits: () => true,
      }),
    );
    act(() => result.current.handleError(makeResponse()));

    let outcome: { applied: boolean } = { applied: false };
    await act(async () => {
      outcome = await result.current.apply({ confirmedConcurrentEdit: true });
    });
    expect(outcome.applied).toBe(true);
    expect(onApply).toHaveBeenCalledOnce();
    expect(result.current.isOpen).toBe(false);
  });

  it('applies without confirmation when editor has no unsaved edits', async () => {
    const onApply = vi.fn();
    const { result } = renderHook(() =>
      useRecoveryPanel({
        onApplyState: onApply,
        hasUnsavedEditorEdits: () => false,
      }),
    );
    act(() => result.current.handleError(makeResponse()));

    let outcome: { applied: boolean } = { applied: false };
    await act(async () => {
      outcome = await result.current.apply({});
    });
    expect(outcome.applied).toBe(true);
    expect(onApply).toHaveBeenCalledOnce();
  });
});

describe('useRecoveryPanel — discard (spec §7.6)', () => {
  it('closes the panel without invoking onApplyState', async () => {
    const onApply = vi.fn();
    const { result } = renderHook(() => useRecoveryPanel({ onApplyState: onApply }));
    act(() => result.current.handleError(makeResponse()));
    expect(result.current.isOpen).toBe(true);

    act(() => result.current.discard());
    expect(result.current.isOpen).toBe(false);
    expect(onApply).not.toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run tests**

```bash
cd src/elspeth/web/frontend && npm test -- useRecoveryPanel
```
Expected: PASS — the previous task's hook implementation already covers this.

If any test fails, fix the hook accordingly.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/__tests__/useRecoveryPanel.test.ts
git commit -m "test(frontend): concurrent-edit guard and discard semantics (composer-progress-persistence phase 4)"
```

---

## Task 4: `RecoveryDiff` component

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx`

- [ ] **Step 1: Write the failing test**

Create `src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryDiff.test.tsx`:

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import { RecoveryDiff } from '../RecoveryDiff';


describe('RecoveryDiff (spec §7.4)', () => {
  it('shows added/removed/changed sections', () => {
    const current = { source: { type: 'csv' }, nodes: [], edges: [] };
    const partial = {
      source: { type: 'csv' },
      nodes: [{ id: 'classify' }, { id: 'results' }],
      edges: [],
      version: 2,
    };
    render(<RecoveryDiff current={current} partial={partial} />);
    expect(screen.getByText(/classify/)).toBeInTheDocument();
    expect(screen.getByText(/results/)).toBeInTheDocument();
  });

  it('renders empty state when current and partial are identical', () => {
    const x = { source: null, nodes: [], edges: [], version: 0 };
    render(<RecoveryDiff current={x} partial={x} />);
    expect(screen.getByText(/no changes/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryDiff
```
Expected: FAIL.

- [ ] **Step 3: Implement the component**

```typescript
import { type FC, useMemo } from 'react';
import type { PartialState } from './recoveryTypes';


interface RecoveryDiffProps {
  current: { source: unknown; nodes: ReadonlyArray<{ id: string }>; edges: unknown[] };
  partial: PartialState;
}


interface DiffEntry {
  kind: 'added' | 'removed' | 'changed';
  category: 'source' | 'node' | 'edge';
  label: string;
  detail?: string;
}


function computeDiff(
  current: RecoveryDiffProps['current'],
  partial: PartialState,
): DiffEntry[] {
  const entries: DiffEntry[] = [];

  // Source diff.
  if (JSON.stringify(current.source) !== JSON.stringify(partial.source)) {
    if (current.source == null && partial.source != null) {
      entries.push({ kind: 'added', category: 'source', label: 'source' });
    } else if (current.source != null && partial.source == null) {
      entries.push({ kind: 'removed', category: 'source', label: 'source' });
    } else {
      entries.push({ kind: 'changed', category: 'source', label: 'source' });
    }
  }

  // Node diff.
  const currentNodeIds = new Set((current.nodes as ReadonlyArray<{ id: string }>).map(n => n.id));
  const partialNodeIds = new Set(
    (partial.nodes as ReadonlyArray<{ id: string }>).map(n => n.id),
  );
  for (const id of partialNodeIds) {
    if (!currentNodeIds.has(id)) entries.push({ kind: 'added', category: 'node', label: id });
  }
  for (const id of currentNodeIds) {
    if (!partialNodeIds.has(id)) entries.push({ kind: 'removed', category: 'node', label: id });
  }

  // (edges diff omitted from the example — implementer extends as needed)

  return entries;
}


export const RecoveryDiff: FC<RecoveryDiffProps> = ({ current, partial }) => {
  const entries = useMemo(() => computeDiff(current, partial), [current, partial]);

  if (entries.length === 0) {
    return <div className="recovery-diff recovery-diff--empty">No changes</div>;
  }

  return (
    <ul className="recovery-diff" aria-label="Pipeline diff">
      {entries.map((entry, i) => (
        <li key={i} className={`recovery-diff__entry recovery-diff__entry--${entry.kind}`}>
          <span aria-hidden="true">
            {entry.kind === 'added' ? '+ ' : entry.kind === 'removed' ? '- ' : '~ '}
          </span>
          <span className="recovery-diff__category">{entry.category}:</span>
          <span className="recovery-diff__label">{entry.label}</span>
          {entry.detail ? (
            <span className="recovery-diff__detail">{entry.detail}</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
};
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryDiff
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryDiff.test.tsx
git commit -m "feat(frontend): RecoveryDiff component (composer-progress-persistence phase 4)"
```

---

## Task 5: `RecoveryTranscript` component

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx`
- Create: `src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryTranscript.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';

import { RecoveryTranscript } from '../RecoveryTranscript';
import type { TranscriptRow } from '../recoveryTypes';


function fakeFetch(rows: TranscriptRow[]) {
  return vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ messages: rows }),
  });
}


describe('RecoveryTranscript (spec §7.5)', () => {
  it('fetches the audit-grade transcript and renders one row per role=tool', async () => {
    const rows: TranscriptRow[] = [
      {
        id: 'a1', role: 'assistant', sequence_no: 1, content: '',
        tool_calls: [
          { id: 'tc_1', function: { name: 'list_transforms', arguments: {} } },
          { id: 'tc_2', function: { name: 'wire_secret_ref', arguments: {} } },
        ],
        composition_state_id: null, created_at: '2026-04-30T00:00:00Z',
      },
      {
        id: 't1', role: 'tool', tool_call_id: 'tc_1', parent_assistant_id: 'a1',
        sequence_no: 2, content: '{"count":12}',
        composition_state_id: null, created_at: '2026-04-30T00:00:01Z',
      },
    ];
    const fetchFn = fakeFetch(rows);
    render(
      <RecoveryTranscript
        transcriptUrl="/api/sessions/s/messages?since=u_1&include_tool_rows=true"
        fetch={fetchFn}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/list_transforms/)).toBeInTheDocument();
      expect(screen.getByText(/wire_secret_ref/)).toBeInTheDocument();
    });
    expect(fetchFn).toHaveBeenCalledWith(
      '/api/sessions/s/messages?since=u_1&include_tool_rows=true',
      expect.any(Object),
    );
  });

  it('shows ✗ for tool calls that have no corresponding tool row', async () => {
    const rows: TranscriptRow[] = [
      {
        id: 'a1', role: 'assistant', sequence_no: 1, content: '',
        tool_calls: [
          { id: 'tc_1', function: { name: 'list_transforms', arguments: {} } },
          { id: 'tc_2', function: { name: 'wire_secret_ref', arguments: {} } },
        ],
        composition_state_id: null, created_at: '2026-04-30T00:00:00Z',
      },
      // tc_2 has no tool row — it didn't complete.
    ];
    render(
      <RecoveryTranscript
        transcriptUrl="/u"
        fetch={fakeFetch(rows)}
      />,
    );
    await waitFor(() => {
      expect(screen.getByText(/wire_secret_ref/)).toBeInTheDocument();
      expect(screen.getByText(/did not complete/)).toBeInTheDocument();
    });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryTranscript
```
Expected: FAIL.

- [ ] **Step 3: Implement the component**

```typescript
import { type FC, useEffect, useState } from 'react';
import type { TranscriptRow, AssistantTranscriptRow, ToolTranscriptRow } from './recoveryTypes';


interface RecoveryTranscriptProps {
  transcriptUrl: string;
  fetch?: typeof globalThis.fetch;
}


export const RecoveryTranscript: FC<RecoveryTranscriptProps> = ({ transcriptUrl, fetch: fetchFn = fetch }) => {
  const [rows, setRows] = useState<TranscriptRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchFn(transcriptUrl, { credentials: 'include' })
      .then(async r => {
        if (!r.ok) {
          throw new Error(`transcript fetch failed: ${r.status}`);
        }
        const body = await r.json();
        if (!cancelled) {
          setRows(body.messages);
        }
      })
      .catch(err => {
        if (!cancelled) setError(String(err));
      });
    return () => { cancelled = true; };
  }, [transcriptUrl, fetchFn]);

  if (error) {
    return <div role="alert" className="recovery-transcript__error">{error}</div>;
  }
  if (rows === null) {
    return <div className="recovery-transcript__loading">Loading transcript…</div>;
  }

  // Build a map from tool_call_id -> tool row (if it exists).
  const toolRowsByCallId = new Map<string, ToolTranscriptRow>();
  for (const row of rows) {
    if (row.role === 'tool') {
      toolRowsByCallId.set(row.tool_call_id, row);
    }
  }

  // Walk assistant rows in order; for each tool_call, render either the
  // tool response (if persisted) or an ✗ marker.
  const entries: Array<{ key: string; name: string; status: 'ok' | 'missing'; detail?: string }> = [];
  for (const row of rows) {
    if (row.role !== 'assistant') continue;
    const a = row as AssistantTranscriptRow;
    for (const call of a.tool_calls) {
      const toolRow = toolRowsByCallId.get(call.id);
      entries.push({
        key: call.id,
        name: call.function.name,
        status: toolRow ? 'ok' : 'missing',
        detail: toolRow ? `seq ${toolRow.sequence_no}` : undefined,
      });
    }
  }

  return (
    <ul className="recovery-transcript" aria-label="Tool transcript">
      {entries.map(entry => (
        <li key={entry.key} className={`recovery-transcript__entry recovery-transcript__entry--${entry.status}`}>
          <span aria-hidden="true">{entry.status === 'ok' ? '✓ ' : '✗ '}</span>
          <span className="recovery-transcript__name">{entry.name}</span>
          {entry.status === 'missing' ? (
            <span className="recovery-transcript__detail"> tool did not complete</span>
          ) : entry.detail ? (
            <span className="recovery-transcript__detail"> {entry.detail}</span>
          ) : null}
        </li>
      ))}
    </ul>
  );
};
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryTranscript
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryTranscript.test.tsx
git commit -m "feat(frontend): RecoveryTranscript component (composer-progress-persistence phase 4)"
```

---

## Task 6: `RecoveryPanel` modal

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryPanel.test.tsx`

- [ ] **Step 1: Write the failing tests**

```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import { RecoveryPanel } from '../RecoveryPanel';
import type { ComposerErrorResponse } from '../recoveryTypes';


function makeResponse(): ComposerErrorResponse {
  return {
    error: 'X', reason: 'convergence_wall_clock_timeout',
    headline: 'Wall-clock timeout', evidence: ['turn 12'],
    partial_state: { source: null, nodes: [], edges: [], version: 1 },
    failed_turn: {
      assistant_message_id: 'a_1',
      tool_calls_attempted: 4,
      tool_responses_persisted: 3,
      transcript_url: '/api/sessions/s/messages?since=u_1&include_tool_rows=true',
    },
  };
}


describe('RecoveryPanel (spec §7.3)', () => {
  it('renders headline, reason badge, diff section, transcript section, and three buttons', () => {
    render(
      <RecoveryPanel
        response={makeResponse()}
        currentEditorState={{ source: null, nodes: [], edges: [] }}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        fetch={vi.fn().mockResolvedValue({ ok: true, json: async () => ({ messages: [] }) })}
      />,
    );
    expect(screen.getByText(/Wall-clock timeout/)).toBeInTheDocument();
    expect(screen.getByLabelText(/reason: convergence_wall_clock_timeout/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Pipeline diff/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Tool transcript/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /apply partial draft/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /discard/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /view raw transcript/i })).toBeInTheDocument();
  });

  it('reason badge has both colour AND text label (spec §7.7 — no colour-only signalling)', () => {
    render(
      <RecoveryPanel
        response={makeResponse()}
        currentEditorState={{ source: null, nodes: [], edges: [] }}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        fetch={vi.fn().mockResolvedValue({ ok: true, json: async () => ({ messages: [] }) })}
      />,
    );
    const badge = screen.getByLabelText(/reason: convergence_wall_clock_timeout/i);
    // Text label is visible to screen readers AND visually.
    expect(badge.textContent).toMatch(/wall.clock|timeout/i);
  });

  it('Apply requires explicit click (no Enter auto-apply)', () => {
    const onApply = vi.fn();
    render(
      <RecoveryPanel
        response={makeResponse()}
        currentEditorState={{ source: null, nodes: [], edges: [] }}
        onApply={onApply}
        onDiscard={vi.fn()}
        fetch={vi.fn().mockResolvedValue({ ok: true, json: async () => ({ messages: [] }) })}
      />,
    );
    fireEvent.keyDown(document.body, { key: 'Enter' });
    expect(onApply).not.toHaveBeenCalled();
  });

  it('uses aria-modal so screen readers know the modal is modal', () => {
    render(
      <RecoveryPanel
        response={makeResponse()}
        currentEditorState={{ source: null, nodes: [], edges: [] }}
        onApply={vi.fn()}
        onDiscard={vi.fn()}
        fetch={vi.fn().mockResolvedValue({ ok: true, json: async () => ({ messages: [] }) })}
      />,
    );
    expect(screen.getByRole('dialog')).toHaveAttribute('aria-modal', 'true');
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryPanel
```
Expected: FAIL.

- [ ] **Step 3: Implement the component**

```typescript
import { type FC, useRef, useEffect } from 'react';

import { Modal } from '../shared/Modal';   // existing project Modal pattern
import { RecoveryDiff } from './RecoveryDiff';
import { RecoveryTranscript } from './RecoveryTranscript';
import type { ComposerErrorResponse, PartialState } from './recoveryTypes';


// Spec §7.7: reason badge has both colour and a human-readable label.
// No colour-only signalling.
const REASON_LABELS: Record<string, string> = {
  convergence_wall_clock_timeout: 'Wall-clock timeout',
  convergence_discovery_budget: 'Discovery budget exhausted',
  convergence_composition_budget: 'Composition budget exhausted',
  client_cancelled: 'Cancelled by client',
  runtime_preflight_failed: 'Runtime preflight failed',
  tool_call_cap_exceeded: 'Tool-call cap exceeded',
};


interface RecoveryPanelProps {
  response: ComposerErrorResponse;
  currentEditorState: { source: unknown; nodes: ReadonlyArray<{ id: string }>; edges: unknown[] };
  onApply: (partial: PartialState) => void;
  onDiscard: () => void;
  fetch?: typeof globalThis.fetch;
}


export const RecoveryPanel: FC<RecoveryPanelProps> = ({
  response,
  currentEditorState,
  onApply,
  onDiscard,
  fetch: fetchFn,
}) => {
  const applyButtonRef = useRef<HTMLButtonElement>(null);

  // Spec §7.7: focus-trapped modal with explicit Apply (no Enter auto-apply).
  useEffect(() => {
    applyButtonRef.current?.focus();
  }, []);

  if (response.partial_state == null || response.failed_turn == null) {
    return null;
  }

  const reasonLabel = REASON_LABELS[response.reason] ?? response.reason;

  return (
    <Modal aria-modal="true" role="dialog" aria-labelledby="recovery-panel-title">
      <header className="recovery-panel__header">
        <h2 id="recovery-panel__title">Recover composer draft</h2>
        <span
          className={`recovery-panel__reason-badge recovery-panel__reason-badge--${response.reason}`}
          aria-label={`reason: ${response.reason}`}
        >
          {reasonLabel}
        </span>
      </header>

      <p className="recovery-panel__headline">{response.headline}</p>
      {response.evidence.length > 0 ? (
        <ul className="recovery-panel__evidence">
          {response.evidence.map((item, i) => <li key={i}>{item}</li>)}
        </ul>
      ) : null}

      <section>
        <h3>Pipeline diff</h3>
        <RecoveryDiff current={currentEditorState} partial={response.partial_state} />
      </section>

      <section>
        <h3>
          Tool transcript ({response.failed_turn.tool_responses_persisted} of{' '}
          {response.failed_turn.tool_calls_attempted} completed)
        </h3>
        <RecoveryTranscript transcriptUrl={response.failed_turn.transcript_url} fetch={fetchFn} />
      </section>

      <footer className="recovery-panel__actions">
        <button
          ref={applyButtonRef}
          type="button"
          onClick={() => onApply(response.partial_state!)}
        >
          Apply partial draft
        </button>
        <button type="button" onClick={onDiscard}>
          Discard
        </button>
        <button type="button" onClick={() => {/* expand raw transcript view; see §7.6 */}}>
          View raw transcript
        </button>
      </footer>
    </Modal>
  );
};
```

(The exact Modal import path depends on the existing project component. If the project does not have a `Modal` primitive, use `<div role="dialog" aria-modal="true">` directly and add focus-trap behaviour via a small utility hook.)

- [ ] **Step 4: Run tests to verify pass**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryPanel
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx src/elspeth/web/frontend/src/components/recovery/__tests__/RecoveryPanel.test.tsx
git commit -m "feat(frontend): RecoveryPanel modal (composer-progress-persistence phase 4)"
```

---

## Task 7: Wire `useComposer.sendMessage` `onError` to open the panel

**Files:**
- Modify: `src/elspeth/web/frontend/src/hooks/useComposer.ts`

- [ ] **Step 1: Locate the existing `onError` path**

```bash
grep -n "onError\|sendMessage" src/elspeth/web/frontend/src/hooks/useComposer.ts
```

- [ ] **Step 2: Write a failing integration test**

Create or extend `src/elspeth/web/frontend/src/hooks/__tests__/useComposer.test.ts`:

```typescript
import { describe, it, expect, vi } from 'vitest';

// Test the integration: when sendMessage's response carries
// partial_state + failed_turn, the recovery-panel handler is invoked.

describe('useComposer integration with recovery panel', () => {
  it('invokes recoveryPanelOpen when both fields are present', async () => {
    const onRecoveryError = vi.fn();
    // ... render the hook with the recovery handler injected ...
    // ... mock the API to return a 422 with partial_state + failed_turn ...
    // ... call sendMessage ...
    // ... expect(onRecoveryError).toHaveBeenCalledWith(...) ...
  });

  it('falls back to the existing toast when partial_state is absent', async () => {
    // ... mock 422 without partial_state; expect toast invoked, not recovery handler ...
  });
});
```

(Implementation depends on the existing `useComposer` shape; flesh out per the project's testing conventions.)

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npm test -- useComposer
```
Expected: FAIL.

- [ ] **Step 4: Wire the recovery handler into `useComposer`**

In `src/elspeth/web/frontend/src/hooks/useComposer.ts`:

```typescript
async function sendMessage(...) {
  try {
    // ... existing happy-path logic ...
  } catch (err) {
    if (err instanceof ComposerError && err.response.partial_state && err.response.failed_turn) {
      // Spec §7.2: only the (present, present) case opens the panel.
      onRecoveryError(err.response);
      return;
    }
    onToastError(err);
  }
}
```

(Adjust to match the existing error-class shape used by `useComposer`.)

- [ ] **Step 5: Run test to verify pass**

```bash
cd src/elspeth/web/frontend && npm test -- useComposer
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useComposer.ts src/elspeth/web/frontend/src/hooks/__tests__/useComposer.test.ts
git commit -m "feat(frontend): wire useComposer.sendMessage onError to recovery panel (composer-progress-persistence phase 4)"
```

---

## Task 8: Manual smoke test against staging

- [ ] **Step 1: Build the frontend**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 2: Start the dev server**

```bash
cd src/elspeth/web/frontend && npm run dev
```

- [ ] **Step 3: Trigger a failed compose**

In a browser, against local dev or staging:
1. Open a session.
2. Submit a prompt that will exceed the wall-clock budget (or use the new `tool_call_cap_exceeded` path with a stress prompt).
3. Verify the recovery panel opens with the headline, reason badge, diff, and transcript visible.

- [ ] **Step 4: Smoke test the Apply path**

1. Make an edit in the editor before clicking Apply.
2. Click Apply — confirm the concurrent-edit confirmation appears.
3. Cancel the confirmation; the panel stays open.
4. Click Apply again, confirm — verify the editor state updates to `partial_state`.

- [ ] **Step 5: Smoke test the Discard path**

1. Trigger another failed compose.
2. Click Discard — panel closes; editor state unchanged.
3. Refresh the page; via the existing chat history view, confirm the failed-turn rows still exist (Discard is a UI choice, not a delete).

- [ ] **Step 6: Run an accessibility audit**

```bash
cd src/elspeth/web/frontend && npx @axe-core/cli http://localhost:5173/<path-to-recovery-panel>
```

Or use Lighthouse via Chrome devtools. Expected: no critical accessibility violations (focus trap working, aria-modal set, reason badge has text label, no colour-only signalling).

- [ ] **Step 7: No commit required if the smoke tests reveal no issues**

If issues are found, file separate commits to fix them and rerun the smoke tests.

---

## Task 9: Final Phase 4 CI run

- [ ] **Step 1: Run the full frontend test suite**

```bash
cd src/elspeth/web/frontend && npm test
```
Expected: PASS.

- [ ] **Step 2: TypeScript check**

```bash
cd src/elspeth/web/frontend && npm run typecheck
```
Expected: clean.

- [ ] **Step 3: Linter**

```bash
cd src/elspeth/web/frontend && npm run lint
```
Expected: clean.

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "feat(composer): progress persistence phase 4 — frontend recovery panel" --body "$(cat <<'EOF'
## Summary

Phase 4 of composer-progress-persistence (spec §11):
- New `RecoveryPanel` modal that opens when a compose request fails with
  `partial_state` + `failed_turn` in the response body
- Pipeline diff against current editor state via `RecoveryDiff`
- Tool transcript fetched from `?include_tool_rows=true` via `RecoveryTranscript`
- `useRecoveryPanel` hook with concurrent-edit guard and discard semantics
- `useComposer.sendMessage` wired to trigger the panel
- Accessibility: focus-trap modal, aria-modal, reason badge with text label,
  no colour-only signalling, explicit Apply (no Enter auto-apply)

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 4 §7.

## Depends on

Phase 3 PR (compose-loop persistence — provides `failed_turn` field and
`?include_tool_rows=true` query parameter).

## VAL gate

This PR delivers VER (every component test passes; manual smoke against
local dev confirmed). VAL — "the user can actually recover from a
failure" — is owned by [elspeth-599ecf69fa](filigree:elspeth-599ecf69fa)
which must carry the `blocks-rc5.1` label.

## Test plan

- [x] Gating matrix: only (partial_state present, failed_turn present) opens panel
- [x] Concurrent-edit guard requires confirmation
- [x] Discard does not invoke onApplyState
- [x] RecoveryDiff handles add/remove/change
- [x] RecoveryTranscript fetches and renders tool rows
- [x] RecoveryPanel renders all four sections with accessibility hooks
- [x] useComposer integration: 422 with partial_state opens panel; 422 without falls back to toast
- [x] Manual smoke against local dev: failed compose triggers panel; Apply/Discard paths work
- [x] Accessibility audit (axe / Lighthouse) clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: After PR merges, close the parent ticket**

Once Phase 4 has merged AND `elspeth-599ecf69fa` (the staging-replay VAL ticket) has confirmed the recovery flow end-to-end, close the parent ticket `elspeth-90b4542b63` with a closing comment that cites all four merged PRs.

The parent epic `elspeth-528bde62bb` may then close as well, contingent on its other component tickets.

---

## Phase 4 Done When

All 9 tasks above are complete. Specifically:

1. ✅ All frontend unit tests pass.
2. ✅ TypeScript and lint clean.
3. ✅ Manual smoke test against local dev confirms the panel opens, diff renders, transcript loads, Apply and Discard paths work.
4. ✅ Accessibility audit (axe / Lighthouse) reveals no critical violations.
5. ✅ `useComposer.sendMessage` `onError` correctly gates the panel on `partial_state + failed_turn` presence.

VAL — "the user can actually recover from a failure" — remains owned by [elspeth-599ecf69fa](filigree:elspeth-599ecf69fa).

# Phase 2C — Frontend integration: sub-components, panel mount, Validate-button removal, staging smoke

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Every task is TDD-shaped (failing test → run-to-fail → implement → run-to-pass → commit).

**Goal:** Land the remainder of the frontend half of Phase 2 — the real `ReadinessRowDetail` (per-row warning detail + jump-to-component), the real `ExplainDialog` (narrative modal), mounting `AuditReadinessPanel` inside `InspectorPanel.tsx`, removing the standalone Validate button, and the staging smoke that exercises Phase 2 end-to-end.

**Architecture:** Replaces the placeholder sub-components shipped by 14b with full implementations, then integrates the panel into the inspector chrome. The standalone Validate button is removed; the panel's Validation row subsumes it. The `useExecutionStore.validate()` action and its callers stay — Phase 2 does not refactor the execution flow.

**Tech Stack:** React + Zustand + Vitest + testing-library + userEvent.

**Sibling plans:**
- [14a-phase-2a-backend.md](14a-phase-2a-backend.md) — backend response models, service, routes.
- [14b-phase-2b-frontend.md](14b-phase-2b-frontend.md) — frontend foundations (types, API client, store, `AuditReadinessPanel` shell + placeholders).

**Umbrella plan:** [14-phase-2-audit-readiness-panel.md](14-phase-2-audit-readiness-panel.md).

**Design reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**
- Replace the `ReadinessRowDetail.tsx` placeholder shipped by 14b with the real implementation (per-row warning detail + jump-to-component for component-ids that resolve to nodes in `compositionState`).
- Replace the `ExplainDialog.tsx` placeholder shipped by 14b with the real modal implementation (fetches narrative lazily via `useAuditReadinessStore.loadExplain`, caches by composition version, preserves whitespace).
- Mount `<AuditReadinessPanel />` inside `InspectorPanel.tsx` between the header and the tab strip.
- Remove the standalone Validate button UI from `InspectorPanel.tsx` and delete `handleValidate`, `injectSystemMessage`, `sendValidationFeedback`, `canValidate`, and `isValidating` from the inspector component.
- Wire `injectSystemMessage` + `sendValidationFeedback` side effects into `subscriptions.ts` as a `useExecutionStore.validationResult` subscriber, so validation-failure messages continue reaching the LLM agent after the button and handler are removed. Both branches are covered: errors (inject + feedback) and warnings-only (inject only, no feedback).
- Add a vitest assertion that the inspector still renders, the keyboard navigation between tabs still works, and no button labelled "Validate" or "Validate pipeline" exists.
- A manual staging smoke that exercises (a) all-green collapse, (b) provenance-warning auto-expansion, (c) Validate-button-is-gone + system-message-fires-on-invalid-pipeline, (d) error-path recovery.

**Out of scope:**
- Backend (Phase 2A delivered).
- Frontend foundations — types, API client, store, panel shell (14b delivered).
- Telemetry on row-click (Phase 8 — explicit marker in §Task 5).
- Phase 3's side-rail reorganisation. 14c mounts inside the existing tab layout; Phase 3 moves the mount point (the panel component is untouched).
- A per-user retention preference UI (a future phase; the Retention row is informational only).
- LLM-interpretations row content (Phase 5b; the row renders with the `not_applicable` glyph and no detail navigation).

## Sequencing and dependencies

14c **depends on 14b being merged**, because 14c replaces files 14b created. If both halves land on the same working branch, 14b's commits must precede 14c's commits in history; otherwise 14c's "replace placeholder" commits will conflict.

14c's staging smoke (Task 9) depends on **Phase 2A being deployed** to `elspeth.foundryside.dev`. Smoke fails if 2A's routes don't respond.

**Phase 3A Task 8 is now absorbed into Phase 2C.** The handler deletion (`handleValidate`/`injectSystemMessage`/`sendValidationFeedback` from `InspectorPanel.tsx`) and the subscription wiring (`subscriptions.ts`) both land in Phase 2C Task 8. Phase 3A's `15a2-phase-3a-removals-part-2.md` Task 8 needs a corresponding update to mark itself as absorbed — that is a separate edit cycle; a cross-reference note is in the Review history entry below.

**Phase 3B dependency.** Phase 3B's deletion of `InspectorPanel.tsx` (15b2 Task 9) MUST relocate `<AuditReadinessPanel />` into `SideRail`'s `siderail-slot-audit-readiness` div BEFORE deleting the inspector. Phase 3B Task 9 owns this move; 14c documents the dependency. Do not merge Phase 3B's inspector deletion without confirming the panel relocation has landed.

**Task 4C** (migrate `userExpanded` to store) is sequenced after Task 4B and before Task 5. It is independent of Tasks 5–9 — none of those tasks touch `userExpandedBySession`. Task 4C must land before Phase 3B Task 9 Step 4a (the panel relocation) to prevent the remount state-loss described in W3.

## Trust-tier check (per CLAUDE.md)

Same as 14b: reads only data the backend just produced (Tier 1). No new boundary. Notable additions in 14c:

- `compositionState.nodes` resolution in `ReadinessRowDetail` — direct typed access (`compositionState.nodes.some((n) => n.id === id)`); no `.get()` / `getattr`.
- `selectNode(componentId)` — calls the existing sessionStore action without coercing the id (the wire schema guarantees it's a string).

## File structure

**New:**
- `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx` — first tests alongside the real implementation (Task 5).
- `src/elspeth/web/frontend/src/components/audit/ExplainDialog.test.tsx` — first tests alongside the real implementation (Task 6).

**Modified:**
- `src/elspeth/web/frontend/src/test/composerFixtures.ts` — `makeAbortablePromise()` / signal-aware mock helper added by Task 4A's commit `7ba9bc49e` (tracks issue `elspeth-f018ea84c6`).
- `src/elspeth/web/frontend/src/stores/auditReadinessStore.ts` — Task 4C adds `userExpandedBySession: Record<string, boolean>` (7th per-session keyed map) to `AuditReadinessState` and `getInitialState()`; adds `setUserExpanded(sessionId, value)` action; `clearSession` pops `userExpandedBySession[sessionId]` alongside the other six maps.
- `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx` — Task 4A's commit `7ba9bc49e` dropped the cleanup-effect synchronous `setState` workaround (issue `elspeth-f018ea84c6`); Task 4B's commit `b8fb6b23f` replaced the `expanded` useState/useEffect pair with computed `showExpanded = anyActionable || userExpanded` (issue `elspeth-82ef9d5bd0`); Task 4C migrates `userExpanded` from component-local `useState(false)` to `auditReadinessStore.userExpandedBySession` keyed by `activeSessionId`.
- `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx` — Task 4A's commit `7ba9bc49e` migrated the existing tests to the new signal-aware mock helper so the store's AbortError catch arm exercises in tests; Task 4C adds one remount-safety regression test.
- `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx` — replace 14b's placeholder with full implementation (Task 5).
- `src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx` — replace 14b's placeholder with full implementation (Task 6).
- `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — mount `<AuditReadinessPanel />`; remove the standalone Validate button and delete `handleValidate`/`injectSystemMessage`/`sendValidationFeedback`/`canValidate`/`isValidating` (Tasks 7 + 8).
- `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx` — add panel-mount and button-removal assertions; delete any tests that exercised `handleValidate`'s side effects directly (those are now covered by `subscriptions.test.ts`) (Tasks 7 + 8).
- `src/elspeth/web/frontend/src/stores/subscriptions.ts` — add `useExecutionStore.validationResult` subscriber that fires `injectSystemMessage` + `sendValidationFeedback` on result reference change; update `_resetSubscriptionsForTesting()` to also reset the new subscriber and `previousValidationResult` (Task 8).
- `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — add `describe("subscriptions — validation result side effects", ...)` block with three green tests; add `waitFor` and `useExecutionStore` imports (Task 8.5).

**Verified-against-reality (load-bearing — DO NOT silently change without re-running the reality-check against `feat/composer-phase-2a-backend` HEAD):**
- The shared composer fixture module is at `src/elspeth/web/frontend/src/test/composerFixtures.ts` and is imported via the alias `@/test/composerFixtures` (NOT `@/test-utils/composerFixtures`). Confirmed against `AuditReadinessPanel.test.tsx:9`.
- `NodeSpec` (`types/index.ts:105–119`) requires 7 required fields: `id`, `node_type`, `plugin`, `input`, `on_success`, `on_error`, `options`. There is no `config` field on `NodeSpec`.
- `SourceSpec` (`types/index.ts:94–99`) has fields `plugin` and `options` — NOT `kind` / `config`.
- The auditReadiness store exposes per-session keyed maps: `snapshotsBySession`, `explainsBySession`, `isLoadingBySession`, `errorBySession`, `isLoadingExplainBySession`, `explainErrorBySession`. There are NO flat `isLoading` / `error` / `isLoadingExplain` / `explainError` fields. Use the exported `getInitialState()` (`auditReadinessStore.ts:55`) to reset; never write a hand-rolled setState literal with `as never` casts.
- **Task 4C will add `userExpandedBySession: Record<string, boolean>` as a 7th per-session map**, plus a `setUserExpanded(sessionId, value)` action. After Task 4C lands, the store has 7 per-session keyed maps and `AuditReadinessPanel` does not hold `userExpanded` in component-local state.

---

## Task 4A: Signal-aware mock helper + drop AuditReadinessPanel cleanup-effect workaround

> **Issue:** `elspeth-f018ea84c6` (promoted from observation `elspeth-obs-8502e9d4bf`).

**This task is already complete.** Commit `7ba9bc49e` ("refactor(web/frontend): signal-aware mock helper + drop panel cleanup-effect setState workaround (Phase 2C.4A)") landed this work on the current branch. A follow-up scrub commit (`8a0f499d7`) removed issue-id references from inline panel comments.

**What Task 4A did:** Phase 2B's `AuditReadinessPanel.tsx` cleanup ran `ctrl.abort()` AND `useAuditReadinessStore.setState(...)` synchronously. The second write papered over a test-mock limitation: `vi.mock`-ed fetch promises didn't propagate `AbortError` when their signal aborted, so the store's production `AbortError` catch arm never fired in tests. Task 4A added `makeAbortablePromise()` to `composerFixtures.ts` — a helper that rejects with a synthetic `AbortError` when `signal.aborted` flips — migrated all panel mocks to use it, and removed the architecturally-duplicated component-side `setState` from the cleanup closure. The store's production `AbortError` catch arm now fires under test as designed.

- [x] **Step 1: Verify — expect this work to be on disk already (commit `7ba9bc49e`).**

```bash
cd /home/john/elspeth/.worktrees/phase-2a-backend
git log --oneline | grep -F '7ba9bc49e'
grep -n 'export function makeAbortablePromise' src/elspeth/web/frontend/src/test/composerFixtures.ts
grep -n 'AbortError catch arm clears' src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx
```

All three commands must return non-empty. If any returns empty: STOP and investigate — the branch state has drifted from the plan's assumption. Do NOT re-apply the original Task 4A body in that case; consult the operator.

If all three return non-empty: Task 4A is already complete. Mark this checkbox and proceed to Task 4B.

---

## Task 4B: Refactor AuditReadinessPanel `expanded` to computed `showExpanded`

> **Issue:** `elspeth-82ef9d5bd0` (promoted from observation `elspeth-obs-6b5bb1a476`).

**This task is already complete.** Commit `b8fb6b23f` ("refactor(web/frontend): compute AuditReadinessPanel showExpanded instead of useState/useEffect sync (Phase 2C.4B)") landed this work on the current branch.

**What Task 4B did:** The 14b plan prescribed `expanded` as `useState` with a `useEffect` that forced `setExpanded(true)` whenever `anyActionable` became true — the React-docs "derived state via effect" antipattern (an extra render cycle, plus a sticky-expansion bug: once expanded, never auto-collapses on a later all-green snapshot). Task 4B renamed `expanded`→`userExpanded`, computed `showExpanded = anyActionable || userExpanded` atomically, deleted the derived-state effect, and added a regression test asserting the panel auto-collapses when a subsequent refetch returns all-green.

- [x] **Step 1: Verify — expect this work to be on disk already (commit `b8fb6b23f`).**

```bash
cd /home/john/elspeth/.worktrees/phase-2a-backend
git log --oneline | grep -F 'b8fb6b23f'
grep -n 'const \[userExpanded, setUserExpanded\]' src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx
grep -n 'const showExpanded = anyActionable' src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx
grep -n 'auto-collapses when a subsequent refetch returns all-green' src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx
```

All four commands must return non-empty. If any returns empty: STOP and investigate — the branch state has drifted from the plan's assumption. Do NOT re-apply the original Task 4B body in that case; consult the operator.

If all four return non-empty: Task 4B is already complete. Mark this checkbox and proceed to Task 4C.

---

## Task 4C: Migrate AuditReadinessPanel `userExpanded` to per-session store state

**Why this task exists.** Task 4B introduced `const [userExpanded, setUserExpanded] = useState(false)` in `AuditReadinessPanel.tsx` — component-local state that controls whether the user has explicitly chosen to keep the panel expanded. This is correct for Phase 2C's mount location (inside `InspectorPanel.tsx`). However, Phase 3B Task 9 Step 4a (`15b2-phase-3b-side-rail-part-2.md`) relocates `<AuditReadinessPanel />` from `InspectorPanel.tsx` to `App.tsx → SideRail.auditReadinessSlot`. This relocation is a full component remount: the old component unmounts, the new one mounts, and React resets all component-local state to initial values. The result is that `userExpanded` resets to `false` at Phase 3B's deploy boundary. A user who had explicitly expanded the panel — to keep it persistently visible while iterating on a pipeline that cycles between warning and all-green states — will find it collapsed without warning. The store-keyed snapshot data survives (zustand, keyed by sessionId), but the user's preference is silently discarded. In an auditability-first UI, visible state loss at a deploy boundary the user cannot predict undermines trust in the panel's reliability as an indicator.

The fix is to migrate `userExpanded` from component-local `useState` into `auditReadinessStore` as a per-session keyed map. This mirrors the pattern already established for six other per-session values (`snapshotsBySession`, `explainsBySession`, `isLoadingBySession`, `errorBySession`, `isLoadingExplainBySession`, `explainErrorBySession`). The component reads `userExpanded` from the store (keyed by `activeSessionId`) and writes via the store action. After the migration, a Phase 3B remount reads the stored preference and renders correctly.

**Files:**
- Modify: `frontend/src/stores/auditReadinessStore.ts` — add `userExpandedBySession` map + `setUserExpanded` action.
- Modify: `frontend/src/components/audit/AuditReadinessPanel.tsx` — replace component-local `useState` with store selectors.
- Modify: `frontend/src/components/audit/AuditReadinessPanel.test.tsx` — add one remount-safety regression test.

- [x] **Step 1: Write the failing test**

Append ONE new test to the `describe("AuditReadinessPanel", ...)` block in `frontend/src/components/audit/AuditReadinessPanel.test.tsx`:

```typescript
  it("preserves the user's expand preference across component unmount/remount (Phase 3B remount safety)", async () => {
    // Seed a cached all-green snapshot directly — no fetch needed; the
    // component reads from the store, and version parity means loadSnapshot
    // is a no-op. This isolates the test to userExpanded behaviour only.
    useAuditReadinessStore.setState({
      snapshotsBySession: { [SESSION_ID]: allGreenSnapshot(1) },
    });

    const user = userEvent.setup();

    // Mount the panel; with an all-green snapshot it collapses to "Audit ready".
    const { unmount } = render(<AuditReadinessPanel />);
    expect(await screen.findByRole("button", { name: /Audit ready/i })).toBeInTheDocument();

    // User clicks to expand — sets userExpanded=true via the toggle.
    await user.click(screen.getByRole("button", { name: /Audit ready/i }));
    // Confirm expansion: the full row list is visible.
    expect(screen.getByText("Validation")).toBeInTheDocument();

    // Simulate the Phase 3B remount: unmount the current tree, then render a fresh instance.
    unmount();
    render(<AuditReadinessPanel />);

    // The panel must still be expanded — userExpanded survived the remount via the store.
    // With component-local useState this test FAILS: the new instance starts with
    // useState(false), anyActionable is false (all-green), and showExpanded = false.
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Audit ready/i })).not.toBeInTheDocument();
  });
```

This test **FAILS** with the current implementation because the second `render(<AuditReadinessPanel />)` starts a fresh component with `useState(false)`, so `userExpanded` is `false`, `anyActionable` is `false`, and `showExpanded = false` — the panel collapses.

- [x] **Step 2: Run the test — expect FAIL**

```bash
cd /home/john/elspeth/.worktrees/phase-2a-backend
.venv/bin/python -m pytest --co -q 2>/dev/null || true  # backend unaffected
npx --prefix src/elspeth/web/frontend vitest run \
  src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx \
  2>&1 | tail -20
```

Expected: the new "preserves the user's expand preference" test fails; all prior tests in the file pass.

- [x] **Step 3: Implement**

Two changes in parallel:

**3a. `frontend/src/stores/auditReadinessStore.ts`**

1. Add `userExpandedBySession` and `setUserExpanded` to the `AuditReadinessState` interface (after `explainErrorBySession`):

```typescript
  userExpandedBySession: Record<string, boolean>;

  setUserExpanded: (sessionId: string, value: boolean) => void;
```

2. Update the `Omit<>` in `getInitialState`'s return type to include `"setUserExpanded"`:

```typescript
export const getInitialState = (): Omit<AuditReadinessState, "loadSnapshot" | "loadExplain" | "clearSession" | "reset" | "setUserExpanded"> => ({
  // ... existing fields ...
  userExpandedBySession: {},
});
```

3. Add the action implementation inside `create<AuditReadinessState>(...)` (after `reset`):

```typescript
  setUserExpanded(sessionId: string, value: boolean) {
    set((s) => ({
      userExpandedBySession: { ...s.userExpandedBySession, [sessionId]: value },
    }));
  },
```

4. Add `userExpandedBySession` cleanup to `clearSession` — pop the entry alongside the other six maps:

```typescript
  clearSession(sessionId: string) {
    // ... existing abort calls ...
    set((state) => {
      const { [sessionId]: _snap, ...restSnap } = state.snapshotsBySession;
      // ... existing destructures ...
      const { [sessionId]: _ue, ...restUE } = state.userExpandedBySession;
      return {
        snapshotsBySession: restSnap,
        // ... existing fields ...
        userExpandedBySession: restUE,
      };
    });
  },
```

**3b. `frontend/src/components/audit/AuditReadinessPanel.tsx`**

Replace the component-local state declaration:

```typescript
  // BEFORE (component-local — resets on every remount):
  const [userExpanded, setUserExpanded] = useState(false);
```

with store selectors:

```typescript
  // AFTER (per-session store — survives remount):
  const userExpanded = useAuditReadinessStore(
    (s) => activeSessionId ? (s.userExpandedBySession[activeSessionId] ?? false) : false,
  );
  const setUserExpandedInStore = useAuditReadinessStore((s) => s.setUserExpanded);
```

Update the two call sites that call `setUserExpanded(bool)` to call `setUserExpandedInStore(activeSessionId!, bool)` instead:

- Collapsed view: `onClick={() => setUserExpandedInStore(activeSessionId!, true)}`
- Collapse button: `onClick={() => setUserExpandedInStore(activeSessionId!, false)}`

Remove `useState` from the `react` import if `userExpanded` was its only use. (`selectedRowId` and `explainOpen` still use `useState`, so the import is not removed in full — verify before removing.)

- [x] **Step 4: Run tests — expect PASS**

```bash
cd /home/john/elspeth/.worktrees/phase-2a-backend
npx --prefix src/elspeth/web/frontend vitest run \
  src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx \
  2>&1 | tail -20
```

Expected: all tests in the file pass (the new remount-safety test now passes; the existing auto-collapse regression test and all other panel tests remain green).

Run the full frontend suite to confirm no regression:

```bash
npx --prefix src/elspeth/web/frontend vitest run 2>&1 | tail -10
```

- [x] **Step 5: Commit**

```bash
cd /home/john/elspeth/.worktrees/phase-2a-backend
git add \
  src/elspeth/web/frontend/src/stores/auditReadinessStore.ts \
  src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx \
  src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx
git commit -m "feat(web/frontend): persist AuditReadinessPanel userExpanded in store (Phase 2C.4C, Phase 3B remount safety)"
```

---

## Task 5: `ReadinessRowDetail` — per-row warning detail + jump-to-component

**Files:**
- Modify: `frontend/src/components/audit/ReadinessRowDetail.tsx` (replaces the 14b placeholder in place).
- Create: `frontend/src/components/audit/ReadinessRowDetail.test.tsx`

A small drawer/popover. Contents:

- The row's `label` as heading.
- The row's `detail` (multi-line; preserves linebreaks).
- A "Jump to component" button per entry in `component_ids` if the id resolves to a node in `compositionState.nodes`. Otherwise the id is displayed as plain text (so the user can grep their YAML).
- A "Close" button.

> **Phase 8 deferral marker.** No telemetry. When telemetry lands, this is the click-handler that fires the audit-row-click event.

- [x] **Step 1: Write the failing test**

`frontend/src/components/audit/ReadinessRowDetail.test.tsx`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { useSessionStore } from "../../stores/sessionStore";
import { makeComposition } from "@/test/composerFixtures";
import type { ReadinessRow } from "../../types/api";

// CANONICAL FIXTURE — `makeComposition` lives at
// `src/elspeth/web/frontend/src/test/composerFixtures.ts` (alias
// `@/test/composerFixtures`). It already returns the correct `NodeSpec` shape
// (7 fields: id, node_type, plugin, input, on_success, on_error, options) and
// `SourceSpec` shape (plugin, options) — do NOT inline a literal here with
// `as never` casts; that's how drift gets introduced.

const ROW_WITH_NODE: ReadinessRow = {
  id: "provenance",
  label: "Provenance",
  status: "warning",
  summary: "Identity passthrough detected",
  detail: "Identity passthrough — provenance gap on 'select_columns'.\nReplace with a transform that records provenance.",
  component_ids: ["select_columns"],
};

const ROW_WITHOUT_RESOLVABLE_ID: ReadinessRow = {
  id: "secrets",
  label: "Secrets",
  status: "error",
  summary: "Required secret missing",
  detail: "Secret reference 'api_key' is not resolved.",
  component_ids: ["api_key"],
};

const ROW_NO_IDS: ReadinessRow = {
  id: "retention",
  label: "Retention",
  status: "warning",
  summary: "Not configured",
  detail: "No retention configured for a pipeline that handles sensitive data.",
  component_ids: [],
};

describe("ReadinessRowDetail", () => {
  beforeEach(() => {
    // makeComposition(1) returns a composition with one node id="select_columns",
    // which the ROW_WITH_NODE fixture above expects to be jumpable. Confirm by
    // reading composerFixtures.ts before changing the node-id assumption.
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1),
      selectNode: vi.fn(),
    } as never);
  });

  it("renders the row label and detail with preserved linebreaks", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: "Provenance" })).toBeInTheDocument();
    expect(screen.getByText(/Identity passthrough — provenance gap/)).toBeInTheDocument();
    expect(screen.getByText(/Replace with a transform/)).toBeInTheDocument();
  });

  it("renders a Jump to component button for ids that resolve to nodes", async () => {
    const user = userEvent.setup();
    const selectNode = vi.fn();
    useSessionStore.setState({ selectNode } as never);
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    const btn = screen.getByRole("button", { name: /Jump to select_columns/ });
    await user.click(btn);
    expect(selectNode).toHaveBeenCalledWith("select_columns");
  });

  it("renders unresolved ids as plain text (no jump button)", () => {
    render(<ReadinessRowDetail row={ROW_WITHOUT_RESOLVABLE_ID} onClose={() => {}} />);
    expect(screen.getByText("api_key")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Jump to/ })).not.toBeInTheDocument();
  });

  it("omits the component-ids block when component_ids is empty", () => {
    render(<ReadinessRowDetail row={ROW_NO_IDS} onClose={() => {}} />);
    expect(screen.queryByText(/Components/i)).not.toBeInTheDocument();
  });

  it("fires onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders nothing in the detail body when row.detail is null", () => {
    const minimal: ReadinessRow = { ...ROW_NO_IDS, detail: null };
    render(<ReadinessRowDetail row={minimal} onClose={() => {}} />);
    // The summary is still rendered.
    expect(screen.getByText(/Not configured/)).toBeInTheDocument();
  });

  it("uses role=dialog and is labelled by the row heading", () => {
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={() => {}} />);
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-labelledby");
    const labelId = dialog.getAttribute("aria-labelledby")!;
    expect(document.getElementById(labelId)).toHaveTextContent("Provenance");
  });

  // W6 — added 2026-05-17

  it("renders both resolvable and unresolvable ids in the same row (mixed)", () => {
    // One id resolves to a node (select_columns is in makeComposition(1));
    // one id does not resolve (api_key is unknown). Both branches of the
    // resolvable/unresolvable split are exercised within a single render.
    const ROW_MIXED: ReadinessRow = {
      id: "secrets",
      label: "Secrets",
      status: "error",
      summary: "Mixed component refs",
      detail: "One resolvable, one not.",
      component_ids: ["select_columns", "api_key"],
    };
    render(<ReadinessRowDetail row={ROW_MIXED} onClose={() => {}} />);
    // select_columns resolves → Jump button.
    expect(screen.getByRole("button", { name: /Jump to select_columns/ })).toBeInTheDocument();
    // api_key does not resolve → plain text, no button.
    expect(screen.getByText("api_key")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Jump to api_key/ })).not.toBeInTheDocument();
  });

  it("fires onClose when Escape is pressed", async () => {
    // Drawer keyboard-dismiss contract: pressing Escape while the drawer
    // has focus should fire onClose, matching the conventional drawer
    // affordance. The implementation must add an onKeyDown Escape handler
    // to the root div — see Step 3 below.
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<ReadinessRowDetail row={ROW_WITH_NODE} onClose={onClose} />);
    // Move focus inside the drawer so Escape dispatches to the dialog's
    // onKeyDown handler. Without this, userEvent.keyboard dispatches
    // to document.body which may not bubble to the React handler.
    screen.getByRole("button", { name: /Close/i }).focus();
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  // Test 5.C SKIPPED — documented skip.
  // `ReadinessRow.component_ids` is typed `readonly string[]` (types/index.ts:857),
  // which is non-nullable. The implementation's `row.component_ids.length > 0`
  // cannot receive `null` at the TypeScript layer. A null-guard test would
  // only be valid if the backend could send `null` in place of the array,
  // but the type does not allow it and there is no `as never` bypass in the
  // production code path. Adding a `null as never` test would be defensive
  // programming against an impossible state — forbidden by project policy.
  // If the backend schema ever changes to `string[] | null`, update types/index.ts
  // first (which will make the production guard mandatory and the test valid).
});
```

- [x] **Step 2: Run test — expect FAIL**

- [x] **Step 3: Implement**

`frontend/src/components/audit/ReadinessRowDetail.tsx`:

```typescript
/**
 * ReadinessRowDetail (Phase 2B)
 *
 * Drawer/popover content for one row of the audit-readiness panel.
 * Renders the row's detail string (multi-line preserved) and offers a
 * jump-to-component button for each entry in component_ids that
 * resolves to a node in the current composition. Unresolvable ids are
 * shown as plain text — they may refer to source/sink names or YAML
 * fragments the user can grep for.
 *
 * Phase 8 will add a telemetry emit here for audit-row-click. No emit yet.
 */
import { useId } from "react";

import { useSessionStore } from "../../stores/sessionStore";
import type { ReadinessRow } from "../../types/api";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  onClose: () => void;
}

export function ReadinessRowDetail({ row, onClose }: ReadinessRowDetailProps) {
  const compositionState = useSessionStore((s) => s.compositionState);
  const selectNode = useSessionStore((s) => s.selectNode);
  const labelId = useId();

  const nodeIds = new Set(compositionState?.nodes.map((n) => n.id) ?? []);

  function handleJump(componentId: string) {
    selectNode(componentId);
    // Phase 8 deferral: emit telemetry here.
    onClose();
  }

  return (
    <div
      role="dialog"
      aria-labelledby={labelId}
      aria-modal="false"
      className="readiness-row-detail"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
    >
      <header className="readiness-row-detail-header">
        <h3 id={labelId} className="readiness-row-detail-title">
          {row.label}
        </h3>
        <button
          type="button"
          className="readiness-row-detail-close"
          onClick={onClose}
          aria-label="Close detail"
        >
          ×
        </button>
      </header>

      <p className="readiness-row-detail-summary">{row.summary}</p>

      {row.detail && (
        <pre className="readiness-row-detail-body">{row.detail}</pre>
      )}

      {row.component_ids.length > 0 && (
        <section
          aria-label="Components implicated"
          className="readiness-row-detail-components"
        >
          <h4 className="readiness-row-detail-components-heading">Components</h4>
          <ul className="readiness-row-detail-components-list">
            {row.component_ids.map((id) => {
              const resolvable = nodeIds.has(id);
              return (
                <li key={id}>
                  {resolvable ? (
                    <button
                      type="button"
                      className="btn readiness-row-detail-jump-btn"
                      onClick={() => handleJump(id)}
                      aria-label={`Jump to ${id}`}
                    >
                      Jump to {id}
                    </button>
                  ) : (
                    <span className="readiness-row-detail-component-id">{id}</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
```

- [x] **Step 4: Run tests — expect PASS**

- [x] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.test.tsx
git commit -m "feat(web/frontend): add ReadinessRowDetail with jump-to-component (Phase 2C.5)"
```

---

## Task 6: `ExplainDialog` — narrative modal

**Files:**
- Modify: `frontend/src/components/audit/ExplainDialog.tsx` (replaces the 14b placeholder in place).
- Create: `frontend/src/components/audit/ExplainDialog.test.tsx`

The Explain dialog fetches the narrative on first open (via `useAuditReadinessStore.loadExplain`), caches by composition version, and renders the result with preserved whitespace.

- [x] **Step 1: Write the failing test**

`frontend/src/components/audit/ExplainDialog.test.tsx`:

```typescript
// Focus-management contract follows the project's modal-dialog pattern — see CommandPalette.tsx and RecoveryPanel.tsx for canonical examples.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { ExplainDialog } from "./ExplainDialog";
import { useAuditReadinessStore, getInitialState } from "../../stores/auditReadinessStore";
import * as api from "../../api/auditReadiness";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

describe("ExplainDialog", () => {
  beforeEach(() => {
    // Canonical reset: getInitialState() returns the per-session keyed shape
    // (snapshotsBySession / explainsBySession / isLoadingBySession /
    // isLoadingExplainBySession / errorBySession / explainErrorBySession).
    // DO NOT hand-roll a setState literal with `as never` — the store has no
    // flat `isLoading` / `error` / `isLoadingExplain` / `explainError` fields,
    // and `as never` would silently mask the drift.
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("fetches the narrative on mount and renders it", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record:\n\n• Source data — 5 URLs.",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByText(/ELSPETH will record/)).toBeInTheDocument();
    expect(screen.getByText(/Source data — 5 URLs/)).toBeInTheDocument();
  });

  it("uses the cached narrative when present without refetching", async () => {
    useAuditReadinessStore.setState({
      explainsBySession: {
        [SESSION_ID]: {
          session_id: SESSION_ID,
          composition_version: 1,
          narrative: "cached narrative",
        },
      },
    } as never);
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    // Confirm the render has settled before asserting the API was not called.
    // A bare `not.toHaveBeenCalled()` check before render settles races the
    // useEffect; wait for the text to appear first.
    await waitFor(() => expect(screen.getByText("cached narrative")).toBeInTheDocument());
    expect(api.fetchAuditReadinessExplain).not.toHaveBeenCalled();
  });

  it("renders a loading state while the fetch is pending, then transitions to content on resolve", async () => {
    let resolve!: (v: { session_id: string; composition_version: number; narrative: string }) => void;
    vi.mocked(api.fetchAuditReadinessExplain).mockReturnValueOnce(
      new Promise((r) => {
        resolve = r;
      }),
    );
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );

    // Wait for the loadExplain effect to fire and the store to flip
    // isLoadingExplainBySession[SESSION_ID] = true before asserting the
    // loading indicator. A bare synchronous assertion races the effect.
    await waitFor(() =>
      expect(screen.getByText(/Generating explanation/i)).toBeInTheDocument(),
    );

    resolve({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "done.",
    });

    // Confirm the transition from loading to content actually happens —
    // without this waitFor, the test passes regardless of whether the
    // post-resolve render works correctly.
    await waitFor(() =>
      expect(screen.getByText("done.")).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Generating explanation/i)).not.toBeInTheDocument();
  });

  it("renders an error when the fetch fails", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockRejectedValueOnce({
      status: 500,
      detail: "boom",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/boom/);
  });

  it("fires onClose when Close is clicked", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    await user.click(screen.getByRole("button", { name: /Close/i }));
    expect(onClose).toHaveBeenCalled();
  });

  it("uses role=dialog and is labelled by the heading", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    const labelId = dialog.getAttribute("aria-labelledby")!;
    expect(document.getElementById(labelId)).toHaveTextContent(/What this pipeline will record/i);
  });

  it("moves focus into the dialog on mount (focus-trap contract)", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const dialog = screen.getByRole("dialog");
    await waitFor(() => {
      expect(dialog.contains(document.activeElement)).toBe(true);
    });
  });

  it("fires onClose when Escape is pressed", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("restores focus to the opener element when the dialog is closed", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });

    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>
            Open explain
          </button>
          {open ? (
            <ExplainDialog
              sessionId={SESSION_ID}
              compositionVersion={1}
              onClose={() => setOpen(false)}
            />
          ) : null}
        </>
      );
    }

    const user = userEvent.setup();
    render(<Harness />);
    const opener = screen.getByRole("button", { name: "Open explain" });
    opener.focus();
    await user.click(opener);
    // Dialog is open — confirm focus is inside it
    const dialog = screen.getByRole("dialog");
    await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true));
    // Close via Escape
    await user.keyboard("{Escape}");
    // Focus must return to the opener
    await waitFor(() => expect(opener).toHaveFocus());
  });

  // W6 — added 2026-05-17

  it("refetches when compositionVersion changes", async () => {
    // Cache semantics: loadExplain caches by (sessionId, composition_version).
    // The short-circuit is: if cached.composition_version === compositionVersion, return.
    // Therefore a version change 1 → 2 misses the cache and a new fetch fires.
    // (Store code: auditReadinessStore.ts:148–151)
    vi.mocked(api.fetchAuditReadinessExplain)
      .mockResolvedValueOnce({
        session_id: SESSION_ID,
        composition_version: 1,
        narrative: "v1 narrative",
      })
      .mockResolvedValueOnce({
        session_id: SESSION_ID,
        composition_version: 2,
        narrative: "v2 narrative",
      });

    const { rerender } = render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    await screen.findByText("v1 narrative");
    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(1);

    rerender(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={2}
        onClose={() => {}}
      />,
    );
    await screen.findByText("v2 narrative");
    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(2);
  });

  it("closes when the backdrop is clicked", async () => {
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "x",
    });
    const onClose = vi.fn();
    const user = userEvent.setup();
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={onClose}
      />,
    );
    await screen.findByText("x");
    // The backdrop is aria-hidden="true" (decorative) so we query by class.
    // The implementation attaches onClick={onClose} to the backdrop div.
    // If the implementation changes to a different backdrop pattern, update this selector.
    const backdrop = document.querySelector(".explain-dialog-backdrop") as HTMLElement;
    expect(backdrop).not.toBeNull();
    await user.click(backdrop);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders a fallback error message when the ApiError has no detail", async () => {
    // Fallback text: auditReadinessStore.ts:209 → apiErr.detail ?? "Failed to load the explain narrative."
    vi.mocked(api.fetchAuditReadinessExplain).mockRejectedValueOnce({ status: 500 });
    render(
      <ExplainDialog
        sessionId={SESSION_ID}
        compositionVersion={1}
        onClose={() => {}}
      />,
    );
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Failed to load the explain narrative.");
  });
});
```

- [x] **Step 2: Run test — expect FAIL**

- [x] **Step 3: Implement**

Implements the project's standard modal-dialog focus contract via `useFocusTrap` — see CommandPalette/RecoveryPanel for the canonical example.

`frontend/src/components/audit/ExplainDialog.tsx`:

```typescript
/**
 * ExplainDialog (Phase 2B)
 *
 * Modal dialog rendering the narrative explanation of what the current
 * pipeline will record. The narrative is fetched lazily on first open
 * and cached by composition_version in the auditReadinessStore.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 * §"The Explain view".
 */
import { useEffect, useId, useRef } from "react";

import { useAuditReadinessStore } from "../../stores/auditReadinessStore";
import { useFocusTrap } from "@/hooks/useFocusTrap";

export interface ExplainDialogProps {
  sessionId: string;
  compositionVersion: number;
  onClose: () => void;
}

export function ExplainDialog({
  sessionId,
  compositionVersion,
  onClose,
}: ExplainDialogProps) {
  // Store fields are per-session-keyed maps, NOT flat. Reading
  // `s.isLoadingExplain` / `s.explainError` would evaluate to `undefined`
  // at runtime — the dialog would never show loading or error. The correct
  // accessors key by sessionId. See "Verified-against-reality" in File
  // structure above.
  const explain = useAuditReadinessStore((s) => s.explainsBySession[sessionId]);
  const isLoading = useAuditReadinessStore(
    (s) => s.isLoadingExplainBySession[sessionId] ?? false,
  );
  const error = useAuditReadinessStore(
    (s) => s.explainErrorBySession[sessionId] ?? null,
  );
  const loadExplain = useAuditReadinessStore((s) => s.loadExplain);
  const titleId = useId();

  // Focus contract: trap focus inside the dialog, restore to opener on close.
  // useFocusTrap handles: Tab-wrap, initial focus (Close button), and focus
  // restoration on unmount. Escape is handled by a separate onKeyDown because
  // useFocusTrap does not register an Escape listener (matches CommandPalette
  // pattern: onKeyDown Escape → onClose).
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef, true, ".explain-dialog-close");

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="explain-dialog"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.preventDefault();
          onClose();
        }
      }}
    >
      <div className="explain-dialog-backdrop" onClick={onClose} aria-hidden="true" />
      <div className="explain-dialog-content">
        <header className="explain-dialog-header">
          <h2 id={titleId} className="explain-dialog-title">
            What this pipeline will record
          </h2>
          <button
            type="button"
            className="explain-dialog-close"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </header>

        {isLoading && !explain && (
          <p className="explain-dialog-loading">Generating explanation…</p>
        )}

        {error && !explain && (
          <div role="alert" className="explain-dialog-error">
            {error}
          </div>
        )}

        {explain && (
          <pre className="explain-dialog-narrative">{explain.narrative}</pre>
        )}
      </div>
    </div>
  );
}
```

- [x] **Step 4: Run tests — expect PASS**

- [x] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx src/elspeth/web/frontend/src/components/audit/ExplainDialog.test.tsx
git commit -m "feat(web/frontend): add ExplainDialog with version-keyed narrative cache (Phase 2C.6)"
```

---

## Task 7: Mount in `InspectorPanel.tsx`

**Files:**
- Modify: `frontend/src/components/inspector/InspectorPanel.tsx`
- Modify: `frontend/src/components/inspector/InspectorPanel.test.tsx`

The panel mounts **between the inspector header and the tab strip**, so it is visible under every tab. Phase 3's IA cleanup will likely move this mount point, but until then this placement is the cheapest "always visible during composition" location.

- [x] **Step 1: Confirm the mount site**

Read `InspectorPanel.tsx` around line 595 (the closing `</div>` of "Row 1" — the inspector header — and the opening of "Row 2" — the tab strip; per the reconnaissance notes the regions are clearly commented `Row 1: Version selector + validation dot | Validate + Execute` and `Row 2: Tab strip`).

The mount is **after** the Row 1 closing `</div>` and **before** the `role="tablist"` opening. This is also the natural insertion point for the standalone Validate button removal in Task 8 — both edits land in adjacent lines.

- [x] **Step 2: Write the failing test**

Add this test to `frontend/src/components/inspector/InspectorPanel.test.tsx` (a new `describe` block; do not rewrite the existing tests). At the top of `InspectorPanel.test.tsx`, ensure these are present (add if missing):

```typescript
import { fetchAuditReadiness } from "@/api/auditReadiness";
import { makeComposition } from "@/test/composerFixtures";
vi.mock("@/api/auditReadiness", () => ({ fetchAuditReadiness: vi.fn(), fetchAuditReadinessExplain: vi.fn() }));
```

Then add the new describe block:

```typescript
describe("AuditReadinessPanel mount in InspectorPanel", () => {
  beforeEach(() => {
    // Use the canonical fixture; pass an overrides map to clear nodes when
    // the test wants an empty pipeline. SourceSpec is { plugin, options }
    // (NOT kind/config — that shape never existed; the previous draft of
    // this plan inherited the drift from a stale memo).
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1, { nodes: [] }),
    } as never);
    vi.mocked(fetchAuditReadiness).mockResolvedValue({
      session_id: "s-1",
      composition_version: 1,
      rows: [
        { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
        { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
        { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
        { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
        { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
        { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
      ],
    });
  });

  it("renders the audit readiness panel inside the inspector", async () => {
    render(<InspectorPanel />);
    await waitFor(() => {
      expect(screen.getByLabelText("Audit readiness")).toBeInTheDocument();
    });
  });

  it("renders the audit readiness panel above the tab strip", async () => {
    render(<InspectorPanel />);
    const panel = await screen.findByLabelText("Audit readiness");
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    // compareDocumentPosition returns 4 (DOCUMENT_POSITION_FOLLOWING) when
    // the argument follows the receiver — the panel must come first.
    expect(panel.compareDocumentPosition(tablist)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  // W6 — added 2026-05-17

  it("panel remains present when every tab is activated in turn", async () => {
    // The panel mounts above the tab strip (not inside any tab panel), so it
    // must survive tab switching. InspectorPanel has 4 tabs: Spec, Graph, YAML,
    // Runs (InspectorPanel.tsx:32–37). We iterate dynamically over whatever
    // tabs are rendered, so this test is resilient to future tab-list changes.
    const user = userEvent.setup();
    render(<InspectorPanel />);
    // Wait for initial mount and panel to appear.
    await screen.findByLabelText("Audit readiness");
    const tabs = screen.getAllByRole("tab");
    for (const tab of tabs) {
      await user.click(tab);
      // Panel must still be present after each tab switch.
      expect(screen.getByLabelText("Audit readiness")).toBeInTheDocument();
    }
  });

  it("renders without crashing when activeSessionId and compositionState are null", () => {
    // Protects against the null-compositionState path omitted from the initial
    // test set. The panel must not throw on a store that has not yet populated.
    useSessionStore.setState({
      activeSessionId: null,
      compositionState: null,
    } as never);
    // Should not throw.
    render(<InspectorPanel />);
    // The panel is either absent or renders an empty/placeholder state — either
    // is correct. The assertion is that no runtime error is thrown and the
    // tablist still renders (InspectorPanel skeleton remains usable).
    expect(screen.getByRole("tablist", { name: /Inspector tabs/ })).toBeInTheDocument();
  });
});
```

- [x] **Step 3: Run test — expect FAIL** (panel isn't mounted)

- [x] **Step 4: Implement**

In `InspectorPanel.tsx`, add at the top of the imports:

```typescript
import { AuditReadinessPanel } from "../audit/AuditReadinessPanel";
```

Then, after the closing `</div>` of the inspector header (the `<div>` that contains Row 1: version-selector + Catalog/Validate/Execute), and **before** the `role="tablist"` block, insert:

```tsx
        {/* Audit-readiness panel — Phase 2.
            Persistent across tabs. The Validation row inside this panel
            replaces the standalone Validate button (Task 8). */}
        <AuditReadinessPanel />
```

- [x] **Step 5: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx
```

Expected: existing tests still pass; the two new tests pass.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx
git commit -m "feat(web/frontend): mount AuditReadinessPanel above the inspector tab strip (Phase 2C.7)"
```

---

## Task 8: Remove the standalone Validate button and relocate side-effect orchestration

**Files:**
- Modify: `frontend/src/components/inspector/InspectorPanel.tsx`
- Modify: `frontend/src/components/inspector/InspectorPanel.test.tsx`
- Modify: `frontend/src/stores/subscriptions.ts`

The standalone Validate button is subsumed by the audit-readiness panel's Validation row. The `useExecutionStore.validate()` action stays — the panel's backend aggregator calls the same validation route, and `handleExecute` still consults `validationResult` to gate the Execute button.

This task does the full relocation in Phase 2C: it deletes the button, deletes `handleValidate`/`injectSystemMessage`/`sendValidationFeedback` from `InspectorPanel.tsx`, and simultaneously wires the same side effects into `subscriptions.ts` so validation-failure messages continue reaching the LLM agent across the merge boundary.

**Why full relocation now (not deferral to Phase 3A):** The previous plan deferred to Phase 3A under the assumption that keyboard shortcuts or other callers of `handleValidate` would keep the handler live. Verification against the actual source disproves this: `App.tsx:168-179` (`Ctrl+Shift+V`) calls `useExecutionStore.getState().validate(activeSessionId)` directly, not `handleValidate`; `CommandPalette.tsx:88-93` likewise calls `validate(activeSessionId)` directly. The InspectorPanel Validate button was the **only** caller of InspectorPanel's `handleValidate`. Deferring would silently orphan the side effects.

- [x] **Step 1: Write the failing test (negative + smoke)**

In `frontend/src/components/inspector/InspectorPanel.test.tsx`, add:

```typescript
describe("Validate button removal (Phase 2C)", () => {
  beforeEach(() => {
    // Use the canonical fixture (see Task 7's beforeEach for the rationale
    // on SourceSpec/NodeSpec shape).
    useSessionStore.setState({
      activeSessionId: "s-1",
      compositionState: makeComposition(1, { nodes: [] }),
    } as never);
  });

  it("does not render a button labelled 'Validate' (subsumed by audit-readiness panel)", () => {
    render(<InspectorPanel />);
    expect(
      screen.queryByRole("button", { name: /^Validate$/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /^Validate pipeline$/ }),
    ).not.toBeInTheDocument();
  });

  it("still renders Execute and Catalog buttons", () => {
    render(<InspectorPanel />);
    expect(screen.getByRole("button", { name: /^Execute pipeline$/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Catalog/ })).toBeInTheDocument();
  });

  it("the tab strip still renders and arrow navigation still works", async () => {
    render(<InspectorPanel />);
    const tablist = screen.getByRole("tablist", { name: /Inspector tabs/ });
    expect(tablist).toBeInTheDocument();
    const tabs = within(tablist).getAllByRole("tab");
    expect(tabs.length).toBeGreaterThan(1);
    tabs[0].focus();
    fireEvent.keyDown(tabs[0], { key: "ArrowRight" });
    // The next tab should be focused after arrow navigation; the focus
    // assertion verifies the keyboard navigation hasn't regressed when the
    // header layout changed.
    expect(document.activeElement).toBe(tabs[1]);
  });
});
```

Also locate any existing tests that assert the Validate button exists or click it to trigger `handleValidate`'s side effects, and update them:

1. Tests under `describe("InspectorPanel three-state validation indicator", ...)` — these test the validation **dot** (which stays). No change.
2. Tests under `describe("Version selector and catalog", ...)` — none reference Validate. No change.
3. Tests under `describe("InspectorPanel execution feedback", ...)` — any test that asserted the `handleValidate`-driven system-message injection should be **deleted**. The new behaviour (injection on validation failure) is covered by the `subscriptions.test.ts` new describe block added in Task 8.5. Do not attempt to invoke `handleValidate` directly in the test — the handler is being deleted.

- [x] **Step 2: Run tests — expect FAIL on the new negative assertions**

- [x] **Step 3: Implement — remove the Validate button and delete the handlers**

**3a. In `InspectorPanel.tsx`, delete the Validate button block:**

Find and delete the entire block:

```tsx
            {/* Validate button with spinner */}
            <button
              onClick={handleValidate}
              disabled={!canValidate}
              aria-label={isValidating ? "Validating" : "Validate pipeline"}
              className="btn inspector-action-btn"
            >
              {isValidating ? (
                <span
                  className="spinner"
                  role="status"
                  aria-label="Validating"
                />
              ) : (
                "Validate"
              )}
            </button>
```

Keep the surrounding Catalog and Execute buttons untouched.

**3b. Delete `handleValidate`, `injectSystemMessage`, `sendValidationFeedback`, `canValidate`, and `isValidating` from `InspectorPanel.tsx`:**

- Delete the `handleValidate` `useCallback` block (lines 387–424 at the time of writing — verify by grepping before editing).
- Delete the `injectSystemMessage` and `sendValidationFeedback` destructure lines (currently lines 356–357: `const injectSystemMessage = useSessionStore(...)` and `const sendValidationFeedback = useSessionStore(...)`).
- Delete `const isValidating = useExecutionStore(...)` (currently line 361).
- Delete `const canValidate = ...` block (currently lines 381–385).

Before deleting each symbol, run:
```bash
grep -n "injectSystemMessage\|sendValidationFeedback\|handleValidate\|canValidate\|isValidating" \
  src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx
```
to confirm there are no other references. If any symbol has a remaining reference outside the deleted block, investigate before deleting.

`hasCompositionContent` may still be used by the validation dot indicator — do NOT delete it; it is a separate derived variable.

**3c. Wire the side effects into `subscriptions.ts`:**

Add a second subscriber to `useExecutionStore` in `initStoreSubscriptions()`. The outer shape subscribes to `validationResult` directly (we observe results, not drive validation — `validate()` remains at its existing callers in `App.tsx` and `CommandPalette.tsx`). Only the inner side-effect body is lifted from `15a2` Task 8 Step 2:

```typescript
import type { ValidationResult } from "@/types/index";

// Module-level state for the executionStore subscriber.
// Must be reset in _resetSubscriptionsForTesting().
let previousValidationResult: ValidationResult | null = null;
let unsubscribeExecution: (() => void) | null = null;
```

Add the `import type` at the top of `subscriptions.ts` alongside the existing store imports. The path `@/types/index` matches the pattern already used in `executionStore.ts:26`. Add the two `let` variables next to `previousVersion` / `previousSessionIds` / `unsubscribe` at the top of `subscriptions.ts`.

In `initStoreSubscriptions()`, after the existing `unsubscribe = useSessionStore.subscribe(...)` block, add:

```typescript
  const VALIDATION_MSG_ID = "system-validation-current";

  unsubscribeExecution = useExecutionStore.subscribe((state) => {
    const result = state.validationResult;
    // Reference-equality guard: fire only on result change, not on every
    // store update. Prevents duplicate injectSystemMessage / sendValidationFeedback
    // calls when validate() is invoked in quick succession.
    if (result === previousValidationResult) return;
    previousValidationResult = result;

    if (!result) return;

    const sessionStore = useSessionStore.getState();

    if (!result.is_valid && result.errors.length > 0) {
      const lines = ["**Validation failed** — the following errors were sent to the agent:"];
      for (const err of result.errors) {
        lines.push(
          `- **[${err.component_type ?? "unknown"}] ${err.component_id ?? "unknown"}:** ${err.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
      // sendValidationFeedback is fire-and-forget. Per CLAUDE.md audit-primacy,
      // the backend records the validation event in the audit Landscape; a
      // frontend telemetry breadcrumb would duplicate that record. The
      // user-visible system message is already injected above. Phase 8 is
      // the right owner if a frontend operational signal proves useful.
      // Operator adjudication 2026-05-15 (preserved from 15a2 Task 8 Step 2).
      void sessionStore.sendValidationFeedback(result);
    } else if (result.is_valid && result.warnings && result.warnings.length > 0) {
      const lines = ["**Validation passed with warnings:**"];
      for (const warn of result.warnings) {
        lines.push(
          `- **[${warn.component_type ?? "unknown"}] ${warn.component_id ?? "unknown"}:** ${warn.message}`,
        );
      }
      sessionStore.injectSystemMessage(lines.join("\n"), VALIDATION_MSG_ID);
    }
  });
```

Update `_resetSubscriptionsForTesting()` to also tear down the new subscriber and reset the new module variable:

```typescript
export function _resetSubscriptionsForTesting(): void {
  unsubscribe?.();
  unsubscribe = null;
  unsubscribeExecution?.();
  unsubscribeExecution = null;
  previousVersion = null;
  previousValidationResult = null;
  previousSessionIds = new Set();
  initialized = false;
}
```

> **Wire shape note.** The outer subscription glue here differs from `15a2` Task 8 Step 2: Phase 3A subscribed to `compositionState.version` change and drove `validate()` inside the subscription (auto-validate on change). Phase 2C subscribes to `useExecutionStore.validationResult` directly — we observe results and fire side effects, but we do not drive validation. `validate()` continues to be called by `App.tsx:177` (`Ctrl+Shift+V`) and `CommandPalette.tsx:88-93`. The inner side-effect body (the two-branch message-building and `sendValidationFeedback` call) is lifted verbatim from `15a2` Task 8 Step 2:566-600.

- [x] **Step 4: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/inspector/InspectorPanel.test.tsx
```

Expected: all assertions pass, including the new negative tests.

- [x] **Step 5: Toolchain check (tsc + eslint)**

All four tsconfig files in this project enable `noUnusedLocals: true` and `noUnusedParameters: true` (`tsconfig.json`, `tsconfig.app.json`, `tsconfig.test.json`, `tsconfig.e2e.json` — verified 2026-05-16). `handleValidate`, `canValidate`, `isValidating`, `injectSystemMessage`, and `sendValidationFeedback` are deleted outright in Step 3b — no underscore-prefix or `@ts-expect-error` workarounds are needed. If any reference to these symbols remains after the deletion (confirmed by the grep in Step 3b), fix the remaining reference rather than suppressing the error.

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit && npx eslint src/components/inspector/InspectorPanel.tsx
```

Both must pass before committing. Do **not** use `eslint-disable` for symbols that should have been deleted.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx \
        src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx \
        src/elspeth/web/frontend/src/stores/subscriptions.ts
git commit -m "feat(web/frontend): remove Validate button; relocate side-effect orchestration to subscriptions.ts (Phase 2C.8)

Full relocation in Phase 2C (not deferred to Phase 3A). Deletes the
standalone Validate button, handleValidate, injectSystemMessage,
sendValidationFeedback, canValidate, and isValidating from
InspectorPanel.tsx. Adds a useExecutionStore.validationResult subscriber
in subscriptions.ts that fires injectSystemMessage + sendValidationFeedback
on result reference change, preserving the LLM agent's validation-failure
signal across the merge boundary.

Keyboard (Ctrl+Shift+V) and CommandPalette callers call validate()
directly and are unaffected. _resetSubscriptionsForTesting() updated
to also unsubscribe the new execution subscriber."
```

---

## Task 8.5: Subscription side-effect test

**Why this task exists.** Task 8 wires `injectSystemMessage` and `sendValidationFeedback` into `subscriptions.ts`. This task verifies that wiring is correct by adding a green test to the existing `subscriptions.test.ts` suite. The test is a normal passing test — not a guard, not a red gate. It also proactively fixes the B4 isolation defect by calling `_resetSubscriptionsForTesting()` in `beforeEach`, exactly as the existing `subscriptions.test.ts` tests do.

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — add a new `describe` block for validation-result side effects.

> **Note (B4 resolved).** The prior plan drafted a standalone `subscriptions.handoff.test.ts` without `_resetSubscriptionsForTesting()` in `beforeEach`, which would have left `initialized = true` leaking between tests (Zustand module-level state is shared). Co-locating the new `describe` block inside `subscriptions.test.ts` inherits the file's existing `beforeEach` discipline and eliminates the singleton-bleed entirely.

- [x] **Step 1: Add the new describe block**

Add to `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` (after the existing `describe` block):

```typescript
describe("subscriptions — validation result side effects", () => {
  beforeEach(() => {
    _resetSubscriptionsForTesting();
    vi.clearAllMocks();
  });

  it("calls injectSystemMessage and sendValidationFeedback when validation fails", () => {
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn().mockResolvedValue(undefined);
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendValidationFeedback,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // Act: set a failing validation result. The subscriber is synchronous —
    // no need for waitFor (mirrors the pattern in the existing subscriptions.test.ts).
    useExecutionStore.setState({
      validationResult: {
        is_valid: false,
        errors: [
          {
            component_type: "source",
            component_id: "csv_source",
            message: "Required field 'path' is missing",
          },
        ],
        warnings: [],
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendValidationFeedback).toHaveBeenCalled();

    const [message, stableId] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation failed");
    expect(message).toContain("csv_source");
    expect(stableId).toBe("system-validation-current");
  });

  it("calls injectSystemMessage but NOT sendValidationFeedback when validation passes with warnings", () => {
    const injectSystemMessage = vi.fn();
    const sendValidationFeedback = vi.fn();
    useSessionStore.setState({
      activeSessionId: "sess-1",
      injectSystemMessage,
      sendValidationFeedback,
    } as never);
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    useExecutionStore.setState({
      validationResult: {
        is_valid: true,
        errors: [],
        warnings: [
          {
            component_type: "transform",
            component_id: "select_cols",
            message: "Identity passthrough detected",
          },
        ],
      } as never,
    } as never);

    expect(injectSystemMessage).toHaveBeenCalled();
    expect(sendValidationFeedback).not.toHaveBeenCalled();

    const [message] = injectSystemMessage.mock.calls[0] as [string, string];
    expect(message).toContain("Validation passed with warnings");
    expect(message).toContain("select_cols");
  });

  it("fires side effects exactly once when the same result reference is set twice (reference-equality guard)", () => {
    const injectSystemMessage = vi.fn();
    useSessionStore.setState({ activeSessionId: "sess-1", injectSystemMessage } as never);
    const result = {
      is_valid: false,
      errors: [{ component_type: "source", component_id: "s1", message: "boom" }],
      warnings: [],
    };

    // Start from null so the first setState transitions null → result.
    useExecutionStore.setState({ validationResult: null } as never);
    initStoreSubscriptions();

    // First setState: null → result; previousValidationResult becomes result;
    // side effects fire once.
    useExecutionStore.setState({ validationResult: result } as never);

    // Second setState: result === previousValidationResult; guard must prevent
    // a second fire.  The subscriber fires (Zustand 1-arg subscribe fires on
    // every setState), but the reference-equality check should short-circuit.
    useExecutionStore.setState({ validationResult: result } as never);

    // Exactly one call — not zero (first fire happened), not two (second blocked).
    expect(injectSystemMessage).toHaveBeenCalledTimes(1);
  });
});
```

At the top of the file, add `useExecutionStore` to the imports (the existing file does not import it). The tests are synchronous — no `waitFor` is needed, matching the style of the existing `subscriptions.test.ts` tests:

```typescript
import { useExecutionStore } from "./executionStore";
```

- [x] **Step 2: Run the new tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/subscriptions.test.ts
```

Expected: all tests pass, including the three new ones.

- [x] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/subscriptions.test.ts
git commit -m "test(web/frontend): add subscription validation-result side-effect tests (Phase 2C.8.5)

Normal passing tests (no guard, no red gate) asserting that
injectSystemMessage + sendValidationFeedback fire when
useExecutionStore.validationResult transitions to a failing result.
Both branches covered: errors (inject + feedback) and warnings-only
(inject only). Reference-equality guard prevents double-fire.
_resetSubscriptionsForTesting() in beforeEach isolates test state
(resolves B4 singleton-bleed)."
```

---

## Task 9: End-to-end smoke against the staging backend

**Files:** None (manual).

**Prerequisite:** Phase 2A is merged and deployed to staging (`elspeth.foundryside.dev`). The session DB has the new endpoints reachable from the frontend build.

- [ ] **Step 1: Build and deploy the frontend**

Following `project_staging_deployment` memory:

```bash
cd src/elspeth/web/frontend && npm run build
# Then on the deploy host (this machine, per the staging-deployment memory):
sudo systemctl restart elspeth-web.service
```

- [ ] **Step 2: Smoke — all-green collapse**

1. Open staging in the browser; log in as the test user.
2. Create a new session, compose a trivially-valid pipeline (CSV source → CSV sink, no transforms).
3. Verify the audit-readiness panel renders **collapsed** to "Audit ready ✓".
4. Click the collapsed summary; verify all six rows appear.
5. Click "Explain →"; verify the narrative renders within ~1s and the wording matches the Phase 2A `build_narrative` output.

- [ ] **Step 3: Smoke — provenance warning**

1. In the same session, add an identity-passthrough transform (e.g. `select_columns` keeping every column).
2. Verify the panel auto-expands to all rows and the Provenance row shows `⚠`.
3. Click the Provenance row; the detail drawer opens.
4. Verify "Jump to <node-id>" navigates to the node in the Spec tab and selects it.

- [ ] **Step 4: Smoke — Validate button is gone; subscription side-effects fire**

1. Look at the inspector header. There must be **no** Validate button — only Catalog and Execute.
2. Verify Execute is disabled until the panel's Validation row shows `✓`.
3. Enter an invalid pipeline configuration (e.g. add a node referencing a non-existent plugin, or leave a required options field blank).
4. Trigger validation — either via `Ctrl+Shift+V` or by saving the composition (if auto-validate fires on version change). **Acceptance criterion:** a system message appears in the chat panel announcing validation failure with the specific error(s). The message format is "**Validation failed** — the following errors were sent to the agent:" followed by the error list. This confirms the `subscriptions.ts` side-effect wiring is live end-to-end.
5. Fix the invalid configuration; trigger validation again. The system message should update to "**Validation passed**" (or no warning message if no warnings). The Execute button should become enabled.

This smoke step also addresses W2 (subscription wiring observable end-to-end). Do not paper over a missing system message — if the chat panel shows no validation-failure message after triggering validation on an invalid pipeline, the subscription is not firing and the regression is live.

- [ ] **Step 5: Smoke — error path**

1. Stop the backend (`sudo systemctl stop elspeth-web.service`).
2. Reload the page; verify the audit-readiness panel renders an `alert`-role error message rather than crashing the inspector.
3. Restart the backend; reload; verify the panel recovers on next composition change.

- [ ] **Step 6: Sign off**

If all five smoke steps pass, mark Phase 2 complete and merge the umbrella PR. If any step fails, file the failure as an observation (`mcp__filigree__observe`) with `file_path` set to the relevant component and stop — do not paper over a runtime divergence.

---

## What Phase 2C leaves the frontend in

- `components/audit/ReadinessRowDetail.tsx` is the full implementation (jump-to-component for resolvable ids).
- `components/audit/ExplainDialog.tsx` is the full implementation (modal, version-keyed narrative cache). Follows the project's modal-dialog focus contract: `useFocusTrap` traps Tab focus and restores to the opener on close; an `onKeyDown` Escape handler dismisses the dialog (matches the CommandPalette/RecoveryPanel pattern).
- `InspectorPanel.tsx` renders `<AuditReadinessPanel />` above the tab strip; the standalone Validate button, `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` are all deleted from the inspector. The side-effect orchestration (system-message injection on validation failure + `sendValidationFeedback`) is now in `subscriptions.ts`, firing on `useExecutionStore.validationResult` reference change. `canValidate` and `isValidating` are also deleted (their only consumer was the deleted button and `handleValidate`).
- `stores/auditReadinessStore.ts` gains `userExpandedBySession: Record<string, boolean>` (7th per-session keyed map) and a `setUserExpanded(sessionId, value)` action (Task 4C). `getInitialState()` includes `userExpandedBySession: {}`. `clearSession` pops `userExpandedBySession[sessionId]` alongside the other six maps. `api/auditReadiness.ts` is unchanged.
- `AuditReadinessPanel.tsx` reads `userExpanded` from the store (keyed by `activeSessionId`) instead of component-local `useState(false)` (Task 4C). `setUserExpanded` now calls `setUserExpanded(activeSessionId, value)` on the store action. The component's expand/collapse logic (`showExpanded = anyActionable || userExpanded`) is unchanged.
- `stores/subscriptions.ts` gains a `useExecutionStore` subscriber that fires `injectSystemMessage` + `sendValidationFeedback` on validation result change. `_resetSubscriptionsForTesting()` is updated to also unsubscribe the new executionStore listener and reset `previousValidationResult`.
- The Phase 1B preference-bootstrap and Phase 5/6/7/8/9/10 chat flows are **untouched**.
- The Execute button still gates on `validationResult` — wire shape unchanged.
- Phase 2 is now feature-complete pending the staging smoke (Task 9).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Panel auto-fetch races a still-mutating composition | Auto-fetch is keyed on `composition_version` (integer, monotonic). The store short-circuits when the version matches; a mid-mutation render reuses the cached value, and the next render after the version advances triggers a fresh fetch. |
| Subscription fires duplicate `sendValidationFeedback` calls if `validate()` is invoked multiple times in quick succession | Reference-equality transition check on `previousValidationResult` ensures the side effects fire only on result change, not on every store update. `injectSystemMessage`'s stable-id pattern (`VALIDATION_MSG_ID`) coalesces the UI side: a repeated message with the same id replaces the prior one rather than appending. |
| Phase 3 changes the inspector layout and the panel breaks | The panel component is layout-agnostic. Phase 3 only changes the mount point. Phase 2's tests assert the panel's behaviour, not its DOM order; Phase 3's tests re-assert ordering at the new mount site. |
| Per-row "Jump to component" fails when `component_ids` references a source or sink (not a node) | Step 3 of Task 5 renders unresolvable ids as plain text rather than a non-functional button. The user can grep the YAML. Phase 3's side-rail reshape (if it surfaces sources/sinks as selectable) is the place to make those clickable. |
| Backend wire-shape drift breaks the renderer | The discriminated-union `never` arm fails the build; Phase 2A's `_StrictResponse` fails at server-side construction. No silent path. |
| Telemetry can't be added later without rewriting the panel | The row-click handlers are isolated functions in `AuditReadinessPanel` and `ReadinessRowDetail`. Phase 8 telemetry adds one line per handler. |
| User reports "the panel is slow" | The aggregator is the same cost as the standalone Validate it replaced (Phase 2A's risk table). Profile before optimising; do not add a debounce until profiling shows the need. |
| `aria-modal="true"` dialog without focus contract — AT users cannot reach dialog content | `useFocusTrap` covers focus trap + initial focus + focus restoration on close; separate `onKeyDown` Escape handler covers keyboard dismissal. Task 6 tests assert each: focus inside on mount, Escape fires onClose, focus returns to opener on unmount. |
| Phase 3B remount of `<AuditReadinessPanel />` resets `userExpanded` (W3) | Task 4C migrates `userExpanded` from component-local `useState` to `auditReadinessStore.userExpandedBySession` (per-session keyed). The remount-safety regression test in Task 4C unmounts and re-renders the panel and asserts the preference survives via the store. |
| Coverage gap: edge cases (drawer Escape close, version-keyed refetch, backdrop click, no-detail error fallback, mixed resolvable ids, all-tab mount, null-compositionState guard) untested | Addressed by W6's 7 new tests across Tasks 5, 6, and 7 (2026-05-17 revision). Test 5.C (null component_ids) was deliberately skipped — `readonly string[]` is non-nullable in the type; the guard would be defensive against an impossible state. |

## Review history

**2026-05-15** — Panel findings applied: Task 8 now deletes `handleValidate` and callers (`injectSystemMessage`, `sendValidationFeedback`) from `InspectorPanel.tsx` — side effects relocated to Phase 3A `subscriptions.ts` (BLOCKER); test migration policy changed from "inline TODO acceptable" to "block-commit-if-not-migrated" (CRITICAL); Phase 3B mount-point dependency documented in §Sequencing (IMPORTANT); `require()` spy replaced with top-level `vi.mock()` in Task 7 test (IMPORTANT); negative `waitFor` race in ExplainDialog test fixed to wait for render before asserting not-called (from 14b Finding 3, applied here where the code lives).

### 2026-05-16 — 4-reviewer panel verdict CHANGES_REQUESTED → fixes applied

Reviewers: reality, architecture, quality, systems (full report:
`14-phase-2-audit-readiness-panel.review.json`).

Fixes applied to 14c in this revision:
1. Task 8 scope-shrink — remove only the standalone Validate button UI; defer handleValidate/injectSystemMessage/sendValidationFeedback deletion to Phase 3A Task 8 where the relocation into subscriptions.ts is atomic (convergence C3)
2. Task 8.5 added — Phase 3A handoff guard test (no `.skip`) that fails red in CI today because Phase 3A's subscription wiring is absent; turns green when Phase 3A Task 8 lands (mechanical gate replacing prose-only constraint)

Strategic adjudication: scope-shrink (preferred path per review C3) over pulling Phase 3A Task 4+8 into the Phase 2 umbrella PR (would inflate umbrella scope and require coordination with the already-committed 15a plans).

### 2026-05-17 — Reality-check vs landed 2A+2B: NEEDS_MAJOR_REWRITE → fixes applied

Reviewer: `axiom-planning:plan-review-reality`, run against `feat/composer-phase-2a-backend` HEAD (2A+2B both landed and green: 56/56 backend unit + 12/12 backend integration + 585/585 frontend tests). Verdict on 14c: **NEEDS_MAJOR_REWRITE** (2 BLOCKER + 3 MAJOR + 1 NIT drifts) — 14a + 14b implementations were APPROVED with zero blockers.

Fixes applied in this revision:

1. **D1 BLOCKER — Task 6 ExplainDialog selector drift (production code).** The dialog read `s.isLoadingExplain` and `s.explainError`; the store has no such flat fields (per-session keyed maps only). Both selectors would evaluate to `undefined` at runtime and the dialog would never render loading or error states. Rewritten to `s.isLoadingExplainBySession[sessionId] ?? false` and `s.explainErrorBySession[sessionId] ?? null`.
2. **D2 BLOCKER — Task 6 ExplainDialog test beforeEach drift.** The setState literal listed flat fields under `as never`; the cast silently masked the field-name mismatch. Replaced with `useAuditReadinessStore.setState(getInitialState())`, mirroring the pattern 2B already landed in `AuditReadinessPanel.test.tsx:58`.
3. **D3 MAJOR — fixture import-path drift.** `@/test-utils/composerFixtures` → `@/test/composerFixtures` (the real fixture is at `src/elspeth/web/frontend/src/test/composerFixtures.ts`). Confirmed against 2B's `AuditReadinessPanel.test.tsx:9`.
4. **D4 MAJOR — Task 5 inline NodeSpec shape drift.** The plan's reference fixture used `{ id, node_type, plugin, config: {} } as never`; `NodeSpec` actually requires 7 fields (`id`, `node_type`, `plugin`, `input`, `on_success`, `on_error`, `options`) and has no `config` field. Inline fixture removed in favour of `import { makeComposition } from "@/test/composerFixtures"`.
5. **D5 MAJOR — Tasks 7+8 SourceSpec shape drift.** The plan used `source: { kind: "csv_file", config: { path: "x.csv" } } as never`; `SourceSpec` is `{ plugin, options }`. Both setState blocks rewritten to use `makeComposition(1, { nodes: [] })`.
6. **D6 NIT — File-structure / per-task header inconsistency.** "File structure" header was `Modified:` but bullet text said "new file"; per-task `Files:` blocks said `Create:` for files 14b had already shipped as placeholders. Section restructured into explicit `New:` + `Modified:` subsections, with a load-bearing "Verified-against-reality" anchor block recording the store/type/path facts the next reality-check should re-verify.

Two new tasks absorbed (both promoted from 2B observations explicitly deferred to 14c):

- **Task 4A (new) — Signal-aware mock helper + cleanup-effect removal.** Tracks `elspeth-f018ea84c6` (promoted from `elspeth-obs-8502e9d4bf`). Adds `makeAbortablePromise()` to `composerFixtures.ts`; lets the store's `AbortError` catch arm fire in tests; drops the architecturally-duplicated synchronous `setState` in `AuditReadinessPanel.tsx`'s useEffect cleanup. Sequenced before Tasks 5–6 so they inherit the canonical mock pattern.
- **Task 4B (new) — Compute `showExpanded` instead of state-sync useEffect.** Tracks `elspeth-82ef9d5bd0` (promoted from `elspeth-obs-6b5bb1a476`). Renames `expanded`→`userExpanded`, computes `showExpanded = anyActionable || userExpanded`, deletes the derived-state effect. Removes the extra render cycle on warning snapshots and the sticky-expansion bug (panel now auto-collapses when a later snapshot returns all-green unless the user explicitly expanded). A new regression test covers the auto-collapse behaviour.

Strategic adjudication: 2C amendment is iterate-don't-rewrite — the plan's task graph, scope boundaries, sequencing claims, and Tasks 4A/4B/5/6/7 are all sound. All edits in this round are localised to (a) the file-structure block, (b) the new Tasks 4A/4B, and (c) spot fixes inside Tasks 5–8's code blocks. No task was added beyond what 2B observations explicitly deferred to 14c. (Note: the convergence-C3 reasoning in Task 8 was revised in the 4-reviewer re-review cycle that followed — see next entry.)

### 2026-05-17 — 4-reviewer first-principles re-review verdict CHANGES_REQUESTED → B1 fix applied

Reviewers: reality, architecture, quality, systems (second-cycle re-review, after Reality-check D1-D6 fixes landed).

**Root cause of B1:** The plan's convergence-C3 reasoning (Task 8 body, "leaves the frontend in") asserted that `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` "remain wired" after the standalone Validate button is removed because "the button is the only caller removed; keyboard shortcuts or other callers continue to work." This claim was factually false:

- `App.tsx:168-179` — `Ctrl+Shift+V` calls `useExecutionStore.getState().validate(activeSessionId)` directly, not `handleValidate`.
- `CommandPalette.tsx:88-93` — "Validate Pipeline" palette command calls `validate(activeSessionId)` directly, not `handleValidate`.
- `CompletionSummary.tsx:78-80` — has a separate `handleValidate` calling `validate()` directly with no side effects.

The InspectorPanel "Validate" button (InspectorPanel.tsx:558) was the **only** caller of InspectorPanel's `handleValidate`. Removing the button made `handleValidate`, `injectSystemMessage`, and `sendValidationFeedback` dead on merge. The LLM agent would stop receiving validation-failure messages — the audit-readiness panel would show ✓ while the agent had no signal. Silent regression in the core Sense/Decide/Act loop.

**Chosen remedy (one-fix-resolves-many):** Pull Phase 3A Task 8's subscription wiring into Phase 2C. This also collapses B4 (Task 8.5 singleton-bleed becomes moot — replaced by a real green test using `_resetSubscriptionsForTesting()`), B5 (Task 8 Step 5's underscore-prefix/`@ts-expect-error` alternatives become moot — `handleValidate` is now deleted in-PR so `canValidate`/`isValidating` are outright deleted), and W1 (persistent CI red as desensitisation is moot — no intentionally-red gate).

Fixes applied in this revision:
1. **Task 8 rewritten** — full relocation: delete `handleValidate`/`injectSystemMessage`/`sendValidationFeedback` from `InspectorPanel.tsx` in-PR; add subscription to `useExecutionStore.validationResult` in `subscriptions.ts` that fires the same side effects on reference-change transition.
2. **Task 8.5 replaced** — intentionally-red guard test removed; replaced with a normal green test in `subscriptions.test.ts` asserting that `injectSystemMessage` + `sendValidationFeedback` fire on validation failure. `_resetSubscriptionsForTesting()` called in `beforeEach` (fixes B4 isolation proactively).
3. **Task 8 Step 5 rewritten** — outright deletion of `canValidate`, `isValidating`, and `handleValidate`; underscore-prefix and `@ts-expect-error` alternatives dropped (they were only proposed because handleValidate was being deferred).
4. **Scope boundaries updated** — handler orchestration moved into scope; "Out of scope" deferral paragraph removed.
5. **"Leaves the frontend in" updated** — handlers deleted, orchestration in `subscriptions.ts`.
6. **Task 9 Step 4 updated** — ShortcutsHelp grep replaced with concrete end-to-end smoke criterion.
7. **Sequencing updated** — Phase 3A no longer owns the handler relocation.
8. **Risks table updated** — "Phase 3A Task 8 deletes ... in a different PR" row removed (moot).

**Cross-reference:** The Phase 3A plan (`15a2-phase-3a-removals-part-2.md`) needs a corresponding update to mark its Task 8 as "absorbed into Phase 2C 14c". That update is a separate cycle — do not edit 15a2 in this pass.

### 2026-05-17 — B3 fix: ExplainDialog focus contract added (useFocusTrap + Escape handler)

Reviewer finding (B3): Plan Task 6 set `role="dialog" aria-modal="true"` in `ExplainDialog.tsx` without focus trap, Escape handler, initial focus, or focus restoration. `aria-modal="true"` is a binding WAI-ARIA contract — screen readers (NVDA, JAWS, VoiceOver) restrict virtual cursor navigation to the dialog's subtree. Without a focus trap, keyboard-only and AT users cannot reach dialog content or dismiss the dialog by keyboard.

Fix applied: Task 6 implementation prescription updated to match the project's established modal-dialog pattern (CommandPalette, RecoveryPanel, ComposerPreferencesPanel, ShortcutsHelp, SecretsPanel):
1. `useRef<HTMLDivElement>(null)` (`dialogRef`) created and attached to the outer dialog `div`.
2. `useFocusTrap(dialogRef, true, ".explain-dialog-close")` called — traps Tab focus, moves initial focus to the Close button, restores focus to the opener on unmount.
3. `onKeyDown` Escape handler added to the dialog `div` — `useFocusTrap` does not register an Escape listener; this mirrors CommandPalette's inline handler (lines 267–270).
4. Three new tests added to Task 6's test block: focus moves into dialog on mount; Escape fires onClose; focus returns to opener after close (via Harness pattern matching RecoveryPanel.test.tsx).
5. Risks table updated with the accessibility-contract row.
6. "What Phase 2C leaves the frontend in" updated to document the focus contract.

Note: `active` flag is `true` (not tied to an `isOpen` prop — ExplainDialog renders only while open, so `active=true` is always correct here, unlike CommandPalette which passes `isOpen`). This divergence is documented in the implementation comment.

### 2026-05-17 — B7 fix: stale line citations in Verified-against-reality anchor block corrected

`NodeSpec` citation corrected from `types/index.ts:133–134` to `types/index.ts:105–119`; `SourceSpec` citation corrected from `types/index.ts:94–96` to `types/index.ts:94–99` (both verified against disk). "requires 7 fields" tightened to "requires 7 required fields" to prevent misreading total field count as 7.

### 2026-05-17 — W3 fix: Task 4C inserted to pre-empt Phase 3B userExpanded state-loss

Systems reviewer finding W3: `AuditReadinessPanel` stores `userExpanded` as component-local `useState(false)`. Phase 3B Task 9 Step 4a relocates `<AuditReadinessPanel />` from `InspectorPanel.tsx` to `App.tsx → SideRail.auditReadinessSlot` — a full component remount — which resets `userExpanded` to `false`. A user who had explicitly expanded the panel to keep it persistently visible would find it collapsed after Phase 3B's deploy. The store-keyed snapshot data survives (zustand, keyed by sessionId), but the UI preference is silently discarded.

Fix: Task 4C inserted between Task 4B and Task 5. Task 4C migrates `userExpanded` from component-local `useState` into `auditReadinessStore` as a per-session keyed map (`userExpandedBySession: Record<string, boolean>`) with a matching `setUserExpanded(sessionId, value)` action. The new test (remount-safety regression) unmounts and re-renders the panel and asserts `userExpanded` survives via the store.

### 2026-05-17 — Tasks 4A and 4B collapsed to verification stubs (already committed on branch)

Git confirms both tasks landed in HEAD before this plan revision:
- `7ba9bc49e` — Task 4A: signal-aware mock helper + drop panel cleanup-effect setState workaround.
- `b8fb6b23f` — Task 4B: compute showExpanded instead of useState/useEffect sync.
- `8a0f499d7` — follow-up scrub: removed issue-id references from inline panel comments (does not affect the structural outcome).

Tasks 4A and 4B were accordingly collapsed from full TDD task bodies (~115 and ~90 lines respectively) to single-step verification stubs. The stubs instruct an implementer to confirm the work is on disk and proceed; they do not re-apply the implementation. Issue traceability (`elspeth-f018ea84c6`, `elspeth-82ef9d5bd0`) is preserved in both stubs. File-structure bullets updated to past tense with commit hashes.

One drift found during reality-check: the original task brief's third verification grep for 4A (`'Task 4A Step 2 fixed that at the mock layer'`) no longer matches disk — commit `8a0f499d7` scrubbed that comment text. Substituted with `'AbortError catch arm clears'` (line 100 of `AuditReadinessPanel.tsx`), which is stable post-scrub evidence that the cleanup-effect refactor landed.

### 2026-05-17 — W5 fix: Task 6 loading-state test wrapped in `waitFor` + post-resolve transition assertion added

Quality reviewer finding W5: Task 6's loading-state test (`"renders a loading state while the fetch is pending"`) had two defects:

1. **Synchronous assertion raced the effect.** The bare `expect(screen.getByText(/Generating explanation/i)).toBeInTheDocument()` fired immediately after `render()`, before the `useEffect` that calls `loadExplain` had run. The test was asserting on the pre-effect initial render, not on the loading state the effect sets. It would pass trivially if the component rendered the loading indicator unconditionally on mount — including if the loading indicator was never cleared.
2. **`resolve()` never awaited.** The promise was resolved at the end of the synchronous test body, but without a subsequent `await waitFor(...)`, the component's post-resolve re-render was never observed. The test therefore did not verify that the loading indicator disappears and content appears after the fetch completes.

Fix applied to Task 6 Step 1's third test prescription:
1. `it` callback made `async`.
2. Loading-indicator assertion wrapped in `await waitFor(...)` — waits for the effect to fire and the store to set `isLoadingExplainBySession[SESSION_ID] = true`.
3. After `resolve(...)`, `await waitFor(() => expect(screen.getByText("done.")).toBeInTheDocument())` added — confirms the transition from loading to content.
4. `expect(screen.queryByText(/Generating explanation/i)).not.toBeInTheDocument()` added immediately after — confirms the loading indicator is gone post-resolve.
5. Test title extended to `"renders a loading state while the fetch is pending, then transitions to content on resolve"` to reflect the broader coverage.

The `waitFor` idiom matches the canonical form already used throughout `AuditReadinessPanel.test.tsx` (e.g. lines 67, 104, 128, 151, 303, 320).

### 2026-05-17 — W6 fix: 7 new test cases added across Tasks 5, 6, and 7 (Quality reviewer coverage gaps)

Quality reviewer finding W6: the initial test sets for Tasks 5, 6, and 7 left 8 edge cases uncovered. One was determined moot on fact-checking; 7 new tests were added.

**Task 5 — 2 tests added:**

1. `"renders both resolvable and unresolvable ids in the same row (mixed)"` — single fixture with one resolvable id (`select_columns`) and one unresolvable id (`api_key`); asserts Jump button for the resolvable one, plain text for the other. Replaces the two separate fixtures that each tested only one branch.
2. `"fires onClose when Escape is pressed"` — presses `{Escape}` and asserts `onClose` called once. Required a corresponding implementation update: `onKeyDown` Escape handler added to the root `div` of `ReadinessRowDetail.tsx` (see Step 3 implementation block — aria-modal is `"false"` for drawers, so `useFocusTrap` was not appropriate; a bare `onKeyDown` handler matches the conventional drawer affordance).

**Test 5.C skipped (null component_ids):** `ReadinessRow.component_ids` is `readonly string[]` in `types/index.ts:857` — non-nullable. The type makes the null-guard test redundant (defensive programming against an impossible state — forbidden by project policy). If the backend schema ever changes to `string[] | null`, update the type first.

**Task 6 — 3 tests added:**

3. `"refetches when compositionVersion changes"` — verifies the store's version-keyed cache semantics: changing `compositionVersion` from 1 → 2 misses the cache (`cached.composition_version === 2` is false at line 150 of `auditReadinessStore.ts`) and fires a new fetch. Two sequential mock responses return different narratives; the test asserts both fetch calls and both narrative renders.
4. `"closes when the backdrop is clicked"` — clicks the `.explain-dialog-backdrop` div (which has `onClick={onClose}` in the implementation) and asserts `onClose` was called. The backdrop is `aria-hidden="true"` (decorative), so the selector is by CSS class.
5. `"renders a fallback error message when the ApiError has no detail"` — rejects with `{ status: 500 }` (no `detail`); asserts the `role="alert"` element contains the store's fallback string `"Failed to load the explain narrative."` (confirmed at `auditReadinessStore.ts:209`: `apiErr.detail ?? "Failed to load the explain narrative."`).

**Task 7 — 2 tests added:**

6. `"panel remains present when every tab is activated in turn"` — iterates `getAllByRole("tab")` dynamically and clicks each; asserts `getByLabelText("Audit readiness")` is still present after each click. Tabs confirmed as Spec/Graph/YAML/Runs (`InspectorPanel.tsx:32–37`); dynamic iteration makes the test resilient to future tab-list changes.
7. `"renders without crashing when activeSessionId and compositionState are null"` — sets both to `null` in the store; asserts no throw and tablist still renders. Protects against the null-compositionState path the original tests did not exercise.

Risks table updated with a W6 row. Implementation update: Step 3 of Task 5 receives the Escape handler addition. No other implementation blocks modified.

### 2026-05-17 — Final polish pass: Phase 2C commit-message labels + Test 5.B focus-dispatch fix

Two convergent minor findings resolved:

1. **Phase 2C commit-message labels (Finding 1).** Tasks 5, 6, and 7 Step 5/6 commit messages incorrectly labelled the commits `(Phase 2B.5)`, `(Phase 2B.6)`, and `(Phase 2B.7)`. Corrected to `(Phase 2C.5)`, `(Phase 2C.6)`, and `(Phase 2C.7)`. The existing commit `084b8c34b "feat(web/frontend): wire clearSession on session removal (Phase 2B.5)"` on `RC5.2` uses the 2B.5 label for a real Phase 2B.5 commit; the old labels would have collided with that history and confused `git log --grep "Phase 2C"` searches. Historical references in prose that correctly describe 14b's foundation work were left untouched.

2. **Test 5.B focus-dispatch reliability (Finding 2).** `"fires onClose when Escape is pressed"` called `user.keyboard("{Escape}")` without first placing focus inside the drawer. In jsdom, `userEvent.keyboard` dispatches to `document.activeElement`; after `render()` that is `document.body`, so the keydown may not bubble to the drawer's React `onKeyDown` handler and the test could pass vacuously (false-positive on a broken Escape contract). Fixed by adding `screen.getByRole("button", { name: /Close/i }).focus()` before the keyboard call, with a comment explaining the jsdom dispatch semantics.

## Memory references

- `project_composer_personas` — Linda-vocabulary row labels.
- `project_staging_deployment` — staging is a source-checkout systemd/Caddy deploy; `npm run build` + `systemctl restart elspeth-web.service`.
- `feedback_no_calendar_shipping_commitments` — no SLAs in this plan.
- `feedback_default_is_fix_not_ticket` — Step 5 of Task 8 is the place this rule bites: lint errors aren't suppressed, they're fixed; if the dead-code analysis surfaces a regression, fix it in-task rather than filing.
- `feedback_repeated_out_of_scope_is_underscoping.md` — Task 8 scope was scope-shrunk in the 2026-05-16 revision (convergence C3), then expanded back in the 2026-05-17 revision when the B1 analysis confirmed the button was the only `handleValidate` caller and deferral would silently orphan the side effects. Both the scope-shrink and the re-expansion were correctness-driven; neither was underscoping.

# Phase 5b — Surface the LLM's interpretation (frontend)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans`. Implement task-by-task,
> checkbox-tracked. The overview, B2 verdict, scope, trust-tier check, and
> risks are in [18-phase-5b-surface-llm-interpretation.md](18-phase-5b-surface-llm-interpretation.md);
> read that first. The backend half is in
> [18a-phase-5b-backend.md](18a-phase-5b-backend.md) and **MUST land
> before this plan starts** — this plan consumes the wire shapes,
> routes, and tool surface introduced in 18a.

**Status header — frontend consumes 18a's wire surface.** Phase 5b
frontend ships AFTER Phase 5b backend. Pre-flight check before starting
any task here: verify 18a's contract module is importable and the audit
spot-check passes:

```bash
# Function-existence check (more reliable than commit-message grep):
python -c "from elspeth.web.sessions.service import SessionServiceImpl; \
  assert hasattr(SessionServiceImpl, 'create_pending_interpretation_event'), \
  'Backend 18a Task 4 not landed yet'"

# Integration gate:
.venv/bin/python -m pytest \
  tests/integration/composer/test_interpretation_audit_spotcheck.py -x
```

If either check fails, return to 18a.

---

## Tech Stack (frontend slice)

- React + Zustand + Vitest + testing-library, matching the rest of
  `web/frontend/src`.
- TypeScript types regenerated to mirror the new pydantic wire schemas
  introduced in 18a-Task-3.
- No new third-party JS dependencies.
- The composer-skill markdown (`pipeline_composer.md`) is touched only
  by 18a-Task-8.

## File structure (frontend changes)

```text
src/elspeth/web/frontend/src/
  types/
    api.ts                                                      MODIFY    (Task 1 — wire types)
    guided.ts                                                   MODIFY    (Task 1 — new TurnType value)
    interpretation.ts                                           CREATE    (Task 1 — domain types)
  api/
    client.ts                                                   MODIFY    (Task 2 — API methods)
    client.test.ts                                              MODIFY    (Task 2)
  stores/
    interpretationStore.ts                                      CREATE    (Task 3)
    interpretationStore.test.ts                                 CREATE    (Task 3)
    sessionStore.ts                                             MODIFY    (Task 3 — wire pending events into session reload)
  components/chat/
    InterpretationReviewInlineMessage.tsx                       CREATE    (Task 5 — freeform mode variant)
    InterpretationReviewInlineMessage.test.tsx                  CREATE    (Task 5)
    ChatPanel.tsx                                               MODIFY    (Tasks 4 + 5 — dispatch)
    ChatPanel.test.tsx                                          MODIFY    (Tasks 4 + 5)
    guided/
      InterpretationReviewTurn.tsx                              CREATE    (Task 4 — guided mode widget)
      InterpretationReviewTurn.test.tsx                         CREATE    (Task 4)
      GuidedTurn.tsx                                            MODIFY    (Task 4 — dispatch to InterpretationReviewTurn)
      GuidedTurn.test.tsx                                       MODIFY    (Task 4)
  components/audit/
    AuditReadinessPanel.tsx                                     MODIFY    (Task 7 — interpretations row)
    AuditReadinessPanel.test.tsx                                MODIFY    (Task 7)
  test/
    interpretationIntegration.test.tsx                          CREATE    (Task 6)

docs/composer/ux-redesign-2026-05/
  18b-phase-5b-frontend.md                                      THIS FILE
```

No backend Python files change in this plan (18a owns those).

---

## Task 1 — TypeScript wire types

**Goal.** Mirror 18a's pydantic schemas in TypeScript. Source-of-truth
pairing follows the existing pattern at
`src/elspeth/web/frontend/src/types/guided.ts` lines 1-12 (the file
references the Python source-of-truth modules in a header comment).

**Files:**

- Modify: `src/elspeth/web/frontend/src/types/api.ts` — add
  `InterpretationEventResponse`, `InterpretationResolveRequest`,
  `InterpretationResolveResponse`, `InterpretationOptOutResponse`,
  `ListInterpretationEventsResponse`.
- Modify: `src/elspeth/web/frontend/src/types/guided.ts` — extend
  `TurnType` union with the new value `"interpretation_review"`.
- Create: `src/elspeth/web/frontend/src/types/interpretation.ts` — the
  domain-shaped TS types referenced by the store and components, plus
  the closed-enum `InterpretationChoice` (mirror of 18a's
  `InterpretationChoice` StrEnum).

### Types

```typescript
// types/interpretation.ts
//
// Source of truth (pydantic):
//   src/elspeth/web/sessions/schemas.py — InterpretationEvent* models
// Source of truth (contracts):
//   src/elspeth/contracts/composer_interpretation.py

export type InterpretationChoice =
  | "pending"
  | "accepted_as_drafted"
  | "amended"
  | "opted_out";

export interface InterpretationEvent {
  id: string;
  session_id: string;
  composition_state_id: string;
  affected_node_id: string;
  tool_call_id: string;
  user_term: string;
  llm_draft: string;
  accepted_value: string | null;
  choice: InterpretationChoice;
  created_at: string;
  resolved_at: string | null;
  actor: string;
}
```

### Step 1 — RED

Tests in `types/interpretation.test.ts` (or extending `guided.test.ts`):

1. `InterpretationChoice` union has exactly 4 values.
2. `InterpretationEvent` has all 13 fields (six required-by-spec +
   seven operational).
3. `TurnType` union now has 7 values (was 6, adds
   `"interpretation_review"`).

### Step 2 — GREEN

Apply the type additions.

### Step 3 — Commit

`frontend(types): wire types for interpretation events + new TurnType`.

---

## Task 2 — API client methods

**Goal.** Three new API client methods.

**Files:**

- Modify: `src/elspeth/web/frontend/src/api/client.ts`.
- Modify: `src/elspeth/web/frontend/src/api/client.test.ts`.

### Methods

```typescript
// In api/client.ts:

export async function listInterpretationEvents(
  sessionId: string,
  status: "pending" | "all" = "all",
): Promise<InterpretationEvent[]> { ... }

export async function resolveInterpretation(
  sessionId: string,
  eventId: string,
  body: {
    choice: "accepted_as_drafted" | "amended";
    amended_value?: string;
  },
): Promise<{ event: InterpretationEvent; new_state: CompositionState }> { ... }

export async function optOutOfInterpretations(
  sessionId: string,
): Promise<{ session_id: string; interpretation_review_disabled: boolean; opted_out_at: string }> { ... }
```

### Test shape

`client.test.ts` adds:

1. `listInterpretationEvents` issues `GET /api/sessions/{id}/interpretations`
   with the correct query string.
2. `resolveInterpretation` issues `POST /api/sessions/{id}/interpretations/{event_id}/resolve`
   with the JSON body shape and parses the response into the typed
   object.
3. `optOutOfInterpretations` issues `POST /api/sessions/{id}/interpretations/opt_out`
   with empty body.
4. 422 responses from the backend (e.g., missing amended_value)
   surface as a typed validation error (matches existing client error
   conventions).
5. 409/404 responses surface as named error types matching existing
   conventions.

### Step 1-3 — RED → GREEN → commit

`frontend(api): client methods for interpretation events`.

---

## Task 3 — Zustand store for interpretation events

**Goal.** A small store that holds pending interpretation events per
session, with mutators that call the API methods and update local
state on success. Also wire session reload to fetch pending events.

**Files:**

- Create: `src/elspeth/web/frontend/src/stores/interpretationStore.ts`.
- Create: `src/elspeth/web/frontend/src/stores/interpretationStore.test.ts`.
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` — on
  session load / session reload, call
  `interpretationStore.refreshPending(sessionId)`.

### Store shape

```typescript
interface InterpretationStoreState {
  // Per-session pending events. Keyed by session_id, then event_id.
  pendingBySession: Record<string, Record<string, InterpretationEvent>>;
  // Per-session count of resolved events (for the audit-readiness
  // panel summary; we only keep counts in memory, not the full
  // resolved-event list).
  resolvedCountBySession: Record<string, {
    accepted_as_drafted: number;
    amended: number;
    opted_out: number;
  }>;
  // Per-session opt-out flag (mirror of sessions.interpretation_review_disabled).
  optedOutBySession: Record<string, boolean>;

  refreshPending: (sessionId: string) => Promise<void>;
  refreshAll: (sessionId: string) => Promise<void>; // pending + resolved counts
  resolveEvent: (
    sessionId: string,
    eventId: string,
    body: { choice: "accepted_as_drafted" | "amended"; amended_value?: string },
  ) => Promise<{ new_state: CompositionState }>;
  optOut: (sessionId: string) => Promise<void>;
}
```

The store wraps the API client. On `resolveEvent` success, it removes
the event from `pendingBySession[sessionId]` and increments the
appropriate counter in `resolvedCountBySession[sessionId]`. On
`optOut` success, it flips `optedOutBySession[sessionId]` to true AND
empties `pendingBySession[sessionId]` (the backend may still leave
pending events in the DB but the frontend UX is "interpretations
silenced from now on for this session").

**Tier-discipline note:** the store is treating wire-side
`InterpretationEvent` as "validated-at-the-boundary Tier-2 pipeline
data" — the API client has already parsed the JSON, run Pydantic-mirror
checks (Task 2), and produced a typed object. Inside the store, direct
field access is the right discipline. No `.optional_field()` patterns.

**Telemetry:** NONE — the frontend store mirrors backend audit truth;
it emits no operational signals of its own. The backend Landscape is
the canonical record; client-side telemetry would be redundant duplication.

### Test shape

`interpretationStore.test.ts`:

1. `refreshPending` calls the API method, fills `pendingBySession`.
2. `refreshAll` calls the API method with `status='all'`, fills both
   `pendingBySession` and `resolvedCountBySession`.
3. `resolveEvent` calls the API method, removes the event from
   pending, increments the counter for the chosen kind, returns
   `{ new_state }`.
4. `optOut` calls the API method, flips the opt-out flag, clears
   pending events for the session.
5. Errors from the API surface as rejected promises and DO NOT
   mutate the store (atomicity).

`sessionStore.test.ts` updates:

6. On session load (`loadSession`), the store now also calls
   `interpretationStore.refreshAll(sessionId)`.
7. On session change, the previous session's pending events are NOT
   reset (the store keys by session_id, so they remain accessible if
   the user navigates back).

### Step 1-3 — RED → GREEN → commit

`frontend(stores): interpretation store + sessionStore reload wiring`.

---

## Task 4 — `InterpretationReviewTurn` (guided mode)

**Goal.** The guided-mode turn widget that renders the "Use mine /
Change it" pair. Pattern follows the existing
`InspectAndConfirmTurn.tsx` shape.

**Widget placement.** `InterpretationReviewTurn` is guided-mode-only:
it is dispatched exclusively from `GuidedTurn.tsx` and has no freeform
consumer (freeform uses the separate `InterpretationReviewInlineMessage`
in Task 5). Following the convention established by `InspectAndConfirmTurn`,
`SingleSelectTurn`, and all other guided-only widgets, it lives at
`components/chat/guided/`. `GuidedTurn.tsx` imports it with the same
relative `"./InterpretationReviewTurn"` pattern used for every sibling
widget.

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx` —
  dispatch the new `TurnType="interpretation_review"` to the new widget.
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.test.tsx`.

### Component contract

```typescript
interface InterpretationReviewTurnProps {
  event: InterpretationEvent;       // The pending event to review
  sessionId: string;
  onResolved?: () => void;          // Optional callback after successful resolve
}
```

### Rendered UI

The widget renders a card with:

- Header: "Before we run: when you said *<user_term>*, I read that as
  roughly *<llm_draft>*."
- Two primary buttons:
  - `Use my interpretation` — calls
    `resolveEvent(sessionId, event.id, { choice: 'accepted_as_drafted' })`.
  - `Change it: I meant…` — toggles to inline-edit mode showing a
    textarea pre-filled with `event.llm_draft`, a Submit button that
    calls `resolveEvent(sessionId, event.id, { choice: 'amended',
    amended_value: <textarea value> })`, and a Cancel button that
    reverts to the two-button view.
- A small footer link: "Don't ask me again this session" — opens a
  confirm modal; on confirm, calls `optOut(sessionId)`.

### Accessibility

- The card is `role="region"` with `aria-labelledby` referencing the
  header.
- Buttons have descriptive `aria-label`s ("Accept the LLM's
  interpretation of cool", "Edit the interpretation of cool").
- The textarea is properly labelled.
- Focus moves to the textarea when the "Change it" path is selected
  (per the project memory `feedback_focus_on_step_advance` and the
  accessibility audit per design doc 06 §"Mode discrepancy").
- The "Don't ask me again" link is visually de-emphasised but
  keyboard-accessible.

### Test shape

`InterpretationReviewTurn.test.tsx`:

1. Renders header text containing `user_term` and `llm_draft`.
2. "Use my interpretation" button calls `resolveEvent` with
   `choice: 'accepted_as_drafted'`.
3. "Change it" button reveals a textarea pre-filled with `llm_draft`,
   focus moves to the textarea.
4. Submitting the amended text calls `resolveEvent` with
   `choice: 'amended', amended_value: <text>`.
5. Cancel reverts to the two-button view.
6. After successful resolve, `onResolved` fires; the widget collapses
   into a "Accepted ✓" state (the parent re-renders without the widget
   because the event is no longer pending — but the widget should
   handle the optimistic UI gracefully).
7. Submitting an empty amendment is disabled (Submit button greyed).
8. Submitting an amendment that exceeds the 8KB cap shows a client-side
   validation error before issuing the request.
9. "Don't ask me again this session" opens a confirm modal; on confirm,
   calls `optOut`.
10. While the resolve API request is in flight, both buttons are
    disabled and a spinner shows.
11. If the resolve API returns 409 (TOCTOU — already resolved),
    surface "This interpretation has already been resolved" and
    refresh the store.
12. If the resolve API returns 422 (Pydantic validation),
    surface the validation error.

`GuidedTurn.test.tsx` updates:

13. A turn with `type: "interpretation_review"` dispatches to
    `InterpretationReviewTurn`.

### Step 1-3 — RED → GREEN → commit

`frontend(chat): InterpretationReviewTurn guided-mode widget`.

---

## Task 5 — `InterpretationReviewInlineMessage` (freeform mode)

**Goal.** Per design-spec §Risks: "the surfacing UI may differ (turn
widget in guided, inline message in freeform)." Freeform mode renders
the review affordance as an inline message in the chat history, not a
distinct turn widget.

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/InterpretationReviewInlineMessage.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/InterpretationReviewInlineMessage.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` —
  in freeform mode, render any pending interpretation events from the
  store as inline messages between the assistant's message and the
  chat input. The position is "above the chat input, below the most
  recent assistant message," visually styled like an assistant message
  with a coloured side-bar to signal "action required." The dispatch
  predicate for the interpretation inline message is: the store has
  at least one pending event for the session AND the session is NOT
  in guided mode. This is distinct from the `inline_blob` summary
  banner (which fires for any `inline_blob` proposal regardless of
  interpretation state); the two can coexist. A proposal whose summary
  does NOT match interpretation-event context (e.g., "Created a 5-row
  source from your input") renders as the standard `InlineSourceCreatedTurn`
  banner, not as an `InterpretationReviewInlineMessage`.

### Component contract

```typescript
interface InterpretationReviewInlineMessageProps {
  event: InterpretationEvent;
  sessionId: string;
  onResolved?: () => void;
}
```

### Rendered UI (freeform variant)

- Speech-bubble-styled like an assistant message.
- The body text reads (per design-spec §"Canonical example"):
  > When you said *"<user_term>"*, I read that as roughly
  > "*<llm_draft>*". Want to adjust the definition, or use mine?
- Same two affordances as guided mode (Use mine / Change it) but
  rendered as inline buttons within the message body, not as a
  separate card.
- Same "Don't ask me again this session" footer link.

The behaviour, accessibility requirements, error handling, and store
interaction are IDENTICAL to `InterpretationReviewTurn`. The two
components share a small `useInterpretationResolver()` custom hook
extracted in Task 4 that wraps `interpretationStore.resolveEvent` with
the spinner/disabled state machinery.

### Test shape

`InterpretationReviewInlineMessage.test.tsx`:

1-12 mirror the guided-mode widget tests with the freeform-mode
rendering.

`ChatPanel.test.tsx` updates:

13. In freeform mode with a pending interpretation event in the
    store, the inline message renders above the chat input.
14. In guided mode, the inline message does NOT render (the guided
    turn widget handles it).
15. Two pending events render two inline messages, in
    `created_at`-ascending order.
16. After opt-out, no inline messages render even if pending events
    remain in the store (the opt-out clears them locally; backend
    still has the rows for audit).
17. **Negative-case (routing predicate):** An `inline_blob` proposal
    with a summary like `"Created a 5-row source from your input"` —
    i.e., a summary containing neither `"I read"` nor `"interpreted as"` —
    does NOT render `InterpretationReviewInlineMessage`. The standard
    `InlineSourceCreatedTurn` renders instead. Assert:
    `expect(screen.queryByTestId('interpretation-review-inline-message'))
    .not.toBeInTheDocument()` and
    `expect(screen.getByTestId('inline-source-created-turn')).toBeInTheDocument()`.
    This pins the negative branch of the dispatch predicate and guards
    against future changes that accidentally widen the interpretation-review
    surface.

### Step 1-3 — RED → GREEN → commit

`frontend(chat): InterpretationReviewInlineMessage freeform-mode variant`.

---

## Task 6 — Vitest integration test (both modes)

**Goal.** End-to-end test that mocks the backend and asserts both UIs
drive the same backend contract.

**File:**

- Create: `src/elspeth/web/frontend/src/test/interpretationIntegration.test.tsx`.

### Test shape

**Part A — 5a-then-5b combined sequence (the demo canonical prompt flow).**

The canonical hero prompt drives 5a (inline blob creation) immediately
before 5b (interpretation review). Part A pins the store-hydration
ordering that would silently break if `interpretationStore.refreshPending`
fires before `sessionStore` has hydrated the composition state.

1. Set up a mock API:
   - POST `set_pipeline` (the 5a `inline_blob` tool call response) →
     returns a `CompositionState` that includes the LLM transform node
     with `node_id='llm_rate_coolness'`.
   - `request_interpretation_review` tool-call response → returns a
     pending `InterpretationEvent` with
     `{ user_term: 'cool', llm_draft: 'modern design + clear purpose + interactivity',
       affected_node_id: 'llm_rate_coolness', choice: 'pending' }`.
   - GET `/interpretations?status=all` → same pending event as above
     (called by `interpretationStore.refreshPending` on session reload).
2. Render the composer in guided mode starting from the
   `InlineSourceCreatedTurn` (5a's turn). This corresponds to the
   composer session state just after Phase 5a has placed the inline
   source.
3. Assert `compositionState.nodes` includes `node_id='llm_rate_coolness'`
   BEFORE the interpretation turn renders. This is the hydration-ordering
   assertion: if the store loads in the wrong order, this node will not
   be present when `InterpretationReviewTurn` mounts.
4. Simulate advancing past `InlineSourceCreatedTurn` (user confirms the
   source). The guided turn dispatcher moves to the next turn.
5. Assert `InterpretationReviewTurn` mounts. At mount time,
   `compositionState.nodes` must still include `node_id='llm_rate_coolness'`.
   Assert this explicitly: `expect(compositionState.nodes).toContainEqual(
   expect.objectContaining({ node_id: 'llm_rate_coolness' }))`.
6. Assert that `affected_node_id` on the rendered event matches a node
   that exists in `compositionState.nodes` (no dangling reference).

**Part B — guided mode resolve flow.**

7. From the state reached in step 5, click "Use my interpretation".
8. Assert the POST body sent to
   `POST /interpretations/{id}/resolve` was
   `{ choice: 'accepted_as_drafted' }`.
9. Assert the widget collapses (the pending event is removed from
   `interpretationStore.pendingBySession[sessionId]`).
10. Assert the new composition state returned by the resolve endpoint is
    propagated into `sessionStore` (the session's `composition_state`
    reflects the post-resolve value).

**Part C — freeform mode resolve flow.**

11. Reset the harness; re-render the same session in freeform mode with
    the mock API returning the same pending event.
12. Assert `InterpretationReviewInlineMessage` is visible above the chat
    input.
13. Click "Change it"; type `'highly engaging and accessible'`; click
    Submit.
14. Assert the POST body was
    `{ choice: 'amended', amended_value: 'highly engaging and accessible' }`.
15. Assert the inline message disappears (event removed from
    `pendingBySession`).

### Step 1-3 — RED → GREEN → commit

`frontend(integration): interpretation review covers 5a+5b flow + guided + freeform`.

---

## Task 7 — Audit-readiness panel row

**Goal.** Add a new row to the audit-readiness panel:
"LLM interpretations: <pending count> pending • <accepted count>
accepted • <amended count> amended • <opted-out count> opted-out
(session-scoped)".

Phase 2C has shipped (`project_phase2c_implementation_complete`);
`AuditReadinessPanel.tsx` exists. This task is unconditional.

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx`.

### Row content

The row is shown only when the composition state has at least one LLM
transform OR the session has any interpretation events (pending or
resolved). In all other cases the row is omitted.

**Backend status → frontend rendering mapping.** 18a-Task-10 returns
one of three `ReadinessStatus` values for the `llm_interpretations`
row. The frontend maps each to the row text and glyph colour:

| Backend status | Row text | Visual |
|---|---|---|
| `not_applicable` | Row hidden (or "LLM interpretations: not yet surfaced" if LLM transform with unresolved placeholder exists) | neutral / grey |
| `warning` | "LLM interpretations: ⚠ <P> pending review (<N - P> resolved)" | yellow |
| `ok` | "LLM interpretations: ✓ all <N> resolved" | green |

When the session opt-out flag is set and the backend returns
`not_applicable` with at least one event in the store, the frontend
overrides to: "LLM interpretations: ◎ opted out for this session
(<N> drafted, not reviewed)" — neutral glyph. The store's
`optedOutBySession[sessionId]` flag is the local signal for this.

The row component update must handle the three status values above
as an exhaustive match — no fallthrough to a stale "hardcoded
not_applicable" branch.

### Test shape

`AuditReadinessPanel.test.tsx` adds:

1. Row hidden when no LLM transform and no events.
2. Row "all resolved" when 2 resolved, 0 pending.
3. Row "pending review" when 1 pending, 1 resolved.
4. Row "opted out" when opt-out flag is set, 0 pending, 2 drafted.
5. Row "not yet surfaced" when there's an LLM transform with an
   unresolved placeholder and no events.

### Step 1-3 — RED → GREEN → commit

`frontend(audit-panel): interpretations row`.

---

## Task 8 — Empty-state copy + chat-input placeholder

**Goal.** Two small copy edits to bring the feature's voice into the
chat:

1. When a pending interpretation event exists, the chat-input
   placeholder briefly cues: "Reviewing your interpretation of
   *<user_term>* above — pick Use mine or Change it to continue."
2. After resolution, a one-line confirmation in the assistant's voice
   appears in chat: "Got it — using your interpretation of *<user_term>*."

These are pure UI nudges; no new logic. They DO NOT modify the audit
trail; the audit row is the canonical record, and the chat copy is a
human-readable echo.

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
  — append the confirmation line on resolve success.

### Test shape

1. `ChatInput` placeholder cue fires when the store has a pending
   event for the active session.
2. The confirmation line appears in chat history after a successful
   resolve.
3. Neither copy edit changes any other test's snapshot or behaviour.

### Step 1-3 — RED → GREEN → commit

`frontend(chat): copy nudges for interpretation review`.

---

## Task 9 — Manual smoke test on staging

**Goal.** End-to-end validation per the verification approach in the
overview §"Verification approach."

**Pre-conditions:** `elspeth-web.service` restarted (after 18a-Task-8
skill edits). Frontend `npm run build` deployed to staging
(`elspeth.foundryside.dev` per the project memory
`project_staging_deployment`).

### Smoke checklist

- [ ] Type the canonical hero prompt:
      "create a list of 5 government web pages and use an LLM to rate
      how cool they are".
- [ ] Confirm Phase 5a's inline source flow fires first (5 URLs
      surfaced).
- [ ] Confirm `InterpretationReviewTurn` fires for "cool" in guided
      mode.
- [ ] Click "Change it"; amend; submit. Confirm:
      - The widget collapses.
      - The audit-readiness panel updates to "1 amended".
      - The new composition state's LLM transform prompt template
        contains the user-amended interpretation (open the YAML
        preview or graph inspector).
- [ ] Run the pipeline. Confirm:
      - The runtime LLM-transform `calls` row in the Landscape
        records the resolved prompt template (NOT the placeholder).
      - `explain(run_id, token_id)` walks back to the
        `interpretation_events` row.
- [ ] Open a SECOND session. Type the same prompt. Click "Don't ask
      me again this session." Confirm:
      - The opt-out audit row is in `interpretation_events` with
        `choice='opted_out'` and
        `interpretation_source='auto_interpreted_opt_out'`.
      - The pipeline compose completes without surfacing "cool" again.
      - The audit-readiness panel shows "opted out for this session".
- [ ] Switch the SECOND session to freeform mode (if applicable);
      verify a re-entry surfaces the inline message variant.

If any step fails: STOP, file a Filigree bug, fix in-session per the
project memory `feedback_default_is_fix_not_ticket` (do NOT close the
phase with known-broken work).

---

## Frontend completion criteria

Phase 5b frontend is complete when:

- [ ] Task 1 (types) green, committed.
- [ ] Task 2 (API client) green, committed.
- [ ] Task 3 (store + sessionStore wiring) green, committed.
- [ ] Task 4 (`InterpretationReviewTurn`) green, committed.
- [ ] Task 5 (`InterpretationReviewInlineMessage`) green, committed.
- [ ] Task 6 (Vitest integration test) green, committed.
- [ ] Task 7 (audit-panel row) green, committed.
- [ ] Task 8 (copy nudges) green, committed.
- [ ] Task 9 (manual smoke) passed end-to-end on staging.
- [ ] Vitest `npm run test` passes from `src/elspeth/web/frontend/`.
- [ ] Frontend production build (`npm run build`) succeeds without
      warnings related to new code.
- [ ] No regressions to existing turn widgets (`InspectAndConfirmTurn`,
      `SingleSelectTurn`, etc.) — visual smoke and existing Vitest
      pass.

Then Phase 5b as a whole is done. Update the umbrella PR (or close
the phase under the existing `feat/composer-guided-mode` branch per
project conventions). The phase-implementation-complete memory entry
follows the existing pattern (`project_phase5_implementation_complete`
etc.).

---

## Review history

| Date | Reviewer | Verdict | Finding IDs | Notes |
|------|----------|---------|-------------|-------|
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, I1, I2 | Applied in this revision. B1 (BLOCKER): The finding alleged misplaced shareable-link test stubs at `client.test.ts` lines referencing `GET /api/sessions/:id/shareable-link` and `GET /api/sessions/shared/:token`. Review of this file confirms no such stubs exist — the finding's premise did not hold. The actual test body completeness defect addressed was the fragile preflight check; see I1. I1: replaced `git log --oneline | grep 'interpretation'` preflight grep with a Python import check and a direct pytest invocation of the audit spot-check integration test. I2: added backend-status → frontend rendering mapping table to Task 7, specifying how `not_applicable`/`warning`/`ok` from 18a-Task-10 map to row text and glyph colour; added opt-out override path; required exhaustive status match. |

---

## Frontend open questions

1. **Should the inline-message variant collapse-into-history after
   resolve, or vanish entirely?** Design spec is silent. MVP: vanish
   entirely (keeps the chat history clean; the audit-panel reflects
   the resolved count). Revisit if the manual smoke suggests users
   want to see "I accepted/amended X" inline. File as a Phase 8 polish
   observation if so.
2. **Multi-pending UI in guided mode.** If the LLM surfaces two
   interpretations back-to-back, the guided mode's "one turn at a
   time" model means turn-2 surfaces only after turn-1 resolves. The
   freeform mode's inline-message variant can show both at once.
   Audit-panel shows the count regardless. The asymmetry is acceptable
   per design-spec §Risks. The Vitest integration test (Task 6) does
   not cover multi-pending in guided mode; if usage shows the LLM
   commonly surfaces 2+ interpretations in a single composition,
   re-examine.
3. **Server-Sent-Events / WebSocket for pending-event freshness.**
   Currently the store refreshes on session-load and on explicit
   mutation. If a second tab on the same session resolves an event,
   tab 1 won't know until reload. Phase 5b accepts this as a known
   limitation; cross-tab real-time freshness is a Phase 11 concern.

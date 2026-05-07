# RC4.2 UX Remediation — Chat Polish Implementation Plan

Date: 2026-03-30
Status: Draft
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This plan covers the small, high-leverage chat-surface fixes that share the
same frontend state and rendering paths.

Included requirements:

- `REQ-UX-01` — fix send-state UX
- `REQ-UX-02` — scroll-to-bottom button
- `REQ-UX-04` — copy icons on chat bubbles

Primary files:

- `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`

This plan is intentionally frontend-only. No REST/API changes are required for
the initial pass.

Prerequisite:

- install a frontend test framework before implementation begins
  (`vitest` + `@testing-library/react` or equivalent)

---

## 2. Current State

The current chat surface already has the three core building blocks we need:

- optimistic user-message insertion in `sessionStore.sendMessage()`
- a global `isComposing` flag that drives `ComposingIndicator`
- scroll tracking in `ChatPanel.tsx` via `isUserScrolledUp`

The current rough edges are:

1. User bubbles can show `Sending...` while the assistant typing indicator is
   also visible.
2. Sending a message does not explicitly force the viewport back to the live
   bottom of the chat.
3. When the user scrolls up, there is no quick way to jump back down.
4. Message bubbles do not expose a copy affordance.

---

## 3. Architecture Decisions

### AD-1: Keep the send-state fix frontend-only

The current `POST /messages` endpoint is synchronous, so there is no true
server-acknowledged intermediate state between "request sent" and "assistant
response available." Because of that, the practical fix is:

- keep `local_status: "pending"` as the failure/retry marker
- suppress the visible `Sending...` label while `isComposing` is true
- preserve the failed state for retry if the request errors

This resolves the contradictory UX without introducing SSE or changing the API
contract.

### AD-2: Explicit send actions force-scroll to the live bottom

An explicit send action means the user has rejoined the live conversation. This
is distinct from passively receiving new messages while reading older history.
On send, the viewport should snap to the newest message immediately and
auto-scroll should resume.

### AD-3: Scroll-to-bottom is local UI state, not global session state

The jump-to-bottom affordance depends on the current scroll container position,
not on persisted message/session state. Keep it local to `ChatPanel.tsx`.

### AD-4: Copy affordances should copy user-visible message text only

The copy action should exclude tool-call disclosures and any future debug-only
sections. It should copy the visible message content exactly as the user reads
it.

---

## 4. Implementation Plan

### Phase 1: Send-State Cleanup

#### `sessionStore.ts`

Keep the current optimistic insert pattern, but clarify the visible state model:

- On send:
  - append optimistic user message with `local_status: "pending"`
  - set `isComposing: true`
  - clear any existing error
- On success:
  - clear the optimistic message's `local_status`
  - append the assistant response
  - update composition state
  - set `isComposing: false`
- On failure:
  - keep the optimistic message in place
  - change `local_status` to `"failed"`
  - set `isComposing: false`
  - set user-facing error text

No new backend state is needed.

#### `MessageBubble.tsx`

Change pending-label rendering so:

- `Sending...` is shown only when:
  - the message is a user message
  - `local_status === "pending"`
  - the store is not currently in the composing state for this send flow
- failed messages still show the inline retry affordance

This yields the intended UX:

- user sends message
- user sees their message immediately
- assistant typing indicator appears
- bubble does not falsely present as "unsent"

### Phase 2: Scroll To Live Bottom

#### `ChatPanel.tsx`

Add explicit send-driven scroll behavior:

- When `handleSend()` fires:
  - set `isUserScrolledUp.current = false`
  - scroll the container or end sentinel into view immediately
- Keep existing auto-scroll-on-new-message behavior for the normal live-chat
  path.

This should be immediate, not conditional on the next render pass alone.

### Phase 3: Scroll-To-Bottom Button

#### `ChatPanel.tsx`

Add a floating jump-to-bottom affordance that appears only when the user has
scrolled away from the bottom threshold.

Required behavior:

- visible only when `isUserScrolledUp.current` is true
- positioned at the bottom-center or bottom-right of the message pane
- clicking it:
  - scrolls to bottom
  - resets the scrolled-up state
  - hides the button

The initial pass does not need unread counts.

### Phase 4: Copy Icons On Bubbles

#### `MessageBubble.tsx`

Add a copy affordance for user and assistant bubbles:

- icon/button attached to each non-system bubble
- copies `message.content` only
- reuses the existing lightweight "Copied!" feedback pattern already used
  elsewhere in the frontend

Mobile/touch should still be able to access the button even if hover styling is
used on desktop.

---

## 5. File-Level Work

### `src/elspeth/web/frontend/src/stores/sessionStore.ts`

Expected changes:

- keep optimistic insertion logic
- preserve retry semantics
- avoid any new backend/API dependency

Potential additions:

- helper to mark an optimistic message pending/failed/cleared
- clearer local naming for message lifecycle transitions

### `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`

Expected changes:

- force-scroll on send
- floating jump-to-bottom button
- local visibility state for the button if refs alone become awkward

### `src/elspeth/web/frontend/src/components/chat/MessageBubble.tsx`

Expected changes:

- conditionally suppress `Sending...` while composing
- add copy button
- keep retry affordance for failed user messages

---

## 6. Testing Plan

### Manual checks

1. Send a message while already at bottom.
   - Message appears immediately.
   - View stays pinned to the bottom.
   - `Sending...` does not linger while the assistant typing indicator is
     visible.

2. Force a failed send.
   - User message remains visible.
   - Failed inline state appears.
   - Retry button works.

3. Scroll far up, then receive or render more content.
   - Jump-to-bottom button appears.
   - Clicking it returns to the live bottom and hides the button.

4. Copy user and assistant messages.
   - Clipboard gets message text only.
   - Tool disclosure content is excluded.

### Unit/component tests

Test prerequisite:

- the repo does not currently have a frontend unit/component test runner
- before implementing this subplan, add and standardize:
  - `vitest`
  - `@testing-library/react`
  - a Zustand store test pattern for isolated state-machine tests

Add focused tests around:

- send-state transitions in `sessionStore`
- pending-label suppression in `MessageBubble`
- jump-to-bottom button visibility toggling
- copy action behavior

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Pending-label suppression hides too much state | Low | Keep failed/retry state explicit; rely on `ComposingIndicator` for active work |
| Send-scroll fights user-controlled scroll | Low | Only force-scroll on explicit send action |
| Copy button hurts mobile layout | Low | Use compact placement and verify narrow widths |
| Jump-to-bottom button obscures content | Low | Use conservative positioning and spacing from input area |

---

## 8. Sequencing

Recommended order inside this subplan:

1. Send-state cleanup
2. Force-scroll on send
3. Jump-to-bottom button
4. Copy icons

This order delivers the highest-signal UX fixes first and keeps regressions easy
to isolate.

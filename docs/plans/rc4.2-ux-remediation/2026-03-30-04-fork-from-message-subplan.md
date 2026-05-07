# RC4.2 UX Remediation — Fork From Message Implementation Plan

Date: 2026-03-30
Status: Draft
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This subplan covers editing a prior user message and creating a new session fork
from that point, with preserved provenance and inherited composition state.

Included requirement:

- `REQ-UX-03` — chat bubble edit and fork from here

Primary surfaces:

- session/message persistence
- composition-state provenance
- fork API
- chat/session frontend UX

---

## 2. Goals

- Let users branch from an earlier point without mutating history.
- Preserve the original session as a complete, auditable timeline.
- Make the forked session clearly identifiable in the UI.
- Ensure forked state reflects the selected historical point, not the latest
  session state.

Non-goals for this phase:

- destructive rollback of the original session
- branching within a single session timeline
- trusting client-provided state lineage without server verification

---

## 3. Architecture Decisions

- Add explicit state provenance to messages.
- Treat forking as session creation with inherited context, not rollback.
- Keep provenance visible in the forked session header or metadata area.

### AD-1: Forking uses explicit pre-send and post-turn state provenance

Add `composition_state_id` to `chat_messages` with role-dependent semantics:

- user message: state visible before the message was sent
- assistant message: state after the composition loop for that turn

This must be treated as correctness-critical, not best-effort metadata.

### AD-2: The client may propose the starting state, but the server remains authoritative

To avoid races between “what the user was editing” and “what the server now
thinks is current,” the send/fork flows should carry explicit state identity
from the client:

- send/fork request includes the state ID or version the user was looking at
- server verifies that identity against the session timeline
- server persists the appropriate message provenance explicitly

This is better than querying “current state” at persistence time and hoping it
still matches the user’s edit context.

### AD-3: NULL provenance on new post-migration messages is a bug

Historical backfill may legitimately leave older messages with
`composition_state_id = NULL`, but once the new schema is live:

- newly created user messages should always get pre-send provenance
- newly created assistant messages should always get post-turn provenance

The implementation and tests should treat unexpected NULL on new messages as a
correctness failure, not a graceful fallback.

### AD-4: Forking from a user bubble uses the pre-send state for that bubble

When the user edits an earlier user message and forks:

- inherited history includes messages before that fork point
- the forked session copies the composition state linked to that user bubble’s
  pre-send `composition_state_id`
- the new edited user message is appended in the forked session
- the original assistant response and later timeline stay only in the original
  session

### AD-5: Blob inheritance remains copy-on-fork and must honor blob quotas

If the inherited state references blobs, the fork operation copies those blobs
into the new session and must respect the blob-manager quota policy rather than
duplicating without limit

---

## 4. Dependencies

- Blob-manager decisions, if forked sessions must also inherit input/output
  blobs cleanly.

---

## 5. Implementation Shape

### Backend

- schema changes:
  - `sessions.forked_from_session_id`
  - `sessions.forked_from_message_id`
  - `chat_messages.composition_state_id`
- fork endpoint with explicit message/state provenance handling
- server-side verification of the client-provided starting state
- copy-on-fork blob hook for inherited blob-backed state

### Frontend

- edit affordance on eligible user bubbles
- submit-edited-message-as-fork flow
- navigation to the newly created fork
- visible provenance banner in the forked session

### Persistence rule

The `send_message` path should persist:

1. user message with pre-send `composition_state_id`
2. assistant message with post-compose `composition_state_id`

That ordering should be tested directly.

---

## 6. Implementation Phases

### Phase 1: Schema and migration groundwork

- add `forked_from_session_id` and `forked_from_message_id`
- add `chat_messages.composition_state_id`
- define migration/backfill behavior for existing rows

Deliverable:

- persistence can represent fork provenance and per-message state boundaries

### Phase 2: Message provenance plumbing

- update send-message request/handler shape to carry client-observed state
  identity
- verify that identity server-side
- persist pre-send and post-turn `composition_state_id` correctly

Deliverable:

- new messages record correct state provenance deterministically

### Phase 3: Fork backend flow

- add fork endpoint
- copy inherited history and state from the selected message boundary
- integrate copy-on-fork blob handling

Deliverable:

- backend can create a coherent forked session from a user message

### Phase 4: Frontend fork UX

- add edit affordance on user bubbles
- submit edited message as fork
- switch to the new session and surface provenance visibly

Deliverable:

- users can branch from earlier messages without mutating history

---

## 7. Testing Requirements

- unit/integration test asserting:
  - user message provenance equals the pre-send state
  - assistant message provenance equals the post-compose state
- regression test ensuring post-migration new messages never persist with NULL
  `composition_state_id`
- fork test ensuring the fork uses the edited user bubble’s pre-send state, not
  the latest session state
- concurrency test covering two sends/forks near the same state boundary
- blob inheritance test verifying copied blobs get new session ownership and
  respect blob quota policy

## 8. Open Questions

- Exact fork payload/response shape once the request includes explicit state
  identity.
- Whether state identity should be by `state_id`, `version`, or both.
- Whether the forked session should also include a synthetic “forked from”
  transcript message in addition to header metadata.

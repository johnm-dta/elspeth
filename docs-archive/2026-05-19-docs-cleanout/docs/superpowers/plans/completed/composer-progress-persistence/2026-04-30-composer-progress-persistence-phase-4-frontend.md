# Composer Progress Persistence — Phase 4: Frontend Recovery Panel

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision history:** Rev 1 was authored before the Phase 3 route/API shape landed. Rev 2 (2026-05-14) re-baselines this frontend plan against the live code: `failed_turn.transcript_url` is currently `null`, `GET /api/sessions/{sid}/messages` supports `include_tool_rows=true` but not a `since` cursor, `ApiError` parsing drops recovery payload fields, tests are co-located rather than under `__tests__`, and the existing modal pattern is `useFocusTrap` plus local dialog markup rather than a shared `Modal` component. Rev 3 (2026-05-14) adds Phase 4A: first clean up the Phase 3 composer-service module-tail monkeypatching before starting the frontend panel. Rev 4 (2026-05-14) applies focused review repairs: direct config assignment with no defensive fallback, broader backend verification, structural recovery-state clearing, and composition-version based concurrent-edit protection. Rev 5 (2026-05-14) closes the rev-4 repass findings: `clearedRecoveryState()` is also spread into `initialState` so `reset()` (logout) clears recovery fields, and the concurrent-edit test description is corrected to reflect that `compositionState.version` advances only on server compose responses, not on local edits.

**Goal:** First normalize the Phase 3 composer-service implementation shape, then build the React recovery surface that opens when a compose request fails with both `partial_state` and `failed_turn` in the response body. Show a pipeline diff against the editor's current state and a tool transcript, then let the user Apply the partial draft with a concurrent-edit confirmation or Discard the panel without deleting audit data.

**Architecture:** Phase 4A is a backend-structure cleanup only: move Phase 3's test driver, redaction serialization helper, state-payload helper, and constructor initialization into normal `ComposerServiceImpl` class code without changing route/API behaviour. Phase 4B is frontend/client recovery: adapt the current TypeScript API layer to preserve recovery fields, derive transcript requests from the active session id, and render the recovery panel from typed frontend state.

**Tech Stack:** Python, pytest, React, TypeScript, Zustand, Vitest/Testing Library, existing `useFocusTrap`, existing `ConfirmDialog` styling patterns, and the current `sessionStore` / `useComposer` split.

**Spec sections:** §7 (Frontend Recovery Surface), §8.4 (frontend tests), §11 Phase 4 scope. Accessibility requirements: §7.7.

---

## Current Code Baseline

Validated against the current tree before this rev:

- `src/elspeth/web/sessions/routes.py::_failed_turn_response_body()` emits `transcript_url: None`; the frontend must not require a non-null URL.
- `GET /api/sessions/{sid}/messages` accepts `include_tool_rows=true`, `limit`, `offset`, `include_llm_audit`, and `include_raw_content`; it does not accept `since`.
- `ChatMessageResponse` already includes `tool_call_id`, `parent_assistant_id`, and `sequence_no`, but `src/elspeth/web/frontend/src/types/index.ts` does not yet model tool/audit rows.
- `src/elspeth/web/frontend/src/api/client.ts::parseResponse()` reads nested FastAPI `detail` objects but currently preserves only selected fields (`error_type`, provider diagnostics, fanout guard, validation errors). It must copy `partial_state`, `failed_turn`, and `partial_state_save_failed` recovery fields into `ApiError`.
- `src/elspeth/web/frontend/src/hooks/useComposer.ts` has no `onError` callback path. It only wraps `sessionStore.sendMessage()` / `retryMessage()` with an `AbortController`.
- The real error dispatch and optimistic-message failure state live in `src/elspeth/web/frontend/src/stores/sessionStore.ts`.
- Existing frontend tests are co-located (`src/.../Thing.test.tsx`), not placed in `__tests__/`.
- Existing modal/focus primitives are `src/elspeth/web/frontend/src/hooks/useFocusTrap.ts` and dialog markup in `ConfirmDialog`, `ShortcutsHelp`, `CommandPalette`, `CatalogDrawer`, and `SecretsPanel`; there is no `components/shared/Modal` module.
- `src/elspeth/web/composer/service.py` still has Phase 3 module-tail monkeypatching: `_run_one_turn_for_test` is attached at module tail, `_serialize_response_via_walker` and `_state_payload_for_compose_turn_for_test` are attached the same way, and `ComposerServiceImpl.__init__` is replaced after class definition. This is the first Phase 4 cleanup item.

---

## File Structure

### Files to create

- `tests/unit/web/composer/test_compose_service_structure.py`
- `src/elspeth/web/frontend/src/types/recovery.ts`
- `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts`
- `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.test.ts`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.test.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.test.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx`
- `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.test.tsx`

### Files to modify

- `src/elspeth/web/composer/service.py` — move Phase 3 module-tail helper assignments into normal class methods / constructor code.
- `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-3-compose-loop.md` — fix the Phase 2 manifest-entry appendix count if it still says 38 while live parity is 39.
- `src/elspeth/web/frontend/src/types/index.ts` — widen `ChatMessage.role` to include `"tool"` and `"audit"` if needed by shared transcript typing, and add optional `raw_content`, `tool_call_id`, `parent_assistant_id`, and `sequence_no` fields matching `ChatMessageResponse`.
- `src/elspeth/web/frontend/src/types/api.ts` — re-export recovery types.
- `src/elspeth/web/frontend/src/api/client.ts` — preserve recovery fields in `ApiError` and add a typed transcript fetch helper.
- `src/elspeth/web/frontend/src/api/client.guided.test.ts` or a new adjacent `client.recovery.test.ts` — cover recovery-field parsing.
- `src/elspeth/web/frontend/src/stores/sessionStore.ts` — add recovery-panel state/actions and open the panel from the existing `sendMessage` / `retryMessage` catch paths.
- `src/elspeth/web/frontend/src/stores/sessionStore.test.ts` — cover the recovery error path and fallback error path.
- `src/elspeth/web/frontend/src/hooks/useComposer.ts` and `src/elspeth/web/frontend/src/hooks/useComposer.test.ts` only if a thin selector API is needed after the store owns recovery state.
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` or `src/elspeth/web/frontend/src/App.tsx` — render the recovery panel at a stable top-level location.
- `src/elspeth/web/frontend/src/App.css` — add recovery panel styles using existing design tokens/classes.

### Files NOT touched in Phase 4

- Backend route/API behaviour. Phase 4A may edit backend Python structure in `ComposerServiceImpl`, but it must not change HTTP response shape, persistence semantics, audit/logging semantics, database schema, or public composer tool behaviour.
- Live chat transcript default behaviour. `components/chat/` must continue to fetch normal messages without `include_tool_rows=true`.
- The closed staging replay task `elspeth-599ecf69fa`; current VAL ownership for this feature is `elspeth-90b4542b63`.

---

## Task 0: Phase 3 composer-service structure cleanup

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Create: `tests/unit/web/composer/test_compose_service_structure.py`
- Modify: `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-3-compose-loop.md`
- Re-run focused existing tests named below

- [ ] **Step 1: Write the failing structural guard**

Create `tests/unit/web/composer/test_compose_service_structure.py` with an AST/source guard that fails on the current module-tail monkeypatch pattern:

```python
from __future__ import annotations

import ast
from pathlib import Path


SERVICE_PATH = Path("src/elspeth/web/composer/service.py")


def test_composer_service_impl_not_patched_at_module_tail() -> None:
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))
    forbidden_assignments = {
        ("ComposerServiceImpl", "_run_one_turn_for_test"),
        ("ComposerServiceImpl", "_serialize_response_via_walker"),
        ("ComposerServiceImpl", "_state_payload_for_compose_turn_for_test"),
        ("ComposerServiceImpl", "__init__"),
    }
    hits: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and (target.value.id, target.attr) in forbidden_assignments
            ):
                hits.append(f"{target.value.id}.{target.attr}")
            if isinstance(target, ast.Name) and target.id.startswith("_PHASE3_"):
                hits.append(target.id)
    assert hits == []
```

Expected failure before implementation: the test reports all current Phase 3 tail assignments plus module-level `_PHASE3_*` capture state such as `_PHASE3_ORIGINAL_COMPOSER_INIT`.

- [ ] **Step 2: Move constructor initialization into `ComposerServiceImpl.__init__`**

Fold the `_phase3_composer_init` body into the real class constructor. Keep these initialized on every instance:

- `_max_tool_calls_per_turn = self._settings.composer_max_tool_calls_per_turn`
- `_telemetry`
- `_redaction_telemetry = OtelRedactionTelemetry()`
- `_phase3_last_tool_outcomes`
- `_phase3_last_expected_current_state_id`
- `_phase3_last_redacted_assistant_tool_calls`
- `_phase3_last_redacted_tool_rows`
- `_phase3_last_audit_outcome`

Do not add `try`/`except`, `getattr`, or any runtime fallback around `_max_tool_calls_per_turn`; `WebSettings.composer_max_tool_calls_per_turn` is unconditionally present with its Pydantic field default. Do not leave `_PHASE3_ORIGINAL_COMPOSER_INIT` or `ComposerServiceImpl.__init__ = ...` in the module.

- [ ] **Step 3: Move Phase 3 helper functions into class methods**

Move the bodies of these functions into `ComposerServiceImpl` as normal methods:

- `_run_one_turn_for_test`
- `_serialize_response_via_walker`
- `_state_payload_for_compose_turn_for_test`

Keep the method names stable because existing tests call `_run_one_turn_for_test` directly. Remove the tail assignments:

- `ComposerServiceImpl._run_one_turn_for_test = ...`
- `ComposerServiceImpl._serialize_response_via_walker = ...`
- `ComposerServiceImpl._state_payload_for_compose_turn_for_test = ...`

Do not change the semantics of `_compose_loop`, `persist_compose_turn_async`, failed-turn metadata, or redaction telemetry.

- [ ] **Step 4: Fix the stale Phase 3 manifest appendix count**

In `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-3-compose-loop.md`, update the appendix note that still says Phase 2 shipped with 38 manifest entries. The current verified parity is 39 registry names and 39 manifest entries. Keep the note approximate where it splits type-driven vs declarative entries unless the live code is re-counted in the same task.

Commit this as a standalone doc-only change before the service refactor:

```bash
git add docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-3-compose-loop.md
git commit -m "docs(plan): refresh phase 3 manifest count"
```

- [ ] **Step 5: Verify focused cleanup**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/web/composer/ \
  tests/property/web/composer/ \
  tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

Expected: all pass. This intentionally covers every current unit/property composer test plus the pipeline characterization slice because many tests instantiate or read `ComposerServiceImpl` attributes. If failures show behavioural drift, fix the refactor rather than relaxing assertions.

- [ ] **Step 6: Verify no module-tail monkeypatch remains**

```bash
! rg -n 'ComposerServiceImpl\._run_one_turn_for_test =|ComposerServiceImpl\._serialize_response_via_walker =|ComposerServiceImpl\._state_payload_for_compose_turn_for_test =|ComposerServiceImpl\.__init__ =|^_PHASE3_' src/elspeth/web/composer/service.py
```

Expected: no matches.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_compose_service_structure.py
git commit -m "refactor(composer): normalize phase 3 service helpers (composer-progress-persistence phase 4A)"
```

---

## Task 1: Recovery API and TypeScript contracts

**Files:**
- Create: `src/elspeth/web/frontend/src/types/recovery.ts`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/frontend/src/types/api.ts`
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Test: `src/elspeth/web/frontend/src/api/client.guided.test.ts` or `src/elspeth/web/frontend/src/api/client.recovery.test.ts`

- [ ] **Step 1: Write failing tests for recovery payload parsing**

Cover both FastAPI shapes used by the current backend:

- top-level JSON with `partial_state` / `failed_turn`
- nested `{"detail": {"partial_state": ..., "failed_turn": ...}}`

The parsed `ApiError` must preserve:

- `partial_state`
- `failed_turn`
- nested `failed_turn.assistant_message_id`
- nested `failed_turn.tool_calls_attempted`
- nested `failed_turn.tool_responses_persisted`
- nested `failed_turn.transcript_url`, including the current `null` case
- `partial_state_save_failed`
- `partial_state_save_error`

Also assert that non-recovery errors keep the existing provider/validation/fanout parsing unchanged.

- [ ] **Step 2: Define recovery types**

`FailedTurn.transcript_url` must be `string | null`, not `string`. The live backend returns `null` today to avoid baking a replay URL into route metadata.

`RecoveryTranscriptRow` must match `ChatMessageResponse` fields from `src/elspeth/web/sessions/schemas.py`, including:

- `role: "assistant" | "tool"` for the recovery transcript surface
- `tool_calls`
- `tool_call_id`
- `parent_assistant_id`
- `sequence_no`
- `composition_state_id`

Use `CompositionState` from the existing frontend type module instead of duplicating a partial state shape. Do not reference a nonexistent `src/types/state.ts`.

The current `partial_state` payload does not include a dedicated frontend schema-version field. Treat that as an accepted low-likelihood compatibility gap for Phase 4; if a later backend adds schema metadata, add it to `CompositionState` and recovery parsing in the same change.

- [ ] **Step 3: Implement client parsing and transcript helper**

Add a helper shaped like:

```typescript
export async function fetchRecoveryTranscript(
  sessionId: string,
  opts: { limit?: number; offset?: number } = {},
): Promise<RecoveryTranscriptRow[]> {
  const params = new URLSearchParams({
    include_tool_rows: "true",
    limit: String(opts.limit ?? 500),
    offset: String(opts.offset ?? 0),
  });
  const response = await fetch(`/api/sessions/${sessionId}/messages?${params}`, {
    headers: authHeaders(),
  });
  return parseResponse<RecoveryTranscriptRow[]>(response);
}
```

Do not add `since=` unless the backend endpoint has been changed in an earlier PR. If a future backend adds a cursor, update this helper and its tests in the same PR.

- [ ] **Step 4: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- client.recovery client.guided
```

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/types/recovery.ts src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/types/api.ts src/elspeth/web/frontend/src/api/client.ts src/elspeth/web/frontend/src/api/client*.test.ts
git commit -m "feat(frontend): parse composer recovery errors (composer-progress-persistence phase 4)"
```

---

## Task 2: `useRecoveryPanel` hook

**Files:**
- Create: `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts`
- Create: `src/elspeth/web/frontend/src/hooks/useRecoveryPanel.test.ts`

- [ ] **Step 1: Write failing tests**

Cover the spec §7.2 / §7.6 behaviours:

- opens only when both `partial_state` and `failed_turn` are present
- stays closed for the other three boolean-pair states
- Apply returns `needsConfirmation=true` when the editor changed after the failed compose started
- confirmed Apply calls the provided `onApplyState` with the typed `CompositionState`
- Discard only closes local UI state; it must not call the API or delete persisted rows

The concurrent-edit guard must compare `compositionState.version` against the composition version captured when the failed compose request started. Do not invent a separate editor revision/edit counter.

- [ ] **Step 2: Implement the hook**

Return stable fields/actions:

- `isOpen`
- `recoveryError`
- `openFromError(apiError)`
- `requestApply()`
- `confirmApply()`
- `cancelApplyConfirmation()`
- `discard()`
- `needsApplyConfirmation`

The hook may be pure UI state, but the source of truth for global recovery payloads and the compose-start `recoveryStartedCompositionVersion` should live in `sessionStore` if the panel is rendered outside `ChatPanel`.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- useRecoveryPanel
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/hooks/useRecoveryPanel.ts src/elspeth/web/frontend/src/hooks/useRecoveryPanel.test.ts
git commit -m "feat(frontend): add recovery panel hook (composer-progress-persistence phase 4)"
```

---

## Task 3: Recovery store wiring

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.test.ts`

- [ ] **Step 1: Write failing store tests**

Extend the current `sendMessage` / `retryMessage` error tests:

- a rejected `ApiError` with both `partial_state` and `failed_turn` opens recovery state
- the optimistic user message is still marked failed with a user-facing error string
- a 422 convergence error without recovery fields uses the existing generic convergence copy and does not open recovery state
- `applyRecoveredState()` replaces `compositionState`, clears stale validation through the execution store, clears selected node when necessary, and closes the panel
- `discardRecovery()` closes the panel without API calls and without mutating `compositionState`
- `discardRecovery()` clears both `recoveryError` and `recoveryStartedCompositionVersion`
- a recovery failure after discard replaces the previous recovery state instead of appending to it
- `discardRecovery()` is idempotent when no recovery panel is open
- `selectSession()` clears stale recovery state
- `createSession()`, `archiveSession()` when archiving the active session, and `forkFromMessage()` clear stale recovery state
- `reset()` (called by `authStore` on logout) clears recovery state because `initialState` spreads `clearedRecoveryState()`
- a second successful compose response arrives while the recovery panel is open and advances `compositionState.version`; the next Apply requires confirmation because the stored compose-start version no longer matches the current version

- [ ] **Step 2: Implement store state/actions**

Add a small store slice:

- `recoveryError: ComposerRecoveryError | null`
- `recoveryStartedCompositionVersion: number | null`
- `openRecoveryFromError(error: ApiError, recoveryStartedCompositionVersion: number | null)`
- `applyRecoveredState()`
- `discardRecovery()`

At the start of each `sendMessage()` / `retryMessage()` compose request, snapshot the current `compositionState?.version ?? null` into a local request variable. If the request rejects with a recovery-shaped `ApiError`, store that compose-start version as `recoveryStartedCompositionVersion` alongside `recoveryError`. When Apply is requested, compare that snapshot with the current `compositionState?.version ?? null`; require confirmation if they differ.

Add a helper paralleling the existing guided-state reset helper:

```typescript
const clearedRecoveryState = () => ({
  recoveryError: null,
  recoveryStartedCompositionVersion: null,
});
```

Thread this helper through every session transition that already clears or replaces session-scoped UI state: `createSession()`, `archiveSession()` when the active session is archived, `selectSession()`, and `forkFromMessage()`. Additionally, spread `...clearedRecoveryState()` into `initialState` so that the `reset()` action (invoked by `authStore` on logout) also clears recovery fields — this mirrors the existing `...clearedGuidedState()` spread in `initialState` and prevents recovery fields from going `undefined` after logout. This prevents applying session A's partial compose state into session B after navigation and prevents stale recovery state surviving a logout/login cycle.

Reuse the existing successful-message state-update logic where practical so selection/validation invalidation behaves like a normal composer state update. Avoid duplicating drift-prone snippets.

`applyRecoveredState()` is local frontend state recovery only. It must not issue a `PATCH`, `PUT`, compose retry, or transcript mutation request; the audit-of-record for the failed turn is the already-persisted `composition_states` row and associated transcript rows. `discardRecovery()` is also intentionally local-only and non-audited at the user-action layer.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- sessionStore
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/sessionStore.ts src/elspeth/web/frontend/src/stores/sessionStore.test.ts
git commit -m "feat(frontend): store composer recovery payloads (composer-progress-persistence phase 4)"
```

---

## Task 4: `RecoveryDiff`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx`
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.test.tsx`

- [ ] **Step 1: Write failing tests**

Cover:

- added / removed / changed source entries
- `null` or missing source collections in `partial_state`
- added / removed / changed nodes by `id`
- added / removed / changed edges by stable edge identity
- empty state when current and partial match
- large-state sanity: thousands of nodes do not render thousands of full JSON blobs by default

- [ ] **Step 2: Implement**

Use typed `CompositionState` inputs. Build structured deltas rather than dumping full JSON. For large diffs, show a compact summary plus expandable details so the panel satisfies the spec's 500 ms recovery TTI sanity target.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryDiff
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.tsx src/elspeth/web/frontend/src/components/recovery/RecoveryDiff.test.tsx
git commit -m "feat(frontend): render composer recovery diff (composer-progress-persistence phase 4)"
```

---

## Task 5: `RecoveryTranscript`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx`
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.test.tsx`

- [ ] **Step 1: Write failing tests**

Cover:

- calls `fetchRecoveryTranscript(sessionId, { limit: 500 })`
- the underlying request URL includes `include_tool_rows=true`
- fetch is called exactly once per panel open with stable dependencies, not once per render
- filters to the assistant row matching `failed_turn.assistant_message_id`
- matches tool rows by `parent_assistant_id` and `tool_call_id`
- renders completed and missing tool calls from the assistant row's `tool_calls`
- renders redaction marker text for redacted persisted tool content
- does not render `raw_content` or provider payload fields even if a fixture accidentally includes them
- handles `failed_turn.assistant_message_id === null` by showing the documented degraded diagnostic instead of throwing
- handles `failed_turn.assistant_message_id` being absent from the first 500 transcript rows by showing a compact degraded diagnostic with retry/reopen guidance rather than an empty transcript
- handles `failed_turn.transcript_url === null`
- shows loading and error states

- [ ] **Step 2: Implement**

Do not fetch `failed_turn.transcript_url` directly. Treat that field as optional future metadata. The live route contract is:

```text
GET /api/sessions/{sessionId}/messages?include_tool_rows=true&limit=500
```

The transcript component receives `sessionId` and `failedTurn`, calls the API helper, then derives entries locally. Keep the fetch effect dependencies stable, such as `[sessionId, failedTurn.assistant_message_id]`, so opening the panel performs one transcript read. Keep raw transcript expansion redacted-only; never request `include_raw_content`, never render hidden chain-of-thought, and never render provider payloads.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryTranscript
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.tsx src/elspeth/web/frontend/src/components/recovery/RecoveryTranscript.test.tsx
git commit -m "feat(frontend): render composer recovery transcript (composer-progress-persistence phase 4)"
```

---

## Task 6: `RecoveryPanel`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx`
- Create: `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.test.tsx`
- Modify: `src/elspeth/web/frontend/src/App.css`

- [ ] **Step 1: Write failing tests**

Cover:

- renders headline, reason badge, evidence, diff, transcript, Apply, Discard, and View raw transcript controls
- uses `role="dialog"` and `aria-modal="true"`
- uses `useFocusTrap` and restores focus on close
- reason badge has visible text plus accessible label; no colour-only signalling
- Enter key does not auto-apply the partial draft
- Apply opens the concurrent-edit confirmation when required
- Discard closes without invoking apply
- focus is restored for Discard, Apply with no conflict, Apply then confirm, and Apply then cancel-confirmation

- [ ] **Step 2: Implement**

Follow existing local dialog patterns instead of importing a nonexistent `components/shared/Modal`. Use `useFocusTrap` directly and borrow class naming/layout conventions from `ConfirmDialog`, `ShortcutsHelp`, and `CatalogDrawer`.

Set the recovery panel/backdrop z-index to the existing dialog layer token (`--z-dialog`, currently 201) so it stacks consistently above the workspace and below any deliberately higher emergency surfaces.

Use a `ConfirmDialog` or equivalent nested confirmation for concurrent edits. If nesting dialogs causes focus restoration bugs, keep the confirmation inside the recovery dialog body with `role="alert"` and explicit confirm/cancel buttons.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- RecoveryPanel
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.test.tsx src/elspeth/web/frontend/src/App.css
git commit -m "feat(frontend): add composer recovery panel (composer-progress-persistence phase 4)"
```

---

## Task 7: Render and trigger the panel

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` or `src/elspeth/web/frontend/src/App.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` or `src/elspeth/web/frontend/src/App.test.tsx`
- Modify: `src/elspeth/web/frontend/src/hooks/useComposer.ts` / `useComposer.test.ts` only if adding a thin selector wrapper remains useful

- [ ] **Step 1: Write failing integration tests**

Cover:

- a recovery-shaped rejected send opens the panel in the rendered app/chat surface
- applying the panel updates the inspector/editor state
- discarding leaves the current editor state unchanged
- a non-recovery 422 still shows the existing error path
- Apply and Discard stay local-only: no compose retry, `PATCH`, `PUT`, transcript mutation, or audit-write request is sent from these button handlers

- [ ] **Step 2: Implement rendering**

Prefer rendering the panel near `App` or `Layout` so it can overlay the full three-panel workspace and access current `compositionState`, active session id, and store actions. Keep `ChatPanel`'s normal message fetch path unchanged. Do not add a feature flag for Phase 4; the panel is always available and opens only when the backend response shape carries both recovery fields.

- [ ] **Step 3: Verify**

```bash
cd src/elspeth/web/frontend && npm test -- App ChatPanel useComposer sessionStore RecoveryPanel
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/App.tsx src/elspeth/web/frontend/src/App.test.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx src/elspeth/web/frontend/src/hooks/useComposer.ts src/elspeth/web/frontend/src/hooks/useComposer.test.ts
git commit -m "feat(frontend): trigger composer recovery panel (composer-progress-persistence phase 4)"
```

Only stage files that actually changed.

---

## Task 8: Manual frontend smoke and accessibility

- [ ] **Step 1: Build**

```bash
cd src/elspeth/web/frontend && npm run build
```

- [ ] **Step 2: Local smoke**

Start the local frontend dev server if needed:

```bash
cd src/elspeth/web/frontend && npm run dev
```

Exercise a mocked or real failed compose response carrying `partial_state` and `failed_turn`:

1. Panel opens with headline, reason badge, diff, and transcript.
2. Transcript fetch hits `include_tool_rows=true` and records audit-grade access server-side.
3. Apply updates the editor state.
4. Concurrent edit confirmation blocks accidental overwrite.
5. Discard closes the panel and does not delete persisted rows.
6. A fanout guard surface and the recovery panel can both be triggered in the same smoke session without z-index, focus, or stale-state interference.

- [ ] **Step 3: Accessibility audit**

Use axe or Lighthouse against the route where the panel is visible. Expected: no critical violations for focus trap, modal semantics, button labels, contrast, or colour-only signalling.

---

## Task 9: Final Phase 4 gate

- [ ] **Step 0: Backend composer regression suite**

```bash
.venv/bin/python -m pytest -q \
  tests/unit/web/composer/ \
  tests/property/web/composer/ \
  tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

- [ ] **Step 1: Frontend unit suite**

```bash
cd src/elspeth/web/frontend && npm test
```

- [ ] **Step 2: TypeScript check**

```bash
cd src/elspeth/web/frontend && npm run typecheck
```

- [ ] **Step 3: Lint**

```bash
cd src/elspeth/web/frontend && npm run lint
```

- [ ] **Step 4: PR**

Open the Phase 4 PR only after the gates above pass. The PR body must cite:

- this plan file
- the overview file
- the Phase 4A cleanup commit that removed Phase 3 module-tail monkeypatching
- the Phase 3 commit/PR that provided `failed_turn` and `include_tool_rows=true`
- `elspeth-90b4542b63` as the live VAL owner

Do not cite `elspeth-599ecf69fa` as a future gate; it is already closed and belongs to an older staging replay.

---

## Phase 4 Done When

1. All frontend unit tests pass.
2. TypeScript and lint are clean.
3. Phase 3's module-tail monkeypatching is gone from `src/elspeth/web/composer/service.py`; constructor initialization and helper methods live on `ComposerServiceImpl` normally.
4. The backend composer regression suite in Task 9 Step 0 still passes after Phase 4A and frontend integration.
5. API parsing preserves recovery fields from nested FastAPI `detail` bodies.
6. `sessionStore` opens recovery only for `(partial_state present, failed_turn present)` and keeps existing error behaviour for the other cases.
7. Recovery transcript fetches through `include_tool_rows=true` using the active session id, not a required `failed_turn.transcript_url`.
8. The recovery panel renders diff and transcript, supports Apply/Discard/View raw transcript, and passes accessibility checks.
9. No feature flag is required; the panel is gated solely by the recovery response shape.
10. Manual smoke confirms panel open, diff render, transcript render, Apply, concurrent-edit confirmation, Discard, and coexistence with the fanout guard surface.
11. VAL — "the user can actually recover from a failure" — remains owned by [elspeth-90b4542b63](filigree:elspeth-90b4542b63).

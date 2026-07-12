# Freeform Introduction and Run History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the interactive freeform starter splash with an account-dismissible guide and give run-history rows human-readable ordinal and timestamp labels.

**Architecture:** Extend the existing account-level composer-preferences row with one nullable dismissal timestamp and thread it through the existing strict backend/frontend contracts. Render a focused empty-state component from `ChatPanel`, and derive deterministic run labels locally from each session’s existing `Run.started_at` values without changing run storage.

**Tech Stack:** FastAPI, Pydantic v2, SQLAlchemy Core, Zustand, React, TypeScript, Vitest, Testing Library, Playwright.

---

### Task 1: Persist the freeform introduction dismissal

**Files:**
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/preferences/models.py`
- Modify: `src/elspeth/web/preferences/service.py`
- Test: `tests/unit/web/preferences/test_models.py`
- Test: `tests/unit/web/preferences/test_schema.py`
- Test: `tests/unit/web/preferences/test_service.py`
- Test: `tests/integration/web/test_preferences_routes.py`

- [ ] **Step 1: Write failing contract tests**

Add assertions that GET defaults `freeform_intro_dismissed_at` to null, PATCH persists a timezone-aware timestamp, an absent field preserves it, explicit null clears it, and one user’s write does not affect another user.

```python
stamp = datetime(2026, 7, 12, 5, 0, tzinfo=UTC)
payload = UpdateComposerPreferencesRequest(freeform_intro_dismissed_at=stamp)
assert payload.freeform_intro_dismissed_at == stamp
assert "freeform_intro_dismissed_at" in payload.model_fields_set
```

- [ ] **Step 2: Run the narrow backend tests and confirm RED**

Run:

```bash
uv run pytest tests/unit/web/preferences/test_models.py tests/unit/web/preferences/test_schema.py tests/unit/web/preferences/test_service.py tests/integration/web/test_preferences_routes.py -q
```

Expected: failures naming the missing preference field/column.

- [ ] **Step 3: Add the nullable field through every strict boundary**

Add `Column("freeform_intro_dismissed_at", DateTime(timezone=True), nullable=True)`, bump `SESSION_SCHEMA_EPOCH` from 26 to 27 with an epoch-history note, and add the field to the response/request models plus every service select, insert, update, return, and audit-summary projection.

```python
freeform_intro_dismissed_at: datetime | None
# request field: absent = unchanged, timestamp = dismissed, null = reset
freeform_intro_dismissed_at: datetime | None = None
```

- [ ] **Step 4: Run the narrow backend tests and confirm GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the backend contract**

```bash
git add src/elspeth/web/sessions/models.py src/elspeth/web/preferences/models.py src/elspeth/web/preferences/service.py tests/unit/web/preferences/test_models.py tests/unit/web/preferences/test_schema.py tests/unit/web/preferences/test_service.py tests/integration/web/test_preferences_routes.py
git commit -m "feat(web): persist freeform intro dismissal"
```

### Task 2: Add frontend preference state and dismissal action

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/api.ts`
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.ts`
- Test: `src/elspeth/web/frontend/src/stores/preferencesStore.test.ts`

- [ ] **Step 1: Write failing store tests**

Cover bootstrap, successful server-confirmed dismissal, in-flight disabled state, failure rollback/error text, reset, and cross-tab storage synchronisation.

```ts
expect(usePreferencesStore.getState().freeformIntroDismissedAt).toBeNull();
await usePreferencesStore.getState().dismissFreeformIntro();
expect(updateUserComposerPreferences).toHaveBeenCalledWith({
  freeform_intro_dismissed_at: expect.any(String),
});
```

- [ ] **Step 2: Run the store test and confirm RED**

```bash
npm test -- --run src/stores/preferencesStore.test.ts
```

Expected: missing state/action failures.

- [ ] **Step 3: Implement the typed store field and action**

Add the wire fields to both API payload interfaces, mirror the server value during bootstrap and every full preference response, add `dismissFreeformIntro()`, and broadcast the resolved timestamp through a versioned localStorage key. Do not optimistically remove the card: set only `writing` before PATCH and publish `freeformIntroDismissedAt` from the successful response.

- [ ] **Step 4: Run the store test and confirm GREEN**

Run the Step 2 command. Expected: all preference-store tests pass.

- [ ] **Step 5: Commit the frontend preference contract**

```bash
git add src/elspeth/web/frontend/src/types/api.ts src/elspeth/web/frontend/src/stores/preferencesStore.ts src/elspeth/web/frontend/src/stores/preferencesStore.test.ts
git commit -m "feat(web): expose freeform intro preference"
```

### Task 3: Replace the freeform starter splash

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/chat.css`
- Delete: `src/elspeth/web/frontend/src/components/chat/TemplateCards.tsx`
- Delete: `src/elspeth/web/frontend/src/components/chat/TemplateCards.test.tsx`
- Delete: `src/elspeth/web/frontend/src/components/chat/templates_data.ts`
- Test: `src/elspeth/web/frontend/src/components/chat/FreeformIntroduction.test.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
- Modify: `src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx`

- [ ] **Step 1: Write failing component and integration tests**

Assert the approved heading/copy, account-loaded/null gating, `Hiding…` disabled state, successful removal, failed-save persistence, dismissed blank state, and absence of starter-example controls.

```tsx
expect(screen.getByRole("heading", { name: "Build a pipeline" })).toBeVisible();
await user.click(screen.getByRole("button", { name: "Don’t show this again" }));
expect(screen.getByRole("button", { name: "Hiding…" })).toBeDisabled();
```

- [ ] **Step 2: Run the narrow component tests and confirm RED**

```bash
npm test -- --run src/components/chat/FreeformIntroduction.test.tsx src/components/chat/ChatPanel.test.tsx src/test/a11y/components.a11y.test.tsx
```

- [ ] **Step 3: Implement the calm guide and remove template UI**

Render `FreeformIntroduction` only for an empty freeform conversation with successfully loaded preferences and a null dismissal timestamp. Use existing button/focus/error conventions and design tokens; after dismissal render nothing in the conversation background. Remove obsolete template imports, selection callbacks, CSS, data, and tests once `rg` confirms no remaining runtime consumer.

- [ ] **Step 4: Run the narrow component tests and confirm GREEN**

Run the Step 2 command. Expected: all selected tests pass with no accessibility violations.

- [ ] **Step 5: Commit the freeform empty state**

```bash
git add src/elspeth/web/frontend/src/components/chat src/elspeth/web/frontend/src/test/a11y/components.a11y.test.tsx
git commit -m "feat(web): simplify freeform empty state"
```

### Task 4: Give run history human-readable identity

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx`
- Modify: `src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/header/header.css`

- [ ] **Step 1: Write failing run-label tests**

Use out-of-order fixtures and equal timestamps to prove ordinal assignment is `started_at` ascending then ID, display is newest first, the visible primary label contains full local date/time, GUID remains secondary, and accessible detail/cancel names contain both identifiers.

```tsx
expect(screen.getByText(/Run 2 · .*2026.*3:45/)).toBeVisible();
expect(screen.getByText("c5f713ed-3bef-40d1-adda-7669d573efad")).toBeVisible();
```

- [ ] **Step 2: Run the drawer test and confirm RED**

```bash
npm test -- --run src/components/execution/RunsHistoryDrawer.test.tsx
```

- [ ] **Step 3: Implement deterministic labels and compact hierarchy**

Derive ordinal metadata with a stable sort, format `started_at` using `Intl.DateTimeFormat` with date, year, hour, and minute, render `Run history · N` with a trailing close control, and wrap row metadata/actions responsively. Keep diagnostics, outputs, cancellation, status glyphs, focus trap, and Escape handling unchanged.

- [ ] **Step 4: Run the drawer test and confirm GREEN**

Run the Step 2 command. Expected: all drawer tests pass.

- [ ] **Step 5: Commit run-history presentation**

```bash
git add src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.tsx src/elspeth/web/frontend/src/components/execution/RunsHistoryDrawer.test.tsx src/elspeth/web/frontend/src/components/header/header.css
git commit -m "fix(web): clarify run history identity"
```

### Task 5: Verify, deploy, and close tracking

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run backend verification**

```bash
make test-fast
```

Expected: exit 0.

- [ ] **Step 2: Run frontend verification and build**

```bash
cd src/elspeth/web/frontend
npm test -- --run
npm run lint
npm run lint:css
npm run build
```

Expected: all tests and linters pass; Vite build exits 0 (existing chunk-size warnings are non-failing).

- [ ] **Step 3: Recreate the pre-release session database if epoch 27 requires it, then restart**

Follow `docs/runbooks/staging-session-db-recreation.md`, then run the passwordless service restart command and probe `/api/system/status` until it returns 200.

- [ ] **Step 4: Verify the live UX with Playwright**

Check desktop and 390px layouts, dismissal persistence after reload/new session, blank dismissed state, run ordinal/timestamp/GUID hierarchy, keyboard focus, and zero console errors.

- [ ] **Step 5: Close the Filigree issue with the final commit anchor**

Add verification evidence and close only after the live checks pass.

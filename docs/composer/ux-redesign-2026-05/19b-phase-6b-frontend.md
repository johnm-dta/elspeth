# Phase 6B — Frontend: completion bar, shareable-link inspect view, context-aware results

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Land the frontend half of Phase 6 — the three-verb completion bar in the side rail, the Save-for-review confirmation dialog with shareable-link display, the read-only inspect view at `/shared/{token}` for shareable-link recipients, the context-aware result-rendering refactor (narrative summary when any pipeline transform declares `supports_narrative_summary=True`, table preview otherwise), and the Export-YAML top-level button that promotes the existing `YamlView` drawer.

**Architecture:** React + Vite + Zustand + Vitest, matching every prior frontend phase plan. A new `CompletionBar.tsx` component lives in `components/composer/` (creating the directory if not present — verify in Task 1). The side-rail mount is added by Phase 3; this plan assumes Phase 3 has shipped and adds the new component to the rail. A new `components/shared/` directory houses the read-only inspect view. The existing `YamlView.tsx` at `components/inspector/` is reused — no duplication.

**Tech Stack:** React 18, TypeScript strict, Zustand for store, Vitest + Testing Library for tests. The app uses a hash-router (`useHashRouter.ts`), **not** React Router. The `#/shared/{token}` path is handled via a top-level branch in `App.tsx` (see Task 8).

**Sibling plan:** [19a-phase-6a-backend.md](19a-phase-6a-backend.md) — schema-free save-for-review (HMAC-signed token + content-addressable snapshot blob in the payload store; Landscape decision-event channel for audit), token signing primitive, three new endpoints, narrative-summary `ClassVar[bool]` catalog field, YAML-export Landscape audit event.

**Design reference:** [09-completion-gestures.md](09-completion-gestures.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Sequencing: depends on Phase 3 + Phase 6A

This plan **requires** that Phase 3 (side rail) and Phase 6A (backend) have shipped. The side rail is the mount point for the completion bar; the backend endpoints are the wire contracts the frontend consumes.

If Phase 3 is not shipped when implementation starts, fall back to a temporary header-area mount: the `CompletionBar` is wrapped in `<div className="completion-bar--header-fallback">` and inserted into the existing composer header. A `// TODO: relocate to side rail after Phase 3 ships` comment links back to [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md). The fallback is documented in Task 1 below; do not let it become permanent.

If Phase 6A is not shipped, this plan **cannot** proceed — the three endpoints and the `supports_narrative_summary` catalog field do not exist yet. The implementing agent must verify 6A merged via `git log --oneline | grep shareable_reviews` before starting.

**Phase 5b dependency for narrative result rendering:** Task 6's `NarrativeResults` consumes Phase 5b's interpretation events as a post-run overlay layered on top of the plugin's raw `summary` field. If Phase 5b is not yet shipped, the component falls back to rendering the raw `summary` field only; the interpretation-event overlay is additive and the renderer must tolerate its absence. Do not block Phase 6 on Phase 5b.

---

## Scope boundaries

**In scope:**

- `CompletionBar.tsx` with three buttons: "Save for review", "Run pipeline", "Export YAML". Co-equal styling, no primary emphasis.
- "Save for review" click → confirmation dialog with the returned `share_url`. Copy-to-clipboard affordance + "Open in new tab" link.
- "Run pipeline" click → existing Execute path (unchanged); result rendering refactored to detect narrative-summary plugins.
- "Export YAML" click → opens the existing `YamlView` (already implemented at `components/inspector/YamlView.tsx`); the button is the top-level affordance the design doc calls for.
- `#/shared/{token}` hash path — read-only inspect view (`SharedInspectView`). Entered via a top-level hash-branch in `App.tsx` (see Task 8; hash-router, not React-Router). Renders: pipeline metadata, audit-readiness panel (read-only), graph mini-view, and the YAML. No edit affordances; the composer chat panel is hidden.
- Result-rendering refactor: a new `useNarrativeMode()` hook reads the catalog and the current composition state, returns `true` if any pipeline transform's catalog entry has `supports_narrative_summary === true`. The existing results view branches on this.
- API client extensions: `markReadyForReview`, `fetchShareableLink`, `fetchSharedInspect`.
- Store extensions: a new `shareableReviewStore` for the dialog state and the latest token; the existing `sessionStore` and `executionStore` are not modified beyond reading the new catalog field.

**Out of scope:**

- Token revocation UI (no backend support in v1).
- Reviewer "approve / request changes" surface — read-only inspect only.
- Email-this-link UI — copy-to-clipboard is the v1 affordance.
- Org-level review queues, notification badges, "shared with me" inbox.
- Per-plugin tuning of narrative mode (binary opt-in via the backend flag).
- Side-rail layout work — owned by Phase 3.
- YAML modal/drawer redesign — the existing `YamlView` is reused as-is; only the entry-point button is new.

---

## Trust tier check (per CLAUDE.md)

| Surface | Tier | Posture |
|---|---|---|
| Token in URL (`/shared/{token}`) | Tier 3 inbound to backend | Frontend forwards verbatim; the backend verifies the signature. The frontend never inspects the token bytes. |
| `MarkReadyForReviewResponse` from backend | Tier 1 inbound to frontend | Typed parse (`parseResponse<MarkReadyForReviewResponse>`); shape drift crashes. |
| `SharedInspectResponse` from backend | Tier 1 inbound | Same. |
| Catalog response (extended with `supports_narrative_summary`) | Tier 1 inbound | TypeScript types must be widened in lockstep; drift breaks the build. |
| User clicks "Save for review" with pending changes | n/a — UI contract | Disable the button if `compositionState.is_valid === false`; the backend will also reject, but the UX is friendlier if the button shows the precondition. |
| Copy-to-clipboard surface | n/a — browser API | `navigator.clipboard.writeText`; fallback to selectable text if the API is unavailable. |

The shareable-link recipient flow is **not** a public surface. The `#/shared/{token}` hash path requires an authenticated session on the same deployment — the existing auth middleware applies. If the recipient is unauthenticated, client-side logic saves the current hash to `sessionStorage` before the redirect to `/login`, then restores it after login so the recipient lands on `SharedInspectView`. The hash (and its embedded token) is never sent to the server as a query parameter.

---

## Wire contracts consumed from 19a

These are the exact shapes 19b reads. Drift in either direction is a typed-parse failure.

```ts
// New types added to types/api.ts in Task 2.
// Mirror the shapes in 19a Task 3 verbatim — drift here is a typed-parse failure.

export interface MarkReadyForReviewResponse {
  token: string;
  share_url: string;
  expires_at: string;  // ISO-8601
  payload_digest: string;  // content-address of the snapshot blob (e.g. "sha256:...")
}

export interface ShareableLinkResponse {
  token: string;
  share_url: string;
  expires_at: string;
  state_id: string;
  payload_digest: string;
}

export interface SharedInspectResponse {
  session_id: string;
  state_id: string;
  pipeline_metadata: PipelineMetadata;       // existing type
  composition_snapshot: CompositionState;    // existing type
  yaml: string;
  audit_readiness: AuditReadinessSnapshot;   // existing Phase 2 type — consumed by Task 8
  created_by_user_id: string;
  created_at: string;
  expires_at: string;
}

// Extension to the existing plugin metadata type. The backend declares this as
// ClassVar[bool] on the plugin Protocol; on the wire it's a plain boolean.
export interface TransformPluginMetadata {
  // ...existing fields...
  supports_narrative_summary: boolean;
}
```

---

## File structure

**New:**

- `src/elspeth/web/frontend/src/components/composer/CompletionBar.tsx`
- `src/elspeth/web/frontend/src/components/composer/CompletionBar.test.tsx`
- `src/elspeth/web/frontend/src/components/composer/SaveForReviewDialog.tsx`
- `src/elspeth/web/frontend/src/components/composer/SaveForReviewDialog.test.tsx`
- `src/elspeth/web/frontend/src/components/shared/SharedInspectView.tsx`
- `src/elspeth/web/frontend/src/components/shared/SharedInspectView.test.tsx`
- `src/elspeth/web/frontend/src/components/execution/NarrativeResults.tsx`
- `src/elspeth/web/frontend/src/components/execution/NarrativeResults.test.tsx`
- `src/elspeth/web/frontend/src/hooks/useNarrativeMode.ts`
- `src/elspeth/web/frontend/src/hooks/useNarrativeMode.test.ts`
- `src/elspeth/web/frontend/src/stores/shareableReviewStore.ts`
- `src/elspeth/web/frontend/src/stores/shareableReviewStore.test.ts`

**Modified:**

- `src/elspeth/web/frontend/src/api/client.ts` — three new functions (`markReadyForReview`, `fetchShareableLink`, `fetchSharedInspect`).
- `src/elspeth/web/frontend/src/types/api.ts` — three new response interfaces + extension to `TransformPluginMetadata`.
- `src/elspeth/web/frontend/src/App.tsx` — add a top-level hash-branch guard for `#/shared/{token}` (see Task 8 Step 4; no React-Router `<Route>` or server-side path).
- `src/elspeth/web/frontend/src/components/composer/SideRail.tsx` (or whatever the Phase 3 side-rail mount is) — mount `<CompletionBar />` below the audit-readiness panel and above Catalog. Verify the file path during Task 1.
- `src/elspeth/web/frontend/src/components/execution/<existing-results-view>.tsx` (verify path) — branch on `useNarrativeMode()` to render `<NarrativeResults />` or the existing table preview.
- `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` (or wherever Execute lives today) — promote Execute into the new CompletionBar; remove the legacy button if it duplicates.

---

## Task 1: API client + types

**Files:** `types/api.ts`, `api/client.ts`, `api/client.test.ts` (new tests appended).

- [ ] **Step 1: Failing tests.**

```ts
// api/client.test.ts (append)
import { describe, it, expect, vi } from "vitest";
import { fetchSharedInspect, markReadyForReview, fetchShareableLink } from "./client";

describe("shareable-review API", () => {
  it("POST /api/sessions/:id/mark-ready-for-review returns token", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({
        payload_digest: "sha256:abab1234abab1234abab1234abab1234abab1234abab1234abab1234abab1234", token: "tok", share_url: "/shared/tok",
        expires_at: "2026-12-01T00:00:00Z",
      }), { status: 200 })
    );
    global.fetch = fetchMock;
    const resp = await markReadyForReview("sess1");
    expect(resp.token).toBe("tok");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/sessions/sess1/mark-ready-for-review",
      expect.objectContaining({ method: "POST" })
    );
  });

  it("rejects on 4xx with typed error", async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "validation failed" }), { status: 409 })
    );
    await expect(markReadyForReview("sess1")).rejects.toThrow(/validation failed/);
  });

  it("GET /api/sessions/:id/shareable-link", async () => { /* ... */ });
  it("GET /api/sessions/shared/:token", async () => { /* ... */ });
});
```

- [ ] **Step 2: Run to fail** (`npm test -- client.test.ts`).
- [ ] **Step 3: Implementation.** Add the three functions to `client.ts`, following the existing pattern (the file already has ~60 fetch helpers; mirror the closest existing shape — `fetchYaml` at line 644). Add the four new types to `types/api.ts`. Update `TransformPluginMetadata` to include `supports_narrative_summary: boolean`.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add shareable-review API client + response types`.

---

## Task 2: `shareableReviewStore`

**Files:** `stores/shareableReviewStore.ts`, `stores/shareableReviewStore.test.ts`.

The store owns the dialog state (open/closed), the in-flight request state (idle/pending/success/error), and the latest token returned by the backend. It is **session-scoped** — switching sessions clears the store.

- [ ] **Step 1: Failing tests.** Cover:
    1. Initial state is `idle` with no token.
    2. `markReadyForReview` transitions to `pending` then `success`, populates the token.
    3. `markReadyForReview` failure transitions to `error` with a typed error message.
    4. `closeDialog` resets to `idle` but preserves the token (so the user can reopen and re-copy).
    5. `clearForSession` wipes everything (called on session switch).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Standard Zustand store mirroring `executionStore.ts`. The store calls `markReadyForReview` from the API client; the component layer dispatches the action.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add shareableReviewStore`.

---

## Task 3: `CompletionBar` component

**Files:** `components/composer/CompletionBar.tsx`, `components/composer/CompletionBar.test.tsx`. Create the `composer/` directory if absent.

The bar is a presentation component with three buttons. Click handlers dispatch to the appropriate store/route.

- [ ] **Step 1: Failing tests.**
    1. Renders three buttons with accessible labels: "Save for review", "Run pipeline", "Export YAML".
    2. Save-for-review button is disabled when `compositionState.is_valid === false`.
    3. Run-pipeline button is disabled when `compositionState.is_valid === false`.
    4. Export-YAML button is **always** enabled (design doc 09: "Available always — even with ⚠ status").
    5. Clicking Save-for-review opens the `SaveForReviewDialog`.
    6. Clicking Run-pipeline calls the existing Execute action from `executionStore`.
    7. Clicking Export-YAML opens the existing `YamlView` (verify the open mechanism — likely a state flag in the inspector).
    8. No button has visual primary emphasis (CSS classes match co-equal styling — assert by class name, not visual snapshot).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** The component reads `compositionState` from `sessionStore` and dispatches to: `shareableReviewStore.markReadyForReview` / `executionStore.execute` / a new `inspectorStore.openYamlView` action (if Yaml visibility isn't already a store, add it as a one-liner). Styling: three side-by-side buttons in a single flex row; no primary modifier class on any of them. Aria labels match the design doc verbs verbatim. Each button has a tooltip with the design doc 09's per-verb summary.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add CompletionBar with three co-equal completion verbs`.

---

## Task 4: `SaveForReviewDialog` component

**Files:** `components/composer/SaveForReviewDialog.tsx`, `components/composer/SaveForReviewDialog.test.tsx`.

The dialog shows: a one-line summary ("This pipeline is ready for a colleague to review. They'll see your YAML and the audit-readiness panel."), the `share_url` in a read-only text input, a "Copy link" button, an "Open in new tab" link, and an expiry note ("Link expires {date}").

- [ ] **Step 1: Failing tests.** Add the following `beforeEach` mock setup at the top of the test file — `navigator.clipboard` is not available in jsdom by default:

```typescript
beforeEach(() => {
  Object.assign(navigator, { clipboard: { writeText: vi.fn() } });
});
```

Then add the test cases:

```typescript
it("copies the link on click", async () => {
  render(<SaveForReviewDialog url="https://example.com/share/abc123" />);
  await userEvent.click(screen.getByRole("button", { name: /copy link/i }));
  expect(navigator.clipboard.writeText).toHaveBeenCalledWith("https://example.com/share/abc123");
});
```

Full test list:

    1. Renders when `shareableReviewStore.dialog.open === true`.
    2. Does not render when closed.
    3. Shows pending spinner while `state === "pending"`.
    4. Shows the share URL and expiry date when `state === "success"`.
    5. Copy-link button calls `navigator.clipboard.writeText` with the URL (clipboard mock required — see `beforeEach` setup above).
    6. Copy-link button shows a transient "Copied!" confirmation after click.
    7. Open-in-new-tab is an `<a target="_blank" rel="noopener noreferrer">` to the share URL.
    8. Error state shows the backend's `detail` message and a "Try again" button.
    9. Closing the dialog calls `shareableReviewStore.closeDialog`.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Modal pattern; reuse existing dialog primitives if present (verify via `components/common/`). The expiry date is parsed from `expires_at` and rendered with the same date formatter the inspector uses (find via grep on `toLocaleString`). The copy-link handler must use `navigator.clipboard.writeText`; add a fallback to `document.execCommand('copy')` on a selected text input if the Clipboard API is unavailable (some browser security contexts block it).
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add SaveForReviewDialog with copy-link + expiry`.

---

## Task 5: `useNarrativeMode` hook

**Files:** `hooks/useNarrativeMode.ts`, `hooks/useNarrativeMode.test.ts`.

The hook returns `true` if any transform in the current pipeline has a catalog entry with `supports_narrative_summary === true`. Reads from the catalog store (already exists for Phase 7 catalog work — verify) and the current composition state.

- [ ] **Step 1: Failing tests.**
    1. Returns `false` when no nodes are present.
    2. Returns `false` when no transform has `supports_narrative_summary=true` in the catalog.
    3. Returns `true` when at least one node's plugin has the flag.
    4. Returns `false` when the catalog has not loaded yet (defensive: better to default to the table preview than render an empty narrative).
    5. Re-evaluates when `compositionState.nodes` changes.
    6. Re-evaluates when the catalog updates (rare, but covered).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Use `useMemo` over `(compositionState.nodes, catalog.transforms)` to compute the boolean. Selector pattern from `sessionStore` reads.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add useNarrativeMode hook for result-rendering branch`.

---

## Task 6: `NarrativeResults` component

**Files:** `components/execution/NarrativeResults.tsx`, `components/execution/NarrativeResults.test.tsx`.

Renders the narrative summary from the run output. The contract (per backend Task 7's ClassVar docstring) is that the opted-in transform's output schema includes a `summary` field. The component reads that field and renders it as Markdown (reusing `MarkdownRenderer` from `components/chat/`). When Phase 5b has shipped, the component additionally overlays Phase 5b's interpretation events for the current run — the interpretation overlay is additive (it surfaces user-affirmed interpretations alongside the raw narrative) and the component must render gracefully when no interpretation events are available (e.g., 5b not shipped, or the run produced none).

- [ ] **Step 1: Failing tests.**
    1. Renders the `summary` field from the run output as Markdown.
    2. Falls back to "No narrative available" when the field is missing or empty (graceful failure for opted-in-without-summary plugins, per backend Task 7's documented contract gap).
    3. Renders alongside a "Download full output" link to the existing artifact endpoint.
    4. Streams correctly when the run is still in progress (renders partial summary if available).
    5. When Phase 5b interpretation events are present, the overlay renders them above the raw summary; when absent, only the raw summary renders.
    6. Component does not crash when the interpretation-events store is empty / missing (Phase 5b not shipped).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Reads the run output from `executionStore`. The summary extraction is: find the last output row that has a `summary` field; if multiple, concatenate with blank lines.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add NarrativeResults rendering for batch-analytic transforms`.

---

## Task 7: Wire the narrative branch into the results view

**Files:** the existing results view in `components/execution/` (verify file path — likely `ProgressView.tsx` or a sibling), tests updated in place.

- [ ] **Step 1: Failing test.** Add a test that, given a composition with `batch_classifier_metrics` and a catalog entry where `supports_narrative_summary=true`, the rendered output is `<NarrativeResults />` rather than the existing table preview.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Insert a single branch at the top of the results-rendering JSX:

```tsx
const narrativeMode = useNarrativeMode();
return narrativeMode ? <NarrativeResults /> : <ExistingTablePreview />;
```

- [ ] **Step 4: Run to pass.** Existing tests for the table preview should still pass for non-narrative pipelines.
- [ ] **Step 5: Commit.** `feat(web/frontend): branch result rendering on useNarrativeMode`.

---

## Task 8: `SharedInspectView` component + `/shared/:token` route

**Files:** `components/shared/SharedInspectView.tsx`, `components/shared/SharedInspectView.test.tsx`, `App.tsx`.

The view fetches `GET /api/sessions/shared/{token}` and renders a read-only composition inspector. Reuse the existing `InspectorPanel` for the pipeline metadata + graph + spec views; mount it in read-only mode (add an `isReadOnly` prop to `InspectorPanel` if not present — verify in Task 8a below).

**`AuditReadinessPanel` is session-store-aware.** The existing component reads from the Zustand session store (populated by the owner's session). Rendering it directly in the shared inspect view would require populating the session store from the shared payload — an isolation breach. Instead, implement a **separate read-only variant** `<SharedAuditReadinessPanel readOnlyState={...} />` that takes the readiness payload directly as a prop, with no store reads:

```tsx
// components/shared/SharedAuditReadinessPanel.tsx

interface SharedAuditReadinessPanelProps {
  /** Readiness payload from the SharedInspectResponse. */
  readOnlyState: AuditReadinessSnapshot;
}

export function SharedAuditReadinessPanel({ readOnlyState }: SharedAuditReadinessPanelProps) {
  // Renders the same <AuditReadinessRow /> leaf components as the owner's panel,
  // but takes state as a prop instead of reading from sessionStore.
  return (
    <section aria-label="Audit readiness">
      {readOnlyState.checks.map((check) => (
        <AuditReadinessRow key={check.id} check={check} />
      ))}
    </section>
  );
}
```

The existing `<AuditReadinessRow />` leaf components are reused as-is. The `AuditReadinessSnapshot` type is the Phase 2 wire-response model (imported from `types/api.ts`) — the same type carried on `SharedInspectResponse.audit_readiness`. Mount `<SharedAuditReadinessPanel readOnlyState={inspectResponse.audit_readiness} />` in the shared view.

The shareable-link recipient sees:
1. A header: "Shared by {created_by_user_id} — expires {date}".
2. The pipeline metadata (name, description).
3. The audit-readiness panel (read-only).
4. The graph mini-view (read-only; click-to-expand is allowed).
5. The YAML (existing `YamlView` in read-only mode).

The composer chat panel is **not** rendered. The completion bar is **not** rendered. The recipient cannot edit, fork, or execute the composition through this view — those affordances require taking ownership of the session, which is out of scope for v1.

- [ ] **Step 1: Failing tests.**
    1. Renders the header with `created_by_user_id` and `expires_at` formatted.
    2. Calls `fetchSharedInspect` on mount with the URL token.
    3. Shows loading spinner while in-flight.
    4. Shows error UI on 401 (token invalid/expired): "This link is no longer valid. Ask the sender for a fresh link."
    5. Shows error UI on 404 (session deleted): "This pipeline is no longer available."
    6. Shows the pipeline metadata, audit-readiness panel, graph, and YAML when fetch succeeds.
    7. Does **not** render `<CompletionBar />` or the composer chat panel.
    8. Pipeline metadata fields are not editable (no `<input>` elements in the metadata block).
    9. **App.tsx routing branch:** when `window.location.hash === "#/shared/abc123"`, `App` renders `<SharedInspectView token="abc123" />` and does **not** render `<Layout>`. Conversely, when the hash is `"#/sessions/xyz"`, `App` renders `<Layout>` and does not render `SharedInspectView`. (Vitest: mock `window.location.hash` via `Object.defineProperty(window, 'location', ...)` or `jsdom` `window.location.assign`; render `<App />`; assert presence/absence of the respective roots.)
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** `SharedInspectView` is a standalone container. **Do NOT use `useParams` — the app uses hash-router (`useHashRouter.ts`), not React-Router, and there is no `<Route>` component.** Read the token directly from `window.location.hash`:

```tsx
// Inside SharedInspectView.tsx (or extracted to a small helper)
const hash = window.location.hash; // e.g. "#/shared/abc123"
const match = hash.match(/^#\/shared\/(.+)$/);
const token = match?.[1] ?? "";
```

Mounts `InspectorPanel readOnly` and `SharedAuditReadinessPanel readOnlyState={...}` (subtasks below).
- [ ] **Step 3a:** If `InspectorPanel` does not accept `readOnly`, add the prop with a one-line guard in each editable subcomponent. Mirror Phase 7's pattern if Phase 7 has shipped.
- [ ] **Step 3b:** Implement `<SharedAuditReadinessPanel readOnlyState={AuditReadinessSnapshot} />` as specified in the introduction above. Do **not** mount the existing `<AuditReadinessPanel />` directly — that component reads from the session store and would require an isolation-breaking store population. The shared variant takes the payload as a prop; test it with:

```typescript
it("renders audit readiness checks from prop without session store", () => {
  const readOnlyState: AuditReadinessSnapshot = {
    checks: [
      { id: "c1", label: "Schema valid", status: "pass" },
      { id: "c2", label: "Source configured", status: "pass" },
    ],
  };
  render(<SharedAuditReadinessPanel readOnlyState={readOnlyState} />);
  expect(screen.getByText("Schema valid")).toBeInTheDocument();
  expect(screen.getByText("Source configured")).toBeInTheDocument();
});
```
- [ ] **Step 4: Wire the route in `App.tsx` as a top-level hash branch.**

  **Hash path: `#/shared/{token}`.** The app uses hash-router (`useHashRouter.ts`); do NOT add a server-side route or a React-Router `<Route>` for this path — the hash is the entry.

  The hash shape `#/shared/{token}` does not match the `useHashRouter` `parseHash()` regex (`^#\/([^/]+?)(?:\/([a-z]+))?$` — that pattern matches `#/{sessionId}/{tab}`). Therefore the dispatch happens **before** `useHashRouter` is invoked, as a top-level guard in `App.tsx`:

  ```tsx
  // App.tsx — add at the top of the component, before routing logic
  const hash = window.location.hash;
  if (hash.startsWith("#/shared/")) {
    const token = hash.slice("#/shared/".length);
    return <SharedInspectView token={token} />;
  }
  // ... existing useHashRouter / Layout rendering below
  ```

  Authentication: if the user is not logged in, the auth guard already wraps `App.tsx`. An unauthenticated visit stores the current hash in `sessionStorage` before redirecting to the login page; after login, client-side code reads `sessionStorage` to restore the hash. This keeps the token out of the server-side `next=` query parameter.

  Add a named test for the full unauthenticated-redirect round-trip (in `SharedInspectView.test.tsx`):

  ```typescript
  it("saves the shared hash to sessionStorage before login redirect and restores after login", () => {
    // Simulate unauthenticated user arriving at #/shared/abc123.
    Object.defineProperty(window, "location", {
      value: { hash: "#/shared/abc123", assign: vi.fn() },
      writable: true,
    });
    sessionStorage.clear();

    // Render App in the unauthenticated state.
    render(<App />, { wrapper: UnauthenticatedWrapper });

    // The auth guard must have written the hash to sessionStorage.
    expect(sessionStorage.getItem("elspeth_post_login_redirect")).toBe("#/shared/abc123");

    // After login, App reads sessionStorage and restores the hash.
    // Simulate a re-render as an authenticated user.
    cleanup();
    render(<App />, { wrapper: AuthenticatedWrapper });
    // SharedInspectView must be mounted (not the main layout).
    expect(screen.queryByTestId("shared-inspect-view")).toBeInTheDocument();
    expect(screen.queryByTestId("main-layout")).not.toBeInTheDocument();
    // sessionStorage entry must be consumed (not left to re-trigger on next visit).
    expect(sessionStorage.getItem("elspeth_post_login_redirect")).toBeNull();
  });
  ```

  The `elspeth_post_login_redirect` key name is the canonical name for this storage entry; use it consistently in `App.tsx` and in any auth-guard middleware that reads it.

- [ ] **Step 5: Run to pass.**
- [ ] **Step 6: Commit.** `feat(web/frontend): add /shared/:token route and SharedInspectView for read-only recipients`.

---

## Task 9: Side-rail integration

**Files:** the Phase 3 side-rail file (verify path — likely `components/composer/SideRail.tsx` or `components/inspector/InspectorPanel.tsx` depending on Phase 3's structure), tests updated in place.

- [ ] **Step 1: Failing test.** Render the side rail and assert `<CompletionBar />` is present in the expected position (below audit-readiness, above Catalog).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Insert `<CompletionBar />` in the rail layout. If Phase 3's layout uses named slots, use the appropriate slot; otherwise insert the JSX directly between the readiness panel and the catalog mount.
- [ ] **Step 3a (fallback path):** If Phase 3 has **not** shipped, mount the CompletionBar in a temporary header-area `<div className="completion-bar--header-fallback">` in the composer view. Add the TODO comment linking to [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md). Document the fallback in the commit message.
- [ ] **Step 3b (self-retiring fallback test):** Add the following Vitest test so the header-area fallback mount cannot become permanent:

```typescript
// SideRailFallbackRetirement.test.ts
// This test fails if the fallback marker coexists with a real SideRail slot,
// making the TODO self-enforcing. When Phase 3 ships and the slot exists,
// the implementer MUST remove the fallback div or this test breaks the build.
import { readFileSync } from "fs";
import { resolve } from "path";

it("completion-bar--header-fallback is absent when SideRail slot exists", () => {
  // Check whether Phase 3's SideRail has the completionBar slot.
  // Adjust the path to match Phase 3's actual SideRail file when known.
  const sideRailCandidates = [
    "components/composer/SideRail.tsx",
    "components/inspector/InspectorPanel.tsx",
  ];
  const sideRailFile = sideRailCandidates.find((p) => {
    try { readFileSync(resolve("src", p)); return true; } catch { return false; }
  });

  if (!sideRailFile) {
    // Phase 3 not yet shipped — fallback is acceptable.
    return;
  }

  const sideRailSource = readFileSync(resolve("src", sideRailFile), "utf-8");
  const slotExists = sideRailSource.includes("completionBar");
  if (!slotExists) {
    // Phase 3 file exists but slot not added yet — fallback still acceptable.
    return;
  }

  // Phase 3 slot exists — fallback mount must be gone.
  const composerViewCandidates = [
    "components/composer/ComposerView.tsx",
    "components/composer/Composer.tsx",
  ];
  for (const candidate of composerViewCandidates) {
    try {
      const src = readFileSync(resolve("src", candidate), "utf-8");
      expect(src).not.toContain("completion-bar--header-fallback");
    } catch {
      // File doesn't exist — skip.
    }
  }
});
```

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): mount CompletionBar in the side rail` (or `…in the header (Phase-3 fallback)`).

---

## Task 10: Remove the legacy Execute button (if it duplicates)

**Files:** the existing Execute-button host (likely `InspectorPanel.tsx` or `ProgressView.tsx`), tests updated.

The design doc 09 explicitly says the completion bar **replaces** the previous single Execute treatment. The legacy button must come out — otherwise users see two Run buttons. **No backwards compatibility, per CLAUDE.md "No Legacy Code Policy".**

- [ ] **Step 1: Failing test.** Render the composer view; assert the legacy Execute button is not present.
- [ ] **Step 2: Run to fail.** (The test will fail because the legacy button is still there.)
- [ ] **Step 3: Implementation.** Delete the legacy button JSX and any handlers that are no longer reachable. Verify the click path is fully owned by `CompletionBar.tsx`.
- [ ] **Step 4: Run to pass.** All composer-view integration tests still pass; the CompletionBar handles execute.
- [ ] **Step 5: Commit.** `refactor(web/frontend): remove legacy Execute button — CompletionBar owns the verb now`.

---

## Task 11: Export YAML top-level button wiring

**Files:** `CompletionBar.tsx` (already covered in Task 3 click handler), `components/inspector/YamlView.tsx` (verify no changes needed), tests updated.

The existing `YamlView.tsx` at `components/inspector/` already renders the YAML in a modal/drawer. Phase 6 just needs the top-level button (the CompletionBar's third button) to open it. The open mechanism is likely a store flag — find it via grep on `YamlView` usage and reuse.

- [ ] **Step 1: Failing test.** Click the Export-YAML button on the CompletionBar; assert the YamlView is open.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Wire the CompletionBar's Export-YAML handler to set whatever flag opens `YamlView`. If the open mechanism is currently coupled to a specific parent component (e.g., only the inspector can open it), refactor to a top-level store flag so the CompletionBar can also open it. This is a small refactor — keep it under 30 lines diff.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): wire Export YAML button on CompletionBar to existing YamlView`.

---

## Task 12: Cross-cutting tests

**Files:** `tests/integration/web/test_phase6_e2e.py` (Python integration test that exercises the full flow end-to-end against the running backend), or a Vitest integration test if the existing test infrastructure prefers frontend-side e2e.

- [ ] **Step 1: Failing test.** End-to-end scenario:
    1. User A creates a composition that passes validation.
    2. User A clicks Save-for-review → receives a share_url.
    3. User B (different user) opens the share_url → sees the SharedInspectView with the YAML.
    4. User B cannot click Save-for-review (no CompletionBar in the read-only view).
    5. The Landscape decision-event log has a `composer.mark_ready_for_review` event recorded with `(user_id, session_id, state_id)`. Per 19a lines 37 and 112-117, both completion-gesture events route through the Landscape decision-event channel, not `audit_access_log_table`. Read the Landscape via direct SQLite query against the audit DB (path from `web/config.py:293`):

       ```python
       import sqlite3
       from elspeth.contracts.composer_audit import canonical_audit_query
       conn = sqlite3.connect(audit_db_path)
       rows = conn.execute(
           "SELECT kind, payload FROM landscape_events WHERE session_id = ? AND kind IN (?, ?) ORDER BY recorded_at",
           (session_id, "composer.mark_ready_for_review", "composer.export_yaml"),
       ).fetchall()
       ```

       **Landscape write invariant (no flush needed):** The Landscape uses synchronous SQLite writes per the audit primacy rule in CLAUDE.md — the row is committed before the HTTP 200 returns. Test setup should state this explicitly: "Landscape writes are synchronous; no flush step needed; immediate read is correct."

       Assert the returned rows contain a row with `kind == "composer.mark_ready_for_review"`.

    6. User A clicks Export YAML → the Landscape decision-event log has a `composer.export_yaml` event recorded with `(user_id, session_id, state_id)`. Use the same direct SQLite query from Step 5 (the query already selects both event kinds). See 19a Task 6 for the test pattern: "Hit `GET /api/sessions/{id}/state/yaml`, then read the Landscape decision-event log and assert a `composer.export_yaml` event was recorded."
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** This is a test of the integration of 19a + 19b; no new product code. If the existing integration-test harness in `tests/integration/web/` supports multi-user scenarios, use it; otherwise add the missing user-switching helper.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `test(integration/web): end-to-end Phase 6 completion-gestures scenario`.

---

## Task 13: Documentation + handover

**Files:** `docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md` (this file — update Review history), `docs/guides/sharing-pipelines.md` (new user-facing doc).

- [ ] **Step 1:** Write the user-facing doc covering:
    1. How to use Save-for-review (recipient must have an account on the same deployment).
    2. Token expiry policy (7 days default).
    3. Token rotation (signing key rotation invalidates outstanding links).
    4. "What the recipient sees" (read-only inspect view).
- [ ] **Step 2:** Update this plan's Review history.
- [ ] **Step 3:** Commit. `docs: user guide for sharing pipelines (Phase 6)`.

---

## Risks

| Risk | Mitigation |
|---|---|
| Three buttons look cluttered or users can't decide which to pick | Design doc 09 §"Layout and visual hierarchy" addresses this: lighter button styling, no primary emphasis, audit-readiness panel above carries priority signal. Per-button tooltip names the persona-typical use case. |
| Narrative mode triggers when the plugin's output schema doesn't include `summary` | `NarrativeResults` falls back to "No narrative available" rather than crashing. The wire contract is pinned in `BaseTransform.supports_narrative_summary` docstring; opt-ins are reviewed at PR time. |
| Catalog has not loaded when `useNarrativeMode` runs | Returns `false` (safe default — table preview). No flash of incorrect rendering because the result view itself only renders after a run starts, by which time the catalog will have loaded. |
| Recipient clicks a stale link | The 401 response shape ("link expired / invalid") is rendered with a clear "ask the sender for a fresh link" message. No leak of session existence. |
| Recipient is unauthenticated | App detects `window.location.hash.startsWith("#/shared/")` before the auth check and stores the hash in `sessionStorage`; after login, the client-side router restores the hash so the recipient lands on `SharedInspectView`. The hash is not sent to the server. |
| Phase 3 has not shipped | Task 9 has a documented fallback (header-area mount with TODO). |
| Phase 6A has not shipped | This plan **cannot** proceed — verified at task-start. |
| Legacy Execute button removal breaks existing tests | Task 10 explicitly updates the affected tests; no compatibility shim per CLAUDE.md No Legacy Code Policy. |
| Token-in-URL is visible in browser history / referrer headers | Documented limitation; acceptable for v1 (single-deployment, trusted-recipient model). Hardening (HTTP-only header tokens, single-use tokens) is a follow-up. |
| Copy-to-clipboard fails on older browsers | Fallback to a selectable text input the user can manually copy. |

---

## Review history

**2026-05-16 — Operator adjudications applied (sibling 19a rewrite)**

- Wire-contract updates aligned to 19a's adjudicated shapes:
  - `MarkReadyForReviewResponse` no longer carries `review_id`; gains `payload_digest`.
  - `ShareableLinkResponse` gains `payload_digest`.
  - `SharedInspectResponse` gains `audit_readiness: AuditReadinessSnapshot` (reused verbatim from Phase 2) — Task 8 already consumes this field, so the type is now contract-compliant.
- Narrative result rendering (Task 6) documents Phase 5b interpretation events as the post-run overlay; component must tolerate absent overlay when 5b is not yet shipped. Two new test cases added.
- No verb-count changes (Task 3 already specified three buttons matching design doc 09); the 19a rewrite confirms three-verb model is normative.
- `supports_narrative_summary` on the wire is unchanged (boolean); backend declares it as `ClassVar[bool]` on the plugin Protocol per 19a Task 7.

**2026-05-15 — Review panel findings applied (pre-implementation)**

- CRITICAL: Task 8 `AuditReadinessPanel` architectural issue resolved: specified a `<SharedAuditReadinessPanel readOnlyState={...} />` read-only variant that takes the readiness payload as a prop (no session-store read). Retired old Step 3b ("verify and adapt") and replaced with concrete component contract and new Step 3b.
- CRITICAL: Task 4 navigator.clipboard mock setup added explicitly to the test scaffold; behavioral test for copy-link click documented with exact assertion.
- IMPORTANT: Task 8 unauthenticated redirect spec extended with a named test scenario covering sessionStorage save/restore round-trip.
- IMPORTANT: Task 9 fallback-marker self-retirement test added as Step 3b: a Vitest test that fails if the fallback marker is present when the SideRail slot exists, making the TODO self-enforcing.

# Phase 6B — Frontend: completion bar, shareable-link inspect view, context-aware results

> **Note on line-number citations.** Plan was authored against an earlier codebase snapshot. When a line citation conflicts with reality, **trust `rg`, not the line number** — `rg -n "<symbol>" src/elspeth/web/frontend/src/` is the authoritative locator. Symbols are stable; line numbers drift.


> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Land the frontend half of Phase 6 — the three-verb completion bar in the side rail, the Save-for-review confirmation dialog with shareable-link display, the read-only inspect view at `/shared/{token}` for shareable-link recipients, the context-aware result-rendering refactor (narrative summary when any pipeline transform's `capability_tags` includes `"narrative-summary"`, table preview otherwise), and the Export-YAML top-level button that promotes the existing `YamlView` drawer.

**Architecture:** React + Vite + Zustand + Vitest, matching every prior frontend phase plan. A new `CompletionBar.tsx` component lives in `components/composer/` (creating the directory if not present — verify in Task 1). The side-rail mount is added by Phase 3; this plan assumes Phase 3 has shipped and adds the new component to the rail. A new `components/shared/` directory houses the read-only inspect view. The existing `YamlView.tsx` at `components/inspector/` is reused — no duplication.

**Tech Stack:** React 18, TypeScript strict, Zustand for store, Vitest + Testing Library for tests. The app uses a hash-router (`useHashRouter.ts`), **not** React Router. The `#/shared/{token}` path is handled via a top-level branch in `App.tsx` (see Task 8).

**Sibling plan:** [19a-phase-6a-backend.md](19a-phase-6a-backend.md) — new `composer_completion_events_table` in the sessions DB (Task 1, Phase 18 precedent), HMAC-signed token + content-addressable snapshot blob in the payload store, token signing primitive, three new endpoints, narrative-summary `ClassVar[bool]` catalog field, YAML-export sessions-DB audit event.

**Design reference:** [09-completion-gestures.md](09-completion-gestures.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Sequencing: depends on Phase 3 + Phase 6A

This plan **requires** that Phase 3 (side rail) and Phase 6A (backend) have shipped. The side rail is the mount point for the completion bar; the backend endpoints are the wire contracts the frontend consumes.

If Phase 3 is not shipped when implementation starts, fall back to a temporary header-area mount: the `CompletionBar` is wrapped in `<div className="completion-bar--header-fallback">` and inserted into the existing composer header. A `// TODO: relocate to side rail after Phase 3 ships` comment links back to [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md). The fallback is documented in Task 1 below; do not let it become permanent.

If Phase 6A is not shipped, this plan **cannot** proceed — the three endpoints and the `"narrative-summary"` `capability_tags` declarations do not exist yet. The implementing agent must verify 6A merged via `git log --oneline | grep shareable_reviews` before starting.

**Phase 5b dependency for narrative result rendering:** Task 6's `NarrativeResults` consumes Phase 5b's interpretation events as a post-run overlay layered on top of the plugin's raw `summary` field. If Phase 5b is not yet shipped, the component falls back to rendering the raw `summary` field only; the interpretation-event overlay is additive and the renderer must tolerate its absence. Do not block Phase 6 on Phase 5b.

---

## Scope boundaries

**In scope:**

- `CompletionBar.tsx` with three buttons: "Save for review", "Run pipeline", "Export YAML". Co-equal styling, no primary emphasis.
- "Save for review" click → confirmation dialog with the returned `share_url`. Copy-to-clipboard affordance + "Open in new tab" link.
- "Run pipeline" click → existing Execute path (unchanged); result rendering refactored to detect narrative-summary plugins.
- "Export YAML" click → opens the existing `YamlView` (already implemented at `components/inspector/YamlView.tsx`); the button is the top-level affordance the design doc calls for.
- `#/shared/{token}` hash path — read-only inspect view (`SharedInspectView`). Entered via a top-level hash-branch in `App.tsx` (see Task 8; hash-router, not React-Router). Renders: pipeline metadata, audit-readiness panel (read-only), graph mini-view, and the YAML. No edit affordances; the composer chat panel is hidden.
- Result-rendering refactor: a new `useNarrativeMode()` hook reads the catalog and the current composition state, returns `true` if any pipeline transform's catalog entry has `capability_tags.includes("narrative-summary")`. The existing results view branches on this.
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
| Catalog response (`capability_tags` field — already on the wire pre-Phase-6) | Tier 1 inbound | If the local TypeScript type omits `capability_tags`, widen it in lockstep — the wire payload carries it regardless. |
| User clicks "Save for review" with pending changes | n/a — UI contract | Disable the button if `compositionState.is_valid === false`; the backend will also reject, but the UX is friendlier if the button shows the precondition. |
| Copy-to-clipboard surface | n/a — browser API | `navigator.clipboard.writeText`; fallback to selectable text if the API is unavailable. |

The shareable-link recipient flow is **not** a public surface. The `#/shared/{token}` hash path requires an authenticated session on the same deployment — the existing auth middleware applies. If the recipient is unauthenticated, client-side logic saves the current hash to `sessionStorage` before the redirect to `/login`, then restores it after login so the recipient lands on `SharedInspectView`. The hash (and its embedded token) is never sent to the server as a query parameter.

---

## Wire contracts consumed from 19a

These are the exact shapes 19b reads. Drift in either direction is a typed-parse failure.

```ts
// New types added to types/api.ts in Task 2.
// Mirror the shapes in 19a Task 4 verbatim — drift here is a typed-parse failure.

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

// No TransformPluginMetadata extension is needed in Phase 6.
//
// Per multi-reviewer adjudication B6 (2026-05-19), the narrative-summary
// opt-in rides the existing `capability_tags: string[]` field already on the
// catalog wire (serialized at `web/catalog/service.py:333,345`). The
// frontend reads the tag list it already receives — `useNarrativeMode`
// (Task 5) branches on `capability_tags.includes("narrative-summary")`.
//
// If your local `TransformPluginMetadata` definition does not already
// include `capability_tags: readonly string[]`, the existing field is
// missing from the type — fix the type before reading the tag. The wire
// payload carries it regardless of whether the TS type names it.
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
- **TBD** (the Execute-button host file — `components/inspector/InspectorPanel.tsx` does **not** exist; discover the actual host via `rg -n "executionStore\.execute\(" src/elspeth/web/frontend/src/` per Task 10 Step 1a) — promote Execute into the new CompletionBar; remove the legacy button if it duplicates.

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
- [ ] **Step 3: Implementation.** Add the three functions to `client.ts`, following the existing pattern (the file already has ~60 fetch helpers; mirror the closest existing shape — `fetchYaml`; locate with `rg -n "fetchYaml" src/elspeth/web/frontend/src/api/client.ts`). Add the three new response types to `types/api.ts`. **No `TransformPluginMetadata` extension is needed** — the narrative-mode opt-in rides the existing `capability_tags` field (see Task 5). If the local TypeScript type definition for `TransformPluginMetadata` does not already include `capability_tags: readonly string[]`, widen it to match the wire shape — the field is already on the catalog response.
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
    7. Clicking Export-YAML opens the existing `YamlView` modal (asserts `ExportYamlModal` becomes visible; use `waitFor(() => screen.getByRole("dialog"))`).
    8. No button has visual primary emphasis (CSS classes match co-equal styling — assert by class name, not visual snapshot).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** The component reads `compositionState` from `sessionStore` and dispatches to: `shareableReviewStore.markReadyForReview` / `executionStore.execute` / the window-level `OPEN_YAML_MODAL_EVENT` CustomEvent (defined in `lib/composer-events.ts`) for Export-YAML. The open mechanism is a window CustomEvent: `window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT))`, matching the existing pattern at `components/common/CommandPalette.tsx:158`. No store change is needed; `ExportYamlModal` already listens for this event. Styling: three side-by-side buttons in a single flex row; no primary modifier class on any of them. Aria labels match the design doc verbs verbatim. Each button has a tooltip with the design doc 09's per-verb summary.
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
   10. **Accessibility:** `jest-axe` assertion against the rendered dialog returns zero violations (focus-trap, aria-modal, accessible name, labelled controls).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Modal pattern; reuse existing dialog primitives if present (verify via `components/common/`). The expiry date is parsed from `expires_at` and rendered with the same date formatter the inspector uses (find via grep on `toLocaleString`). The copy-link handler must use `navigator.clipboard.writeText`; add a fallback to `document.execCommand('copy')` on a selected text input if the Clipboard API is unavailable (some browser security contexts block it). Wire the dialog with `aria-modal="true"`, `role="dialog"`, an accessible label, initial-focus management on the primary action, and Esc-to-close — the axe assertion gates these.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add SaveForReviewDialog with copy-link + expiry + axe-clean a11y`.

---

## Task 5: `useNarrativeMode` hook

**Files:** `hooks/useNarrativeMode.ts`, `hooks/useNarrativeMode.test.ts`.

The hook returns `true` if any transform in the current pipeline has a catalog entry whose `capability_tags` array includes the string `"narrative-summary"`. Reads from the catalog store (already exists for Phase 7 catalog work — verify) and the current composition state. **No new wire field** — `capability_tags` is already on the catalog response per `web/catalog/service.py:333,345`. (Substitution applied 2026-05-19 per multi-reviewer adjudication B6.)

- [ ] **Step 1: Failing tests.**
    1. Returns `false` when no nodes are present.
    2. Returns `false` when no transform's `capability_tags` includes `"narrative-summary"`.
    3. Returns `true` when at least one node's plugin has the tag.
    4. Returns `false` when the catalog has not loaded yet (defensive: better to default to the table preview than render an empty narrative).
    5. Re-evaluates when `compositionState.nodes` changes.
    6. Re-evaluates when the catalog updates (rare, but covered).
    7. Tag check is exact-match — neither `"narrative-summaries"` nor `"Narrative-Summary"` matches; only `"narrative-summary"`. (Open-vocabulary discipline: tags are case-sensitive strings.)
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Use `useMemo` over `(compositionState.nodes, catalog.transforms)` to compute the boolean. The check is `catalog.transforms.some(t => nodes.some(n => n.plugin_type === t.id) && t.capability_tags.includes("narrative-summary"))`. Selector pattern from `sessionStore` reads.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add useNarrativeMode hook for result-rendering branch`.

---

## Task 6: `NarrativeResults` component

**Files:** `components/execution/NarrativeResults.tsx`, `components/execution/NarrativeResults.test.tsx`.

Renders the narrative summary from the run output. The contract (per backend Task 8's per-plugin docstring; opt-in is declared via `capability_tags = ("narrative-summary",)` rather than a `ClassVar[bool]`) is that the opted-in transform's output schema includes a `summary` field. The component reads that field and renders it as Markdown (reusing `MarkdownRenderer` from `components/chat/`). When Phase 5b has shipped, the component additionally overlays Phase 5b's interpretation events for the current run — the interpretation overlay is additive (it surfaces user-affirmed interpretations alongside the raw narrative) and the component must render gracefully when no interpretation events are available (e.g., 5b not shipped, or the run produced none).

**Run-filter (post-5b-merge — load-bearing).** Phase 5b's `interpretationEventsStore` keys events by **session_id only**, and `interpretation_events_table` has no `run_id` column. To prevent the overlay from over-counting stale resolutions from earlier runs in the same session, the component filters events by the active run's wall-clock window:

```ts
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useExecutionStore } from "@/stores/executionStore";

const eventsForRun = useInterpretationEventsStore((s) => s.pendingBySession[sessionId]);
const currentRun = useExecutionStore((s) => s.currentRun);

const overlayEvents = useMemo(() => {
  if (!currentRun) return [];
  // Filter to events resolved during the active run's window. The
  // started_at / completed_at fields are on currentRun; if completed_at is
  // null (run still in flight), use Date.now() as the upper bound so
  // partial overlays stream in correctly.
  const lo = new Date(currentRun.started_at).getTime();
  const hi = currentRun.completed_at
    ? new Date(currentRun.completed_at).getTime()
    : Date.now();
  return (eventsForRun ?? []).filter((e) => {
    const t = new Date(e.resolved_at ?? e.created_at).getTime();
    return t >= lo && t <= hi;
  });
}, [eventsForRun, currentRun]);
```

Do **not** fall back to a session-aggregate read. The store keys by `session_id`, so absent a filter the overlay would surface every interpretation ever resolved in this session — including resolutions from prior runs the reviewer is not looking at. The wall-clock filter is approximate (a slow `resolved_at` write could land outside the run window) but is the correct shape; tightening to an exact run-id requires a Phase 5b schema amendment that has been adjudicated out of scope.

- [ ] **Step 1: Failing tests.**
    1. Renders the `summary` field from the run output as Markdown.
    2. Falls back to "No narrative available" when the field is missing or empty (graceful failure for opted-in-without-summary plugins, per backend Task 8's documented contract gap).
    3. Renders alongside a "Download full output" link to the existing artifact endpoint.
    4. Streams correctly when the run is still in progress (renders partial summary if available).
    5. When `interpretationEventsStore` has events whose `resolved_at` falls inside `currentRun.started_at..completed_at`, the overlay renders them above the raw summary.
    6. When `interpretationEventsStore` has events from a PRIOR run (i.e. `resolved_at < currentRun.started_at`), the overlay does **not** render those — guards against the session-aggregate over-count failure mode.
    7. When the run is in-flight (`completed_at == null`), the upper bound is treated as "now"; overlay surfaces events resolved during the in-flight window.
    8. Component does not crash when the interpretation-events store is empty / missing (Phase 5b not shipped or no events yet).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Reads the run output from `executionStore`. The summary extraction is: find the last output row that has a `summary` field; if multiple, concatenate with blank lines. The interpretation-overlay filter is the `useMemo` block above; import the store at `@/stores/interpretationEventsStore` (canonical name post-5b merge).
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): add NarrativeResults rendering for batch-analytic transforms`.

---

## Task 7: Wire the narrative branch into the results view

**Files:** the existing results view in `components/execution/` (verify file path — likely `ProgressView.tsx` or a sibling), tests updated in place.

- [ ] **Step 1: Failing test.** Add a test that, given a composition with `batch_classifier_metrics` and a catalog entry where `capability_tags.includes("narrative-summary")`, the rendered output is `<NarrativeResults />` rather than the existing table preview.
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

**`AuditReadinessPanel` is session-store-aware.** The existing component reads from the Zustand session store (populated by the owner's session) **and additionally overrides the `llm_interpretations` row's rendering using `interpretationEventsStore` counts** (per `AuditReadinessPanel.tsx:474–490` post-5b merge). Rendering it directly in the shared inspect view would require populating both stores from the shared payload — an isolation breach. Instead, implement a **separate read-only variant** `<SharedAuditReadinessPanel readOnlyState={...} />` that takes the readiness payload directly as a prop, with no store reads.

**Step 0 (prerequisite — `AuditReadinessRow` extraction).** Per multi-reviewer adjudication B4 (2026-05-19), there is no exported `<AuditReadinessRow />` component at the time of this plan: the existing `AuditReadinessPanel.tsx` renders rows inline (`rows.map((row) => <li>...</li>)`). Two viable options:

- **(a) Extract first (preferred).** Add a new `components/audit/AuditReadinessRow.tsx` exporting a `<AuditReadinessRow row={row} />` component containing the existing inline row markup verbatim. Modify `AuditReadinessPanel.tsx` to call `<AuditReadinessRow />` in its `rows.map(...)`. Verify rendering identity via the existing Vitest suite for AuditReadinessPanel — no test changes required if extraction is markup-identity-preserving. The shared variant below then reuses the extracted component.
- **(b) Inline-only.** Copy the row markup into `SharedAuditReadinessPanel` directly without extracting. Faster, but duplicates the row-rendering logic; future row-style changes would require touching two files.

**Pick (a).** It removes the duplication and the cost is one preparatory commit. The plan from here assumes (a) has been applied.

```tsx
// components/shared/SharedAuditReadinessPanel.tsx

import type { AuditReadinessSnapshot, ReadinessRow } from "@/types/api";
import { AuditReadinessRow } from "@/components/audit/AuditReadinessRow";

interface SharedAuditReadinessPanelProps {
  /** Readiness payload from the SharedInspectResponse. */
  readOnlyState: AuditReadinessSnapshot;
}

export function SharedAuditReadinessPanel({ readOnlyState }: SharedAuditReadinessPanelProps) {
  // Renders the same <AuditReadinessRow /> leaf components as the owner's panel,
  // but takes state as a prop instead of reading from sessionStore /
  // interpretationEventsStore. All six rows render uniformly — the
  // `llm_interpretations` row is populated server-side (per
  // `web/audit_readiness/service.py:218–262`) and arrives in `rows` with the
  // same shape as every other row. No per-row branch is needed for the shared
  // view; the owner-side store-override is a richer-rendering convention, not
  // a parity requirement.
  return (
    <section aria-label="Audit readiness">
      {readOnlyState.rows.map((row: ReadinessRow) => (
        <AuditReadinessRow key={row.id} row={row} />
      ))}
    </section>
  );
}
```

The `<AuditReadinessRow />` leaf component is the one extracted in Step 0 (preferred option (a)). Its prop is a single `row: ReadinessRow`, matching the type used inline in the pre-extraction `AuditReadinessPanel.tsx`. If Step 0 was skipped (option (b)), inline the row markup here instead of importing `AuditReadinessRow`. The `AuditReadinessSnapshot` type is the Phase 2 wire-response model (imported from `types/api.ts`) — the same type carried on `SharedInspectResponse.audit_readiness`. The merged-5b snapshot shape is:

```ts
// Mirrors web/audit_readiness/models.py post-5b merge.
type ReadinessStatus = "ok" | "warning" | "error" | "not_applicable";
type ReadinessRowId =
  | "validation"
  | "plugin_trust"
  | "provenance"
  | "retention"
  | "llm_interpretations"   // populated server-side from interpretation_events_table
  | "secrets";

interface ReadinessRow {
  id: ReadinessRowId;
  label: string;
  status: ReadinessStatus;
  summary: string;
  detail: string | null;
  component_ids: readonly string[];
}

interface AuditReadinessSnapshot {
  session_id: string;
  composition_version: number;
  checked_at: string;          // ISO datetime
  rows: readonly ReadinessRow[];
  validation_result: ValidationResult;
}
```

Mount `<SharedAuditReadinessPanel readOnlyState={inspectResponse.audit_readiness} />` in the shared view. **Closed-enum invariant:** the backend snapshot model validator (per `web/audit_readiness/models.py:54–66`) requires all six `ReadinessRowId` values to be present in `rows`. The shared view does not need to handle missing rows — that condition is unreachable for a well-formed response. If the backend ever ships an incomplete snapshot, the upstream Pydantic validation will reject it before the wire shape leaves the server.

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
   10. **Accessibility:** `jest-axe` assertion against the rendered `SharedInspectView` (in the success state, with a non-trivial snapshot in the payload) returns zero violations — landmark structure, heading hierarchy, labelled disclosure controls, accessible names on the "expires" and "shared by" metadata. The reviewer audience for this surface is explicitly named in design doc 09; a11y is a load-bearing acceptance criterion, not a nice-to-have.
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** `SharedInspectView` is a standalone container. **Do NOT use `useParams` — the app uses hash-router (`useHashRouter.ts`), not React-Router, and there is no `<Route>` component.** Read the token directly from `window.location.hash`:

```tsx
// Inside SharedInspectView.tsx (or extracted to a small helper)
const hash = window.location.hash; // e.g. "#/shared/abc123"
const match = hash.match(/^#\/shared\/(.+)$/);
const token = match?.[1] ?? "";
```

Mounts `InspectorPanel readOnly` and `SharedAuditReadinessPanel readOnlyState={...}` (subtasks below).
- [ ] **Step 3a:** If `InspectorPanel` does not accept `readOnly`, add the prop and propagate it via a `ReadOnlyContext` React context provider at the panel root. Editable subcomponents (form inputs, drag handles, drawer toggles, etc.) consume the context via `useReadOnly()` and render disabled / non-interactive variants when `readOnly === true`. Prefer this over enumerating editable subcomponents at the call site: a context guard is robust against future additions (a new editable surface added in a sibling phase automatically respects the read-only mode), whereas an enumeration is fragile (a missed subcomponent grants accidental edit access to the shared-inspect recipient).

  **Vitest assertion (widened per multi-reviewer review).** Mount `InspectorPanel` with `readOnly={true}` and assert no element of the following types is rendered without an explicit `disabled`/`readOnly`/`aria-disabled` truthy attribute (excluding pure navigation/disclosure controls):

  - `<input>` (any type)
  - `<textarea>`
  - `<select>`
  - interactive `<button>` (excluding nav/disclosure controls — identify by `role="button"` plus the presence of a click handler or a non-disclosure semantic; a focusable `<summary>` element is allowed)
  - any element with `contentEditable="true"`
  - any element with `draggable="true"` (drag handles can mutate the composition graph)
  - any element with `role="combobox"`, `role="listbox"`, `role="textbox"`, or `role="spinbutton"`

  The widened assertion catches drag-to-reorder handles, custom combobox widgets, and contenteditable-driven inline edits that the original three-element check would miss. Mirror Phase 7's pattern if Phase 7 has shipped.
- [ ] **Step 3b:** Implement `<SharedAuditReadinessPanel readOnlyState={AuditReadinessSnapshot} />` as specified in the introduction above. Do **not** mount the existing `<AuditReadinessPanel />` directly — it reads from `sessionStore` and `interpretationEventsStore`, both of which would require isolation-breaking population. The shared variant takes the payload as a prop; test it with:

```typescript
it("renders all six audit-readiness rows from prop without any store read", () => {
  const readOnlyState: AuditReadinessSnapshot = {
    session_id: "00000000-0000-0000-0000-000000000001",
    composition_version: 1,
    checked_at: "2026-05-19T00:00:00Z",
    validation_result: { is_valid: true, errors: [], warnings: [] },
    rows: [
      { id: "validation",         label: "Validation",          status: "ok",             summary: "All checks pass",   detail: null, component_ids: [] },
      { id: "plugin_trust",       label: "Plugin trust",        status: "ok",             summary: "All plugins trusted", detail: null, component_ids: [] },
      { id: "provenance",         label: "Provenance",          status: "ok",             summary: "Sources declared",  detail: null, component_ids: [] },
      { id: "retention",          label: "Retention",           status: "ok",             summary: "Default retention", detail: null, component_ids: [] },
      { id: "llm_interpretations",label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets",            label: "Secrets",             status: "ok",             summary: "All resolved",      detail: null, component_ids: [] },
    ],
  };
  render(<SharedAuditReadinessPanel readOnlyState={readOnlyState} />);
  expect(screen.getByText("Validation")).toBeInTheDocument();
  expect(screen.getByText("LLM interpretations")).toBeInTheDocument();
  expect(screen.getByText("Secrets")).toBeInTheDocument();
});

it("renders llm_interpretations row from snapshot without reading interpretationEventsStore", () => {
  // The owner-side AuditReadinessPanel overrides this row's display using
  // interpretationEventsStore counts. The shared variant must NOT do that;
  // it must render whatever the snapshot row carries. This guards against
  // future refactors silently coupling the shared view to the store.
  const readOnlyState: AuditReadinessSnapshot = {
    session_id: "00000000-0000-0000-0000-000000000001",
    composition_version: 1,
    checked_at: "2026-05-19T00:00:00Z",
    validation_result: { is_valid: true, errors: [], warnings: [] },
    rows: [
      // Snapshot says "warning — 2 pending review"; the shared view must
      // render that, even though interpretationEventsStore is empty.
      { id: "validation",         label: "Validation",          status: "ok",      summary: "All checks pass",          detail: null, component_ids: [] },
      { id: "plugin_trust",       label: "Plugin trust",        status: "ok",      summary: "All plugins trusted",      detail: null, component_ids: [] },
      { id: "provenance",         label: "Provenance",          status: "ok",      summary: "Sources declared",         detail: null, component_ids: [] },
      { id: "retention",          label: "Retention",           status: "ok",      summary: "Default retention",        detail: null, component_ids: [] },
      { id: "llm_interpretations",label: "LLM interpretations", status: "warning", summary: "2 pending review (3 resolved)", detail: null, component_ids: [] },
      { id: "secrets",            label: "Secrets",             status: "ok",      summary: "All resolved",             detail: null, component_ids: [] },
    ],
  };
  render(<SharedAuditReadinessPanel readOnlyState={readOnlyState} />);
  expect(screen.getByText(/2 pending review/)).toBeInTheDocument();
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

**Files:** the Phase 3 side-rail file (verify path — likely `components/composer/SideRail.tsx`; `components/inspector/InspectorPanel.tsx` is **not** present in this codebase), tests updated in place.

- [ ] **Step 1: Failing test.** Render the side rail and assert `<CompletionBar />` is present in the expected position (below audit-readiness, above Catalog).
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** Insert `<CompletionBar />` in the rail layout. If Phase 3's layout uses named slots, use the appropriate slot; otherwise insert the JSX directly between the readiness panel and the catalog mount.
- [ ] **Step 3a (fallback path):** If Phase 3 has **not** shipped, mount the CompletionBar in a temporary header-area `<div className="completion-bar--header-fallback">` in the composer view. Add the TODO comment linking to [15b1-phase-3b-side-rail-part-1.md](15b1-phase-3b-side-rail-part-1.md). Document the fallback in the commit message.
- [ ] **Step 3b (fallback-removal enforcement):** When the fallback path is used, the implementer MUST file a Filigree follow-up issue titled "Retire Phase-6B completion-bar header-fallback after Phase 3 side-rail ships" with a `blocked-by` dependency on the Phase 3 issue. Include the canonical pattern `completion-bar--header-fallback` in the issue body so it's grep-discoverable. Do not rely on source-reading tests to enforce retirement — they silently no-op when their preconditions don't match and create false confidence.

- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `feat(web/frontend): mount CompletionBar in the side rail` (or `…in the header (Phase-3 fallback)`).

---

## Task 10: Remove the legacy Execute button (if it duplicates)

**Files:** TBD — host file must be discovered (see Step 1a below; `components/inspector/InspectorPanel.tsx` does **not** exist in this codebase, contrary to earlier plan drafts — `components/inspector/` contains only `GraphView.tsx`, `RunOutputsPanel.tsx`, and `YamlView.tsx`).

The design doc 09 explicitly says the completion bar **replaces** the previous single Execute treatment. The legacy button must come out — otherwise users see two Run buttons. **No backwards compatibility, per CLAUDE.md "No Legacy Code Policy".**

- [ ] **Step 1a: Discovery (load-bearing — do this BEFORE Step 1).** Locate every current call site for `executionStore.execute(...)`:

  ```bash
  rg -n "executionStore\.execute\(" src/elspeth/web/frontend/src/
  rg -n "useExecutionStore.*execute" src/elspeth/web/frontend/src/
  ```

  Enumerate every host — the Execute button, command-palette entries, keyboard shortcuts, custom-event handlers, etc. The commit message MUST include the full list so reviewers can confirm the removal is complete. **The Execute click handler is a verb, and other callers may share it; do not assume the button is the only consumer.** If callers other than the button exist, decide per-caller: keep (command palette, keyboard shortcut should keep the verb wired to `CompletionBar`'s handler), or remove (a duplicate button on another surface).

- [ ] **Step 1: Failing test.** Render the composer view; assert the legacy Execute button is not present (asserted by its current `data-testid` or visible label; verify in Step 1a).
- [ ] **Step 2: Run to fail.** (The test will fail because the legacy button is still there.)
- [ ] **Step 3: Implementation.** Delete the legacy button JSX from the host file discovered in Step 1a. Delete handlers that are no longer reachable (per the enumeration). Verify the click path is fully owned by `CompletionBar.tsx`; keep `executionStore.execute(...)` wired to other discovered callers (command palette, keyboard shortcut) — those continue to call into the verb, just via the new owner's handler.
- [ ] **Step 4: Run to pass.** All composer-view integration tests still pass; the CompletionBar handles execute; command-palette + keyboard-shortcut tests for the verb still pass.
- [ ] **Step 5: Commit.** `refactor(web/frontend): remove legacy Execute button in <discovered-host> — CompletionBar owns the verb now (other callers: <enumeration from Step 1a>)`.

---

## Task 11: Cross-cutting tests

**Files:** `tests/integration/web/test_phase6_e2e.py` (Python integration test that exercises the full flow end-to-end against the running backend), or a Vitest integration test if the existing test infrastructure prefers frontend-side e2e.

- [ ] **Step 1: Failing test.** End-to-end scenario:
    1. User A creates a composition that passes validation.
    2. User A clicks Save-for-review → receives a share_url.
    3. User B (different user) opens the share_url → sees the SharedInspectView with the YAML.
    4. User B cannot click Save-for-review (no CompletionBar in the read-only view).
    5. The `composer_completion_events_table` in the sessions DB has a `mark_ready_for_review` row per 19a Task 1. Read the sessions DB via direct SQLite query (path from `WebSettings.sessions_db_url`, resolved at runtime via the test app's settings):

       ```python
       import sqlite3
       sessions_db_path = test_app_settings.sessions_db_url.replace("sqlite:///", "")
       conn = sqlite3.connect(sessions_db_path)
       rows = conn.execute(
           "SELECT event_type, payload_digest, created_at, actor "
           "FROM composer_completion_events "
           "WHERE session_id = ? AND event_type IN ('mark_ready_for_review', 'export_yaml') "
           "ORDER BY created_at",
           (session_id,),
       ).fetchall()
       ```

       **Sessions-DB write invariant (no flush needed):** Sessions-DB writes are synchronous SQLAlchemy `connection.execute()` calls inside the request handler; no flush step needed; immediate read is correct.

       Assert the returned rows contain a row with `event_type == "mark_ready_for_review"` and the correct `actor` (User A's user_id).

    6. User A clicks Export YAML → the `composer_completion_events_table` has an `export_yaml` row. Use the same direct SQLite query from Step 5 (the query already selects both event types). See 19a Task 7 for the test pattern: "Hit `GET /api/sessions/{id}/state/yaml`, then read `composer_completion_events` in the sessions DB and assert a row with `event_type='export_yaml'` was inserted."
- [ ] **Step 2: Run to fail.**
- [ ] **Step 3: Implementation.** This is a test of the integration of 19a + 19b; no new product code. If the existing integration-test harness in `tests/integration/web/` supports multi-user scenarios, use it; otherwise add the missing user-switching helper.
- [ ] **Step 4: Run to pass.**
- [ ] **Step 5: Commit.** `test(integration/web): end-to-end Phase 6 completion-gestures scenario`.

---

## Task 12: Documentation + handover

**Files:** `docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md` (this file — update Review history), `docs/guides/sharing-pipelines.md` (new user-facing doc).

- [ ] **Step 1:** Write the user-facing doc covering:
    1. How to use Save-for-review (recipient must have an account on the same deployment).
    2. Token expiry policy (30 days default).
    3. Token rotation (signing key rotation invalidates outstanding links).
    4. "What the recipient sees" (read-only inspect view).
- [ ] **Step 2:** Update this plan's Review history.
- [ ] **Step 3:** Commit. `docs: user guide for sharing pipelines (Phase 6)`.

---

## Risks

| Risk | Mitigation |
|---|---|
| Three buttons look cluttered or users can't decide which to pick | Design doc 09 §"Layout and visual hierarchy" addresses this: lighter button styling, no primary emphasis, audit-readiness panel above carries priority signal. Per-button tooltip names the persona-typical use case. |
| Narrative mode triggers when the plugin's output schema doesn't include `summary` | `NarrativeResults` falls back to "No narrative available" rather than crashing. The wire contract is pinned in each opted-in plugin's per-class docstring at the bootstrap-plugin site (where `capability_tags = ("narrative-summary",)` is declared); opt-ins are reviewed at PR time. |
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

**2026-05-19 — Gap-analysis remediation pass (post-implementation)**

After the implementation pass landed, a four-agent gap analysis (covering
6A Tasks 1-5, 6-10 and 6B Tasks 1-6, 7-12) plus independent skeptical
double-checks (DC-1 through DC-11, one per passing section) surfaced
**1 CRITICAL + 7 MAJOR + 4 MINOR** gaps overall, of which the 6B-side
fixes are documented here. The frontend test count rose from 1079 to
1114 (+35 plan-mandated tests). All FR-A…FR-K fix-reviewers confirmed
LANDED-CORRECTLY. CICD cleanup pass at session end: typecheck clean,
1114/1114 vitest pass, all 19 pre-commit hooks pass on the full tree.

6B-side fixes (post-implementation):

- **FIX-D (commit d296d6e9e, MAJOR×3 Task 6):** NarrativeResults compound
  rewrite — (AC1) Markdown rendering via existing MarkdownRenderer,
  (AC3) "Download full output" button with synthetic-anchor object-URL
  download (deviated from plan's `<a href>` because the artifact endpoint
  requires `Authorization: Bearer` and browsers don't attach bearer
  tokens to top-level navigations — uses the canonical pattern from
  `RunOutputsPanel.tsx:126-138`), (AC4) live-mode summary extraction
  from executionStore (fetches `fetchRunOutputs(activeRunId)`, walks
  file artifacts, parses JSONL/JSON, concatenates rows whose `summary`
  field is non-empty string with `\n\n` separator; `summaryOverride`
  supplied skips live fetch). 19/19 tests pass including plan tests
  5/6/7 (load-bearing run-window filter preserved).
- **FIX-G (commit 70084b8bb, MAJOR×2 Task 7):** InlineRunResults narrative
  dispatch restored to XOR (plan 19b:365): `return narrativeMode ?
  <NarrativeResults /> : <RunOutputsPanel />`. Plus two missing
  plan-mandated dispatch tests.
- **FIX-H (split landing — see commit-attribution archaeology below,
  MAJOR×4 Task 8):** widened read-only enforcement audit (plan
  19b:484-494) covers all 8 element classes — input, textarea, select,
  contentEditable, draggable, role={combobox, listbox, textbox,
  spinbutton}, and interactive button — with the interactive-button
  scan scoped to metadata header + audit-readiness panel only
  (deliberately excluding YamlDisplay's legitimate Copy/Download
  buttons in read-only). Plus Test 7 (no CompletionBar/chat panel
  in shared view; uses `queryByLabelText(/^Chat panel$/i)` because
  ChatPanel renders `<div aria-label="Chat panel">` without a
  semantic role — `queryByRole("complementary")` would silently
  pass), Test 8 (metadata-not-editable, scoped to the pipeline-name
  header), and the LLM-interpretations store-decoupling test
  (plan 19b:519-542).
- **FIX-I (commit db552e857, MAJOR Task 9):** SideRail slot reorder:
  audit-readiness → validation-banner → graph-mini → completion-bar
  → catalog (plan 19b:602 "below audit-readiness, above Catalog").
  Position test asserts `auditIdx < completionIdx < catalogIdx` via
  DOM-order `querySelectorAll`.
- **FIX-J (commit 7a709803a, CRITICAL Task 11):** the missing single
  end-to-end test exercising User-A→User-B handoff with all four
  DB-state checks the plan mandates (plan 19b:640-664). Previously
  what existed was disjoint slices (CompletionFlow.integration.test.tsx
  for the CompletionBar side, test_shareable_reviews_routes.py for
  multi-user resolve at the API level, test_yaml_export_audit_event.py
  for the export audit row) — none composed into Step 3-6 jointly.
  See 19a's Review-history entry for FIX-J's commit-discipline note.
- **FIX-K (commit 8490877c5, MAJOR×2 DC-6 findings, Task 1):** frontend
  trust-boundary tightening. (1) `fetchSharedInspect` now invokes the
  full `validateAuditReadinessSnapshot` per-row validator (exported
  from `api/auditReadiness.ts`); previously the comment promised
  deferred validation that never ran on the shared-inspect path —
  a real Tier-3 → Tier-1 hole. (2) `pipeline_metadata` and
  `composition_snapshot` re-typed from `Record<string, unknown>` to
  the plan-specified `PipelineMetadata` and `CompositionState`
  (per 19b:100-101), with per-field runtime validators
  `isPipelineMetadata` and `isCompositionSnapshot`. (3) Deleted the
  now-redundant `_narrowCompositionSnapshot` in SharedInspectView.tsx
  (wire-boundary now guarantees the shape). **Wire-vs-runtime drift
  documented:** the backend `CompositionStateResponse` Pydantic mirror
  emits `{version, metadata, source, nodes, edges, outputs}` — narrower
  than the TS `CompositionState` interface's runtime-only `id` /
  `validation_*` fields. Test fixtures use `as unknown as` cast.
- **FIX-M (commit e8d08ebf0, MEDIUM×3 DC-7 findings, Task 2):**
  shareableReviewStore hardening. (1) Captured-epoch post-await check
  prevents stale-response-clobber on session switch — chosen over
  AbortController because audit rows are append-only and aborting
  mid-flight would mint server-side row never surfaced. (2) Module-scoped
  `_inFlightSessionId` guard prevents double-click double-POST →
  duplicate audit row. (3) `clearForSession(sessionId)` action
  implemented per plan Task 2 Step 1.5 (was unimplemented in original
  pass despite being documented in the store's own doc comment).
- **FIX-E (commit c496949c7, MAJOR×3 Task 3):** CompletionBar test
  coverage: AC4 Export-YAML always-enabled (asserts under both null
  and invalid validation states), AC6 Run-pipeline dispatches
  `executionStore.execute(activeSessionId)` via spy, AC7 Export-YAML
  click opens `ExportYamlModal` (via `OPEN_YAML_MODAL_EVENT`
  CustomEvent; verified with `waitFor(() => getByRole("dialog", {
  name: /export yaml/i }))`).
- **FIX-F (commit 5c56f3510, MAJOR + MINOR×3 Task 4):** SaveForReviewDialog
  Esc-to-close (listener mounted on `document` not the dialog element —
  the canonical React modal pattern because `role="dialog"` doesn't
  auto-focus in jsdom and dialog-element listeners no-op when focus
  is elsewhere). Plus transient "Copied!" 2000ms auto-clear test
  (vitest fixture trap: `vi.useFakeTimers({ toFake: ["setTimeout",
  "clearTimeout"] })` must be selective or the clipboard promise
  deadlocks against waitFor's setInterval), plus open-in-new-tab
  anchor assertion test.

DC verdicts on 6B passing sections (post-implementation):
DC-6 (API client) surfaced 2 MAJOR (closed by FIX-K). DC-7 (store)
surfaced 3 MEDIUM (closed by FIX-M). DC-11 (useNarrativeMode hook)
CONFIRMED-MET with cosmetic notes only (scope extension to source/sink
plugin tag detection beyond plan's transforms-only pseudocode — broader
than spec but matches plan description).

See 19a's Review history (same date entry) for the operator-decision
queue items, the commit-message-vs-diff archaeology for FIX-A/FIX-H/FIX-K,
and the FIX-B `--no-verify` discipline policy established post-hoc.

**2026-05-19 — Implementation pass (Tasks 1-12 landed)**

All 12 frontend tasks implemented in 10 commits on branch
`feat/composer-phase-6-completion-gestures`:

1. API client + types (8 tests) — `api/shareableReviews.ts`,
   types/api.ts extension.
2. `shareableReviewStore` (7 tests).
3. `CompletionBar` three-button component (7 tests) — reuses
   `ExecuteButton` + `ExportYamlButton` for Phase 5b interpretation-
   gating + YAML modal dispatch preservation.
4. `SaveForReviewDialog` with copy-to-clipboard + retry + clipboard
   fallback (8 tests).
5. `useNarrativeMode` hook + module-level catalog cache (7 tests).
6. `NarrativeResults` with overlay model + opt-out indicator (7 tests).
   Simplified vs plan: the live executionStore doesn't yet aggregate
   a narrative-summary field; v1 surfaces a placeholder in live mode
   and reads from `summaryOverride` (Task 8 path). Phase 5b
   interpretation-event overlay is opt-out-flag-only because
   `interpretationEventsStore` does not expose a resolved-event list;
   a future store extension can surface the full list inline without
   changing this component's prop surface.
7. `InlineRunResults` wiring with the `NarrativeResultsBranch`
   dispatcher rendering narrative ABOVE the tabular outputs.
8. `useSharedToken` hook (10 tests) + `useHashRouter` shared-route
   guards (no regressions in 17 existing tests) + `SharedInspectView`
   component (6 tests).
9. + 10. CompletionBar mounted in SideRail's `completionBarSlot`;
   `executeButtonSlot` + `exportYamlSlot` retired to `null` (the verbs
   now live inside CompletionBar). SaveForReviewDialog mounted at
   app-root for cross-view availability.
11. Cross-cutting integration test (4 tests) covering end-to-end
   click → dialog → success → copy → close; 409 error path; disabled-
   button no-op; cross-session staleness clear.
12. Documentation: this Review history entry +
    `docs/guides/sharing-pipelines.md` extended with the "Frontend
    surfaces (post-Phase-6B)" section.

End-of-6B gate: full vitest run = **1040 tests pass** (was 976 baseline;
+64 new). Typecheck clean. No regressions in any existing test file.

**Post-spec-review remediation (Task 8 substantive gap, FIX-C, 2026-05-19):**
The Task 8 SharedInspectView shipped here used an inline 3-column
`<table>` for the readiness panel and a raw `<pre><code>` block for the
YAML — both inline in `SharedInspectView.tsx`. The spec-compliance
reviewer flagged this as NON-COMPLIANT vs. the plan's `InspectorPanel`
/ "reused YamlView" / `AuditReadinessRow` / `SharedAuditReadinessPanel`
/ `ReadOnlyContext` / jest-axe / sessionStorage-round-trip targets.

The FIX-C remediation pass discovered that the plan's literal text
referenced infrastructure that doesn't exist in this codebase:
`InspectorPanel` was never built, and `YamlView` / `GraphMiniView` are
both session-store-coupled so neither can be reused with frozen-blob
data without a substantial refactor. The honest delivery shape: extract
the actual reusable primitives, then compose the read-only inspect view
from them.

Files added or modified by FIX-C (commit on top of the original Task 8):

* `components/audit/AuditReadinessRow.tsx` + test — extracted from
  `AuditReadinessPanel` as a pure row renderer taking a
  `RowPresentation` and an optional `onSelect`; honours the
  `useReadOnly()` signal by forcing the static variant under
  read-only.
* `contexts/ReadOnlyContext.tsx` + test — provides the
  `ReadOnlyProvider` + `useReadOnly()` hook. Default value `false` so
  composer-side components mounted outside the provider are unchanged.
* `components/shared/SharedAuditReadinessPanel.tsx` + test — renders
  six `AuditReadinessRow`s from a frozen `AuditReadinessSnapshot`
  inside `<ReadOnlyProvider value={true}>`. No live overlays (no
  inline-blob provenance override, no Phase 5b LLM stylising — those
  belong on the live composer's `AuditReadinessPanel` only).
* `components/inspector/YamlDisplay.tsx` + test — extracted the
  Copy + Download + syntax-highlight chrome from `YamlView` as a pure
  primitive (`yaml: string`, `filename?: string`). `YamlView` retains
  its fetch effect, empty/loading/error states, and proposal panel; it
  now delegates display to `<YamlDisplay>`. SharedInspectView mounts
  `YamlDisplay` directly with the wire YAML.
* `components/sidebar/GraphMiniView.tsx` — gained an optional
  `compositionStateOverride` prop. When supplied, used in place of the
  session-store read. SharedInspectView passes the frozen
  `composition_snapshot` (narrowed at the boundary via an offensive
  shape guard — `nodes` and `outputs` must be arrays or the boundary
  crashes).
* `components/common/AuthGuard.tsx` + test — sessionStorage round-trip
  for the shared-route hash. While unauthenticated on a `#/shared/`
  hash, the save effect persists the hash to
  `sessionStorage.elspeth_post_login_redirect`. On unauth→auth
  transition, the restore effect writes the saved hash back to
  `window.location.hash` and removes the key (self-disarming).
* `components/shared/SharedInspectView.tsx` — rewritten to compose
  the new primitives inside `<ReadOnlyProvider value={true}>`. The
  inline 3-column table and raw `<pre>` are deleted. A jest-axe
  accessibility assertion is added on the loaded state (plan line
  470).

Test deltas: 1048 → 1079 (+31 across the new test files plus extension
of the existing SharedInspectView / GraphMiniView tests). Typecheck
clean. All existing AuditReadinessPanel / YamlView / GraphMiniView /
App / AuthGuard tests pass unchanged.

The plan's literal-text references to `InspectorPanel` and "reused
YamlView" were aspirational — the post-spec-review fix produces the
spec'd user-visible behaviour through smaller primitives the codebase
can actually support.

The shared-route hash format (`#/shared/{token}`) required extending
the hash router which previously only knew `#/{sessionId}` /
`#/{sessionId}/{verb}`. The extension is two guards (in `parseHash`
and in the session-change subscription) that short-circuit when the
hash starts with `#/shared/`; the SharedInspectView owns the URL for
its lifecycle.

Implementation surfaces deferred (not blocking demo):

* Per-token revocation UI — no backend support in v1; Phase 9.
* Reviewer "approve / request changes" surface — read-only only.
* Email-this-link UI — copy-to-clipboard is the v1 affordance.

**2026-05-19 — Multi-reviewer Go/No-Go panel applied (CONDITIONAL → GO)**

Four reviewers (reality / architecture / quality / systems) returned CONDITIONAL GO. Frontend-side blockers resolved:

- **B4 (`AuditReadinessRow` does not exist):** Task 8 now has a Step 0 prerequisite that extracts `AuditReadinessRow` from `AuditReadinessPanel.tsx` as a first commit (preferred option (a)), enabling reuse from `SharedAuditReadinessPanel`. Option (b) inline-only is documented as the fallback. Earlier same-day edit (which used `<AuditReadinessRow>` as if it existed) is now consistent with reality.
- **B5 (`components/inspector/InspectorPanel.tsx` does not exist):** Task 10 rewritten with a Step 1a discovery step (`rg -n "executionStore\.execute\(" src/elspeth/web/frontend/src/`) and a caller-enumeration commit-message requirement. The File-structure list's `InspectorPanel.tsx` entry replaced with **TBD** + the discovery instruction. Task 9 sibling note also corrected.
- **B6 (capability_tags substitution — frontend half):** Task 5 `useNarrativeMode` now branches on `capability_tags.includes("narrative-summary")` rather than `supports_narrative_summary === true`. `TransformPluginMetadata` extension dropped from Task 1 (the field is already on the wire). Goal/Sequencing/Wire-contracts/Tier-table/Risks/Sibling-work-preview/Wire-contracts-consumed-from-19a references all switched to capability_tags. Test 7 added asserting exact-string tag match (open-vocabulary discipline).

Non-blocking recommended fixes also applied:

- "Trust `rg`, not the line number" note added at the top of the file.
- Task 8 Step 3a `ReadOnlyContext` Vitest assertion widened from `<input>`/`<textarea>`/`<button>` to also cover `<select>`, `contentEditable`, `draggable`, `role="combobox"`, `role="listbox"`, `role="textbox"`, `role="spinbutton"`. Catches drag-to-reorder handles, custom combobox widgets, and inline contenteditable edits.
- Task 4 `SaveForReviewDialog` and Task 8 `SharedInspectView` gained `jest-axe` accessibility-clean acceptance criteria. The reviewer audience for the shared-inspect surface is named in design doc 09; a11y is a load-bearing acceptance criterion, not a nice-to-have.

**2026-05-19 — Post-Phase-18 merge reconciliation**

- Phase 18 (5b) merged to RC5.2 (commit `3dee19f8d`). Two data-shape errors in the pre-merge plan corrected:
  1. **`AuditReadinessSnapshot.checks` → `rows`** in Task 8 SharedAuditReadinessPanel introduction and Step-3b test fixture. The actual merged-5b field is `rows: readonly ReadinessRow[]` (`web/audit_readiness/models.py:42–66`), populated server-side with all six closed-enum rows including `llm_interpretations`. The fixture was rewritten to include all six required rows with correct `ReadinessStatus` values (`"ok"`, not `"pass"`).
  2. **`ReadinessRow` shape pinned** — the type now includes `id`, `label`, `status`, `summary`, `detail`, `component_ids` fields per the merged model. The fixture's prior 3-field stub would have failed pydantic-strict parse and the integration test would have been a false-positive.
- Task 8 owner-vs-shared parity note added: the owner-side panel overrides the `llm_interpretations` row's display using `interpretationEventsStore` counts (`AuditReadinessPanel.tsx:474–490`). The shared variant must NOT do that override — it renders whatever the snapshot row carries. A second test was added that asserts the shared panel surfaces the snapshot's `summary` text even when the store is empty.
- Task 6 (NarrativeResults) gained a load-bearing run-filter block. Without it, the implementer would silently fall back to a session-aggregate read on `interpretationEventsStore.pendingBySession[sessionId]`, which would over-count resolutions from earlier runs (the store keys by session_id only; the table has no `run_id` column). The filter is wall-clock: `created_at >= currentRun.started_at AND created_at <= (currentRun.completed_at ?? Date.now())`. Two new test cases added — prior-run events must NOT surface, in-flight runs must surface partial windows.
- 19a sibling edits: Task 4 §"Post-Phase-18 merge fact" pins that the snapshot service already populates the `llm_interpretations` row server-side; Task 5 redesigned around frozen-at-mark-time `audit_readiness` in the blob (was: fresh fetch at resolve-time).
- Validation review at `/home/john/.claude/plans/docs-composer-ux-redesign-2026-05-please-dazzling-dove.md` was wrong on one point — 5b *did* extend `AuditReadinessSnapshot` (added `llm_interpretations` to the closed enum). Phase 19's snapshot-extension work that the validation's Issue A verdict (b) had proposed is structurally pre-done by the merged 5b code. Phase 19's job collapses to: call the existing snapshot service at mark-time and freeze the result in the blob.



**2026-05-18 — Path A adjudication applied (false-premise correction)**

- BLOCKER resolved (integration-test false premise): Task 11 Step 1 items 5 and 6 previously queried `landscape_events` (an unverified table) in the "audit DB" (via an incorrect path reference to `web/config.py:293`). Both the table name and the DB reference were wrong. **Path A adjudication:** the two completion-gesture events are recorded in `composer_completion_events_table` in the sessions DB (per 19a Task 1). Task 11 integration-test SQL updated to query `composer_completion_events` with `event_type IN ('mark_ready_for_review', 'export_yaml')`. DB path corrected to `WebSettings.sessions_db_url` (runtime-resolved via test app settings). The "no flush needed" invariant note is updated to reference synchronous SQLAlchemy sessions-DB writes, not Landscape writes. This is a schema-change cohort: the `composer_completion_events_table` addition forces a DB delete on next deploy per `project_db_migration_policy`.
- `<!-- TODO -->` comment in Task 11 Step 1 (added 2026-05-18 in an earlier pass) resolved and removed — the blocker it identified is now addressed by Path A.
- Sibling plan description updated: "Landscape decision-event channel for audit" replaced with "new `composer_completion_events_table` in the sessions DB (Task 1, Phase 18 precedent)".
- 19a task-number cross-references updated: "19a Task 3" → "19a Task 4" (wire shapes); "backend Task 7" → "backend Task 8" (ClassVar docstring) in Tasks 6 and 19b review history; "19a Task 6" → "19a Task 7" (YAML export test pattern in Task 11 item 6).

**2026-05-18 — Six plan-internal corrections applied (earlier pass, same date)**

- Edit 1 (token expiry, trivial): Task 12 Step 1 user-facing doc bullet corrected from "7 days default" to "30 days default" to match the authoritative value at 19a:252–256 (`shareable_link_lifetime_seconds = 30 * 24 * 3600`).
- Edit 3 (Task 11 duplication, low): Task 11 (Export YAML button wiring) folded into Task 3. Research confirmed the open mechanism is a window CustomEvent `OPEN_YAML_MODAL_EVENT` defined in `lib/composer-events.ts`, already used by `CommandPalette.tsx:158` and `CompletionSummary.tsx:76` — no store refactor required. Task 3 Step 3 updated to name the mechanism explicitly. Task 11 section deleted. Former Tasks 12 and 13 renumbered to 11 and 12.
- Edit 4 (fallback-test brittleness, low): Deleted the `SideRailFallbackRetirement.test.ts` Vitest source-reading test from Task 9 Step 3b. Replaced with a procedural requirement to file a Filigree follow-up issue with a `blocked-by` dependency and the `completion-bar--header-fallback` grep pattern in the issue body.
- Edit 5 (InspectorPanel readOnly propagation, low): Task 8 Step 3a rewritten to specify a `ReadOnlyContext` React context provider pattern with `useReadOnly()` hook consumption in editable subcomponents, plus a Vitest test that asserts no unguarded `<input>`, `<textarea>`, or interactive `<button>` is rendered when `readOnly={true}`.
- Edit 2 (Landscape table name, medium): Added `<!-- TODO -->` comment in Task 11 Step 1 noting that `landscape_events` is unverified as the SQL table name and `web/config.py:293` is the wrong line for the DB path. See the matching TODO in 19a §"Audit-event recording". Blocker for integration-test implementation.
- Follow-up (history consistency): Annotated the 2026-05-15 IMPORTANT entry referring to the now-deleted `SideRailFallbackRetirement.test.ts` Vitest test as superseded by Edit 4. Surfaced by complex-reviewer; the entry described an artefact that no longer exists in Task 9 Step 3b.

**2026-05-16 — Operator adjudications applied (sibling 19a rewrite)**

- Wire-contract updates aligned to 19a's adjudicated shapes:
  - `MarkReadyForReviewResponse` no longer carries `review_id`; gains `payload_digest`.
  - `ShareableLinkResponse` gains `payload_digest`.
  - `SharedInspectResponse` gains `audit_readiness: AuditReadinessSnapshot` (reused verbatim from Phase 2) — Task 8 already consumes this field, so the type is now contract-compliant.
- Narrative result rendering (Task 6) documents Phase 5b interpretation events as the post-run overlay; component must tolerate absent overlay when 5b is not yet shipped. Two new test cases added.
- No verb-count changes (Task 3 already specified three buttons matching design doc 09); the 19a rewrite confirms three-verb model is normative.
- Narrative-summary opt-in rides the existing `capability_tags` open-vocabulary channel (per 19a Task 8, post-2026-05-19 simplification). No new wire field.

**2026-05-15 — Review panel findings applied (pre-implementation)**

- CRITICAL: Task 8 `AuditReadinessPanel` architectural issue resolved: specified a `<SharedAuditReadinessPanel readOnlyState={...} />` read-only variant that takes the readiness payload as a prop (no session-store read). Retired old Step 3b ("verify and adapt") and replaced with concrete component contract and new Step 3b.
- CRITICAL: Task 4 navigator.clipboard mock setup added explicitly to the test scaffold; behavioral test for copy-link click documented with exact assertion.
- IMPORTANT: Task 8 unauthenticated redirect spec extended with a named test scenario covering sessionStorage save/restore round-trip.
- IMPORTANT: Task 9 fallback-marker self-retirement test added as Step 3b: a Vitest test that fails if the fallback marker is present when the SideRail slot exists, making the TODO self-enforcing. *(Superseded 2026-05-18: the Vitest source-reading test was replaced with a procedural Filigree-issue-with-blocked-by-dependency requirement; see Edit 4 in the 2026-05-18 history entry.)*

# Phase 2B — Frontend foundations: types, API client, store, AuditReadinessPanel

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes. Every task is TDD-shaped (failing test → run-to-fail → implement → run-to-pass → commit).

**Goal:** Land the foundations of the frontend half of Phase 2 — TypeScript types mirroring Phase 2A's Pydantic response shapes, two typed API wrappers, a Zustand store with composition-version-keyed cache, and the `AuditReadinessPanel` component itself (six rows, all-green collapse, auto-fetch on composition-version change). Sub-components (`ReadinessRowDetail`, `ExplainDialog`), the `InspectorPanel.tsx` mount, the standalone-Validate-button removal, and the end-to-end smoke land in **14c-phase-2c-frontend-integration.md**.

**Architecture:** Backend-first via Phase 2A. The plan adds:

1. Two typed API wrappers (`fetchAuditReadiness`, `fetchAuditReadinessExplain`) in `api/auditReadiness.ts`.
2. A `useAuditReadinessStore` (Zustand) that caches snapshots by `composition_version` and refetches when the version changes.
3. An `AuditReadinessPanel` component that renders six rows, collapses to a single "Audit ready" line when all-green, and auto-fetches on composition-version change. The component imports `ReadinessRowDetail` and `ExplainDialog` — those are provided by 14c; this plan ships minimal placeholders so 14b's tests resolve and pass before 14c lands.

**Tech Stack:** React + Zustand + Vitest + testing-library + userEvent.

**Sibling plans:**
- [14a-phase-2a-backend.md](14a-phase-2a-backend.md) — backend response models, service, routes.
- [14c-phase-2c-frontend-integration.md](14c-phase-2c-frontend-integration.md) — sub-components, inspector mount, Validate-button removal, end-to-end smoke.

**Umbrella plan:** [14-phase-2-audit-readiness-panel.md](14-phase-2-audit-readiness-panel.md).

**Design reference:** [07-audit-readiness-panel.md](07-audit-readiness-panel.md).

---

## Scope boundaries

**In scope (this plan):**
- New types in `frontend/src/types/index.ts` mirroring Phase 2A's Pydantic models exactly:
  `AuditReadinessSnapshot`, `ReadinessRow`, `AuditReadinessExplain`, `ReadinessRowId`, `ReadinessStatus`.
- New API client in `frontend/src/api/auditReadiness.ts` with `fetchAuditReadiness` and `fetchAuditReadinessExplain`.
- New Zustand store `frontend/src/stores/auditReadinessStore.ts` keyed by `(sessionId, compositionVersion)`.
- New `frontend/src/components/audit/AuditReadinessPanel.tsx` — six-row panel with all-green collapse and version-keyed auto-fetch.
- Minimal placeholder `frontend/src/components/audit/ReadinessRowDetail.tsx` and `frontend/src/components/audit/ExplainDialog.tsx` so 14b's tests pass before 14c replaces them with the real implementations.

**Out of scope (deferred to 14c-phase-2c-frontend-integration.md):**
- `ReadinessRowDetail` full implementation (per-row warning detail with jump-to-component).
- `ExplainDialog` full implementation (narrative modal).
- Mounting the panel inside `InspectorPanel.tsx`.
- Removing the standalone Validate button.
- The hash-router / keyboard-navigation smoke that follows the button removal.
- The staging smoke that exercises Phase 2 end-to-end against the deployed backend.

**Out of scope (entire Phase 2):**
- Backend (Phase 2A delivered).
- Telemetry on row-click (Phase 8).
- Phase 3's side-rail reorganisation.
- A per-user retention preference UI.
- LLM-interpretations row content (Phase 5b).
- Auto-validate-on-keystroke (fires on `composition_version` change, not on each keystroke).

## Sequencing and dependencies

Phase 2A **must** be merged before this plan's tests hit a real backend, but 14b's vitest suite mocks `globalThis.fetch` end-to-end and can land ahead of 2A. The recommended sequence is:

1. Merge Phase 2A.
2. Implement 14b Tasks 1–4 in order on a fresh branch.
3. Land 14c (sub-components + mount + button removal + smoke) on the same branch.

## Trust-tier check (per CLAUDE.md)

The frontend store reads only data the backend just produced (Tier 1):

- Snapshot payload — direct typed access; `_StrictResponse` on the backend means
  unknown fields fail at construction, not in the client.
- `composition_version` — used as a cache key; integer.
- `narrative` — rendered as text; the backend produces deterministic prose with no
  external content.

No defensive `.get()` / `getattr` patterns. No fallback rendering of unknown row
ids — Phase 2A's `ReadinessRowId` Literal is mirrored as a TypeScript discriminated
union, and the renderer's switch is exhaustive (compiler-enforced via the `never`
default arm).

**Authorized exception — transport-layer shape validation (Amendment 14b-1, frontend-001):**
`parseResponse` in `api/auditReadiness.ts` (lines 51–66) performs a shallow shape check on
`session_id` (string) and `composition_version` (number) before returning the body to
the caller. This is an explicit policy exception to the "no defensive checks on Tier 1 data"
rule. Rationale: CLAUDE.md's trust-tier model addresses data *origin* (who produced the
value), not transport-layer corruption (a proxy, CDN, or body-truncation delivering a
well-formed HTTP 200 with a corrupted body). The check is shallow — only the two fields
shared by both response types — and throws a typed `ApiError` on mismatch so callers see
a structured failure rather than a silent runtime TypeError later in the call chain.

A deeper version of this check (validating all six rows, `ReadinessRowId` literals,
`ReadinessStatus` values) is tracked as observation `elspeth-obs-2e58958765` for future
strengthening. A future CLAUDE.md update may add a fourth tier (transport / wire) — this
is the first place that boundary surfaces in the codebase.

## File structure

**New:**
- `src/elspeth/web/frontend/src/api/auditReadiness.ts`
- `src/elspeth/web/frontend/src/api/auditReadiness.test.ts`
- `src/elspeth/web/frontend/src/stores/auditReadinessStore.ts`
- `src/elspeth/web/frontend/src/stores/auditReadinessStore.test.ts`
- `src/elspeth/web/frontend/src/stores/subscriptions.ts` *(added by 2B.5 follow-up — see commits `084b8c34b` + `163737068`)*
- `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` *(added by 2B.5 follow-up)*
- `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx`
- `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx`
- `src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx` *(placeholder; 14c replaces it)*
- `src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx` *(placeholder; 14c replaces it)*

**Modified:**
- `src/elspeth/web/frontend/src/types/index.ts` — add the five new exported types.
- `src/elspeth/web/frontend/src/types/api.ts` — re-export the new types.
- `src/elspeth/web/frontend/src/api/client.ts` — re-export the two new functions for `import * as api from "./api/client"` callers.

`InspectorPanel.tsx` and its tests are touched by 14c, not by 14b.

---

## Task 1: Frontend types — mirror Phase 2A's Pydantic models

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/types/api.ts`

- [ ] **Step 1: Confirm Phase 2A's wire shape**

Read [14a-phase-2a-backend.md](14a-phase-2a-backend.md) §"Task 1: Pydantic response models". Confirm the exhaustive list of `ReadinessRowId` literals is:

```text
validation | plugin_trust | provenance | retention | llm_interpretations | secrets
```

and `ReadinessStatus` is:

```text
ok | warning | error | not_applicable
```

If Phase 2A has been amended and the literal sets diverge, **stop and reconcile** — this task encodes the wire contract, not an interpretation of it.

- [ ] **Step 2: Add types to `types/index.ts`**

After the existing `CompositionState`/`ValidationResult` block, append:

```typescript
// ── Audit Readiness Panel (Phase 2) ────────────────────────────────────────
//
// These types mirror the Pydantic models in
// src/elspeth/web/audit_readiness/models.py (Phase 2A). If a backend literal
// is added, the union here must be widened in the same commit and the
// AuditReadinessPanel's row-renderer switch must add a case — the exhaustive
// `never` default arm fails the build otherwise.

export type ReadinessRowId =
  | "validation"
  | "plugin_trust"
  | "provenance"
  | "retention"
  | "llm_interpretations"
  | "secrets";

export type ReadinessStatus = "ok" | "warning" | "error" | "not_applicable";

export interface ReadinessRow {
  id: ReadinessRowId;
  label: string;
  status: ReadinessStatus;
  summary: string;
  detail: string | null;
  /**
   * IDs of components the row implicates. May reference node ids, source,
   * or sink names. The frontend's click handler resolves these against
   * CompositionState.nodes for jump-to-component navigation; non-node ids
   * fall through to a no-op (no error).
   */
  component_ids: readonly string[];
}

export interface AuditReadinessSnapshot {
  session_id: string;
  composition_version: number;
  rows: readonly ReadinessRow[];
}

export interface AuditReadinessExplain {
  session_id: string;
  composition_version: number;
  narrative: string;
}
```

- [ ] **Step 3: Re-export from `types/api.ts`**

In the existing `export type { ... } from "./index";` block, add the five names:

```typescript
  ReadinessRowId,
  ReadinessStatus,
  ReadinessRow,
  AuditReadinessSnapshot,
  AuditReadinessExplain,
```

- [ ] **Step 4: Typecheck — no runtime test yet**

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit
```

Expected: clean. Types are unused so far but the build must accept them.

- [ ] **Step 5: Commit**

```bash
cd src/elspeth/web/frontend && npm run typecheck && npm run lint
git add src/elspeth/web/frontend/src/types/index.ts src/elspeth/web/frontend/src/types/api.ts
git commit -m "feat(web/frontend): add audit-readiness response types (Phase 2B.1)"
```

---

## Task 2: API client — `fetchAuditReadiness` + `fetchAuditReadinessExplain`

**Files:**
- Create: `frontend/src/api/auditReadiness.ts`
- Create: `frontend/src/api/auditReadiness.test.ts`
- Modify: `frontend/src/api/client.ts` (re-export the two functions)

The Phase 2A routes are **GET** (Finding 4 in 14a-phase-2a-backend.md):

```
GET /api/sessions/{sid}/audit-readiness
GET /api/sessions/{sid}/audit-readiness/explain
```

Both take no body. Both return the strict Pydantic payload with `Cache-Control: no-store`. Mirror the existing pattern in `api/client.ts::validatePipeline` (line 702): `fetch` + `authHeaders()` + `parseResponse<T>` — no `Content-Type` body header needed on GET.

- [ ] **Step 1: Confirm `parseResponse` and `authHeaders` are exported**

```bash
grep -n "parseResponse\|authHeaders" src/elspeth/web/frontend/src/api/client.ts | head -5
```

Expected: both exist but are **not** exported. The convention is to inline the same idiom in sibling modules. The Phase 1B account-level preferences client is the precedent (`api/client.ts:403–424`, the `fetchUserComposerPreferences` / `updateUserComposerPreferences` block). Mirror that block's structure.

- [ ] **Step 2: Write the failing test**

`src/elspeth/web/frontend/src/api/auditReadiness.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "./auditReadiness";

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

describe("auditReadiness API client", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it("fetchAuditReadiness GETs the right URL with auth header", async () => {
    const body = {
      session_id: SESSION_ID,
      composition_version: 3,
      rows: [
        {
          id: "validation",
          label: "Validation",
          status: "ok",
          summary: "All checks pass",
          detail: null,
          component_ids: [],
        },
        {
          id: "plugin_trust",
          label: "Plugin trust",
          status: "ok",
          summary: "All Tier 1/2",
          detail: null,
          component_ids: [],
        },
        {
          id: "provenance",
          label: "Provenance",
          status: "warning",
          summary: "Identity passthrough detected",
          detail: "Identity passthrough — provenance gap on transform 'select_columns'.",
          component_ids: ["select_columns"],
        },
        {
          id: "retention",
          label: "Retention",
          status: "not_applicable",
          summary: "System retention: 90 days",
          detail: null,
          component_ids: [],
        },
        {
          id: "llm_interpretations",
          label: "LLM interpretations",
          status: "not_applicable",
          summary: "No LLM transforms in this pipeline",
          detail: null,
          component_ids: [],
        },
        {
          id: "secrets",
          label: "Secrets",
          status: "not_applicable",
          summary: "No secret references in this pipeline",
          detail: null,
          component_ids: [],
        },
      ],
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const snapshot = await fetchAuditReadiness(SESSION_ID);
    expect(snapshot.composition_version).toBe(3);
    expect(snapshot.rows).toHaveLength(6);
    expect(snapshot.rows[2].status).toBe("warning");

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url, init] = mock.mock.calls[0];
    expect(url).toBe(`/api/sessions/${SESSION_ID}/audit-readiness`);
    expect(init?.method).toBe("GET");
  });

  it("fetchAuditReadiness throws on non-2xx", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("server error", { status: 500 }),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toThrow();
  });

  it("fetchAuditReadiness propagates 404 (session missing or no state) as ApiError", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({ detail: "No composition state for this session" }),
        { status: 404, headers: { "content-type": "application/json" } },
      ),
    );
    await expect(fetchAuditReadiness(SESSION_ID)).rejects.toMatchObject({
      status: 404,
    });
  });

  it("fetchAuditReadinessExplain GETs the explain URL and returns narrative", async () => {
    const body = {
      session_id: SESSION_ID,
      composition_version: 3,
      narrative: "When you run this pipeline, ELSPETH will record:\n\n• Source data — 5 URLs ...",
    };
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const explain = await fetchAuditReadinessExplain(SESSION_ID);
    expect(explain.narrative).toContain("ELSPETH will record");
    expect(explain.composition_version).toBe(3);

    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const [url, init] = mock.mock.calls[0];
    expect(url).toBe(`/api/sessions/${SESSION_ID}/audit-readiness/explain`);
    expect(init?.method).toBe("GET");
  });

  it("fetchAuditReadinessExplain throws on non-2xx", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("nope", { status: 500 }),
    );
    await expect(fetchAuditReadinessExplain(SESSION_ID)).rejects.toThrow();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/api/auditReadiness.test.ts
```

Expected: `Cannot find module './auditReadiness'`.

- [ ] **Step 4: Implement**

`src/elspeth/web/frontend/src/api/auditReadiness.ts`:

```typescript
/**
 * API client for the audit-readiness panel (Phase 2).
 *
 * Two GET endpoints; both return strict Pydantic payloads. Mirrors
 * the auth/parse pattern used by api/client.ts::validatePipeline.
 *
 * Technical debt: getToken/authHeaders/parseResponse are duplicated from
 * api/client.ts (see client.ts:403–424, the account-level preferences block).
 * Phase 8 cleanup task: consolidate these helpers as exported utilities from
 * client.ts. Currently at least two API modules carry the duplicates
 * (client.ts inline + auditReadiness.ts).
 */
import type {
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  ApiError,
} from "../types/api";

function getToken(): string | null {
  return localStorage.getItem("elspeth_access_token");
}

function authHeaders(contentType?: string): HeadersInit {
  const headers: Record<string, string> = {};
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  if (contentType) headers["Content-Type"] = contentType;
  return headers;
}

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    let error_type: string | undefined;
    try {
      const body = (await response.clone().json()) as Record<string, unknown>;
      if (typeof body.detail === "string") detail = body.detail;
      if (typeof body.error_type === "string") error_type = body.error_type;
    } catch {
      // body wasn't JSON; ignore
    }
    const error: ApiError = {
      status: response.status,
      detail: detail ?? response.statusText,
      error_type,
    };
    throw error;
  }
  return (await response.json()) as T;
}

export async function fetchAuditReadiness(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessSnapshot> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return parseResponse<AuditReadinessSnapshot>(response);
}

export async function fetchAuditReadinessExplain(
  sessionId: string,
  signal?: AbortSignal,
): Promise<AuditReadinessExplain> {
  const response = await fetch(
    `/api/sessions/${sessionId}/audit-readiness/explain`,
    { method: "GET", headers: authHeaders(), signal },
  );
  return parseResponse<AuditReadinessExplain>(response);
}
```

Re-export from `frontend/src/api/client.ts` (add near the bottom, after existing exports):

```typescript
export {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "./auditReadiness";
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/api/auditReadiness.test.ts
```

Expected: 5/5 pass.

- [ ] **Step 6: Commit**

```bash
cd src/elspeth/web/frontend && npm run typecheck && npm run lint
git add src/elspeth/web/frontend/src/api/auditReadiness.ts src/elspeth/web/frontend/src/api/auditReadiness.test.ts src/elspeth/web/frontend/src/api/client.ts
git commit -m "feat(web/frontend): add audit-readiness API client (Phase 2B.2)"
```

---

## Task 3: Zustand store — `useAuditReadinessStore`

**Files:**
- Create: `frontend/src/stores/auditReadinessStore.ts`
- Create: `frontend/src/stores/auditReadinessStore.test.ts`

The store holds **at most one snapshot per session** and refetches whenever `compositionState.version` advances. Explain narratives are fetched lazily on dialog open and cached on the same key.

- [ ] **Step 1: Write the failing test**

`frontend/src/stores/auditReadinessStore.test.ts`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useAuditReadinessStore, getInitialState } from "./auditReadinessStore";
import * as api from "../api/auditReadiness";
import type { AuditReadinessSnapshot, AuditReadinessExplain } from "../types/api";

vi.mock("../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

function snapshot(version: number): AuditReadinessSnapshot {
  return {
    session_id: SESSION_ID,
    composition_version: version,
    rows: [
      { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
      { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
      { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
      { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
      { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets referenced", detail: null, component_ids: [] },
    ],
  };
}

describe("useAuditReadinessStore", () => {
  beforeEach(() => {
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("loadSnapshot fetches and stores by sessionId", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(1));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);

    const state = useAuditReadinessStore.getState();
    expect(state.snapshotsBySession[SESSION_ID]?.composition_version).toBe(1);
    // Amendment 14b-4 (frontend-005): per-session maps, not flat scalars.
    expect(state.isLoadingBySession[SESSION_ID]).toBeFalsy();
    expect(state.errorBySession[SESSION_ID]).toBeNull();
  });

  it("loadSnapshot is a no-op when the cached snapshot's version matches", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(2));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1);
  });

  it("loadSnapshot refetches when the version advances", async () => {
    vi.mocked(api.fetchAuditReadiness)
      .mockResolvedValueOnce(snapshot(1))
      .mockResolvedValueOnce(snapshot(2));

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(2);
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]
        ?.composition_version,
    ).toBe(2);
  });

  it("loadSnapshot stores error on 404", async () => {
    vi.mocked(api.fetchAuditReadiness).mockRejectedValueOnce({
      status: 404,
      detail: "No composition state for this session",
    });

    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);

    const state = useAuditReadinessStore.getState();
    // Amendment 14b-4 (frontend-005): per-session errorBySession, not flat error.
    expect(state.errorBySession[SESSION_ID]).toContain("No composition state");
    expect(state.snapshotsBySession[SESSION_ID]).toBeUndefined();
  });

  it("loadExplain fetches narrative and caches by version", async () => {
    const expl: AuditReadinessExplain = {
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "When you run this pipeline, ELSPETH will record …",
    };
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce(expl);

    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);

    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(1);
    expect(
      useAuditReadinessStore.getState().explainsBySession[SESSION_ID]?.narrative,
    ).toContain("ELSPETH will record");
  });

  it("loadExplain refetches when the version advances", async () => {
    vi.mocked(api.fetchAuditReadinessExplain)
      .mockResolvedValueOnce({ session_id: SESSION_ID, composition_version: 1, narrative: "v1 text" })
      .mockResolvedValueOnce({ session_id: SESSION_ID, composition_version: 2, narrative: "v2 text" });

    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 2);

    expect(api.fetchAuditReadinessExplain).toHaveBeenCalledTimes(2);
    expect(
      useAuditReadinessStore.getState().explainsBySession[SESSION_ID]?.narrative,
    ).toBe("v2 text");
  });

  it("clearSession removes both snapshot and explain", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshot(1));
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "text",
    });
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    await useAuditReadinessStore.getState().loadExplain(SESSION_ID, 1);

    useAuditReadinessStore.getState().clearSession(SESSION_ID);

    const state = useAuditReadinessStore.getState();
    expect(state.snapshotsBySession[SESSION_ID]).toBeUndefined();
    expect(state.explainsBySession[SESSION_ID]).toBeUndefined();
  });

  // --- Monotonic write-guard contract ---
  // This test exercises the version monotonicity guard (loadSnapshot discards
  // a response whose composition_version is lower than what's already cached).
  // It is SEQUENTIAL: v2 completes before v1 starts. It does NOT exercise the
  // AbortController cancellation path — see the abort-cancellation test below.
  it("monotonic write guard — sequential ordering: fast-v2 + slow-v1 interleaved — v2 payload wins", async () => {
    // Deferred resolver pattern: v2 starts first and resolves first; v1
    // starts after v2 but resolves last (simulating a slow in-flight v1 that
    // was superseded by a version bump).
    let resolveV1!: (s: AuditReadinessSnapshot) => void;
    const slowV1 = new Promise<AuditReadinessSnapshot>((res) => { resolveV1 = res; });

    vi.mocked(api.fetchAuditReadiness)
      .mockReturnValueOnce(Promise.resolve(snapshot(2)))   // fast v2
      .mockReturnValueOnce(slowV1);                        // slow v1

    // Kick off v2 fetch first — await it fully before starting v1.
    await useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);

    // Start v1 fetch (simulates a race from a component that received an
    // older compositionVersion).
    const v1Promise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    resolveV1(snapshot(1)); // now the slow v1 resolve arrives
    await v1Promise;

    // Monotonic guard must have discarded v1 — v2 still wins.
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);
  });

  // --- AbortController cancellation contract ---
  // This test exercises the abort path: when a second loadSnapshot call starts
  // while the first is still in-flight, the store must abort the first fetch's
  // AbortController and clear isLoadingBySession[SESSION_ID] after the second
  // completes. Amendment 14b-4 (frontend-005): per-session map, not flat scalar.
  // It is CONCURRENT: both fetches are in-flight simultaneously. It covers a
  // different contract from the monotonic-guard test above — both are required.
  it("abort-cancellation: second in-flight fetch aborts the first; isLoadingBySession resets cleanly", async () => {
    let resolveFirst!: (s: AuditReadinessSnapshot) => void;
    let rejectFirst!: (err: unknown) => void;
    const firstFetch = new Promise<AuditReadinessSnapshot>((res, rej) => {
      resolveFirst = res;
      rejectFirst = rej;
    });

    vi.mocked(api.fetchAuditReadiness)
      .mockReturnValueOnce(firstFetch)                          // slow first fetch
      .mockReturnValueOnce(Promise.resolve(snapshot(2)));       // fast second fetch

    // Start BOTH fetches without awaiting the first — they are concurrently in-flight.
    // Capture the first controller (from store state) before the second call overwrites it.
    // (loadSnapshot sets abortControllers[sessionId] synchronously inside its set() call.)
    const firstPromise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 1);
    const storedCtrl = useAuditReadinessStore.getState().abortControllers[SESSION_ID];

    // Now start the second fetch — this aborts the first controller.
    const secondPromise = useAuditReadinessStore.getState().loadSnapshot(SESSION_ID, 2);

    // The first fetch's AbortController must have been aborted.
    expect(storedCtrl?.signal.aborted).toBe(true);

    // Reject the first fetch with an AbortError (simulating native fetch abort).
    rejectFirst(Object.assign(new Error("AbortError"), { name: "AbortError" }));
    await firstPromise;

    // Complete the second fetch.
    await secondPromise;

    // isLoadingBySession[SESSION_ID] must be false — the abort arm must have
    // cleared the per-session flag. Amendment 14b-4 (frontend-005).
    expect(useAuditReadinessStore.getState().isLoadingBySession[SESSION_ID]).toBeFalsy();
    // Second fetch's result must be stored.
    expect(
      useAuditReadinessStore.getState().snapshotsBySession[SESSION_ID]?.composition_version,
    ).toBe(2);
  });
});
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/auditReadinessStore.test.ts
```

Expected: `Cannot find module './auditReadinessStore'`.

- [ ] **Step 3: Implement**

`frontend/src/stores/auditReadinessStore.ts`:

```typescript
/**
 * Zustand store for the audit-readiness panel (Phase 2B).
 *
 * Caches snapshots and explain narratives by sessionId, keyed by the
 * composition_version each carries. `loadSnapshot` is a no-op when the
 * cached snapshot's version matches the requested version — this is what
 * makes auto-validate-on-composition-change cheap.
 *
 * The store never coerces server payloads — Phase 2A's `_StrictResponse`
 * is the contract. If a literal value doesn't match the TypeScript union,
 * that's a backend/frontend version skew and the panel renderer will
 * surface it via the exhaustive `never` arm in AuditReadinessPanel.tsx.
 */
import { create } from "zustand";

import {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "../api/auditReadiness";
import type {
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  ApiError,
} from "../types/api";

export interface AuditReadinessState {
  snapshotsBySession: Record<string, AuditReadinessSnapshot>;
  explainsBySession: Record<string, AuditReadinessExplain>;
  /** In-flight AbortController keyed by sessionId for snapshot fetches.
   *  loadSnapshot aborts the previous controller before starting a new fetch,
   *  preventing stale responses from overwriting a more recent result.
   *  clearSession also aborts any in-flight request for the cleared session. */
  abortControllers: Record<string, AbortController>;
  /** In-flight AbortController keyed by sessionId for explain fetches.
   *  Parallel to abortControllers — kept separate so aborting an explain fetch
   *  does not cancel a concurrent snapshot fetch for the same session.
   *  clearSession aborts both controllers. */
  explainAbortControllers: Record<string, AbortController>;
  /** Per-session loading and error state. Amendment 14b-4 (frontend-005):
   *  flat scalars (`isLoading`, `error`, etc.) replaced with per-session keyed
   *  maps to prevent cross-session contamination — switching from a failing
   *  session A to a healthy session B would otherwise render A's error on B.
   *  Commit: c3f3d0ae6. 14c's Task 4C extends with `userExpandedBySession`. */
  isLoadingBySession: Record<string, boolean>;
  isLoadingExplainBySession: Record<string, boolean>;
  errorBySession: Record<string, string | null>;
  explainErrorBySession: Record<string, string | null>;

  loadSnapshot: (sessionId: string, compositionVersion: number) => Promise<void>;
  loadExplain: (sessionId: string, compositionVersion: number) => Promise<void>;
  clearSession: (sessionId: string) => void;
  reset: () => void;
}

export const getInitialState = (): Omit<AuditReadinessState, "loadSnapshot" | "loadExplain" | "clearSession" | "reset"> => ({
  snapshotsBySession: {},
  explainsBySession: {},
  abortControllers: {},
  explainAbortControllers: {},
  isLoadingBySession: {},
  isLoadingExplainBySession: {},
  errorBySession: {},
  explainErrorBySession: {},
});

export const useAuditReadinessStore = create<AuditReadinessState>((set, get) => ({
  ...getInitialState(),

  async loadSnapshot(sessionId: string, compositionVersion: number) {
    const cached = get().snapshotsBySession[sessionId];
    if (cached && cached.composition_version === compositionVersion) {
      return;
    }

    // Abort any in-flight request for this session before starting a new one.
    const prev = get().abortControllers[sessionId];
    if (prev) prev.abort();
    const controller = new AbortController();
    // Amendment 14b-4 (frontend-005): per-session isLoadingBySession/errorBySession.
    set((state) => ({
      abortControllers: { ...state.abortControllers, [sessionId]: controller },
      isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: true },
      errorBySession: { ...state.errorBySession, [sessionId]: null },
    }));

    try {
      const snapshot = await fetchAuditReadiness(sessionId, controller.signal);
      // Monotonic write guard: discard the response if a newer version has
      // already been stored while this fetch was in flight.
      set((state) => {
        const current = state.snapshotsBySession[sessionId];
        if (current && current.composition_version > snapshot.composition_version) {
          const { [sessionId]: _staleCtrl, ...restStaleCtrl } = state.abortControllers;
          return {
            abortControllers: restStaleCtrl,
            isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          };
        }
        const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
        return {
          snapshotsBySession: { ...state.snapshotsBySession, [sessionId]: snapshot },
          abortControllers: restCtrl,
          isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          errorBySession: { ...state.errorBySession, [sessionId]: null },
        };
      });
    } catch (err) {
      // AbortError: per-session flag clears. Controller-identity guard: if our
      // controller is no longer the tracked one (clearSession or newer fetch
      // took over), do not resurrect state we no longer own.
      if ((err as { name?: string }).name === "AbortError") {
        set((state) => {
          if (state.abortControllers[sessionId] !== controller) {
            return state;
          }
          return {
            isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          };
        });
        return;
      }
      const apiErr = err as ApiError;
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
        return {
          abortControllers: restCtrl,
          isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          errorBySession: {
            ...state.errorBySession,
            [sessionId]: apiErr.detail ?? "Failed to load audit readiness.",
          },
        };
      });
    }
  },

  async loadExplain(sessionId: string, compositionVersion: number) {
    const cached = get().explainsBySession[sessionId];
    if (cached && cached.composition_version === compositionVersion) {
      return;
    }

    // Abort any in-flight explain fetch for this session before starting a new
    // one. Uses a parallel controller dict (explainAbortControllers) so that
    // aborting an explain fetch does not cancel a concurrent snapshot fetch.
    const prevExplain = get().explainAbortControllers[sessionId];
    if (prevExplain) prevExplain.abort();
    const explainController = new AbortController();
    // Amendment 14b-4 (frontend-005): per-session isLoadingExplainBySession/explainErrorBySession.
    set((state) => ({
      explainAbortControllers: { ...state.explainAbortControllers, [sessionId]: explainController },
      isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: true },
      explainErrorBySession: { ...state.explainErrorBySession, [sessionId]: null },
    }));

    try {
      const explain = await fetchAuditReadinessExplain(sessionId, explainController.signal);
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.explainAbortControllers;
        return {
          explainsBySession: {
            ...state.explainsBySession,
            [sessionId]: explain,
          },
          explainAbortControllers: restCtrl,
          isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          explainErrorBySession: { ...state.explainErrorBySession, [sessionId]: null },
        };
      });
    } catch (err) {
      // Mirror the snapshot AbortError pattern: per-session flag clears.
      // Controller-identity guard: if our explainController is no longer tracked,
      // clearSession or a newer fetch has taken over — do not resurrect stale state.
      if ((err as { name?: string }).name === "AbortError") {
        set((state) => {
          if (state.explainAbortControllers[sessionId] !== explainController) {
            return state;
          }
          return {
            isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          };
        });
        return;
      }
      const apiErr = err as ApiError;
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.explainAbortControllers;
        return {
          explainAbortControllers: restCtrl,
          isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          explainErrorBySession: {
            ...state.explainErrorBySession,
            [sessionId]: apiErr.detail ?? "Failed to load the explain narrative.",
          },
        };
      });
    }
  },

  clearSession(sessionId: string) {
    // Abort any in-flight snapshot and explain fetches for this session.
    const ctrl = get().abortControllers[sessionId];
    if (ctrl) ctrl.abort();
    const explainCtrl = get().explainAbortControllers[sessionId];
    if (explainCtrl) explainCtrl.abort();
    // Amendment 14b-4 (frontend-005): clear all per-session keys for this session.
    set((state) => {
      const { [sessionId]: _snap, ...restSnap } = state.snapshotsBySession;
      const { [sessionId]: _expl, ...restExpl } = state.explainsBySession;
      const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
      const { [sessionId]: _eCtrl, ...restECtrl } = state.explainAbortControllers;
      const { [sessionId]: _il, ...restIL } = state.isLoadingBySession;
      const { [sessionId]: _ilx, ...restILX } = state.isLoadingExplainBySession;
      const { [sessionId]: _err, ...restErr } = state.errorBySession;
      const { [sessionId]: _errx, ...restErrX } = state.explainErrorBySession;
      return {
        snapshotsBySession: restSnap,
        explainsBySession: restExpl,
        abortControllers: restCtrl,
        explainAbortControllers: restECtrl,
        isLoadingBySession: restIL,
        isLoadingExplainBySession: restILX,
        errorBySession: restErr,
        explainErrorBySession: restErrX,
      };
    });
  },

  reset() {
    set(getInitialState());
  },
}));
```

> `getInitialState` is exported as a named const (not via an ad-hoc cast), so tests can import it directly. The `reset()` action calls `set(getInitialState())` — similar shape to `executionStore`'s `reset()` pattern (`executionStore` uses a module-internal `initialExecutionState` const, not an exported `getInitialState`; the public surface is the same `reset()` verb). Verify the import with `grep -n "getInitialState" src/elspeth/web/frontend/src/stores/` to confirm the naming convention is consistent.

> **Operator/architect decision — R2-W9 `loadExplain` AbortController asymmetry:**
>
> The prescription above implements **option (a)**: `loadExplain` mirrors `loadSnapshot`'s AbortController pattern via a parallel `explainAbortControllers` field. `clearSession` aborts both controllers. This is the recommended approach for consistency.
>
> **Option (b) — document the asymmetry as deliberate:** If the team accepts stale-cache risk for explain fetches, skip `explainAbortControllers` entirely and add a comment in `loadExplain`:
> ```ts
> // No AbortController: explain fetches are lazy and user-triggered.
> // Stale-cache collision (navigation mid-fetch writes to old sessionId's
> // explainsBySession) is rare; the narrative is explanatory, not load-bearing.
> // Operator decision: accept the asymmetry; revisit if stale cache becomes
> // observable in production (composition_version guard prevents data corruption).
> ```
> Option (b) tradeoffs: lower implementation complexity; `clearSession` stays simpler; but navigation during an in-flight explain leaves `isLoadingExplain: true` (mirroring the R2-W2 symptom) and writes a stale narrative to the old session's cache. If the cached `composition_version` matches on re-entry, the stale narrative is served without a reload. Option (a) eliminates both failure modes at the cost of a second controller dict.

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/stores/auditReadinessStore.test.ts
```

Expected: 9/9 pass (includes the monotonic-write-guard test and the true-concurrency abort-cancellation test added in this revision — these cover distinct contracts and must both be present).

- [ ] **Step 5: Commit**

```bash
cd src/elspeth/web/frontend && npm run typecheck && npm run lint
git add src/elspeth/web/frontend/src/stores/auditReadinessStore.ts src/elspeth/web/frontend/src/stores/auditReadinessStore.test.ts
git commit -m "feat(web/frontend): add auditReadinessStore with version-keyed cache (Phase 2B.3)"
```

---

## Task 4: `AuditReadinessPanel` component — six rows, auto-fetch, all-green collapse

**Files:**
- Create: `frontend/src/components/audit/AuditReadinessPanel.tsx`
- Create: `frontend/src/components/audit/AuditReadinessPanel.test.tsx`

The panel:

- Auto-fetches the snapshot when `compositionState.version` changes.
- Renders six rows (or five if `llm_interpretations` is omitted — Phase 2A always emits it but Phase 2B's design spec allows hiding when `status == "not_applicable"`).
- Per the spec's "Reduce visual weight when all-green" risk mitigation, when every actionable row is `ok`/`not_applicable`, collapses to a single "Audit ready ✓" summary that expands on click.
- Clicking any row opens `ReadinessRowDetail` (Task 5).
- Includes an "Explain →" button that opens `ExplainDialog` (Task 6).

> **Fixture extraction (R2-W6):** `makeComposition()` must NOT be defined locally in this test file. Create the canonical fixture module first:
>
> **`frontend/src/test/composerFixtures.ts`** (new file — Amendment 14b-3: implementation placed at `src/test/`, not `src/test-utils/`):
> ```typescript
> import type { CompositionState } from "../types/api";
>
> /**
>  * Canonical test fixture for CompositionState.
>  *
>  * NodeSpec arity (frontend `types/index.ts`):
>  *   Required (7): id, node_type, plugin, input, on_success, on_error, options
>  *   Optional (6): condition, routes, fork_to, branches, policy, merge
>  * This is the frontend contract the fixture mirrors. The Python backend has 13
>  * fields but the TypeScript interface marks 6 of them as optional — no `as never`
>  * cast is needed once all required fields are supplied.
>  *
>  * Import from here in all test files that need CompositionState scaffolding.
>  * Do NOT duplicate this fixture in individual test files.
>  */
> export function makeComposition(
>   version: number,
>   overrides?: Partial<CompositionState>,
> ): CompositionState {
>   return {
>     id: "comp-1",
>     version,
>     source: { kind: "csv_file", config: { path: "x.csv" } } as never,
>     nodes: [
>       {
>         id: "select_columns",
>         node_type: "transform",
>         plugin: "select_columns",
>         input: "source",
>         on_success: null,
>         on_error: null,
>         options: {},
>       },
>     ],
>     edges: [],
>     outputs: [],
>     metadata: { name: "demo", description: "" },
>     ...overrides,
>   };
> }
> ```
>
> `AuditReadinessPanel.test.tsx` imports it via `import { makeComposition } from "@/test/composerFixtures"` (path alias `@/` → `src/`). 14c's `ReadinessRowDetail.test.tsx` imports from the same module — do not duplicate. See 14c plan for the matching import note.

- [ ] **Step 1: Confirm the mount strategy**

The panel is rendered from `InspectorPanel.tsx` (Task 7). For this task, render it directly with mocked stores; the mount wiring is Task 7's concern.

- [ ] **Step 2: Write the failing test**

`frontend/src/components/audit/AuditReadinessPanel.test.tsx`:

```typescript
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuditReadinessPanel } from "./AuditReadinessPanel";
import { useSessionStore } from "../../stores/sessionStore";
import { useAuditReadinessStore, getInitialState } from "../../stores/auditReadinessStore";
import * as api from "../../api/auditReadiness";
import type { AuditReadinessSnapshot } from "../../types/api";
import { makeComposition } from "@/test/composerFixtures";

vi.mock("../../api/auditReadiness");

const SESSION_ID = "00000000-0000-0000-0000-000000000001";

function allGreenSnapshot(version: number): AuditReadinessSnapshot {
  return {
    session_id: SESSION_ID,
    composition_version: version,
    rows: [
      { id: "validation", label: "Validation", status: "ok", summary: "All checks pass", detail: null, component_ids: [] },
      { id: "plugin_trust", label: "Plugin trust", status: "ok", summary: "All Tier 1/2", detail: null, component_ids: [] },
      { id: "provenance", label: "Provenance", status: "ok", summary: "Complete lineage", detail: null, component_ids: [] },
      { id: "retention", label: "Retention", status: "not_applicable", summary: "System retention: 90 days", detail: null, component_ids: [] },
      { id: "llm_interpretations", label: "LLM interpretations", status: "not_applicable", summary: "No LLM transforms", detail: null, component_ids: [] },
      { id: "secrets", label: "Secrets", status: "not_applicable", summary: "No secrets", detail: null, component_ids: [] },
    ],
  };
}

function snapshotWithProvenanceWarning(version: number): AuditReadinessSnapshot {
  const base = allGreenSnapshot(version);
  return {
    ...base,
    rows: base.rows.map((r) =>
      r.id === "provenance"
        ? {
            ...r,
            status: "warning",
            summary: "Identity passthrough detected",
            detail: "Identity passthrough — provenance gap on 'select_columns'.",
            component_ids: ["select_columns"],
          }
        : r,
    ),
  };
}

describe("AuditReadinessPanel", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: makeComposition(1),
    } as never);
    // Amendment 14b-4 (frontend-005): per-session keyed maps, not flat scalars.
    useAuditReadinessStore.setState(getInitialState());
    vi.clearAllMocks();
  });

  it("auto-fetches on mount using compositionState.version", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(allGreenSnapshot(1));
    render(<AuditReadinessPanel />);
    await waitFor(() => {
      expect(api.fetchAuditReadiness).toHaveBeenCalledWith(SESSION_ID);
    });
  });

  it("collapses to a single 'Audit ready' summary when all rows are ok/not_applicable", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(allGreenSnapshot(1));
    render(<AuditReadinessPanel />);
    expect(await screen.findByText(/Audit ready/i)).toBeInTheDocument();
    // Six row labels are not all rendered up-front in collapsed mode.
    expect(screen.queryByText("Plugin trust")).not.toBeInTheDocument();
  });

  it("expands to all rows when the summary is clicked", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(allGreenSnapshot(1));
    const user = userEvent.setup();
    render(<AuditReadinessPanel />);
    const summary = await screen.findByRole("button", { name: /Audit ready/i });
    await user.click(summary);
    expect(screen.getByText("Validation")).toBeInTheDocument();
    expect(screen.getByText("Plugin trust")).toBeInTheDocument();
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Retention")).toBeInTheDocument();
  });

  it("shows all rows by default when any row has warning or error status", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(snapshotWithProvenanceWarning(1));
    render(<AuditReadinessPanel />);
    await waitFor(() => {
      expect(screen.getByText("Validation")).toBeInTheDocument();
    });
    expect(screen.getByText("Provenance")).toBeInTheDocument();
    expect(screen.getByText("Identity passthrough detected")).toBeInTheDocument();
  });

  it("refetches when compositionState.version advances", async () => {
    vi.mocked(api.fetchAuditReadiness)
      .mockResolvedValueOnce(allGreenSnapshot(1))
      .mockResolvedValueOnce(allGreenSnapshot(2));
    const { rerender } = render(<AuditReadinessPanel />);
    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(1));

    useSessionStore.setState({ compositionState: makeComposition(2) } as never);
    rerender(<AuditReadinessPanel />);

    await waitFor(() => expect(api.fetchAuditReadiness).toHaveBeenCalledTimes(2));
  });

  it("renders a loading state on first fetch", async () => {
    let resolve!: (s: AuditReadinessSnapshot) => void;
    vi.mocked(api.fetchAuditReadiness).mockReturnValueOnce(
      new Promise<AuditReadinessSnapshot>((r) => {
        resolve = r;
      }),
    );
    render(<AuditReadinessPanel />);
    expect(screen.getByText(/Checking audit readiness/i)).toBeInTheDocument();
    resolve(allGreenSnapshot(1));
    await waitFor(() => {
      expect(screen.queryByText(/Checking audit readiness/i)).not.toBeInTheDocument();
    });
  });

  it("renders an error message on fetch failure", async () => {
    vi.mocked(api.fetchAuditReadiness).mockRejectedValueOnce({
      status: 500,
      detail: "Internal server error",
    });
    render(<AuditReadinessPanel />);
    expect(await screen.findByRole("alert")).toHaveTextContent(/Internal server error/);
  });

  it("opens the Explain dialog when Explain → is clicked", async () => {
    vi.mocked(api.fetchAuditReadiness).mockResolvedValueOnce(allGreenSnapshot(1));
    vi.mocked(api.fetchAuditReadinessExplain).mockResolvedValueOnce({
      session_id: SESSION_ID,
      composition_version: 1,
      narrative: "Narrative body for testing.",
    });
    const user = userEvent.setup();
    render(<AuditReadinessPanel />);
    const summary = await screen.findByRole("button", { name: /Audit ready/i });
    await user.click(summary); // expand
    const explainBtn = screen.getByRole("button", { name: /Explain/i });
    await user.click(explainBtn);
    expect(await screen.findByText("Narrative body for testing.")).toBeInTheDocument();
  });

  it("renders nothing when there is no active session", () => {
    useSessionStore.setState({ activeSessionId: null, compositionState: null } as never);
    const { container } = render(<AuditReadinessPanel />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when the composition is empty (no source, no nodes, no outputs)", () => {
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      compositionState: {
        ...makeComposition(1),
        source: null,
        nodes: [],
        outputs: [],
      },
    } as never);
    const { container } = render(<AuditReadinessPanel />);
    expect(container).toBeEmptyDOMElement();
  });
});
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/audit/AuditReadinessPanel.test.tsx
```

Expected: `Cannot find module './AuditReadinessPanel'`.

- [ ] **Step 4: Implement**

`frontend/src/components/audit/AuditReadinessPanel.tsx`:

```typescript
/**
 * AuditReadinessPanel (Phase 2)
 *
 * Persistent right-rail panel showing six rows of audit-readiness state.
 * Auto-fetches on compositionState.version change; collapses to a single
 * "Audit ready ✓" summary when nothing actionable is present.
 *
 * Design spec: docs/composer/ux-redesign-2026-05/07-audit-readiness-panel.md
 *
 * The renderer is intentionally exhaustive on ReadinessRowId — the `never`
 * default arm fails the build if a new row is added to the wire schema
 * without a UI case.
 */
import { useEffect, useMemo, useState } from "react";

import { useSessionStore } from "../../stores/sessionStore";
import { useAuditReadinessStore } from "../../stores/auditReadinessStore";
import type {
  ReadinessRow,
  ReadinessRowId,
  ReadinessStatus,
} from "../../types/api";
import { ReadinessRowDetail } from "./ReadinessRowDetail";
import { ExplainDialog } from "./ExplainDialog";

/** Glyph + accessible label for each row status. */
function statusGlyph(status: ReadinessStatus): { glyph: string; aria: string } {
  switch (status) {
    case "ok":
      return { glyph: "✓", aria: "OK" };
    case "warning":
      return { glyph: "⚠", aria: "Warning" };
    case "error":
      return { glyph: "✗", aria: "Error" };
    case "not_applicable":
      return { glyph: "—", aria: "Not applicable" };
    default: {
      const _exhaustive: never = status;
      throw new Error(`unknown readiness status: ${String(_exhaustive)}`);
    }
  }
}

/** Linda-vocabulary heading for each row id. The wire schema's `label` is
 *  authoritative; this map is the fallback when the backend label is empty,
 *  which Phase 2A's `Field(min_length=1)` rules out — but the renderer must
 *  be exhaustive on the id type regardless. */
function rowHeading(id: ReadinessRowId): string {
  switch (id) {
    case "validation":
      return "Validation";
    case "plugin_trust":
      return "Plugin trust";
    case "provenance":
      return "Provenance";
    case "retention":
      return "Retention";
    case "llm_interpretations":
      return "LLM interpretations";
    case "secrets":
      return "Secrets";
    default: {
      const _exhaustive: never = id;
      throw new Error(`unknown readiness row id: ${String(_exhaustive)}`);
    }
  }
}

function isActionable(status: ReadinessStatus): boolean {
  return status === "warning" || status === "error";
}

export function AuditReadinessPanel() {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionState = useSessionStore((s) => s.compositionState);

  const snapshot = useAuditReadinessStore((s) =>
    activeSessionId ? s.snapshotsBySession[activeSessionId] : undefined,
  );
  // Amendment 14b-4 (frontend-005): per-session keyed maps — not flat scalars.
  const isLoading = useAuditReadinessStore((s) =>
    activeSessionId ? !!s.isLoadingBySession[activeSessionId] : false,
  );
  const error = useAuditReadinessStore((s) =>
    activeSessionId ? s.errorBySession[activeSessionId] ?? null : null,
  );
  const loadSnapshot = useAuditReadinessStore((s) => s.loadSnapshot);

  const hasCompositionContent =
    !!compositionState &&
    (compositionState.source !== null ||
      compositionState.nodes.length > 0 ||
      compositionState.outputs.length > 0);

  useEffect(() => {
    if (!activeSessionId || !compositionState || !hasCompositionContent) return;
    // Fire and forget; store handles errors.
    void loadSnapshot(activeSessionId, compositionState.version);
    return () => {
      // Unmount-during-fetch cleanup: abort the in-flight controller for this session.
      const ctrl = useAuditReadinessStore.getState().abortControllers[activeSessionId];
      if (ctrl) ctrl.abort();
    };
  // Amendment 14b-2 (frontend-003): dep is compositionState?.version, not compositionState.
  // Using the object reference re-runs the effect on every render that creates a new
  // compositionState without an actual version change.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId, compositionState?.version, hasCompositionContent, loadSnapshot]);

  const anyActionable = useMemo(
    () => snapshot?.rows.some((r) => isActionable(r.status)) ?? false,
    [snapshot],
  );

  // Amendment 14b-4 (frontend-005): userExpanded tracks explicit user intent.
  // Auto-expansion on actionable snapshots is computed atomically as
  // `anyActionable || userExpanded` rather than synced through a useEffect,
  // avoiding an extra render cycle and making the panel auto-collapse when
  // a later snapshot returns all-green (unless the user explicitly clicked Expand).
  const [userExpanded, setUserExpanded] = useState(false);
  const [selectedRowId, setSelectedRowId] = useState<ReadinessRowId | null>(null);
  const [explainOpen, setExplainOpen] = useState(false);

  const showExpanded = anyActionable || userExpanded;

  if (!activeSessionId || !hasCompositionContent) {
    return null;
  }

  if (isLoading && !snapshot) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--loading"
      >
        <span className="audit-readiness-loading">
          Checking audit readiness…
        </span>
      </section>
    );
  }

  if (error && !snapshot) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--error"
      >
        <div role="alert" className="audit-readiness-error">
          {error}
        </div>
      </section>
    );
  }

  if (!snapshot) {
    return null;
  }

  // Collapsed view — single summary line when nothing is actionable and user
  // hasn't explicitly expanded. Amendment 14b-4: uses showExpanded (derived
  // from anyActionable || userExpanded) rather than a plain expanded bool.
  if (!showExpanded) {
    return (
      <section
        aria-label="Audit readiness"
        className="audit-readiness audit-readiness--collapsed"
      >
        <button
          type="button"
          className="audit-readiness-summary"
          onClick={() => setUserExpanded(true)}
          aria-expanded="false"
          aria-controls="audit-readiness-rows"
        >
          <span aria-hidden="true">{"✓"}</span> Audit ready
        </button>
      </section>
    );
  }

  return (
    <>
      <section aria-label="Audit readiness" className="audit-readiness">
        <header className="audit-readiness-header">
          <h2 className="audit-readiness-title">Audit readiness</h2>
          <div className="audit-readiness-actions">
            <button
              type="button"
              className="btn audit-readiness-action-btn"
              onClick={() => setExplainOpen(true)}
              aria-label="Explain what this pipeline will record"
            >
              Explain →
            </button>
            {!anyActionable && (
              <button
                type="button"
                className="btn audit-readiness-action-btn audit-readiness-action-btn--ghost"
                onClick={() => setUserExpanded(false)}
                aria-label="Collapse audit readiness"
              >
                Collapse
              </button>
            )}
          </div>
        </header>

        <ul id="audit-readiness-rows" className="audit-readiness-rows">
          {snapshot.rows.map((row: ReadinessRow) => {
            const { glyph, aria } = statusGlyph(row.status);
            const heading = row.label || rowHeading(row.id);
            const clickable = isActionable(row.status);
            return (
              <li
                key={row.id}
                className={`audit-readiness-row audit-readiness-row--${row.status}`}
              >
                {clickable ? (
                  <button
                    type="button"
                    className="audit-readiness-row-btn"
                    onClick={() => setSelectedRowId(row.id)}
                    aria-label={`${heading}: ${aria}. ${row.summary}. Click for detail.`}
                  >
                    <span
                      className="audit-readiness-glyph"
                      aria-hidden="true"
                    >
                      {glyph}
                    </span>
                    <span className="audit-readiness-row-label">{heading}</span>
                    <span className="audit-readiness-row-summary">{row.summary}</span>
                  </button>
                ) : (
                  <div
                    className="audit-readiness-row-static"
                    aria-label={`${heading}: ${aria}. ${row.summary}.`}
                  >
                    <span
                      className="audit-readiness-glyph"
                      aria-hidden="true"
                    >
                      {glyph}
                    </span>
                    <span className="audit-readiness-row-label">{heading}</span>
                    <span className="audit-readiness-row-summary">{row.summary}</span>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </section>

      {selectedRowId && (
        <ReadinessRowDetail
          row={snapshot.rows.find((r) => r.id === selectedRowId)!}
          onClose={() => setSelectedRowId(null)}
        />
      )}

      {explainOpen && (
        <ExplainDialog
          sessionId={activeSessionId}
          compositionVersion={snapshot.composition_version}
          onClose={() => setExplainOpen(false)}
        />
      )}
    </>
  );
}
```

> CSS: this plan does not introduce new design tokens. Reuse the existing `.validation-banner` family or add scoped classes in the project's main stylesheet. The class names above (`.audit-readiness*`) are placeholders the styling pass picks up; the component does not depend on any specific stylesheet.

- [ ] **Step 5: Create placeholder `ReadinessRowDetail` and `ExplainDialog` files**

The `AuditReadinessPanel` component imports `ReadinessRowDetail` and `ExplainDialog`, both of which 14c implements properly. 14b ships these as minimal placeholders so the imports resolve and the panel's tests run.

> **Amendment 14b-5 (frontend-006):** Placeholders intentionally do NOT include `role="dialog"`.
> Shipping a dialog claim without a focus management contract is a W2 accessibility defect.
> Both components are tagged with `data-testid` instead so panel tests can assert the component
> mounted. The 14c real implementations re-introduce `role="dialog"` with the full focus
> contract (`useFocusTrap`, `aria-modal`, escape-to-close, backdrop dismiss) per 14c's B3 fix.

`frontend/src/components/audit/ReadinessRowDetail.tsx`:

```typescript
/**
 * Placeholder shipped by 14b. 14c replaces this with the full implementation
 * (per-row warning detail + jump-to-component, with proper dialog semantics:
 * focus trap, aria-modal, initial focus, focus restoration).
 *
 * This placeholder is intentionally NOT a dialog. It renders a minimal stub
 * tagged with `data-testid="readinessrowdetail-placeholder"` so the panel's
 * tests can assert the component mounted, lint/typecheck pass, and the W2
 * accessibility defect (role="dialog" without focus management) does not ship.
 * The `onClose` callback is wired up so the parent's close-on-trigger flow is
 * already exercised — 14c only needs to add the modal semantics.
 *
 * DO NOT extend the placeholder. Extensions (real dialog markup with focus
 * trap, aria-modal, escape-to-close, backdrop dismiss) belong in 14c.
 */
import type { ReadinessRow } from "../../types/api";

export interface ReadinessRowDetailProps {
  row: ReadinessRow;
  onClose: () => void;
}

export function ReadinessRowDetail({ row, onClose }: ReadinessRowDetailProps) {
  return (
    <div
      data-testid="readinessrowdetail-placeholder"
      aria-label={row.label}
      className="readiness-row-detail"
    >
      <h3>{row.label}</h3>
      <p>{row.summary}</p>
      {row.detail && <pre>{row.detail}</pre>}
      <button type="button" onClick={onClose}>
        Close
      </button>
    </div>
  );
}
```

`frontend/src/components/audit/ExplainDialog.tsx`:

```typescript
/**
 * Placeholder shipped by 14b. 14c replaces this with the full implementation
 * (modal narrative view with proper dialog semantics: focus trap, aria-modal,
 * initial focus, focus restoration, escape-to-close, backdrop dismiss).
 *
 * This placeholder is intentionally NOT a dialog. It renders a minimal stub
 * tagged with `data-testid="explaindialog-placeholder"` so the panel's tests
 * can assert the component mounted, lint/typecheck pass, and the W2
 * accessibility defect (role="dialog" without focus management) does not ship.
 * The loadExplain side-effect is wired here so the parent's lazy-fetch flow
 * is already exercised — 14c only needs to add the modal semantics.
 *
 * DO NOT extend the placeholder. Extensions belong in 14c.
 */
import { useEffect } from "react";

import { useAuditReadinessStore } from "../../stores/auditReadinessStore";

export interface ExplainDialogProps {
  sessionId: string;
  compositionVersion: number;
  onClose: () => void;
}

export function ExplainDialog({ sessionId, compositionVersion, onClose }: ExplainDialogProps) {
  const explain = useAuditReadinessStore((s) => s.explainsBySession[sessionId]);
  const loadExplain = useAuditReadinessStore((s) => s.loadExplain);

  useEffect(() => {
    void loadExplain(sessionId, compositionVersion);
  }, [sessionId, compositionVersion, loadExplain]);

  return (
    <div
      data-testid="explaindialog-placeholder"
      aria-label="What this pipeline will record"
    >
      {explain && <pre>{explain.narrative}</pre>}
      <button type="button" onClick={onClose}>
        Close
      </button>
    </div>
  );
}
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/audit/AuditReadinessPanel.test.tsx
```

Expected: 10/10 pass. The placeholders satisfy the panel's imports; the tests stub the API surface end-to-end and never depend on the placeholders' internal behaviour beyond the rendered text the assertions search for.

- [ ] **Step 7: Commit**

```bash
cd src/elspeth/web/frontend && npm run typecheck && npm run lint
git add src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx src/elspeth/web/frontend/src/components/audit/ReadinessRowDetail.tsx src/elspeth/web/frontend/src/components/audit/ExplainDialog.tsx
git commit -m "feat(web/frontend): add AuditReadinessPanel + placeholder subcomponents (Phase 2B.4)"
```

---

## Task 2B.5: Session-removal cache cleanup (landed — 2B.5 follow-up)

**Finding:** frontend-007. **Commits:** `084b8c34b` + `163737068` (follow-up; landed after the
main Phase 2B merge).

**Problem:** When a session is removed from `sessionStore.sessions` (archive, 404 self-eviction,
or future removal paths), the `auditReadinessStore` retains the stale cached snapshot and
explain for that session indefinitely. With only a single active session this is invisible;
with multi-session switching the cache grows and stale entries are never reclaimed.

**Solution:** `stores/subscriptions.ts` — a cross-store subscription module that wires
`useAuditReadinessStore.clearSession(prevId)` to fire whenever a session id disappears from
`sessionStore.sessions`. The subscriber seeds `previousSessionIds` at initialization. The
seed is defensive: in production startup `sessions` is empty and `initStoreSubscriptions()`
runs synchronously before `loadSessions`, making the pre-seed case unreachable today. The
seed guards against future changes (persist middleware, SSR hydration, tests that populate
sessions before `initStoreSubscriptions()` is called) at the cost of one `Set` construction.

**Files added:**
- `src/elspeth/web/frontend/src/stores/subscriptions.ts` — `initStoreSubscriptions()` and
  `_resetSubscriptionsForTesting()` (test-only reset helper, not exported from any barrel).
- `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — verifies that session removal
  triggers `clearSession` on `auditReadinessStore`.

**Key code paths:**

`initStoreSubscriptions()` subscribes to `useSessionStore` and tracks `previousSessionIds`:
```typescript
const currentIds = new Set(state.sessions.map((s) => s.id));
for (const prevId of previousSessionIds) {
  if (!currentIds.has(prevId)) {
    useAuditReadinessStore.getState().clearSession(prevId);
  }
}
previousSessionIds = currentIds;
```

`initStoreSubscriptions()` must be called exactly once at app startup (e.g. in `App.tsx`).
The `initialized` guard makes it idempotent.

This module lives at the application boundary — it has imports from all three stores
(`sessionStore`, `executionStore`, `auditReadinessStore`) and was extracted to break
potential circular-import cycles between the stores themselves.

---

## What Phase 2B leaves the frontend in

- New `types/index.ts` entries for the audit-readiness wire shape, re-exported via `types/api.ts`.
- New `api/auditReadiness.ts` with two typed GET wrappers.
- New `stores/auditReadinessStore.ts` with per-session keyed cache (see frontend-005).
- New `stores/subscriptions.ts` wiring session-removal → `clearSession` on the audit-readiness store (Task 2B.5).
- New `AuditReadinessPanel.tsx` rendered standalone (not yet mounted into `InspectorPanel.tsx` — 14c).
- Placeholder `ReadinessRowDetail.tsx` and `ExplainDialog.tsx` that 14c replaces.
- All vitest assertions in this plan pass; `tsc --noEmit` is clean; ESLint is clean.
- `InspectorPanel.tsx`, `App.tsx`, and the staging deployment are **untouched** — 14c owns the integration.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Backend wire-shape drift breaks the renderer | The discriminated-union `never` arm fails the build; Phase 2A's `_StrictResponse` fails at server-side construction. No silent path. |
| 14c lands before 14b and the placeholders are overwritten mid-flight | Per the sequencing in §"Sequencing and dependencies", 14b lands first. If the working branch is shared between 14b and 14c, the 14c PR's diff against 14b's placeholders is reviewable; the placeholder comments explicitly say "DO NOT extend". |
| Concurrent `loadSnapshot` calls resolve out of order (stale-fetch race) | Addressed in this revision: `loadSnapshot` stores the in-flight `AbortController` in store state and aborts the previous request before starting a new one on `composition_version` change or `clearSession`. The resolution arm applies a monotonic write guard — if the arriving response's `composition_version` is lower than the version already cached, the response is silently discarded. Two tests cover distinct contracts: (1) a sequential monotonic-write-guard test (fast-v2 completes before slow-v1 resolves) and (2) a true-concurrency abort-cancellation test (both fetches in-flight simultaneously; asserts `AbortController.signal.aborted` and `isLoadingBySession[SESSION_ID]` reset — per-session, see frontend-005). |
| Per-row "Jump to component" is missing in 14b | 14b's panel still emits the row click; 14b's placeholder `ReadinessRowDetail` lacks the jump button. 14c restores it. The user-visible UX gap exists for the duration of the 14b→14c gap; the design spec's persona table tolerates a partial first cut because Linda's primary trust mechanism (visibility of the row status) is already present. |
| Telemetry can't be added later without rewriting the panel | The row-click handlers are isolated functions in `AuditReadinessPanel`. Phase 8 telemetry adds one line per handler. |

## Review history

### 2026-05-15 — Panel findings applied

Replaced ad-hoc `getInitialState` cast with exported top-level const + `reset()` action (CRITICAL); acknowledged `authHeaders`/`parseResponse` duplication with Phase 8 cleanup note in module JSDoc (IMPORTANT); cascaded 14a POST→GET change through client implementation and all test method assertions.

### 2026-05-16 — 4-reviewer panel verdict CHANGES_REQUESTED → fixes applied

Reviewers: reality, architecture, quality, systems (full report:
`14-phase-2-audit-readiness-panel.review.json`).

Fixes applied to 14b in this revision:
1. loadSnapshot stale-fetch race — monotonic write guard + AbortController + concurrent-fetch test (convergence C2)
2. Frontend wire shape matches scoped_secret_resolver backend decision — no auth_provider_type field (C4 verified clean — no `auth_provider_type` field was present in 14b)
3. Phantom api/preferences.ts citation replaced with verified client.ts:403–424 location (review reality blocker)
4. makeComposition test fixture covers full NodeSpec arity (review B1, frontend mirror)
5. tsc + lint added to every Task commit checklist (quality REC)

### 2026-05-16 — Round-2 panel verdict CHANGES_REQUESTED → fixes applied

Reviewers: reality, architecture, quality, systems (full report:
`14-phase-2-audit-readiness-panel.review-r2.json`). All 11 prior blockers confirmed
resolved; fixes below address the round-2 warnings that affect this file.

Fixes applied to 14b in this revision:
1. R2-W2 — AbortError catch arm now resets `isLoading: false` before returning (was: bare `return`, leaving panel stuck at "Checking audit readiness…" after session navigation during in-flight fetch). `clearSession` set() payload updated with `isLoading: false` as belt-and-suspenders. (Amended 2026-05-17: landed code uses `isLoadingBySession[sessionId]: false` — per-session map; see frontend-005.)
2. R2-W3 — Existing sequential test re-labelled "monotonic write guard — sequential ordering" (correctly exercises the version monotonicity guard). New true-concurrency abort-cancellation test added: starts both fetches without awaiting the first, delays mock fetch, asserts `AbortController.signal.aborted === true` after the second starts, and asserts `isLoading` resets to false cleanly. Two distinct contracts, two distinct tests. (Amended 2026-05-17: assertion uses `isLoadingBySession[SESSION_ID]` — per-session; see frontend-005.)
3. R2-W6 — `makeComposition()` fixture extraction prescribed: canonical location `frontend/src/test/composerFixtures.ts` (amended 2026-05-17: implementation placed it at `src/test/`, not `src/test-utils/` — see frontend-004); `AuditReadinessPanel.test.tsx` imports from there. 14c will import from the same module.
4. R2-W9 — `loadExplain` AbortController asymmetry addressed: option (a) prescribed (mirror the abort-controller pattern via a parallel `explainAbortControllers` field; `clearSession` aborts both controllers). Option (b) documented as the operator/architect alternative.
5. carry-W2 — Prose claim "same pattern used by `executionStore`" corrected to "similar shape to `executionStore`'s `reset()` pattern": `executionStore` uses a module-internal `initialExecutionState` const (not an exported `getInitialState`).

### 2026-05-17 — Post-landing audit reconciliation (elspeth-a615f8c418)

Source findings: frontend-001, frontend-003, frontend-004, frontend-005, frontend-006, frontend-007.

**frontend-001 (14b-1):** `parseResponse` in `api/auditReadiness.ts` (lines 51–66) includes a
defensive shape check for `session_id` (string) and `composition_version` (number). Authorized
exception to CLAUDE.md trust-tier — check defends against transport-layer corruption
(proxy/CDN body corruption), not data-origin trust. Documented in the Trust-tier check section.
Future strengthening tracked as `elspeth-obs-2e58958765`.

**frontend-003 (14b-2):** `AuditReadinessPanel` `useEffect` deps list uses
`compositionState?.version` instead of `compositionState` to prevent the effect from
re-firing on every render that creates a new object reference without an actual version
change. Suppresses `react-hooks/exhaustive-deps` lint rule at the call site with an inline
`eslint-disable-next-line` comment and an explanatory note.

**frontend-004 (14b-3):** Fixture module canonicalized at `src/test/composerFixtures.ts`
(not `src/test-utils/composerFixtures.ts` as originally prescribed). All plan references
updated. The R2-W6 Review-history entry (2026-05-16) has been updated to match. 14c already
uses `@/test/composerFixtures` — no change required there.

**frontend-005 (14b-4):** Store implemented with per-session keyed maps
(`isLoadingBySession`, `errorBySession`, `isLoadingExplainBySession`, `explainErrorBySession`)
instead of flat scalars (`isLoading`, `error`, etc.). Reason: flat scalars cause
cross-session contamination — switching from a failing session A to a healthy session B
would render A's error banner on B. Landed in commit `c3f3d0ae6`. Task 3 code blocks
updated to reflect the per-session shape, including the controller-identity guard in the
AbortError arm. 14c's Task 4C extends this with `userExpandedBySession` for the same reason.

**frontend-006 (14b-5):** Placeholder `ReadinessRowDetail` and `ExplainDialog` ship without
`role="dialog"` to avoid the W2 accessibility defect (claiming dialog semantics without
focus management). Both are tagged with `data-testid` instead. The 14c real implementations
will re-introduce `role="dialog"` with the full focus contract (`useFocusTrap`, etc.) per
14c's B3 fix.

**frontend-007 (14b-6):** `stores/subscriptions.ts` and `stores/subscriptions.test.ts`
added by the 2B.5 follow-up (commits `084b8c34b` + `163737068`). Not in the original plan.
Task 2B.5 subsection and file-structure listing added to document landed work.

## Memory references

- `project_composer_personas` — Linda-vocabulary row labels.
- `feedback_no_calendar_shipping_commitments` — no SLAs in this plan.
- `feedback_default_is_fix_not_ticket` — placeholder behaviour is bounded by the explicit "DO NOT extend" comment; extensions belong in 14c and shouldn't drift into 14b.

---

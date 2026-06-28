# Phase 1B — Frontend: default-mode wiring, opt-out surfaces, banner, smoke deploy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the frontend half of Phase 1 — typed API client, Zustand store, bootstrap on auth-success, `createSession` honours the preference, three opt-out surfaces (settings pane, inline checkbox, banner-mentioned link), one-time banner, and the staging smoke test that exercises Phase 1 end-to-end.

**Architecture:** Backend-first via Phase 1A. This plan adds a `preferencesStore` (Zustand) cached from `/api/composer-preferences`, bootstrapped at auth-success and consulted by `sessionStore.createSession`. Three UI surfaces write to the same store, which writes through to the backend. A persistent banner fires for opted-out users until dismissed.

**Tech Stack:** React + Zustand + Vitest + testing-library.

**Sibling plan:** [12-phase-1a-backend.md](12-phase-1a-backend.md) — backend schema, Pydantic models, service, routes.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md).

---

## Scope boundaries

**In scope:**
- New types in `types/api.ts` for the preferences payload.
- New API client wrappers (`fetchUserComposerPreferences`, `updateUserComposerPreferences`) added directly to `api/client.ts` as neighbors of the existing per-session preference helpers (rationale in Task 1's Convention reminder).
- New `preferencesStore` (Zustand) with bootstrap, optimistic write, banner-dismissal.
- `App.tsx` bootstraps preferences on auth-success.
- `sessionStore.createSession` reads `defaultMode` and routes new sessions into guided- or freeform-entry.
- New `ComposerPreferencesPanel.tsx` (settings pane) — radio group writing to the store.
- New `UserMenu.tsx` (top-right dropdown) — entry into Settings.
- New `InlineOptOutCheckbox.tsx` in guided-mode chrome — the inline opt-out.
- New `DefaultModeChangedBanner.tsx` — one-time dismissible banner for opted-out users.
- Staging smoke task (delete DB, deploy, exercise the user journeys).

**Out of scope:**
- Backend (Phase 1A delivered).
- Telemetry on opt-out rate (Phase 8).
- Mode-related layout changes beyond the new components (Phase 3).
- Catalog reshape (Phase 7).

## Sequencing and dependencies

Phase 1A **must** be merged (and the DB delete performed at Task 9 below) before this plan's smoke task can succeed. Tasks 1-8 of this plan can be merged ahead of Phase 1A only if test isolation is maintained — but the recommended sequence is **ship 1A first, then ship 1B's tasks in order, then run Task 9 once both halves are deployed.**

## Verification approach

Each task is TDD-shaped. Task 9 is a manual smoke task that exercises both halves end-to-end. The plan does not declare Phase 1 complete on green tests alone.

---

## Task 1: Frontend types and API client

**Convention reminder (read before writing):** `src/elspeth/web/frontend/src/api/client.ts`
is the single source for HTTP wrappers. The new user-prefs functions MUST live there as
direct neighbors of the existing per-session `fetchComposerPreferences(sessionId)` /
`updateComposerPreferences(sessionId, body)` (`client.ts:371-396`). Three reasons:

1. **`authHeaders` is module-private** (`client.ts:58` is declared `function authHeaders(...)` with
   no `export`). It is not importable from outside `client.ts`. A separate `api/preferences.ts`
   module would either require exporting it (widening the module surface for one caller) or
   re-implementing the helper (drift hazard).
2. **`parseResponse` is module-private and load-bearing.** Every existing exported wrapper goes
   through `parseResponse` (`client.ts:98`), which (a) triggers `useAuthStore.getState().logout()`
   on 401 responses (token-expiry recovery) and (b) decodes the FastAPI envelope into the typed
   `ApiError` shape. Hand-rolling `if (!response.ok) throw new Error(...)` outside `client.ts`
   silently bypasses both. This is a CLAUDE.md "patterns are what you'll replicate" issue —
   the next reviewer will catch it.
3. **Naming symmetry.** The existing per-session pair lives in `client.ts`; placing the
   account-level pair next to it lets the diff for "where do composer prefs live?" surface
   both at once. A separate file fragments the answer.

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/api.ts` — add `ComposerMode` and payload types.
- Modify: `src/elspeth/web/frontend/src/api/client.ts` — add `fetchUserComposerPreferences()`
  and `updateUserComposerPreferences(payload)` as exported functions alongside the existing
  per-session `fetchComposerPreferences` / `updateComposerPreferences`.
- Create: `src/elspeth/web/frontend/src/api/client.preferences.test.ts` — sibling test
  module (mirrors `client.guided.test.ts` / `client.recovery.test.ts` conventions).

- [ ] **Step 1: Confirm the auth pattern in client.ts**

Read `src/elspeth/web/frontend/src/api/client.ts:58` and `:98`. Confirm:
- `authHeaders(contentType?: string): HeadersInit` is **module-private** (no `export`).
- `parseResponse<T>(response)` is **module-private** and handles 401-logout + envelope decoding.
- The existing exported wrappers (e.g. `fetchSessions`, `createSession`, `fetchComposerPreferences`)
  all use the pattern: `const response = await fetch(url, { headers: authHeaders(...), ... });
  return parseResponse<T>(response);`. The new functions MUST follow this pattern verbatim.

- [ ] **Step 2: Write the failing test**

Create `src/elspeth/web/frontend/src/api/client.preferences.test.ts`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fetchUserComposerPreferences, updateUserComposerPreferences } from "./client";

describe("preferences API client", () => {
  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  it("GET parses ComposerPreferences payload", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          default_mode: "guided",
          banner_dismissed_at: null,
          updated_at: "2026-05-15T00:00:00Z",
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const prefs = await fetchUserComposerPreferences();
    expect(prefs.default_mode).toBe("guided");
    expect(prefs.banner_dismissed_at).toBeNull();
  });

  it("PATCH sends partial body", async () => {
    const mock = globalThis.fetch as ReturnType<typeof vi.fn>;
    const payload = { default_mode: "freeform" as const, banner_dismissed_at: null, updated_at: "2026-05-15T00:00:00Z" };
    mock.mockResolvedValueOnce(new Response(JSON.stringify(payload), { status: 200, headers: { "content-type": "application/json" } }));

    await updateUserComposerPreferences({ default_mode: "freeform" });

    const init = mock.mock.calls[0][1];
    expect(init?.method).toBe("PATCH");
    expect(JSON.parse(init?.body as string)).toEqual({ default_mode: "freeform" });
  });

  it("GET throws on non-2xx", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("server error", { status: 500 }),
    );
    await expect(fetchUserComposerPreferences()).rejects.toThrow();
  });

  it("PATCH throws on non-2xx (e.g., 422 invalid mode)", async () => {
    (globalThis.fetch as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      new Response("invalid", { status: 422 }),
    );
    // @ts-expect-error -- intentionally invalid mode for the 422 path
    await expect(updateUserComposerPreferences({ default_mode: "kiosk" })).rejects.toThrow();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/api/client.preferences.test.ts`
Expected: FAIL — `fetchUserComposerPreferences` is not exported from `./client`.

- [ ] **Step 4: Add the types**

In `src/elspeth/web/frontend/src/types/api.ts`, append:

```typescript
export type ComposerMode = "guided" | "freeform";

export interface UserComposerPreferencesPayload {
  default_mode: ComposerMode;
  banner_dismissed_at: string | null;
  updated_at: string;
}

export interface UpdateUserComposerPreferencesPayload {
  default_mode?: ComposerMode;
  banner_dismissed_at?: string | null;
}
```

Note: do **not** rename or remove the existing per-session `ComposerPreferences` interface
in `src/elspeth/web/frontend/src/types/index.ts:190` (which carries `trust_mode` /
`density_default`). The two types coexist; `User*` is the disambiguator chosen for the
account-level payload to avoid the name collision.

- [ ] **Step 5: Implement the API functions inside `client.ts`**

In `src/elspeth/web/frontend/src/api/client.ts`, add to the `import type` block at the top:

```typescript
import type {
  UserComposerPreferencesPayload,
  UpdateUserComposerPreferencesPayload,
} from "@/types/api";
```

Place the two functions immediately **after** the existing
`updateComposerPreferences(sessionId, body)` (around `client.ts:397`), so the per-session and
account-level wrappers sit as visible neighbors:

```typescript
/** Get the user's account-level composer preferences (Phase 1A). */
export async function fetchUserComposerPreferences(): Promise<UserComposerPreferencesPayload> {
  const response = await fetch("/api/composer-preferences", {
    headers: authHeaders(),
  });
  return parseResponse<UserComposerPreferencesPayload>(response);
}

/** Update the user's account-level composer preferences (Phase 1A). Partial PATCH. */
export async function updateUserComposerPreferences(
  payload: UpdateUserComposerPreferencesPayload,
): Promise<UserComposerPreferencesPayload> {
  const response = await fetch("/api/composer-preferences", {
    method: "PATCH",
    headers: authHeaders("application/json"),
    body: JSON.stringify(payload),
  });
  return parseResponse<UserComposerPreferencesPayload>(response);
}
```

**Critical:** route the response through `parseResponse<T>(response)`, not a hand-rolled
`if (!response.ok) throw new Error(...)`. `parseResponse` is module-private (`client.ts:98`)
and handles the project's 401-logout interceptor + typed `ApiError` envelope. Bypassing it
breaks token-expiry recovery on preferences calls and yields untyped `Error` instances that
downstream catch blocks cannot triage by `error_type`.

- [ ] **Step 6: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/api/client.preferences.test.ts`
Expected: PASS — 4 tests green.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/types/api.ts src/elspeth/web/frontend/src/api/client.ts src/elspeth/web/frontend/src/api/client.preferences.test.ts
git commit -m "feat(web/frontend): typed user-composer-preferences API helpers (Phase 1B.1)"
```

## Task 2: preferencesStore (Zustand) with bootstrap + optimistic write

**Files:**
- Create: `src/elspeth/web/frontend/src/stores/preferencesStore.ts`.
- Create: `src/elspeth/web/frontend/src/stores/preferencesStore.test.ts`.

**Mock convention.** The project's existing store tests (e.g. `sessionStore.test.ts:15`)
mock `@/api/client` at module load via `vi.mock("@/api/client", () => ({...}))` and
override per-test with `vi.mocked(fn).mockResolvedValueOnce(...)`. The test sketch below
uses `vi.spyOn(preferencesApi, ...)` for brevity, which works in vitest but is **not**
the project convention. Translate to the project pattern when implementing:

```typescript
// Top of preferencesStore.test.ts — project-convention mock layout
vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));
// And in each test:
vi.mocked(preferencesApi.fetchUserComposerPreferences).mockResolvedValueOnce({...});
```

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { usePreferencesStore } from "./preferencesStore";
import { resetStore } from "@/test/store-helpers";
import * as preferencesApi from "@/api/client";

describe("preferencesStore", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.restoreAllMocks();
  });

  it("loads preferences from API on bootstrap", async () => {
    vi.spyOn(preferencesApi, "fetchUserComposerPreferences").mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().bootstrap();

    const state = usePreferencesStore.getState();
    expect(state.defaultMode).toBe("freeform");
    expect(state.bannerDismissedAt).toBeNull();
    expect(state.loaded).toBe(true);
  });

  it("setDefaultMode updates state optimistically and persists", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    vi.spyOn(preferencesApi, "updateUserComposerPreferences").mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().setDefaultMode("freeform");

    expect(usePreferencesStore.getState().defaultMode).toBe("freeform");
    expect(preferencesApi.updateUserComposerPreferences).toHaveBeenCalledWith({
      default_mode: "freeform",
    });
  });

  it("setDefaultMode reverts on error", async () => {
    usePreferencesStore.setState({ defaultMode: "guided", loaded: true });
    vi.spyOn(preferencesApi, "updateUserComposerPreferences").mockRejectedValueOnce(
      new Error("network failure"),
    );

    await expect(
      usePreferencesStore.getState().setDefaultMode("freeform"),
    ).rejects.toThrow("network failure");

    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
    expect(usePreferencesStore.getState().writing).toBe(false);
  });

  it("setDefaultMode ignores concurrent calls while writing", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided", writing: true });
    const spy = vi.spyOn(preferencesApi, "updateUserComposerPreferences");
    await usePreferencesStore.getState().setDefaultMode("freeform");
    expect(spy).not.toHaveBeenCalled();
    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
  });

  it("dismissDefaultChangedBanner persists timestamp", async () => {
    const stamp = "2026-05-15T12:00:00Z";
    usePreferencesStore.setState({ loaded: true, defaultMode: "freeform" });
    vi.spyOn(preferencesApi, "updateUserComposerPreferences").mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: stamp,
      updated_at: stamp,
    });

    await usePreferencesStore.getState().dismissDefaultChangedBanner();

    expect(usePreferencesStore.getState().bannerDismissedAt).toBe(stamp);
  });

  it("dismissDefaultChangedBanner is no-op while another write is in flight", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
      writing: true,
    });
    const spy = vi.spyOn(preferencesApi, "updateUserComposerPreferences");
    await usePreferencesStore.getState().dismissDefaultChangedBanner();
    expect(spy).not.toHaveBeenCalled();
    expect(usePreferencesStore.getState().bannerDismissedAt).toBeNull();
  });

  it("banner reappears if backend dismiss fails (revert-on-error)", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
    });
    vi.spyOn(preferencesApi, "updateUserComposerPreferences").mockRejectedValueOnce(
      new Error("server error"),
    );

    await expect(
      usePreferencesStore.getState().dismissDefaultChangedBanner(),
    ).rejects.toThrow("server error");

    // State reverts to null — banner reappears.
    expect(usePreferencesStore.getState().bannerDismissedAt).toBeNull();
    expect(usePreferencesStore.getState().writing).toBe(false);
  });

  it("bootstrap is re-entrant safe (always fetches; two calls = two API calls)", async () => {
    const spy = vi.spyOn(preferencesApi, "fetchUserComposerPreferences").mockResolvedValue({
      default_mode: "guided", banner_dismissed_at: null, updated_at: "2026-05-15T00:00:00Z",
    });
    await usePreferencesStore.getState().bootstrap();
    await usePreferencesStore.getState().bootstrap();
    expect(usePreferencesStore.getState().loaded).toBe(true);
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the store**

Create `src/elspeth/web/frontend/src/stores/preferencesStore.ts`:

```typescript
// src/stores/preferencesStore.ts
import { create } from "zustand";
import {
  fetchUserComposerPreferences,
  updateUserComposerPreferences,
} from "@/api/client";
import type { ComposerMode } from "@/types/api";

interface PreferencesState {
  /** null until bootstrap completes. Components must gate on loaded === true. */
  defaultMode: ComposerMode | null;
  bannerDismissedAt: string | null;
  loaded: boolean;
  writing: boolean;

  /** Load from the backend. Always re-fetches (re-entrant safe). */
  bootstrap: () => Promise<void>;
  /** Await bootstrap if not loaded; return the resolved mode. Use from createSession. */
  resolveDefaultMode: () => Promise<ComposerMode>;
  /** Write a new default mode. Optimistic update + persist; reverts on error. */
  setDefaultMode: (mode: ComposerMode) => Promise<void>;
  /** Mark the 'default changed' banner as dismissed (one-time per user). */
  dismissDefaultChangedBanner: () => Promise<void>;
  /** Restore initial state — used by test/store-helpers resetStore(). */
  reset: () => void;
}

const INITIAL_STATE = {
  defaultMode: null as ComposerMode | null,
  bannerDismissedAt: null as string | null,
  loaded: false,
  writing: false,
};

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  ...INITIAL_STATE,

  bootstrap: async () => {
    const payload = await fetchUserComposerPreferences();
    set({
      defaultMode: payload.default_mode,
      bannerDismissedAt: payload.banner_dismissed_at,
      loaded: true,
    });
  },

  resolveDefaultMode: async () => {
    const s = get();
    if (s.loaded) return s.defaultMode!;
    await get().bootstrap();
    return get().defaultMode!;
  },

  setDefaultMode: async (mode) => {
    // Guard: no-op while a write is already in flight (prevents race on rapid clicks).
    if (get().writing) return;
    const previous = get().defaultMode;
    set({ defaultMode: mode, writing: true });
    try {
      const payload = await updateUserComposerPreferences({ default_mode: mode });
      set({ defaultMode: payload.default_mode, writing: false });
    } catch (err) {
      set({ defaultMode: previous, writing: false });
      throw err;
    }
  },

  dismissDefaultChangedBanner: async () => {
    // Same writing-guard as setDefaultMode — both writes go through the same
    // PATCH route and share the writing flag. Without this guard, a rapid
    // settings-panel write followed by a banner-dismiss click could race:
    // the second optimistic set + revert path overwrites the first's pending
    // result. The guard makes the second call a no-op; the user's banner
    // will get dismissed on the next click after the first write settles.
    if (get().writing) return;
    const stamp = new Date().toISOString();
    const previous = get().bannerDismissedAt;
    set({ bannerDismissedAt: stamp, writing: true });
    try {
      const payload = await updateUserComposerPreferences({
        banner_dismissed_at: stamp,
      });
      set({
        bannerDismissedAt: payload.banner_dismissed_at,
        writing: false,
      });
    } catch (err) {
      // Revert: banner reappears if the backend call failed.
      set({ bannerDismissedAt: previous, writing: false });
      throw err;
    }
  },

  reset: () => set(INITIAL_STATE),
}));
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts`
Expected: PASS — 8 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/preferencesStore.ts src/elspeth/web/frontend/src/stores/preferencesStore.test.ts
git commit -m "feat(web/frontend): preferencesStore with bootstrap + setDefaultMode (Phase 1B.2)"
```

## Task 3: Wire bootstrap from `App.tsx` on auth-success

**Files:**
- Modify: `src/elspeth/web/frontend/src/App.tsx` — call `bootstrap()` when auth succeeds.
- Modify: `src/elspeth/web/frontend/src/App.test.tsx` — add coverage **and** extend the
  existing `vi.mock("./api/client", ...)` (at `App.test.tsx:104`) to declare
  `fetchUserComposerPreferences: vi.fn()` and `updateUserComposerPreferences: vi.fn()`.
  Without these, the real `usePreferencesStore` module loaded by the new test will see
  `undefined` for the API functions and throw on import.

- [ ] **Step 1: Read App.tsx to find the auth-success hook**

Open `src/elspeth/web/frontend/src/App.tsx`. **Important orientation:** the current App
component does **not** consume `useAuth()` directly — auth gating is handled by
`<AuthGuard>` (the JSX wrapper at the bottom of `App`, line 196), which itself calls
`useAuth()` and renders the login page when unauthenticated. App's body therefore only
runs when authentication has succeeded.

You have two equivalent options for the bootstrap call site:

- **(A) Add `useAuth()` to App and gate the effect on `isAuthenticated`.** Mirrors the
  pattern the plan's test mock expects (`App.test.tsx:81-89` already mocks
  `./hooks/useAuth` to return `{ isAuthenticated: true, ... }`, so the test exercises this
  path directly). Preferred — explicit and test-mock-aligned.
- **(B) Place the effect inside a small new component rendered *inside* `<AuthGuard>`.**
  Equivalent runtime behaviour because AuthGuard children only render once auth succeeds.
  Use this if Task 3's test imports cleanly without modifying App.

Pick (A). The rest of this task assumes (A).

Also locate where `initStoreSubscriptions()` runs (`App.tsx:27` — module load) so you do
not duplicate that pattern.

- [ ] **Step 2: Write the failing test**

Add to `src/elspeth/web/frontend/src/App.test.tsx`. The existing file already mocks `useAuth` via `vi.mock("./hooks/useAuth", ...)` — add a second `describe` block that extends that mock:

```typescript
import { render, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import App from "./App";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

describe("App preferences bootstrap", () => {
  // Zustand stores are module-level singletons; without resetStore() any
  // earlier test in App.test.tsx (or any sibling test imported into the same
  // vitest worker) that touched usePreferencesStore would leak state here.
  // This mirrors the Phase 1A finding 7 pattern — every store-touching
  // describe needs resetStore() in beforeEach.
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.restoreAllMocks();
  });

  it("calls preferences.bootstrap once authenticated", async () => {
    const bootstrap = vi
      .spyOn(usePreferencesStore.getState(), "bootstrap")
      .mockResolvedValueOnce(undefined);

    // useAuth is already mocked in the file's top-level vi.mock to return
    // { isAuthenticated: true, ... }. No additional setup needed here.
    render(<App />);

    await waitFor(() => {
      expect(bootstrap).toHaveBeenCalled();
    });
  });
});
```

The "not authenticated" path is covered implicitly by existing tests that exercise the LoginPage; there is no need to add a second test that directly manipulates auth state.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/App.test.tsx -t "preferences bootstrap"`
Expected: FAIL — bootstrap is never called.

- [ ] **Step 4: Add the bootstrap call**

In `App.tsx`, near where other auth-success bootstrap logic lives:

```typescript
const { isAuthenticated } = useAuth();
const bootstrapPrefs = usePreferencesStore((s) => s.bootstrap);

useEffect(() => {
  if (!isAuthenticated) return;
  bootstrapPrefs().catch((err) => {
    // Preferences failure is non-fatal: the store keeps its defaults
    // (guided / not-dismissed). The error is preserved in DevTools for
    // operators investigating.
    console.error("[preferences] bootstrap failed:", err);
  });
}, [isAuthenticated, bootstrapPrefs]);
```

Use `useAuth()` — not `useAuthStore` directly — to keep the test mock surface stable. `authStore` has no `authenticated` field; `useAuth()` encapsulates the `selectIsAuthenticated` selector.

Place this effect alongside the other auth-gated effects.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/App.test.tsx -t "preferences bootstrap"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/App.tsx src/elspeth/web/frontend/src/App.test.tsx
git commit -m "feat(web/frontend): bootstrap composer prefs on auth success (Phase 1B.3)"
```

## Task 4: `sessionStore.createSession` honours the default mode

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` — `createSession` action.
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.test.ts` — new test cases, and
  **extend the top-of-file `vi.mock("@/api/client", ...)` mock list** to include the two
  new functions:

  ```typescript
  vi.mock("@/api/client", () => ({
    // ...existing mocks...
    fetchUserComposerPreferences: vi.fn(),
    updateUserComposerPreferences: vi.fn(),
  }));
  ```

  Without this, the real `preferencesStore` (imported by the new test cases) will see
  `undefined` for the API functions and the tests will throw at module-load time. The
  same extension applies to any other test file that imports `usePreferencesStore`
  alongside the existing `@/api/client` mock (search for `vi.mock("@/api/client"` to
  audit all call sites).

- [ ] **Step 1: Map the current createSession + guided entry shape**

Read `sessionStore.ts:266` and the surrounding context. Specifically locate:
- The existing `createSession` body.
- The existing "Switch to guided" action (search for `enterGuided`, `bootstrapGuided`, `startGuided`, or similar — Phase A slice 1 of per-step-chat introduced this).
- The state field that tracks "this session is in guided mode" (likely `guidedSession`, which is `null` for freeform sessions).

The bootstrap goal: after a session is created, if `defaultMode === "guided"`, run the same flow the user's "Switch to guided" button runs.

- [ ] **Step 2: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useSessionStore } from "./sessionStore";
import { usePreferencesStore } from "./preferencesStore";
import * as api from "@/api/client";

describe("sessionStore.createSession honours default mode", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useSessionStore.setState({
      activeSessionId: null,
      guidedSession: null,
      // ...whatever else needs resetting; copy the existing test's beforeEach.
    } as never);
  });

  it("leaves guidedSession null when default mode is freeform", async () => {
    usePreferencesStore.setState({
      defaultMode: "freeform",
      bannerDismissedAt: null,
      loaded: true,
      writing: false,
    });
    vi.spyOn(api, "createSession").mockResolvedValueOnce({
      id: "sess-1",
      title: "untitled",
      // ...adapt to actual SessionRecord shape
    } as never);

    await useSessionStore.getState().createSession();

    expect(useSessionStore.getState().guidedSession).toBeNull();
  });

  it("enters guided mode when default mode is guided", async () => {
    usePreferencesStore.setState({
      defaultMode: "guided",
      bannerDismissedAt: null,
      loaded: true,
      writing: false,
    });
    vi.spyOn(api, "createSession").mockResolvedValueOnce({
      id: "sess-2",
      title: "untitled",
    } as never);
    // Spy on whatever guided-entry method exists (adjust name).
    const enterGuided = vi
      .spyOn(useSessionStore.getState(), "enterGuided")
      .mockResolvedValueOnce();

    await useSessionStore.getState().createSession();

    expect(enterGuided).toHaveBeenCalled();
  });

  it("bootstrap race: createSession before bootstrap resolves respects eventual preference", async () => {
    resetStore(usePreferencesStore); // loaded=false, defaultMode=null

    let resolveBootstrap!: () => void;
    vi.spyOn(usePreferencesStore.getState(), "bootstrap").mockReturnValueOnce(
      new Promise<void>((res) => {
        resolveBootstrap = () => { usePreferencesStore.setState({ defaultMode: "guided", loaded: true }); res(); };
      }),
    );
    vi.spyOn(api, "createSession").mockResolvedValueOnce({ id: "sess-3", title: "untitled" } as never);
    const enterGuided = vi.spyOn(useSessionStore.getState(), "enterGuided").mockResolvedValueOnce();

    const createPromise = useSessionStore.getState().createSession();
    resolveBootstrap();
    await createPromise;

    expect(enterGuided).toHaveBeenCalled(); // createSession waited and entered guided
  });
});
```

The exact action name (`enterGuided` vs. `startGuided`) is identified in Step 1. Adapt the test before running it. Add `import { resetStore } from "@/test/store-helpers";` to the imports.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/sessionStore.test.ts -t "honours default mode"`
Expected: FAIL — current `createSession` ignores preferences entirely.

- [ ] **Step 4: Implement the read in createSession**

The current `createSession` (`sessionStore.ts:266-294`) wraps the entire body in a single
`try { ... } catch { set({ error: ... }) }`. The new logic must be placed **inside** the
existing `try` block, immediately after the `set(...)` that populates `activeSessionId`,
so that:

- A failure of `api.createSession()` short-circuits before the guided-entry attempt
  (no `enterGuided` on an unset session).
- An `enterGuided` failure surfaces through the existing `catch` and sets the same
  `state.error`. This is acceptable for Phase 1 (the user can retry); a tighter error
  attribution is a Phase 3 follow-up filed as an observation in Task 9.

```typescript
async createSession() {
  try {
    const session = await api.createSession();
    clearComposerProgressPollTimer();
    set((state) => ({
      sessions: [session, ...state.sessions],
      activeSessionId: session.id,
      // ...existing reset fields...
    }));

    // resolveDefaultMode() awaits bootstrap if not yet loaded, closing the
    // Ctrl+N race (user hits "new session" before App.tsx's bootstrap effect
    // has resolved). usePreferencesStore.getState() is the bare imperative
    // store-to-store call — not a React hook.
    const mode = await usePreferencesStore.getState().resolveDefaultMode();
    if (mode === "guided") {
      await get().enterGuided();
    }
  } catch {
    set({ error: "Failed to create session. Please try again." });
  }
},
```

Use `resolveDefaultMode()`, not `getState().defaultMode` directly — the bare field read returns `null` before bootstrap. The existing `enterGuided` action (`sessionStore.ts:1057`) is the correct entry point: it routes between `startGuided` (cold path) and `reenterGuided` (terminal === "exited_to_freeform") based on the live `guidedSession` state, which `createSession` just cleared via `clearedGuidedState()` — so `enterGuided` will take the `startGuided` path.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/sessionStore.test.ts -t "honours default mode"`
Expected: PASS.

Also run the full sessionStore suite to catch regressions:
Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/sessionStore.test.ts`
Expected: PASS — all existing tests still green.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/sessionStore.ts src/elspeth/web/frontend/src/stores/sessionStore.test.ts
git commit -m "feat(web/frontend): new sessions respect composer.default_mode (Phase 1B.4)"
```

## Task 4.5: Integration test — preferences → session wire

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.test.ts` — add one integration test.

Real stores, API-layer mocked. No store-action spies. Asserts the full chain: `preferencesStore.defaultMode = "guided"` → `sessionStore.createSession()` → `enterGuided` called.

- [ ] **Step 1: Add integration test to preferencesStore.test.ts**

```typescript
import { useSessionStore } from "@/stores/sessionStore";
import * as api from "@/api/client";

describe("preferences → session integration (real stores, API mocked)", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    useSessionStore.setState({ activeSessionId: null, guidedSession: null } as never);
    vi.restoreAllMocks();
  });

  it("createSession enters guided when preference is guided", async () => {
    usePreferencesStore.setState({ defaultMode: "guided", loaded: true, writing: false, bannerDismissedAt: null });
    vi.spyOn(api, "createSession").mockResolvedValueOnce({ id: "sess-i", title: "untitled" } as never);
    const enterGuided = vi.spyOn(useSessionStore.getState(), "enterGuided").mockResolvedValueOnce(undefined);

    await useSessionStore.getState().createSession();

    expect(enterGuided).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/stores/preferencesStore.test.ts`
Expected: PASS — all tests including the new integration test.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/preferencesStore.test.ts
git commit -m "test(web/frontend): integration test preferences → session wire (Phase 1B.4.5)"
```

## Task 5: ComposerPreferencesPanel settings pane

**Files:**
- Create: `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx`.
- Create: `src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.test.tsx`.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ComposerPreferencesForm } from "./ComposerPreferencesPanel";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

describe("ComposerPreferencesForm", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    vi.restoreAllMocks();
  });

  it("renders the current default-mode selection", () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/freeform/i)).toBeChecked();
    expect(screen.getByLabelText(/guided/i)).not.toBeChecked();
  });

  it("writes the new default-mode on selection", async () => {
    const setDefault = vi.spyOn(usePreferencesStore.getState(), "setDefaultMode").mockResolvedValueOnce();
    render(<ComposerPreferencesForm />);
    await userEvent.click(screen.getByLabelText(/freeform/i));
    expect(setDefault).toHaveBeenCalledWith("freeform");
  });

  it("disables inputs while writing", () => {
    usePreferencesStore.setState({ writing: true });
    render(<ComposerPreferencesForm />);
    expect(screen.getByLabelText(/guided/i)).toBeDisabled();
    expect(screen.getByLabelText(/freeform/i)).toBeDisabled();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/settings/ComposerPreferencesPanel.test.tsx`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement the component**

```typescript
// src/components/settings/ComposerPreferencesPanel.tsx
import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import type { ComposerMode } from "@/types/api";

/** Inner form — used standalone in tests and embedded in ComposerPreferencesPanel. */
export function ComposerPreferencesForm(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);

  // useCallback must be unconditional (React rules). Early return follows.
  const onChange = useCallback(
    async (mode: ComposerMode) => {
      try { await setDefaultMode(mode); }
      catch (err) { console.error("[preferences] setDefaultMode failed:", err); }
    },
    [setDefaultMode],
  );

  if (!loaded) return null; // defaultMode is null before bootstrap — gate here

  return (
    <fieldset disabled={writing} aria-busy={writing}>
      <legend>Default mode for new sessions</legend>
      <label>
        <input type="radio" name="composer-default-mode" value="guided"
          checked={defaultMode === "guided"} disabled={writing}
          onChange={() => onChange("guided")} />
        <span>Guided (recommended)</span>
      </label>
      <label>
        <input type="radio" name="composer-default-mode" value="freeform"
          checked={defaultMode === "freeform"} disabled={writing}
          onChange={() => onChange("freeform")} />
        <span>Freeform</span>
      </label>
    </fieldset>
  );
}

/**
 * Full panel — matches the SecretsPanel modal pattern verbatim (backdrop +
 * focus trap + Escape-to-close + role=dialog + aria-modal=true). Do not
 * substitute a lighter dialog; the project's a11y convention is the
 * SecretsPanel shape (see src/components/settings/SecretsPanel.tsx).
 */
export function ComposerPreferencesPanel({ onClose }: { onClose: () => void }): JSX.Element {
  const modalRef = useRef<HTMLDivElement>(null);
  // Initial focus selector: the guided radio (which is the default).
  useFocusTrap(modalRef, true, "input[name='composer-default-mode'][value='guided']");

  // Close on Escape — same pattern as SecretsPanel.tsx:83-89.
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <>
      {/* Backdrop — click-to-dismiss, same pattern as SecretsPanel.tsx:122-131. */}
      <div
        role="presentation"
        onClick={onClose}
        style={{ position: "fixed", inset: 0, backgroundColor: "rgba(0,0,0,0.45)", zIndex: 100 }}
      />
      {/* Modal */}
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-label="Composer preferences"
        className="settings-panel"
        /* Inline-style positioning matches SecretsPanel.tsx:139-156. If a shared
           CSS class .settings-panel-modal exists by the time you implement, use
           that instead and drop the inline styles. */
      >
        <header className="settings-panel-header">
          <h2>Composer preferences</h2>
          <button type="button" aria-label="Close" onClick={onClose}>✕</button>
        </header>
        <div className="settings-panel-body">
          <ComposerPreferencesForm />
        </div>
      </div>
    </>
  );
}
```

Imports to add at the top of the file:

```typescript
import { useCallback, useEffect, useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";
```

Note: `disabled` on each `<input>` (in addition to the parent `fieldset`) keeps `toBeDisabled()` assertions direct. The test file imports `ComposerPreferencesForm` for the unit tests; `ComposerPreferencesPanel` is used by App.tsx.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/settings/ComposerPreferencesPanel.test.tsx`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.tsx src/elspeth/web/frontend/src/components/settings/ComposerPreferencesPanel.test.tsx
git commit -m "feat(web/frontend): ComposerPreferencesPanel settings pane (Phase 1B.5)"
```

## Task 6: UserMenu (top-right dropdown)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/UserMenu.tsx`.
- Create: `src/elspeth/web/frontend/src/components/common/UserMenu.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx` — slot in the header.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — open the settings modal on click.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UserMenu } from "./UserMenu";

describe("UserMenu", () => {
  it("shows settings + sign out items when opened", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(screen.getByRole("menuitem", { name: /settings/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /sign out/i })).toBeInTheDocument();
  });

  it("is closed by default", () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    expect(screen.queryByRole("menuitem", { name: /settings/i })).not.toBeInTheDocument();
  });

  it("calls onOpenSettings when 'Settings' is clicked, then closes", async () => {
    const openSettings = vi.fn();
    render(<UserMenu onOpenSettings={openSettings} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /settings/i }));
    expect(openSettings).toHaveBeenCalled();
    expect(screen.queryByRole("menuitem", { name: /settings/i })).not.toBeInTheDocument();
  });

  it("calls onSignOut when 'Sign out' is clicked", async () => {
    const signOut = vi.fn();
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={signOut} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.click(screen.getByRole("menuitem", { name: /sign out/i }));
    expect(signOut).toHaveBeenCalled();
  });

  it("closes when clicking outside", async () => {
    render(
      <div>
        <UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />
        <button>outside</button>
      </div>,
    );
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    expect(screen.getByRole("menuitem", { name: /settings/i })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /outside/i }));
    expect(screen.queryByRole("menuitem", { name: /settings/i })).not.toBeInTheDocument();
  });

  it("Escape key closes the menu and returns focus to trigger", async () => {
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    const trigger = screen.getByRole("button", { name: /account/i });
    await userEvent.click(trigger);
    expect(screen.getByRole("menuitem", { name: /settings/i })).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("menuitem", { name: /settings/i })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("Tab/Shift+Tab navigates between menu items (project convention: Tab not arrows)", async () => {
    // The project menu pattern (see CommandPalette.tsx) uses Tab/Shift+Tab
    // for item navigation, not arrow keys. This test validates that convention.
    render(<UserMenu onOpenSettings={vi.fn()} onSignOut={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: /account/i }));
    await userEvent.tab();
    expect(screen.getByRole("menuitem", { name: /settings/i })).toHaveFocus();
    await userEvent.tab();
    expect(screen.getByRole("menuitem", { name: /sign out/i })).toHaveFocus();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/UserMenu.test.tsx`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement the component**

```typescript
// src/components/common/UserMenu.tsx
import { useState, useCallback, useRef, useEffect } from "react";

interface UserMenuProps {
  onOpenSettings: () => void;
  onSignOut: () => void;
}

export function UserMenu({ onOpenSettings, onSignOut }: UserMenuProps): JSX.Element {
  const [open, setOpen] = useState(false);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const handle = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handle);
    return () => document.removeEventListener("mousedown", handle);
  }, [open]);

  // Close on Escape and return focus to trigger.
  useEffect(() => {
    if (!open) return;
    const handle = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, [open]);

  const onSettings = useCallback(() => {
    setOpen(false);
    onOpenSettings();
  }, [onOpenSettings]);

  const onSignOutClick = useCallback(() => {
    setOpen(false);
    onSignOut();
  }, [onSignOut]);

  return (
    <div ref={wrapperRef} className="user-menu">
      <button
        ref={triggerRef}
        type="button"
        aria-label="account menu"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        Account
      </button>
      {open && (
        <ul role="menu">
          <li role="menuitem" tabIndex={0} onClick={onSettings} onKeyDown={(e) => e.key === "Enter" && onSettings()}>
            Settings
          </li>
          <li role="menuitem" tabIndex={0} onClick={onSignOutClick} onKeyDown={(e) => e.key === "Enter" && onSignOutClick()}>
            Sign out
          </li>
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Mount UserMenu in Layout**

**Important — Layout has no header.** Read `src/elspeth/web/frontend/src/components/common/Layout.tsx`
(currently `Layout.tsx:224-291`). The component is a CSS-grid with three columns
(`layout-sidebar` / `layout-chat` / `layout-inspector`). There is no top-strip "header"
region today. Two acceptable mount sites:

- **(A, preferred) Extend the existing `layout-sidebar-toolbar`** (`Layout.tsx:232-264`),
  which already hosts the sidebar-collapse toggle and the theme-toggle. Add `<UserMenu>`
  as the third element in this toolbar, right-aligned via CSS. This reuses the existing
  vertical-space budget — no new header row, no impact on `calc(100vh - ...)` math.
- **(B) Introduce a new `layout-header` row** spanning all three grid columns. Requires
  updating the CSS grid template and is a larger change. Defer unless Phase 3 already
  plans this.

Pick (A). Thread `onOpenSettings` / `onSignOut` through `LayoutProps`. The UserMenu
component returns a button that opens a dropdown — the dropdown is positioned with
`position: absolute` relative to its wrapper, so it can live inside the sidebar toolbar
without affecting layout flow.

```typescript
// Layout.tsx — extend the props
interface LayoutProps {
  // ...existing...
  onOpenSettings: () => void;
  onSignOut: () => void;
  // ...
}

// Inside the existing layout-sidebar-toolbar, after the theme-toggle button
// (Layout.tsx:263), add:
<UserMenu onOpenSettings={onOpenSettings} onSignOut={onSignOut} />
```

If the sidebar toolbar is too cramped visually, you may right-align the UserMenu using
`margin-left: auto` on the menu wrapper (the toolbar is already flex per its current
class).

- [ ] **Step 5: Wire from App.tsx**

In `App.tsx`, add:

```typescript
const [showComposerSettings, setShowComposerSettings] = useState(false);
const logout = useAuthStore((s) => s.logout);

// Pass these to <Layout>:
<Layout
  onOpenSettings={() => setShowComposerSettings(true)}
  onSignOut={logout}
  /* ...existing props... */
>
  ...children...
</Layout>

{showComposerSettings && (
  <ComposerPreferencesPanel onClose={() => setShowComposerSettings(false)} />
)}
```

`ComposerPreferencesPanel` follows the `SecretsPanel` pattern (a full panel rendered as a Layout sibling with a `onClose` prop and focus trapping). It is structured with a header (title + close button) and a body containing the `<ComposerPreferencesForm />` radio group — mirroring `src/components/settings/SecretsPanel.tsx`. Do not introduce a new modal abstraction; the SecretsPanel pattern is the established project convention.

- [ ] **Step 6: Run tests to verify**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/UserMenu.test.tsx src/components/common/Layout.test.tsx src/App.test.tsx`
Expected: PASS — including any existing `Layout` tests that touch the header.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/UserMenu.tsx src/elspeth/web/frontend/src/components/common/UserMenu.test.tsx src/elspeth/web/frontend/src/components/common/Layout.tsx src/elspeth/web/frontend/src/App.tsx
git commit -m "feat(web/frontend): UserMenu in header with settings entry (Phase 1B.6)"
```

## Task 7: Inline opt-out checkbox on guided-mode chrome

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/guided/InlineOptOutCheckbox.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/guided/InlineOptOutCheckbox.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (or the per-step guided sidecar) — mount the checkbox in the guided-mode chrome.

- [ ] **Step 1: Find the right mount point in ChatPanel**

Read `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`. Find:
- Where guided mode renders (gated on `guidedSession` truthy).
- The guided stepper / header / sidecar — the inline opt-out should be unobtrusive and live near the chrome, not in the chat scrollback.

The recommended location: bottom of the guided sidecar (or a footer area of the guided body), small text, secondary visual weight per [05-modes-and-opt-out.md](05-modes-and-opt-out.md) §1.

- [ ] **Step 2: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineOptOutCheckbox } from "./InlineOptOutCheckbox";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

describe("InlineOptOutCheckbox", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    vi.restoreAllMocks();
  });

  it("is unchecked when default is guided", () => {
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it("is checked when default is freeform", () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).toBeChecked();
  });

  it("writes 'freeform' when ticked from guided default", async () => {
    const setDefault = vi.spyOn(usePreferencesStore.getState(), "setDefaultMode").mockResolvedValueOnce();
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("freeform");
  });

  it("writes 'guided' when re-ticked from freeform default", async () => {
    usePreferencesStore.setState({ defaultMode: "freeform" });
    const setDefault = vi.spyOn(usePreferencesStore.getState(), "setDefaultMode").mockResolvedValueOnce();
    render(<InlineOptOutCheckbox />);
    await userEvent.click(screen.getByRole("checkbox"));
    expect(setDefault).toHaveBeenCalledWith("guided");
  });

  it("disables the checkbox during write", () => {
    usePreferencesStore.setState({ writing: true });
    render(<InlineOptOutCheckbox />);
    expect(screen.getByRole("checkbox")).toBeDisabled();
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/chat/guided/InlineOptOutCheckbox.test.tsx`
Expected: FAIL — component not found.

- [ ] **Step 4: Implement the component**

```typescript
// src/components/chat/guided/InlineOptOutCheckbox.tsx
import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";

export function InlineOptOutCheckbox(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const loaded = usePreferencesStore((s) => s.loaded);
  const writing = usePreferencesStore((s) => s.writing);
  const setDefaultMode = usePreferencesStore((s) => s.setDefaultMode);

  // useCallback must be unconditional (React rules). Early return follows.
  const onToggle = useCallback(async () => {
    const current = usePreferencesStore.getState().defaultMode;
    try { await setDefaultMode(current === "freeform" ? "guided" : "freeform"); }
    catch (err) { console.error("[preferences] inline opt-out failed:", err); }
  }, [setDefaultMode]);

  if (!loaded) return null; // defaultMode is null before bootstrap

  const checked = defaultMode === "freeform"; // ticked = "always start in freeform"

  return (
    <label className="inline-opt-out">
      <input
        type="checkbox"
        checked={checked}
        disabled={writing}
        onChange={onToggle}
      />
      <span>Always start new sessions in freeform mode</span>
    </label>
  );
}
```

- [ ] **Step 5: Mount in ChatPanel guided chrome**

In `ChatPanel.tsx`, near the bottom of the guided body (after the message scrollback, before the chat input), add:

```tsx
{guidedSession && <InlineOptOutCheckbox />}
```

The `guidedSession` truthy check ensures the affordance only appears in guided mode.

- [ ] **Step 6: Run tests to verify**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/chat`
Expected: PASS — including the new component tests and the existing ChatPanel tests.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/chat/guided/InlineOptOutCheckbox.tsx src/elspeth/web/frontend/src/components/chat/guided/InlineOptOutCheckbox.test.tsx src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx
git commit -m "feat(web/frontend): inline opt-out checkbox in guided mode (Phase 1B.7)"
```

## Task 8: One-time "default changed" banner

**Files:**
- Create: `src/elspeth/web/frontend/src/components/common/DefaultModeChangedBanner.tsx`.
- Create: `src/elspeth/web/frontend/src/components/common/DefaultModeChangedBanner.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/Layout.tsx` — mount inside main column.

The banner fires once per user when both: `defaultMode === 'freeform'` (i.e., they have explicitly opted out) AND `bannerDismissedAt === null` (they haven't dismissed it).

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DefaultModeChangedBanner } from "./DefaultModeChangedBanner";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { resetStore } from "@/test/store-helpers";

describe("DefaultModeChangedBanner", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.restoreAllMocks();
  });

  it("does not render before preferences load", () => {
    usePreferencesStore.setState({ defaultMode: "freeform", loaded: false });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render when banner has been dismissed", () => {
    usePreferencesStore.setState({ defaultMode: "freeform", loaded: true, bannerDismissedAt: "2026-05-15T00:00:00Z" });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not render when user is on guided", () => {
    usePreferencesStore.setState({ defaultMode: "guided", loaded: true });
    render(<DefaultModeChangedBanner />);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders for opted-out users who haven't dismissed", () => {
    usePreferencesStore.setState({ defaultMode: "freeform", loaded: true, bannerDismissedAt: null });
    render(<DefaultModeChangedBanner />);
    expect(screen.getByRole("status")).toHaveTextContent(/freeform/i);
  });

  it("dismisses on click and persists", async () => {
    usePreferencesStore.setState({ defaultMode: "freeform", loaded: true, bannerDismissedAt: null });
    const dismiss = vi.spyOn(usePreferencesStore.getState(), "dismissDefaultChangedBanner").mockResolvedValueOnce();
    render(<DefaultModeChangedBanner />);
    await userEvent.click(screen.getByRole("button", { name: /got it|dismiss/i }));
    expect(dismiss).toHaveBeenCalled();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/DefaultModeChangedBanner.test.tsx`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement the banner**

```typescript
// src/components/common/DefaultModeChangedBanner.tsx
import { useCallback } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";

export function DefaultModeChangedBanner(): JSX.Element | null {
  const defaultMode = usePreferencesStore((s) => s.defaultMode);
  const bannerDismissedAt = usePreferencesStore((s) => s.bannerDismissedAt);
  const loaded = usePreferencesStore((s) => s.loaded);
  const dismiss = usePreferencesStore((s) => s.dismissDefaultChangedBanner);

  const visible = loaded && defaultMode === "freeform" && bannerDismissedAt === null;

  const onDismiss = useCallback(() => {
    void dismiss().catch((err) => {
      console.error("[preferences] banner dismiss failed:", err);
    });
  }, [dismiss]);

  if (!visible) return null;

  return (
    <div role="status" className="banner banner-info">
      <p>
        We changed the default for new sessions to <strong>guided mode</strong>.
        Your account is currently set to start new sessions in <strong>freeform</strong>.
        You can switch from the chat panel any time, or change your default in Settings.
      </p>
      <button type="button" onClick={onDismiss}>
        Got it
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Mount inside Layout**

Mount the banner **inside** the `<Layout>` component's main content column, not above it. Placing it above Layout adds uncounted vertical space if Layout uses `calc(100vh - ...)` or a flex height budget. Layout's flexbox must own the full height including the banner.

Sketch (adapt to actual Layout structure):

```tsx
// Inside Layout.tsx, at the top of the main content column:
import { DefaultModeChangedBanner } from "@/components/common/DefaultModeChangedBanner";

// In the main column JSX:
<main>
  <DefaultModeChangedBanner />
  {/* ...rest of main content... */}
</main>
```

Or, if `App.tsx` controls the main column, place the banner as the first child inside the Layout's main slot — whichever ensures Layout's height budget accounts for the banner.

- [ ] **Step 5: Run tests to verify**

Run: `cd src/elspeth/web/frontend && npx vitest run src/components/common/DefaultModeChangedBanner.test.tsx src/App.test.tsx`
Expected: PASS — all green.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/DefaultModeChangedBanner.tsx src/elspeth/web/frontend/src/components/common/DefaultModeChangedBanner.test.tsx src/elspeth/web/frontend/src/components/common/Layout.tsx
git commit -m "feat(web/frontend): one-time default-changed banner (Phase 1B.8)"
```

## Task 9: End-to-end smoke check on staging

**Files:**
- Modify: `docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md` — flip the Phase 1 row to "shipped" on success.
- No code changes; this is the manual verification step that ties Phase 1A + 1B together.

This task is the gate on declaring Phase 1 complete. It performs the operator action `project_db_migration_policy` mandates (delete the sessions DB) and exercises all five user journeys from the personas analysis.

- [ ] **Step 1: Confirm both halves have been built**

Run the full backend + frontend test suites locally:

```bash
.venv/bin/python -m pytest tests/unit/web/preferences/ tests/integration/web/test_preferences_routes.py -v
cd src/elspeth/web/frontend && npm test
```

Expected: PASS in both suites. If anything fails, fix before deploying. Do not deploy with red tests "to see what happens."

- [ ] **Step 2: Delete the staging sessions DB**

Per `project_db_migration_policy` and `project_staging_deployment` (`elspeth.foundryside.dev` is a source-checkout systemd/Caddy deploy on this machine):

```bash
# Stop the web service
sudo systemctl stop elspeth-web.service

# Find the DB path -- it's the auth_session_db_url in settings; typically
# under the runtime data dir of the deploy. Adjust to actual path.
SESSIONS_DB=$(grep -E '^auth_session_db_url' /etc/elspeth/web.yaml | awk '{print $2}' | tr -d '"' | sed 's|sqlite:///||')
echo "Deleting: $SESSIONS_DB"
sudo rm -f "$SESSIONS_DB"

# Build the frontend (per memory project_staging_deployment).
cd /path/to/elspeth/src/elspeth/web/frontend
npm run build

# Restart -- this triggers metadata.create_all() against the empty DB,
# creating user_preferences_table fresh.
sudo systemctl start elspeth-web.service
```

Wait ~10 seconds and verify the service is up: `sudo systemctl status elspeth-web.service`.

- [ ] **Step 3: Smoke-test the user journeys**

Open `elspeth.foundryside.dev` in a fresh incognito window (so localStorage from prior sessions doesn't interfere). Test six scenarios; each should match the recorded behaviour:

0. **Pre-Phase-1 user who never opts out (silent migration):**
   - Log in with a returning account that has prior sessions but has never changed the default-mode setting (no explicit opt-out on record).
   - Create a new session.
   - Expected: opens in **guided** mode (stepper visible) — silent migration to the new guided default.
   - Expected: **no banner shown** (silent migration is the policy-correct behaviour; no friction for users who never expressed a preference).

1. **Brand-new user (zero prior sessions):**
   - Log in with a fresh account.
   - Create a new session.
   - Expected: opens in **guided** mode (stepper visible).
   - Expected: no banner shown.

2. **User who explicitly opted out via Settings:**
   - Setup: complete Journey 1 above (brand-new user account created), then navigate to Account menu → Settings and toggle default to **freeform** (the explicit opt-out).
   - Close settings. Create a new session.
   - Expected: opens in **freeform** mode (chat-driven body, no stepper).
   - Expected: **dismissible banner shows** explaining the opt-out is active ("You've set freeform as your default. Change anytime in Settings.").
   - Click banner → banner disappears.
   - Reload the page → banner does not return.

3. **Settings opt-out:**
   - Click the top-right Account menu → Settings.
   - Toggle default to **freeform** (or guided, whichever flips it).
   - Close settings.
   - Create a new session → verify it opens in the selected mode.

4. **Inline opt-out:**
   - In a guided session, find the "Always start new sessions in freeform mode" checkbox.
   - Tick it.
   - Create a new session → verify it opens in freeform.

5. **Per-session toggle still works (regression check):**
   - In any session, click "Exit to freeform" (or "Switch to guided").
   - Verify the per-session toggle still flips the current session's mode.
   - Create another new session → verify it respects the **account default**, not the per-session switch.

Any failure here is a Phase 1 bug. Diagnose and fix before declaring complete.

- [ ] **Step 4: Mark the roadmap**

Edit `docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md`:

- Find §B "Phase-by-phase readiness summary" and flip the Phase 1 row from "READY" to "**SHIPPED**".
- Below §B, add a one-line note: "Phase 1 shipped (date)" — operator chooses whether to add the date or leave it as a marker per `feedback_no_calendar_shipping_commitments`.

- [ ] **Step 5: Commit and announce**

```bash
git add docs/composer/ux-redesign-2026-05/00-implementation-roadmap.md
git commit -m "docs: mark Phase 1 (default-guided + opt-out) shipped"
```

If the operator wants a deploy announcement (Slack / email / changelog), produce one with the user-facing summary: "New sessions default to guided mode for new users. Existing users continue in freeform — change anytime in Settings."

- [ ] **Step 6: File cross-tab staleness observation**

File a filigree observation to track the deferred two-tab staleness work:

```bash
filigree observe "Two-tab cross-tab preferences staleness: if the user has the app open in two tabs and changes their default mode in one, the other tab shows stale preferences until reload. Server-sent-events sync deferred to Phase 8." --label=deferred
```

If operating without filigree, note this in the Phase 1 retrospective doc instead.

## Task 10: Playwright E2E preference journey

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/composer-preferences.spec.ts`.

Project Playwright config: `playwright.config.ts`. Existing E2E specs in `tests/e2e/`. Isolation via `ELSPETH_WEB__data_dir`. Run with `npx playwright test tests/e2e/composer-preferences.spec.ts`.

- [ ] **Step 1: Write the E2E spec**

```typescript
// tests/e2e/composer-preferences.spec.ts
import { test, expect } from "@playwright/test";

// Journey 1: brand-new user gets guided mode, no banner.
test("new user gets guided mode on first session", async ({ page }) => {
  await page.goto("/");
  await page.fill('[name="username"]', process.env.E2E_USER ?? "test-new");
  await page.fill('[name="password"]', process.env.E2E_PASS ?? "test-pass");
  await page.click('[type="submit"]');
  await page.click('[aria-label="new session"]');
  await expect(page.locator('[data-testid="guided-stepper"]')).toBeVisible();
  await expect(page.locator('[role="status"].banner')).not.toBeVisible();
});

// Journey 2: user who opted out sees banner on next session, dismisses, banner stays gone on reload.
// Journey 3: settings opt-out changes mode for next session.
// Journey 4: inline opt-out checkbox changes mode for next session.
// Add analogous test() blocks — same login → action → assert pattern as Journey 1.
// All journeys mirror the manual smoke steps in Task 9 Step 3.
```

- [ ] **Step 2: Run the spec against staging**

```bash
ELSPETH_WEB__data_dir=/tmp/e2e-prefs-$(date +%s) npx playwright test tests/e2e/composer-preferences.spec.ts --headed
```

Expected: PASS. If any fail, diagnose before declaring Phase 1 complete.

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/composer-preferences.spec.ts
git commit -m "test(e2e): composer preference journey Playwright spec (Phase 1B.10)"
```

---

## What Phase 1B leaves the composer in

After all tasks land + the smoke deploy succeeds:

- New sessions for new users start in **guided mode** — reversing the implicit freeform default from commit `82dd2e73b`.
- Users who explicitly opt out via Settings see subsequent sessions open in **freeform mode** with a dismissible banner; users who never touch the setting silently get the new **guided** default — no banner, no friction.
- Three reachable opt-out surfaces (settings pane, inline checkbox, banner-link) all write to the same backend row.
- Per-session toggles in `ChatPanel.tsx` continue to work, scoped to the session — unchanged.
- The composer is otherwise **unchanged**: the IA cleanup (Phase 3), audit-readiness panel (Phase 2), tutorial (Phase 4), and completion gestures (Phase 6) are all still on the roadmap.

Phase 1 is *intentionally small.* It is the lowest-risk, highest-isolation phase in the redesign and is the cleanest place to introduce the new `user_preferences_table` that future phases (e.g., a tutorial-completed flag, opt-out telemetry) will extend.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Smoke deploy reveals the staging DB had unexpected state | Task 9 deletes the DB; the issue is on a fresh DB. If you cannot delete (production-bound data), this plan is wrong for that environment. |
| Banner is shown to brand-new users (race between bootstrap and createSession) | The banner is gated on `loaded === true`. `App.tsx` boots prefs on auth-success before any session UI mounts. Task 3 enforces this order. |
| Tests pass but in-browser flow breaks | Task 9 is the live smoke test. The plan does not declare success on green tests alone. |
| Two-tab cross-tab staleness | Deferred to Phase 8. File as an observation in Task 9 Step 6 (see above). If both tabs have the app open and the user changes their preference in one tab, the other tab will show stale state until reload. No server-sent-events sync in this phase. |
| Bootstrap failure stalls new-session creation | If `fetchUserComposerPreferences()` fails (network outage, 5xx on `/api/composer-preferences`), the store stays at `{ defaultMode: null, loaded: false }`. Because `createSession` calls `resolveDefaultMode()` which re-attempts `bootstrap()` until it succeeds, every new-session click triggers a fresh network call. On flaky networks the new-session UX stalls behind the prefs call. **Open design question for the operator (see Open Questions below):** should this plan add a localStorage fallback (cf. `useTheme.ts:1-40` precedent and roadmap §D2) so that prefs are read from `localStorage.getItem("composer.default_mode")` after a write, and `resolveDefaultMode()` returns the cached value when bootstrap fails? The current plan deliberately does **not** add this — it ships with the simplest behaviour and treats the staleness/perf concern as Phase 8 telemetry-driven. Confirm with the operator before changing. |

## Memory references

- `project_composer_default_guided_with_opt_out` — the design call this implements.
- `project_db_migration_policy` — informs Task 9's no-Alembic / delete-the-DB pattern.
- `project_staging_deployment` — informs Task 9's deploy steps (source-checkout systemd, npm run build, restart).
- `feedback_no_calendar_shipping_commitments` — no dates in this plan.

## Review history

- **2026-05-15 Panel review (4 reviewers):** 5 BLOCKERs, 2 CRITICALs, 9 IMPORTANTs. Key findings: `authFetch` does not exist (use `authHeaders` + bare `fetch`); `fetchComposerPreferences` / `updateComposerPreferences` name collision with existing session-scoped functions (renamed to `fetchUserComposerPreferences` / `updateUserComposerPreferences`); `ComposerPreferences` type collision (new type is `UserComposerPreferencesPayload`); `useAuthStore` has no `authenticated` field or `signOut` action (use `useAuth()` hook; action is `logout`); bootstrap race in `createSession` (fixed via `resolveDefaultMode()`); settings modal must follow `SecretsPanel` pattern not inline-overlay; `role="alert"` on informational banner should be `role="status"`; banner must mount inside Layout height budget; keyboard navigation tests added to UserMenu; `resetStore()` required in test `beforeEach`; Task 10 Playwright spec added; Task 4.5 integration test added.
- **2026-05-16 Final-validation pass:** Phase 1A migration heuristic retired (Task 5 of plan 12 — the session-count probe is gone). Reconciled Phase 1B's risk table: deleted "Existing-user migration mis-detects 'existing'" row and "Phase 1 introduces a regression in current freeform behaviour for existing users" row (both premised on the retired heuristic). Updated Task 8 banner-gating prose to reflect that the only path to `defaultMode === 'freeform'` is now explicit user opt-out.
- **2026-05-16** — Journey 2 rewritten to reflect retired session-count heuristic; Journey 0 added for silent migration.

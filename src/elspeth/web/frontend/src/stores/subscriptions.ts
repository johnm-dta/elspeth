// stores/subscriptions.ts
//
// Cross-store subscriptions extracted to break circular imports between
// sessionStore, executionStore, and auditReadinessStore. Call
// initStoreSubscriptions() once at app startup (e.g. in App.tsx).

import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";
import { useAuditReadinessStore } from "./auditReadinessStore";

let previousVersion: number | null = null;
let previousSessionIds: Set<string> = new Set();
let initialized = false;
let unsubscribe: (() => void) | null = null;

/**
 * Wire up cross-store subscriptions. Must be called exactly once at
 * application startup, after all stores have been created.
 *
 * Current subscriptions:
 * - Auto-clear validation when compositionState.version changes.
 * - Auto-clear audit-readiness cache for any session that disappears from
 *   sessionStore.sessions (archive, 404 self-eviction, future removal paths).
 *   Uses a previous-id set tracked across firings to detect removals
 *   uniformly — no need to instrument each removal call site.
 */
export function initStoreSubscriptions(): void {
  if (initialized) return;
  initialized = true;

  // Seed previousSessionIds from the current store state. Otherwise the first
  // removal of any session that was already in sessionStore before
  // initStoreSubscriptions() was called would silently no-op (the empty
  // previousSessionIds set has no ids to detect as "removed"). Production
  // startup order makes this unreachable today (sessions starts empty and
  // init runs synchronously before loadSessions), but adding persist
  // middleware, SSR hydration, or a test that seeds sessions before init
  // would expose the gap. The seed costs one set construction.
  previousSessionIds = new Set(useSessionStore.getState().sessions.map((s) => s.id));

  unsubscribe = useSessionStore.subscribe((state) => {
    // Version-change clears validation.
    const currentVersion = state.compositionState?.version ?? null;
    if (previousVersion !== null && currentVersion !== previousVersion) {
      useExecutionStore.getState().clearValidation();
    }
    previousVersion = currentVersion;

    // Session removal clears audit-readiness cache.
    const currentIds = new Set(state.sessions.map((s) => s.id));
    for (const prevId of previousSessionIds) {
      if (!currentIds.has(prevId)) {
        useAuditReadinessStore.getState().clearSession(prevId);
      }
    }
    previousSessionIds = currentIds;
  });
}

/**
 * Test-only helper. Resets the module-level state so each test starts from
 * a clean slate. Unsubscribes the active zustand subscriber so stale
 * callbacks do not accumulate across beforeEach resets. Not exported from
 * any index barrel.
 */
export function _resetSubscriptionsForTesting(): void {
  unsubscribe?.();
  unsubscribe = null;
  previousVersion = null;
  previousSessionIds = new Set();
  initialized = false;
}

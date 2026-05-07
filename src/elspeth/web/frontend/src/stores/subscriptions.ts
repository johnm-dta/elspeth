// stores/subscriptions.ts
//
// Cross-store subscriptions extracted from executionStore to break the
// circular import between sessionStore and executionStore. Call
// initStoreSubscriptions() once at app startup (e.g. in App.tsx).

import { useSessionStore } from "./sessionStore";
import { useExecutionStore } from "./executionStore";

let previousVersion: number | null = null;
let initialized = false;

/**
 * Wire up cross-store subscriptions. Must be called exactly once at
 * application startup, after all stores have been created.
 *
 * Current subscriptions:
 * - Auto-clear validation when compositionState.version changes.
 */
export function initStoreSubscriptions(): void {
  if (initialized) return;
  initialized = true;

  useSessionStore.subscribe((state) => {
    const currentVersion = state.compositionState?.version ?? null;
    if (previousVersion !== null && currentVersion !== previousVersion) {
      useExecutionStore.getState().clearValidation();
    }
    previousVersion = currentVersion;
  });
}

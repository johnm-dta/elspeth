/**
 * Test helpers for Zustand store testing.
 *
 * Pattern: Zustand v5 stores created with `create()` expose a vanilla
 * store API via `.getState()` and `.setState()`. Tests can drive state
 * transitions directly without rendering React components, giving fast
 * synchronous assertions on store logic.
 *
 * For async actions (API calls), mock the API module and then call the
 * store action — the store updates synchronously after the awaited
 * promise resolves.
 *
 * Usage:
 *   import { useSessionStore } from "@/stores/sessionStore";
 *   import { resetStore } from "@/test/store-helpers";
 *
 *   beforeEach(() => resetStore(useSessionStore));
 *
 *   it("sets composing state on send", async () => {
 *     const store = useSessionStore;
 *     // ... drive actions via store.getState().someAction()
 *     // ... assert via expect(store.getState().someField)
 *   });
 */
import type { StoreApi, UseBoundStore } from "zustand";

/**
 * Reset a Zustand store to its initial state between tests.
 *
 * Zustand stores are module-level singletons. Without reset, state
 * leaks between tests. This helper calls `reset()` if the store
 * defines one, otherwise replaces state with the initial snapshot.
 */
export function resetStore<T extends object>(
  store: UseBoundStore<StoreApi<T>>,
): void {
  const state = store.getState();
  if ("reset" in state && typeof state.reset === "function") {
    (state.reset as () => void)();
  } else {
    store.setState(store.getInitialState(), true);
  }
}

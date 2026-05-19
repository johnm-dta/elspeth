// src/hooks/useSession.ts
import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useAuthStore, selectIsAuthenticated } from "@/stores/authStore";


/**
 * Session lifecycle effects that must stay mounted with the app shell.
 * HeaderSessionSwitcher reads store state directly after SessionSidebar removal,
 * so this hook owns the load/reset effects separately from the old sidebar hook.
 *
 * loadSessions is gated on isAuthenticated. Previously SessionSidebar was
 * rendered inside AuthGuard's authenticated tree, so the load effect was
 * implicitly gated by mount location. Phase 3A.7 hoisted the effect to App
 * (which fires hooks regardless of AuthGuard's render decision) — the
 * isAuthenticated guard restores the original semantics. Without it the
 * cold-load fires an unauthenticated GET that the global 401 interceptor
 * traps, and a late-arriving 401 can wipe a freshly-acquired login token.
 */
export function useSessionLifecycle(): void {
  const loadSessions = useSessionStore((s) => s.loadSessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const loadRuns = useExecutionStore((s) => s.loadRuns);
  const isAuthenticated = useAuthStore(selectIsAuthenticated);

  useEffect(() => {
    if (!isAuthenticated) return;
    void loadSessions();
  }, [isAuthenticated, loadSessions]);

  // Reset always (including when activeSessionId becomes null); load runs only
  // when there is a session to load.
  useEffect(() => {
    useExecutionStore.getState().reset();
    if (activeSessionId) {
      void loadRuns(activeSessionId);
    }
  }, [activeSessionId, loadRuns]);
}

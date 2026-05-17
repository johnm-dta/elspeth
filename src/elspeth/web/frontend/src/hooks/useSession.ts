// src/hooks/useSession.ts
import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";

/**
 * Session lifecycle effects that must stay mounted with the app shell.
 * HeaderSessionSwitcher reads store state directly after SessionSidebar removal,
 * so this hook owns the load/reset effects separately from the old sidebar hook.
 */
export function useSessionLifecycle(): void {
  const loadSessions = useSessionStore((s) => s.loadSessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const loadRuns = useExecutionStore((s) => s.loadRuns);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  // Load runs whenever the active session changes.
  // Reset execution state first to prevent stale progress/activeRunId from a
  // previous session bleeding into the new one (H6).
  useEffect(() => {
    if (activeSessionId) {
      useExecutionStore.getState().reset();
      void loadRuns(activeSessionId);
    }
  }, [activeSessionId, loadRuns]);
}

/**
 * Hook for session consumers that need session actions plus lifecycle effects.
 */
export function useSession() {
  useSessionLifecycle();
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const createSession = useSessionStore((s) => s.createSession);
  const selectSession = useSessionStore((s) => s.selectSession);
  const renameSession = useSessionStore((s) => s.renameSession);
  const archiveSession = useSessionStore((s) => s.archiveSession);

  return { sessions, activeSessionId, createSession, selectSession, renameSession, archiveSession };
}

// src/hooks/useSession.ts
import { useEffect } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";

/**
 * Hook for session lifecycle. Loads sessions on mount.
 * When the active session changes, loads runs for that session.
 */
export function useSession() {
  const loadSessions = useSessionStore((s) => s.loadSessions);
  const sessions = useSessionStore((s) => s.sessions);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const createSession = useSessionStore((s) => s.createSession);
  const selectSession = useSessionStore((s) => s.selectSession);
  const archiveSession = useSessionStore((s) => s.archiveSession);
  const loadRuns = useExecutionStore((s) => s.loadRuns);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Load runs whenever the active session changes.
  // Reset execution state first to prevent stale progress/activeRunId from a
  // previous session bleeding into the new one (H6).
  useEffect(() => {
    if (activeSessionId) {
      useExecutionStore.getState().reset();
      loadRuns(activeSessionId);
    }
  }, [activeSessionId, loadRuns]);

  return { sessions, activeSessionId, createSession, selectSession, archiveSession };
}

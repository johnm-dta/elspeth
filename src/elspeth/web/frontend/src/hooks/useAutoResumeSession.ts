import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";

/**
 * Returning-user auto-resume (elspeth-e69642fede).
 *
 * A returning user previously landed on an empty shell whose only action
 * lived inside a header menu. Once the session list has loaded, this hook
 * auto-selects the most recently active (non-archived) session — a single
 * attempt per app mount.
 *
 * It deliberately stands down when:
 * - `enabled` is false — the caller gates on auth, the shared-inspect route,
 *   preferences having settled, and the first-run tutorial (tutorial resume
 *   wins for first-run users; see App.tsx's decision tree);
 * - the URL hash names a session — the hash router owns deep links;
 * - a session is already active (deep link or user action beat us to it);
 * - there is nothing to resume (App renders the empty landing instead).
 */
export function useAutoResumeSession(enabled: boolean): void {
  const attempted = useRef(false);
  const sessionsLoaded = useSessionStore((s) => s.sessionsLoaded);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);

  useEffect(() => {
    if (!enabled || attempted.current || !sessionsLoaded) return;
    // A session-shaped hash (#/{id} or #/{id}/verb) is the hash router's
    // jurisdiction; if the id turns out to be stale the router clears it
    // and the user lands on the switcher-driven flow, not a surprise resume.
    if (/^#\/.+/.test(window.location.hash)) {
      attempted.current = true;
      return;
    }
    if (activeSessionId !== null) {
      attempted.current = true;
      return;
    }
    const candidates = useSessionStore
      .getState()
      .sessions.filter((session) => !session.archived);
    if (candidates.length === 0) {
      attempted.current = true;
      return;
    }
    const mostRecent = [...candidates].sort(
      (a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at),
    )[0];
    attempted.current = true;
    void useSessionStore.getState().selectSession(mostRecent.id);
  }, [enabled, sessionsLoaded, activeSessionId]);
}

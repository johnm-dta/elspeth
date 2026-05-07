/**
 * Hash-based router for session and tab deep linking.
 *
 * Format: #/{sessionId}/{tab}
 *   - #/abc123          → select session abc123, default tab
 *   - #/abc123/graph    → select session abc123, switch to graph tab
 *   - # or empty        → no deep link
 *
 * Session changes push history entries (back button works).
 * Tab changes replace the current entry (don't clutter history).
 *
 * Race condition: on first load, sessions may not be loaded yet.
 * We apply the hash optimistically — selectSession will fetch data
 * from the API regardless. Once sessions load, we re-check.
 */

import { useEffect, useRef } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import { SWITCH_TAB_EVENT } from "@/components/common/CommandPalette";

/** Dispatched by InspectorPanel when the active tab changes. */
export const TAB_CHANGED_EVENT = "elspeth-tab-changed";

interface HashState {
  sessionId: string | null;
  tab: string | null;
}

const VALID_TABS = new Set(["spec", "graph", "yaml", "runs"]);

function parseHash(): HashState {
  const hash = window.location.hash;
  const match = hash.match(/^#\/([^/]+?)(?:\/([a-z]+))?$/);
  if (!match) return { sessionId: null, tab: null };
  const tab = match[2] && VALID_TABS.has(match[2]) ? match[2] : null;
  return { sessionId: match[1], tab };
}

function buildHash(sessionId: string | null, tab: string | null): string {
  if (!sessionId) return "";
  return tab ? `#/${sessionId}/${tab}` : `#/${sessionId}`;
}

export function useHashRouter(): void {
  // Track what we last wrote to avoid reacting to our own updates
  const lastWrittenHash = useRef<string>("");
  // Track the current tab from hash (for preserving across session changes)
  const currentTab = useRef<string | null>(null);
  // Flag to suppress hash write during initial application
  const applying = useRef(false);

  // ── Apply hash state to the app ──────────────────────────────────────
  const applyHash = (state: HashState) => {
    applying.current = true;

    const { sessionId, tab } = state;
    const store = useSessionStore.getState();

    if (sessionId && sessionId !== store.activeSessionId) {
      store.selectSession(sessionId);
    }

    const resolvedTab = tab ?? "spec";
    currentTab.current = resolvedTab;
    window.dispatchEvent(
      new CustomEvent(SWITCH_TAB_EVENT, { detail: resolvedTab }),
    );

    applying.current = false;
  };

  // ── On mount: apply initial hash ─────────────────────────────────────
  useEffect(() => {
    const initial = parseHash();
    if (initial.sessionId) {
      // Pre-set lastWrittenHash so the session subscriber doesn't
      // pushState on the initial load (we want the URL to stay as-is).
      lastWrittenHash.current = window.location.hash;
      applyHash(initial);
    } else {
      // No hash — write current session to hash if one is already active
      const { activeSessionId } = useSessionStore.getState();
      if (activeSessionId) {
        const hash = buildHash(activeSessionId, null);
        lastWrittenHash.current = hash;
        window.history.replaceState(null, "", hash || window.location.pathname);
      }
    }
  }, []);

  // ── Handle browser back/forward ──────────────────────────────────────
  useEffect(() => {
    function handleHashChange() {
      const newHash = window.location.hash;
      // Skip if this was our own write
      if (newHash === lastWrittenHash.current) return;
      // Pre-set so the session subscriber doesn't re-push this hash
      lastWrittenHash.current = newHash;
      applyHash(parseHash());
    }

    // popstate fires on back/forward — more reliable than hashchange
    // for history.pushState entries
    window.addEventListener("popstate", handleHashChange);
    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("popstate", handleHashChange);
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  // ── Sync session changes → hash (pushState) ─────────────────────────
  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (applying.current) return;
      if (state.activeSessionId === prevState.activeSessionId) return;

      const hash = buildHash(state.activeSessionId, currentTab.current);
      if (hash === lastWrittenHash.current) return;
      lastWrittenHash.current = hash;

      if (hash) {
        window.history.pushState(null, "", hash);
      } else {
        window.history.replaceState(null, "", window.location.pathname);
      }
    });
    return unsub;
  }, []);

  // ── Sync tab changes → hash (replaceState) ──────────────────────────
  useEffect(() => {
    function handleTabChanged(e: Event) {
      if (applying.current) return;
      const tab = (e as CustomEvent<string>).detail;
      if (!VALID_TABS.has(tab)) return;
      currentTab.current = tab;

      const { activeSessionId } = useSessionStore.getState();
      if (!activeSessionId) return;

      const hash = buildHash(activeSessionId, tab);
      if (hash === lastWrittenHash.current) return;
      lastWrittenHash.current = hash;
      window.history.replaceState(null, "", hash);
    }

    window.addEventListener(TAB_CHANGED_EVENT, handleTabChanged);
    return () => window.removeEventListener(TAB_CHANGED_EVENT, handleTabChanged);
  }, []);

  // ── Re-check hash once sessions load ─────────────────────────────────
  // On first load, selectSession may be called before sessions are in
  // the store. Once they arrive, verify the hash session is valid.
  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      // Only react to sessions list changing from empty to populated
      if (prevState.sessions.length > 0 || state.sessions.length === 0) return;

      const { sessionId } = parseHash();
      if (!sessionId) return;

      const exists = state.sessions.some((s) => s.id === sessionId);
      if (!exists && state.activeSessionId === sessionId) {
        // Session from hash doesn't exist — clear invalid hash and session
        lastWrittenHash.current = "";
        window.history.replaceState(null, "", window.location.pathname);
        useSessionStore.setState({ activeSessionId: null });
      }
    });
    return unsub;
  }, []);
}

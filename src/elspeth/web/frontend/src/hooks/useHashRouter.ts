/**
 * Hash-based router for session deep linking.
 *
 * Format: #/{sessionId}                 -> canonical
 *         #/{sessionId}/graph           -> open graph modal, then rewrite
 *         #/{sessionId}/yaml            -> open YAML modal, then rewrite
 *         #/{sessionId}/{anything-else} -> silently strip the verb
 *
 * Phase 3B replaced the old inspector-tab vocabulary with action fragments.
 * The fragment is an arrival action, not steady-state URL state.
 */

import { useEffect, useRef } from "react";
import {
  OPEN_GRAPH_MODAL_EVENT,
  OPEN_YAML_MODAL_EVENT,
} from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";

interface HashState {
  sessionId: string | null;
  verb: string | null;
}

interface HashRouterResult {
  redirectToast: RedirectToast | null;
}

interface RedirectToast {
  message: string;
  dismiss: () => void;
}

const ACTION_VERBS: Record<string, string> = {
  graph: OPEN_GRAPH_MODAL_EVENT,
  yaml: OPEN_YAML_MODAL_EVENT,
};

function parseHash(): HashState {
  const hash = window.location.hash;
  const match = hash.match(/^#\/([^/]+?)(?:\/([a-z]+))?$/);
  if (!match) return { sessionId: null, verb: null };
  return { sessionId: match[1], verb: match[2] ?? null };
}

function buildCanonicalHash(sessionId: string | null): string {
  return sessionId ? `#/${sessionId}` : "";
}

export function useHashRouter(): HashRouterResult {
  const lastWrittenHash = useRef<string>("");
  const applying = useRef(false);

  const applyHash = (state: HashState) => {
    applying.current = true;
    const { sessionId, verb } = state;
    const store = useSessionStore.getState();

    if (sessionId && sessionId !== store.activeSessionId) {
      store.selectSession(sessionId);
    }

    if (verb && verb in ACTION_VERBS) {
      const eventName = ACTION_VERBS[verb];
      queueMicrotask(() => window.dispatchEvent(new CustomEvent(eventName)));
    }

    const canonical = buildCanonicalHash(sessionId);
    if (canonical !== window.location.hash) {
      lastWrittenHash.current = canonical;
      window.history.replaceState(
        null,
        "",
        canonical || window.location.pathname,
      );
    }

    applying.current = false;
  };

  useEffect(() => {
    const initial = parseHash();
    if (initial.sessionId) {
      lastWrittenHash.current = window.location.hash;
      applyHash(initial);
    } else {
      const { activeSessionId } = useSessionStore.getState();
      if (activeSessionId) {
        const hash = buildCanonicalHash(activeSessionId);
        lastWrittenHash.current = hash;
        window.history.replaceState(
          null,
          "",
          hash || window.location.pathname,
        );
      }
    }
  }, []);

  useEffect(() => {
    function handleHashChange() {
      const newHash = window.location.hash;
      if (newHash === lastWrittenHash.current) return;
      lastWrittenHash.current = newHash;
      applyHash(parseHash());
    }

    window.addEventListener("popstate", handleHashChange);
    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("popstate", handleHashChange);
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (applying.current) return;
      if (state.activeSessionId === prevState.activeSessionId) return;

      const hash = buildCanonicalHash(state.activeSessionId);
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

  useEffect(() => {
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (prevState.sessions.length > 0 || state.sessions.length === 0) return;

      const { sessionId } = parseHash();
      if (!sessionId) return;

      const exists = state.sessions.some((s) => s.id === sessionId);
      if (!exists && state.activeSessionId === sessionId) {
        lastWrittenHash.current = "";
        window.history.replaceState(null, "", window.location.pathname);
        useSessionStore.setState({ activeSessionId: null });
      }
    });
    return unsub;
  }, []);

  return { redirectToast: null };
}

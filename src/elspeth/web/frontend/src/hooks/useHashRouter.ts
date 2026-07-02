/**
 * Hash-based router for session deep linking.
 *
 * Format: #/{sessionId}                 -> canonical
 *         #/{sessionId}/graph           -> open graph modal, then rewrite
 *         #/{sessionId}/yaml            -> open YAML modal, then rewrite
 *         #/{sessionId}/{anything-else} -> silently strip the verb
 *         #/shared/{token}              -> IGNORED — owned by useSharedToken
 *                                          (Phase 6B Task 8); the hook below
 *                                          short-circuits without touching the
 *                                          hash so SharedInspectView keeps
 *                                          control of the URL.
 *
 * Phase 3B uses action fragments rather than steady-state view fragments.
 * The fragment is an arrival action, not steady-state URL state.
 */

import { useEffect, useRef, useState } from "react";
import {
  OPEN_GRAPH_MODAL_EVENT,
  OPEN_YAML_MODAL_EVENT,
} from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";
import { hasCompositionContent } from "@/utils/compositionState";
import type { CompositionState } from "@/types/api";

interface HashState {
  sessionId: string | null;
  verb: string | null;
}

interface RedirectToast {
  message: string;
  dismiss: () => void;
}

const ACTION_VERBS: Record<string, string> = {
  graph: OPEN_GRAPH_MODAL_EVENT,
  yaml: OPEN_YAML_MODAL_EVENT,
};

/**
 * Verbs that were valid tabs in earlier versions but have since been removed.
 * When detected, a one-time dismissible toast is shown to the user.
 * All other unrecognized verbs are silently stripped (backward compat).
 */
const RETIRED_VERBS: Record<string, string> = {
  runs: "The Runs tab was removed in this update. Showing Graph instead.",
  spec: "The Spec tab was removed in this update. Showing Graph instead.",
};

const TOAST_DISMISSED_KEY = "elspeth_redirect_toast_dismissed";

const SHARED_HASH_PREFIX = "#/shared/";

/** True when the current hash is a Phase 6B shared-inspect route. The
 *  session router short-circuits on this so SharedInspectView keeps
 *  control of the URL and the session store is not mutated. */
function _isSharedRoute(hash: string): boolean {
  return hash.startsWith(SHARED_HASH_PREFIX);
}

function parseHash(): HashState {
  const hash = window.location.hash;
  if (_isSharedRoute(hash)) return { sessionId: null, verb: null };
  const match = hash.match(/^#\/([^/]+?)(?:\/([a-z]+))?$/);
  if (!match) return { sessionId: null, verb: null };
  return { sessionId: match[1], verb: match[2] ?? null };
}

function buildCanonicalHash(sessionId: string | null): string {
  return sessionId ? `#/${sessionId}` : "";
}

/**
 * Dispatch the Export-YAML open event once `sessionId`'s composition state
 * is KNOWN, and only when the pipeline has content — the same
 * hasCompositionContent gate ExportYamlButton applies
 * (elspeth-bff8043d33 residual: the #/{id}/yaml deep link could open the
 * near-empty modal on a pipeline with nothing to export).
 *
 * The deferral matters: on a fresh #/{id}/yaml arrival, selectSession's
 * fetch is still in flight when the verb is parsed, so gating on the
 * instantaneous compositionState would break the deep link for every
 * non-empty pipeline. `compositionStateLoaded` disambiguates "still
 * fetching" from "loaded and empty"; the one-shot subscription resolves
 * when the fetch settles and aborts if the user switches sessions first.
 */
function dispatchYamlWhenCompositionKnown(sessionId: string): void {
  const dispatchIfContent = (state: {
    compositionState: CompositionState | null;
  }) => {
    if (hasCompositionContent(state.compositionState)) {
      window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT));
    }
  };
  const snapshot = useSessionStore.getState();
  if (snapshot.activeSessionId === sessionId && snapshot.compositionStateLoaded) {
    queueMicrotask(() => dispatchIfContent(useSessionStore.getState()));
    return;
  }
  const unsub = useSessionStore.subscribe((state) => {
    if (state.activeSessionId !== sessionId) {
      unsub();
      return;
    }
    if (!state.compositionStateLoaded) return;
    unsub();
    dispatchIfContent(state);
  });
}

interface UseHashRouterOptions {
  enabled?: boolean;
}

export function useHashRouter(
  options: UseHashRouterOptions = {},
): { redirectToast: RedirectToast | null } {
  const enabled = options.enabled ?? true;
  const lastWrittenHash = useRef<string>("");
  const applying = useRef(false);

  // Read dismissal flag once at mount. Using a ref rather than reading
  // localStorage on every applyHash invocation avoids repeated storage reads.
  const dismissedRef = useRef<boolean>(
    typeof window !== "undefined" &&
      window.localStorage.getItem(TOAST_DISMISSED_KEY) === "1",
  );

  const [redirectToast, setRedirectToast] = useState<RedirectToast | null>(
    null,
  );

  const applyHash = (state: HashState) => {
    // Phase 6B Task 8: when the live hash is a shared-inspect route the
    // session router is dormant — SharedInspectView owns the URL. Apply
    // is a no-op in that mode so the hash is preserved verbatim and the
    // active session is not mutated by the (deliberately empty) parsed
    // state we'd otherwise act on.
    if (_isSharedRoute(window.location.hash)) {
      return;
    }
    applying.current = true;
    try {
      const { sessionId, verb } = state;
      const store = useSessionStore.getState();

      if (sessionId && sessionId !== store.activeSessionId) {
        store.selectSession(sessionId);
      } else if (!sessionId && store.activeSessionId) {
        useSessionStore.setState({ activeSessionId: null });
      }

      // Fix A: use hasOwnProperty to avoid prototype-chain walk.
      // The `in` operator walks the prototype chain: `"constructor" in ACTION_VERBS`
      // is true even though "constructor" is not an own property of ACTION_VERBS.
      // `ACTION_VERBS["constructor"]` returns the Object constructor function and
      // `new CustomEvent(fn)` would coerce it to a garbage event name.
      // Object.prototype.hasOwnProperty.call() is the ES2020-compatible guard.
      const hasOwn = Object.prototype.hasOwnProperty;
      if (verb === "yaml" && sessionId) {
        // Export YAML is content-gated; see dispatchYamlWhenCompositionKnown.
        dispatchYamlWhenCompositionKnown(sessionId);
      } else if (verb && hasOwn.call(ACTION_VERBS, verb)) {
        const eventName = ACTION_VERBS[verb];
        queueMicrotask(() => window.dispatchEvent(new CustomEvent(eventName)));
      }

      // Fix C: retired-verb redirect toast. Only "runs" and "spec" trigger
      // the toast; all other unrecognized verbs are silently stripped.
      if (verb && hasOwn.call(RETIRED_VERBS, verb) && !dismissedRef.current) {
        const message = RETIRED_VERBS[verb];
        setRedirectToast({
          message,
          dismiss: () => {
            dismissedRef.current = true;
            window.localStorage.setItem(TOAST_DISMISSED_KEY, "1");
            setRedirectToast(null);
          },
        });
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
    } finally {
      // Fix B: guarantee the flag is cleared even if selectSession throws.
      // Without this, applying.current stays true permanently and the URL-echo
      // subscription (useEffect #3) becomes a no-op until page reload.
      applying.current = false;
    }
  };

  useEffect(() => {
    if (!enabled) return;
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
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
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
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
    const unsub = useSessionStore.subscribe((state, prevState) => {
      if (applying.current) return;
      if (state.activeSessionId === prevState.activeSessionId) return;
      // Phase 6B Task 8: do not mutate the URL while a shared-inspect
      // route is live — SharedInspectView owns the hash. The session
      // store may legitimately change activeSessionId via background
      // bootstrap while a reviewer is on the shared route; the change
      // is irrelevant to the rendered view and must not strip the token.
      if (_isSharedRoute(window.location.hash)) return;

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
  }, [enabled]);

  useEffect(() => {
    if (!enabled) return;
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
  }, [enabled]);

  return { redirectToast };
}

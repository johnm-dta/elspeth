/**
 * useSharedToken — Phase 6B Task 8.
 *
 * Detects when the URL hash is in the form `#/shared/{token}` and returns
 * the decoded token. Otherwise returns null.
 *
 * This hook is a peer of `useHashRouter` and runs BEFORE the regular
 * composer route is mounted. The intended call site is the App
 * component, which short-circuits to `SharedInspectView` when this
 * hook returns a non-null token. The regular `useHashRouter`'s session
 * handling does not fire in shared-route mode because App.tsx renders a
 * different tree.
 *
 * The token is URL-encoded in transit (the share_url is the raw token
 * appended to `/#/shared/`); we decode here so callers receive the
 * canonical token bytes.
 */

import { useEffect, useState } from "react";

const SHARED_HASH_PREFIX = "#/shared/";

function _parseSharedToken(hash: string): string | null {
  if (!hash.startsWith(SHARED_HASH_PREFIX)) return null;
  const raw = hash.slice(SHARED_HASH_PREFIX.length);
  if (raw === "") return null;
  try {
    return decodeURIComponent(raw);
  } catch {
    return null;
  }
}

/** Test helper exposed for unit tests that don't want to round-trip
 *  through window.location.hash setters. */
export const _parseSharedTokenForTesting = _parseSharedToken;

export function useSharedToken(): string | null {
  const [token, setToken] = useState<string | null>(() =>
    _parseSharedToken(typeof window !== "undefined" ? window.location.hash : ""),
  );

  useEffect(() => {
    function handleHashChange() {
      setToken(_parseSharedToken(window.location.hash));
    }
    window.addEventListener("hashchange", handleHashChange);
    window.addEventListener("popstate", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
      window.removeEventListener("popstate", handleHashChange);
    };
  }, []);

  return token;
}

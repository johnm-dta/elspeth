import { useEffect, type ReactNode } from "react";
import { useAuth } from "../../hooks/useAuth";
import { LoginPage } from "../auth/LoginPage";

interface AuthGuardProps {
  children: ReactNode;
}

/**
 * Auth gate component. Renders at the top of the component tree.
 * - Shows a loading spinner while checking stored credentials.
 * - Renders LoginPage if the user is not authenticated.
 * - Renders children if authenticated.
 *
 * Phase 6B FIX-C: shared-route hash round-trip via sessionStorage.
 *
 *   When a reviewer follows a `#/shared/{token}` link without being
 *   logged in, the unauth→auth transition must preserve the hash so
 *   they land back on the shared view after login (plan lines 564-589
 *   of `docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`).
 *   The login flow today does not trigger a hard navigation in
 *   v1 — `LoginPage` flips `authStore` state in-place — so in the
 *   common case the hash is preserved automatically. The
 *   sessionStorage round-trip is the durable fallback for any future
 *   login flow that DOES navigate (OAuth, SSO redirect, full reload)
 *   and for browser-back behaviour where the hash might be lost.
 *
 *   The key `elspeth_post_login_redirect` is self-disarming — the
 *   restore effect removes it after restoring the hash, so a
 *   subsequent auth check on a fresh page load doesn't loop back to a
 *   stale shared-route.
 */
const POST_LOGIN_REDIRECT_KEY = "elspeth_post_login_redirect";
const SHARED_HASH_PREFIX = "#/shared/";

export function AuthGuard({ children }: AuthGuardProps) {
  const { isAuthenticated, isLoading } = useAuth();

  // SAVE: while unauthenticated, persist the current hash if it's a
  // shared-route. We don't save other hashes (the regular composer's
  // session-id hashes are not deep-linkable in the same way; the
  // shared route is the only surface today where the URL alone
  // selects the post-login destination).
  useEffect(() => {
    if (isLoading || isAuthenticated) return;
    const hash = window.location.hash;
    if (!hash.startsWith(SHARED_HASH_PREFIX)) return;
    if (window.sessionStorage.getItem(POST_LOGIN_REDIRECT_KEY) !== null) return;
    window.sessionStorage.setItem(POST_LOGIN_REDIRECT_KEY, hash);
  }, [isAuthenticated, isLoading]);

  // RESTORE: on transition to authenticated, if we saved a hash
  // earlier, restore it and clear the key. The dependency on
  // `isAuthenticated` only means this fires once per unauth→auth
  // transition — the `removeItem` is self-disarming so authenticated
  // re-renders don't re-fire the restore (the key is gone).
  useEffect(() => {
    if (!isAuthenticated) return;
    const saved = window.sessionStorage.getItem(POST_LOGIN_REDIRECT_KEY);
    if (saved === null) return;
    window.sessionStorage.removeItem(POST_LOGIN_REDIRECT_KEY);
    // Only restore if the saved value differs from the current hash —
    // a hard-reload that preserved the hash naturally doesn't need
    // restoration, and re-writing the same hash would dispatch a
    // duplicate `hashchange` event.
    if (window.location.hash !== saved) {
      window.location.hash = saved;
    }
  }, [isAuthenticated]);

  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Checking authentication"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
        }}
      >
        <span className="spinner" style={{ width: 32, height: 32, borderWidth: 3 }} />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <>{children}</>;
}

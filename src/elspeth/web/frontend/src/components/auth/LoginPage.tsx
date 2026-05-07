import { useState, useEffect, type FormEvent } from "react";
import { useAuth } from "../../hooks/useAuth";
import * as api from "../../api/client";
import type { AuthConfig } from "../../types/index";

/**
 * Login page that adapts to the configured auth provider.
 *
 * Fetches GET /api/auth/config on mount to determine provider type:
 * - "local": renders a username/password form
 * - "oidc" or "entra": renders a "Sign in with SSO" button that
 *   redirects to config.authorization_endpoint (resolved from OIDC
 *   discovery by the backend)
 *
 * On return from an OIDC redirect, extracts the token from the URL
 * fragment or query parameter and calls loginWithToken().
 */
export function LoginPage() {
  const { login, loginWithToken, loginError } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);

  // Fetch auth config on mount to determine which login form to show
  useEffect(() => {
    api
      .fetchAuthConfig()
      .then((config) => {
        setAuthConfig(config);
        setConfigLoading(false);
      })
      .catch(() => {
        // If config fetch fails, fall back to local auth
        setAuthConfig({
          provider: "local",
          oidc_issuer: null,
          oidc_client_id: null,
          authorization_endpoint: null,
        });
        setConfigLoading(false);
      });
  }, []);

  // Handle OIDC callback: extract token from URL fragment or query parameter.
  // Verifies the state nonce to prevent CSRF / session-fixation attacks (H2/H3).
  useEffect(() => {
    const savedState = sessionStorage.getItem("oidc_state");
    sessionStorage.removeItem("oidc_state");

    const hash = window.location.hash;
    const params = new URLSearchParams(window.location.search);

    // Check URL fragment first (implicit flow: #access_token=...)
    if (hash) {
      const fragmentParams = new URLSearchParams(hash.substring(1));
      const token = fragmentParams.get("access_token");
      const callbackState = fragmentParams.get("state");
      if (token) {
        // Clean the URL before processing
        window.history.replaceState(null, "", window.location.pathname);
        if (savedState && callbackState === savedState) {
          loginWithToken(token);
        }
        return;
      }
    }

    // Check query parameter (authorization code flow: ?token=...)
    const callbackToken = params.get("token");
    const callbackState = params.get("state");
    if (callbackToken) {
      window.history.replaceState(null, "", window.location.pathname);
      if (savedState && callbackState === savedState) {
        loginWithToken(callbackToken);
      }
    }
  }, [loginWithToken]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username || !password) return;

    setIsSubmitting(true);
    await login(username, password);
    setIsSubmitting(false);
  }

  function handleSsoRedirect() {
    if (!authConfig?.authorization_endpoint || !authConfig?.oidc_client_id) return;

    // Generate OIDC state nonce for CSRF protection (H2)
    const state = crypto.randomUUID();
    sessionStorage.setItem("oidc_state", state);

    const url =
      `${authConfig.authorization_endpoint}` +
      `?client_id=${encodeURIComponent(authConfig.oidc_client_id)}` +
      `&response_type=token` +
      `&redirect_uri=${encodeURIComponent(window.location.origin)}` +
      `&scope=openid profile email` +
      `&state=${encodeURIComponent(state)}` +
      `&nonce=${crypto.randomUUID()}`;
    window.location.href = url;
  }

  if (configLoading) {
    return (
      <div
        role="status"
        aria-label="Loading authentication configuration"
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
        }}
      >
        <span className="spinner" aria-label="Loading" role="status" />
      </div>
    );
  }

  const isOidc =
    authConfig?.provider === "oidc" || authConfig?.provider === "entra";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        backgroundColor: "var(--color-bg)",
      }}
    >
      <div
        style={{
          width: 360,
          padding: 32,
          backgroundColor: "var(--color-surface)",
          borderRadius: 8,
          boxShadow: "0 2px 8px rgba(10, 40, 50, 0.4)",
        }}
      >
        <h1
          style={{
            fontSize: 24,
            marginBottom: 24,
            textAlign: "center",
            color: "var(--color-text)",
          }}
        >
          Sign in to ELSPETH
        </h1>

        {loginError && (
          <div
            role="alert"
            style={{
              padding: "8px 12px",
              marginBottom: 16,
              backgroundColor: "var(--color-error-bg)",
              color: "var(--color-error)",
              borderRadius: 4,
              fontSize: 14,
              border: "1px solid var(--color-error-border)",
            }}
          >
            {loginError}
          </div>
        )}

        {isOidc ? (
          /* OIDC / Entra SSO: single "Sign in with SSO" button */
          <button
            type="button"
            onClick={handleSsoRedirect}
            aria-label="Sign in with single sign-on"
            style={{
              display: "block",
              width: "100%",
              padding: "10px 16px",
              backgroundColor: "var(--color-accent)",
              color: "var(--color-text-inverse)",
              border: "none",
              borderRadius: 4,
              fontSize: 14,
              cursor: "pointer",
            }}
          >
            Sign in with SSO
          </button>
        ) : (
          /* Local auth: username/password form */
          <form onSubmit={handleSubmit}>
            <label
              htmlFor="login-username"
              style={{
                display: "block",
                marginBottom: 4,
                fontSize: 14,
                color: "var(--color-text)",
              }}
            >
              Username
            </label>
            <input
              id="login-username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
              style={{
                display: "block",
                width: "100%",
                padding: "8px 12px",
                marginBottom: 16,
                border: "1px solid var(--color-border-strong)",
                borderRadius: 4,
                fontSize: 14,
                boxSizing: "border-box",
                backgroundColor: "var(--color-surface-elevated)",
                color: "var(--color-text)",
              }}
            />

            <label
              htmlFor="login-password"
              style={{
                display: "block",
                marginBottom: 4,
                fontSize: 14,
                color: "var(--color-text)",
              }}
            >
              Password
            </label>
            <input
              id="login-password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
              style={{
                display: "block",
                width: "100%",
                padding: "8px 12px",
                marginBottom: 24,
                border: "1px solid var(--color-border-strong)",
                borderRadius: 4,
                fontSize: 14,
                boxSizing: "border-box",
                backgroundColor: "var(--color-surface-elevated)",
                color: "var(--color-text)",
              }}
            />

            <button
              type="submit"
              disabled={isSubmitting}
              aria-label={isSubmitting ? "Signing in" : "Sign in"}
              style={{
                display: "block",
                width: "100%",
                padding: "10px 16px",
                backgroundColor: isSubmitting
                  ? "var(--color-surface-elevated)"
                  : "var(--color-accent)",
                color: isSubmitting
                  ? "var(--color-text-muted)"
                  : "var(--color-text-inverse)",
                border: "none",
                borderRadius: 4,
                fontSize: 14,
                cursor: isSubmitting ? "not-allowed" : "pointer",
              }}
            >
              {isSubmitting ? "Signing in..." : "Sign in"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

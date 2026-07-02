import { useState, useEffect, type CSSProperties, type FormEvent } from "react";
import { useAuth } from "../../hooks/useAuth";
import * as api from "../../api/client";
import type { ApiError, AuthConfig } from "../../types/index";
import { Button, Input, AlertBanner, WordMark } from "../ui";

/**
 * Login page that adapts to the configured auth provider.
 *
 * Fetches GET /api/auth/config on mount to determine provider type:
 * - "local": renders a username/password form; when the backend's
 *   registration_mode is "open", also offers a "Create an account"
 *   view (username + password + confirm) that auto-logs the new
 *   account in via the token returned by POST /api/auth/register
 * - "oidc" or "entra": renders a "Sign in with SSO" button that
 *   redirects to config.authorization_endpoint (resolved from OIDC
 *   discovery by the backend)
 *
 * On return from an OIDC redirect, extracts the token from the URL
 * fragment or query parameter and calls loginWithToken().
 *
 * Failed sign-in attempts keep the username and clear only the
 * password (WCAG 3.3.7 Redundant Entry, elspeth-d49f8ad511); the
 * error banner is programmatically associated with the credential
 * fields via aria-invalid + aria-describedby.
 */

/** id linking the sign-in error banner to the credential fields. */
const LOGIN_ERROR_ID = "login-error";
/** id linking the registration error banner to its targeted fields. */
const REGISTER_ERROR_ID = "register-error";

/** Which registration fields the current registration error is about. */
interface RegisterErrorTargets {
  username: boolean;
  password: boolean;
  confirm: boolean;
}

const NO_REGISTER_TARGETS: RegisterErrorTargets = {
  username: false,
  password: false,
  confirm: false,
};

/** Inline copy of the app's link-button idiom (.tutorial-link-button) —
 *  LoginPage is inline-styled and has no dedicated stylesheet. */
const linkButtonStyle: CSSProperties = {
  border: 0,
  background: "transparent",
  color: "var(--color-link)",
  cursor: "pointer",
  font: "inherit",
  padding: 0,
  textDecoration: "underline",
  textUnderlineOffset: 3,
};

export function LoginPage() {
  const { login, loginWithToken, loginError } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [view, setView] = useState<"signin" | "register">("signin");
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [registerErrorTargets, setRegisterErrorTargets] =
    useState<RegisterErrorTargets>(NO_REGISTER_TARGETS);

  // Fetch auth config on mount to determine which login form to show
  useEffect(() => {
    api
      .fetchAuthConfig()
      .then((config) => {
        setAuthConfig(config);
        setConfigLoading(false);
      })
      .catch(() => {
        // If config fetch fails, fall back to local auth. Registration is
        // treated as closed — we don't know the effective mode, so we don't
        // advertise an affordance that may 404.
        setAuthConfig({
          provider: "local",
          registration_mode: "closed",
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
    const succeeded = await login(username, password);
    if (!succeeded) {
      // Keep the username (WCAG 3.3.7 Redundant Entry) — only the
      // rejected password is cleared, per convention.
      setPassword("");
    }
    setIsSubmitting(false);
  }

  async function handleRegister(e: FormEvent) {
    e.preventDefault();
    if (!username || !password || !confirmPassword) return;

    if (password !== confirmPassword) {
      setRegisterError("Passwords do not match.");
      setRegisterErrorTargets({ username: false, password: true, confirm: true });
      return;
    }

    setIsSubmitting(true);
    setRegisterError(null);
    setRegisterErrorTargets(NO_REGISTER_TARGETS);
    try {
      const { access_token } = await api.register(username, password);
      // The backend auto-logs the new account in; adopting the returned
      // token drops the user straight into the app.
      await loginWithToken(access_token);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 409) {
        setRegisterError("That username is not available.");
        setRegisterErrorTargets({ username: true, password: false, confirm: false });
      } else {
        setRegisterError("Registration failed. Please try again.");
        setRegisterErrorTargets(NO_REGISTER_TARGETS);
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  function switchView(next: "signin" | "register") {
    setView(next);
    // Keep the username across the switch (it's the common field);
    // passwords and stale errors don't carry over.
    setPassword("");
    setConfirmPassword("");
    setRegisterError(null);
    setRegisterErrorTargets(NO_REGISTER_TARGETS);
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
        <span className="spinner" aria-hidden="true" />
      </div>
    );
  }

  const isOidc =
    authConfig?.provider === "oidc" || authConfig?.provider === "entra";
  // Registration is a local-auth capability; only advertise it when the
  // backend's effective mode is "open" ("closed" and "email_verified"
  // both render nothing — the endpoint would refuse the request).
  const registrationOpen =
    authConfig?.provider === "local" &&
    authConfig?.registration_mode === "open";
  const showRegister = registrationOpen && view === "register";

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
        <div style={{ textAlign: "center", marginBottom: 24 }}>
          {/* The brand mark is the canonical <WordMark> (mono/uppercase/
              tracked). The positioning line below states what ELSPETH is —
              derived from the product's own "auditable outputs" thesis, in the
              public-service register. Copy is operator/UX-tunable. */}
          <WordMark as="h1" size={22} style={{ margin: 0 }} />
          <p
            style={{
              margin: "var(--space-sm) 0 0",
              fontSize: "var(--font-size-sm)",
              color: "var(--color-text-secondary)",
            }}
          >
            Build and run auditable data pipelines.
          </p>
        </div>

        {isOidc ? (
          <>
            {loginError && <AlertBanner tone="error">{loginError}</AlertBanner>}
            {/* OIDC / Entra SSO: single "Sign in with SSO" button */}
            <Button
              variant="primary"
              type="button"
              onClick={handleSsoRedirect}
              aria-label="Sign in with single sign-on"
            >
              Sign in with SSO
            </Button>
          </>
        ) : showRegister ? (
          /* Local auth: registration form (registration_mode="open" only) */
          <form
            onSubmit={handleRegister}
            aria-label="Create an account"
            style={{ display: "flex", flexDirection: "column", gap: "var(--space-md)" }}
          >
            <h2
              style={{
                margin: 0,
                fontSize: "var(--font-size-md)",
                fontWeight: 600,
              }}
            >
              Create an account
            </h2>

            {registerError && (
              <AlertBanner tone="error" id={REGISTER_ERROR_ID}>
                {registerError}
              </AlertBanner>
            )}

            <Input
              label="Username"
              id="register-username"
              type="text"
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              aria-invalid={registerErrorTargets.username ? true : undefined}
              aria-describedby={
                registerErrorTargets.username ? REGISTER_ERROR_ID : undefined
              }
            />

            <Input
              label="Password"
              id="register-password"
              type="password"
              autoComplete="new-password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              aria-invalid={registerErrorTargets.password ? true : undefined}
              aria-describedby={
                registerErrorTargets.password ? REGISTER_ERROR_ID : undefined
              }
            />

            <Input
              label="Confirm password"
              id="register-confirm-password"
              type="password"
              autoComplete="new-password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              aria-invalid={registerErrorTargets.confirm ? true : undefined}
              aria-describedby={
                registerErrorTargets.confirm ? REGISTER_ERROR_ID : undefined
              }
            />

            <Button
              variant="primary"
              type="submit"
              disabled={isSubmitting}
              aria-label={isSubmitting ? "Creating account" : "Create account"}
            >
              {isSubmitting ? "Creating account…" : "Create account"}
            </Button>

            <p
              style={{
                margin: 0,
                fontSize: "var(--font-size-sm)",
                color: "var(--color-text-secondary)",
                textAlign: "center",
              }}
            >
              Already have an account?{" "}
              <button
                type="button"
                style={linkButtonStyle}
                onClick={() => switchView("signin")}
              >
                Sign in
              </button>
            </p>
          </form>
        ) : (
          /* Local auth: username/password form */
          <>
            {loginError && (
              <AlertBanner tone="error" id={LOGIN_ERROR_ID}>
                {loginError}
              </AlertBanner>
            )}
            <form
              onSubmit={handleSubmit}
              style={{ display: "flex", flexDirection: "column", gap: "var(--space-md)" }}
            >
              {/* The sign-in error is deliberately generic (it never says which
                  field was wrong), so on failure BOTH credential fields are
                  flagged and described by the banner — same idiom as
                  SecretsPanel's form-error wiring. */}
              <Input
                label="Username"
                id="login-username"
                type="text"
                autoComplete="username"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                aria-invalid={loginError ? true : undefined}
                aria-describedby={loginError ? LOGIN_ERROR_ID : undefined}
              />

              <Input
                label="Password"
                id="login-password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                aria-invalid={loginError ? true : undefined}
                aria-describedby={loginError ? LOGIN_ERROR_ID : undefined}
              />

              <Button
                variant="primary"
                type="submit"
                disabled={isSubmitting}
                aria-label={isSubmitting ? "Signing in" : "Sign in"}
              >
                {isSubmitting ? "Signing in…" : "Sign in"}
              </Button>

              {registrationOpen && (
                <p
                  style={{
                    margin: 0,
                    fontSize: "var(--font-size-sm)",
                    color: "var(--color-text-secondary)",
                    textAlign: "center",
                  }}
                >
                  New to ELSPETH?{" "}
                  <button
                    type="button"
                    style={linkButtonStyle}
                    onClick={() => switchView("register")}
                  >
                    Create an account
                  </button>
                </p>
              )}
            </form>
          </>
        )}
      </div>
    </div>
  );
}

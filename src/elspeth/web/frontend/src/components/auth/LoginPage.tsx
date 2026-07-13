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
 *   registration_mode is "open" or "email_verified", also offers a
 *   "Create an account" view. Open registration auto-logs the account in;
 *   email-verified registration waits for the verification link.
 * - "oidc" or "entra": renders a "Sign in with SSO" button that
 *   redirects to config.authorization_endpoint (resolved from OIDC
 *   discovery by the backend)
 *
 * OIDC/Entra uses authorization code + S256 PKCE. On return, the callback
 * query and its single-use transaction are consumed before any decision or
 * network request. Email verification remains a separate local-auth path.
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
const OIDC_TRANSACTION_KEY = "oidc_transaction";
const OIDC_TRANSACTION_VERSION = 1;
const OIDC_TRANSACTION_MAX_AGE_MS = 5 * 60 * 1000;
const OIDC_TRANSACTION_FUTURE_SKEW_MS = 30 * 1000;
const CALLBACK_MAX_BYTES = 64 * 1024;
const ACCESS_TOKEN_MAX_BYTES = 16 * 1024;
const PKCE_VALUE = /^[A-Za-z0-9._~-]{43,128}$/;

interface OidcTransaction {
  version: 1;
  state: string;
  verifier: string;
  created_at: number;
}

interface CallbackCapture {
  kind: "oidc" | "verify";
  params: URLSearchParams;
  transactionJson: string | null;
  started: boolean;
}

// React StrictMode constructs the component twice. The callback must be
// scrubbed during the first render, while the second render must receive the
// same bounded in-memory capture rather than rereading URL/storage secrets.
let pendingCallbackCapture: CallbackCapture | null = null;

function scrubCallbackUrl(): void {
  window.history.replaceState(null, "", window.location.pathname);
}

function captureCallback(): CallbackCapture | null {
  if (pendingCallbackCapture !== null && !pendingCallbackCapture.started) {
    return pendingCallbackCapture;
  }

  const rawSearch = window.location.search;
  const oversizedCallback =
    new TextEncoder().encode(rawSearch).byteLength > CALLBACK_MAX_BYTES;
  const params = new URLSearchParams(
    oversizedCallback ? "?malformed=1" : rawSearch,
  );
  const oidcKeys = ["code", "state", "error", "error_description", "token"];
  const hasOidcSignal =
    oversizedCallback ||
    oidcKeys.some((key) => params.has(key)) ||
    window.location.hash.length > 0;
  const verificationOnly = params.has("verify_token") && !hasOidcSignal;

  if (!hasOidcSignal && !verificationOnly) return null;

  if (verificationOnly) {
    const capture: CallbackCapture = {
      kind: "verify",
      params,
      transactionJson: null,
      started: false,
    };
    scrubCallbackUrl();
    pendingCallbackCapture = capture;
    return capture;
  }

  let transactionJson = sessionStorage.getItem(OIDC_TRANSACTION_KEY);
  sessionStorage.removeItem(OIDC_TRANSACTION_KEY);
  if (
    transactionJson !== null &&
    new TextEncoder().encode(transactionJson).byteLength > CALLBACK_MAX_BYTES
  ) {
    transactionJson = null;
  }
  const capture: CallbackCapture = {
    kind: "oidc",
    params,
    transactionJson,
    started: false,
  };
  scrubCallbackUrl();
  pendingCallbackCapture = capture;
  return capture;
}

function parseOidcTransaction(raw: string | null, now: number): OidcTransaction | null {
  if (raw === null) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    const record = parsed as Record<string, unknown>;
    if (
      Object.keys(record).sort().join(",") !== "created_at,state,verifier,version" ||
      record.version !== OIDC_TRANSACTION_VERSION ||
      typeof record.state !== "string" ||
      !/^[A-Za-z0-9._~-]{8,256}$/.test(record.state) ||
      typeof record.verifier !== "string" ||
      !PKCE_VALUE.test(record.verifier) ||
      typeof record.created_at !== "number" ||
      !Number.isSafeInteger(record.created_at) ||
      now - record.created_at > OIDC_TRANSACTION_MAX_AGE_MS ||
      record.created_at - now > OIDC_TRANSACTION_FUTURE_SKEW_MS
    ) {
      return null;
    }
    return record as unknown as OidcTransaction;
  } catch {
    return null;
  }
}

function randomBase64Url(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function pkceChallenge(verifier: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier));
  let binary = "";
  for (const byte of new Uint8Array(digest)) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function readBoundedTokenResponse(response: Response): Promise<string> {
  if (!response.ok || response.redirected || response.type === "opaqueredirect") {
    throw new Error("token response failed status check");
  }
  const contentType = response.headers.get("Content-Type")?.split(";", 1)[0].trim().toLowerCase();
  if (contentType !== "application/json") {
    throw new Error("token response failed media-type check");
  }
  const contentLength = response.headers.get("Content-Length");
  if (contentLength !== null && /^\d+$/.test(contentLength) && Number(contentLength) > CALLBACK_MAX_BYTES) {
    throw new Error("token response failed size check");
  }

  const reader = response.body?.getReader();
  if (reader === undefined) throw new Error("token response failed body check");
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value !== undefined) {
      total += value.byteLength;
      if (total > CALLBACK_MAX_BYTES) {
        await reader.cancel();
        throw new Error("token response failed size check");
      }
      chunks.push(value);
    }
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  const decoded: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes));
  if (typeof decoded !== "object" || decoded === null || Array.isArray(decoded)) {
    throw new Error("token response failed shape check");
  }
  const token = decoded as Record<string, unknown>;
  if (
    token.token_type !== "Bearer" ||
    typeof token.access_token !== "string" ||
    token.access_token.trim().length === 0 ||
    new TextEncoder().encode(token.access_token).byteLength > ACCESS_TOKEN_MAX_BYTES
  ) {
    throw new Error("token response failed bearer-token check");
  }
  return token.access_token;
}

/** Which registration fields the current registration error is about. */
interface RegisterErrorTargets {
  username: boolean;
  email: boolean;
  password: boolean;
  confirm: boolean;
}

const NO_REGISTER_TARGETS: RegisterErrorTargets = {
  username: false,
  email: false,
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
  const [callbackCapture] = useState(captureCallback);
  const { login, loginWithToken, loginError } = useAuth();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [authConfig, setAuthConfig] = useState<AuthConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [configFailed, setConfigFailed] = useState(false);
  const [view, setView] = useState<"signin" | "register">("signin");
  const [registerError, setRegisterError] = useState<string | null>(null);
  const [registerNotice, setRegisterNotice] = useState<string | null>(null);
  const [verificationError, setVerificationError] = useState<string | null>(
    null,
  );
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
          token_endpoint: null,
        });
        setConfigFailed(true);
        setConfigLoading(false);
      });
  }, []);

  // Process the already-consumed callback. The capture happens synchronously
  // during render, before this or the config-fetch effect can start network IO.
  useEffect(() => {
    if (callbackCapture === null || callbackCapture.started) return;

    if (callbackCapture.kind === "verify") {
      callbackCapture.started = true;
      const verificationTokens = callbackCapture.params.getAll("verify_token");
      const verificationToken = verificationTokens.length === 1 ? verificationTokens[0] : "";
      pendingCallbackCapture = null;
      if (
        !verificationToken ||
        new TextEncoder().encode(verificationToken).byteLength > ACCESS_TOKEN_MAX_BYTES
      ) {
        setVerificationError("Email verification failed. Please request a new link.");
        return;
      }
      api
        .verifyEmail(verificationToken)
        .then(({ access_token }) => loginWithToken(access_token))
        .catch(() => {
          setVerificationError(
            "Email verification failed. Please request a new link.",
          );
        });
      return;
    }

    if (configLoading) return;
    callbackCapture.started = true;
    const fail = () => setVerificationError("Single sign-on failed. Please try again.");
    const finish = () => {
      pendingCallbackCapture = null;
    };

    const codeValues = callbackCapture.params.getAll("code");
    const stateValues = callbackCapture.params.getAll("state");
    const errorValues = callbackCapture.params.getAll("error");
    const descriptionValues = callbackCapture.params.getAll("error_description");
    const transaction = parseOidcTransaction(callbackCapture.transactionJson, Date.now());
    const callbackValid =
      !configFailed &&
      authConfig !== null &&
      (authConfig.provider === "oidc" || authConfig.provider === "entra") &&
      typeof authConfig.token_endpoint === "string" &&
      typeof authConfig.oidc_client_id === "string" &&
      codeValues.length === 1 &&
      stateValues.length === 1 &&
      errorValues.length === 0 &&
      descriptionValues.length === 0 &&
      !callbackCapture.params.has("token") &&
      !callbackCapture.params.has("verify_token") &&
      transaction !== null &&
      codeValues[0].length > 0 &&
      new TextEncoder().encode(codeValues[0]).byteLength <= ACCESS_TOKEN_MAX_BYTES &&
      stateValues[0] === transaction.state;

    if (!callbackValid || transaction === null || authConfig?.token_endpoint == null || authConfig.oidc_client_id == null) {
      fail();
      finish();
      return;
    }

    const redirectUri = new URL(window.location.pathname, window.location.origin).toString();
    const form = new URLSearchParams();
    form.set("grant_type", "authorization_code");
    form.set("code", codeValues[0]);
    form.set("client_id", authConfig.oidc_client_id);
    form.set("redirect_uri", redirectUri);
    form.set("code_verifier", transaction.verifier);

    void fetch(authConfig.token_endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
      credentials: "omit",
      redirect: "error",
      cache: "no-store",
      referrerPolicy: "no-referrer",
    })
      .then(readBoundedTokenResponse)
      .then((accessToken) => loginWithToken(accessToken))
      .catch(fail)
      .finally(finish);
  }, [authConfig, callbackCapture, configFailed, configLoading, loginWithToken]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!username || !password) return;

    setIsSubmitting(true);
    setVerificationError(null);
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
    const emailVerificationRequired =
      authConfig?.registration_mode === "email_verified";
    const trimmedEmail = email.trim();
    if (
      !username ||
      !password ||
      !confirmPassword ||
      (emailVerificationRequired && !trimmedEmail)
    ) {
      return;
    }

    if (password !== confirmPassword) {
      setRegisterError("Passwords do not match.");
      setRegisterErrorTargets({
        username: false,
        email: false,
        password: true,
        confirm: true,
      });
      return;
    }

    setIsSubmitting(true);
    setRegisterError(null);
    setRegisterNotice(null);
    setVerificationError(null);
    setRegisterErrorTargets(NO_REGISTER_TARGETS);
    try {
      const result = emailVerificationRequired
        ? await api.register(username, password, trimmedEmail)
        : await api.register(username, password);
      if ("access_token" in result) {
        // The backend auto-logs open-registration accounts in; adopting the
        // returned token drops the user straight into the app.
        await loginWithToken(result.access_token);
      } else {
        setRegisterNotice(`Check ${result.email} for the verification link.`);
        setPassword("");
        setConfirmPassword("");
      }
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 409) {
        setRegisterError("That username is not available.");
        setRegisterErrorTargets({
          username: true,
          email: false,
          password: false,
          confirm: false,
        });
      } else if (apiErr.status === 422 && emailVerificationRequired) {
        setRegisterError("Enter an email address to verify this account.");
        setRegisterErrorTargets({
          username: false,
          email: true,
          password: false,
          confirm: false,
        });
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
    setEmail("");
    setPassword("");
    setConfirmPassword("");
    setRegisterError(null);
    setRegisterNotice(null);
    setVerificationError(null);
    setRegisterErrorTargets(NO_REGISTER_TARGETS);
  }

  async function handleSsoRedirect() {
    if (!authConfig?.authorization_endpoint || !authConfig?.token_endpoint || !authConfig?.oidc_client_id) return;

    try {
      const state = randomBase64Url();
      const verifier = randomBase64Url();
      const challenge = await pkceChallenge(verifier);
      const transaction: OidcTransaction = {
        version: OIDC_TRANSACTION_VERSION,
        state,
        verifier,
        created_at: Date.now(),
      };
      sessionStorage.setItem(OIDC_TRANSACTION_KEY, JSON.stringify(transaction));

      const redirectUri = new URL(window.location.pathname, window.location.origin).toString();
      const url = new URL(authConfig.authorization_endpoint);
      url.searchParams.set("client_id", authConfig.oidc_client_id);
      url.searchParams.set("response_type", "code");
      url.searchParams.set("redirect_uri", redirectUri);
      url.searchParams.set("scope", "openid profile email");
      url.searchParams.set("state", state);
      url.searchParams.set("code_challenge", challenge);
      url.searchParams.set("code_challenge_method", "S256");

      const anchor = document.createElement("a");
      anchor.href = url.toString();
      anchor.rel = "noreferrer";
      anchor.click();
    } catch {
      sessionStorage.removeItem(OIDC_TRANSACTION_KEY);
      setVerificationError("Single sign-on failed. Please try again.");
    }
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
  // Registration is a local-auth capability; email_verified creates a
  // pending account and completes via the emailed verification link.
  const registrationAvailable =
    authConfig?.provider === "local" &&
    (authConfig?.registration_mode === "open" ||
      authConfig?.registration_mode === "email_verified");
  const emailVerificationRequired =
    authConfig?.registration_mode === "email_verified";
  const showRegister = registrationAvailable && view === "register";

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
            {verificationError && (
              <AlertBanner tone="error">{verificationError}</AlertBanner>
            )}
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
          /* Local auth: registration form */
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
            {registerNotice && (
              <AlertBanner tone="info">{registerNotice}</AlertBanner>
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

            {emailVerificationRequired && (
              <Input
                label="Email"
                id="register-email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                aria-invalid={registerErrorTargets.email ? true : undefined}
                aria-describedby={
                  registerErrorTargets.email ? REGISTER_ERROR_ID : undefined
                }
              />
            )}

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
            {verificationError && (
              <AlertBanner tone="error">{verificationError}</AlertBanner>
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

              {registrationAvailable && (
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

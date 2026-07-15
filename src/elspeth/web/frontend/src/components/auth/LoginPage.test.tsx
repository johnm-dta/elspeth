import { StrictMode } from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LoginPage } from "./LoginPage";
import { AuthGuard } from "../common/AuthGuard";
import * as api from "../../api/client";
import { useAuthStore } from "../../stores/authStore";
import { resetStore } from "../../test/store-helpers";
import type { AuthConfig } from "../../types/index";

// The real authStore + useAuth drive these tests (the field-wipe bug lived
// in the store/AuthGuard seam, so mocking the hook would test nothing);
// only the HTTP layer is mocked.
vi.mock("../../api/client", () => ({
  fetchAuthConfig: vi.fn(),
  login: vi.fn(),
  register: vi.fn(),
  verifyEmail: vi.fn(),
  fetchCurrentUser: vi.fn(),
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

function localConfig(
  mode: AuthConfig["registration_mode"] = "open",
): AuthConfig {
  return {
    provider: "local",
    registration_mode: mode,
    oidc_issuer: null,
    oidc_client_id: null,
    authorization_endpoint: null,
    token_endpoint: null,
  };
}

function oidcConfig(): AuthConfig {
  return {
    provider: "oidc",
    registration_mode: "closed",
    oidc_issuer: "https://cognito-idp.ap-southeast-2.amazonaws.com/pool-id",
    oidc_client_id: "public-client-id",
    authorization_endpoint:
      "https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/authorize",
    token_endpoint:
      "https://example.auth.ap-southeast-2.amazoncognito.com/oauth2/token",
  };
}

function setOidcTransaction(overrides: Record<string, unknown> = {}) {
  sessionStorage.setItem(
    "oidc_transaction",
    JSON.stringify({
      version: 1,
      state: "callback-state",
      verifier: "v".repeat(64),
      created_at: Date.now(),
      ...overrides,
    }),
  );
}

function jsonResponse(body: object): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

async function failOneSignIn(user: ReturnType<typeof userEvent.setup>) {
  vi.mocked(api.login).mockRejectedValue({
    status: 401,
    detail: "Invalid credentials",
  });
  await user.type(await screen.findByLabelText("Username"), "alice");
  await user.type(screen.getByLabelText("Password"), "wrong-password");
  await user.click(screen.getByRole("button", { name: "Sign in" }));
  return screen.findByRole("alert");
}

describe("LoginPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "/");
    sessionStorage.clear();
    localStorage.clear();
    resetStore(useAuthStore);
    // Simulate a completed boot (loadFromStorage resolved, no stored token).
    useAuthStore.setState({ isLoading: false });
    vi.mocked(api.fetchAuthConfig).mockReturnValue(new Promise(() => {}));
    vi.stubGlobal("fetch", vi.fn());
  });

  describe("OIDC authorization code with PKCE", () => {
    it("uses one single-use PKCE transaction and an authorization-code request", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
      let navigatedTo = "";
      const click = vi
        .spyOn(HTMLAnchorElement.prototype, "click")
        .mockImplementation(function (this: HTMLAnchorElement) {
          navigatedTo = this.href;
        });
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(await screen.findByRole("button", { name: /single sign-on/i }));

      expect(click).toHaveBeenCalledTimes(1);
      const request = new URL(navigatedTo);
      const transaction = JSON.parse(sessionStorage.getItem("oidc_transaction") ?? "null");
      expect(request.origin + request.pathname).toBe(oidcConfig().authorization_endpoint);
      expect(request.searchParams.get("response_type")).toBe("code");
      expect(request.searchParams.get("client_id")).toBe("public-client-id");
      expect(request.searchParams.get("state")).toBe(transaction.state);
      expect(request.searchParams.get("code_challenge_method")).toBe("S256");
      expect(request.searchParams.get("code_challenge")).toMatch(/^[A-Za-z0-9_-]{43}$/);
      const digest = await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(transaction.verifier),
      );
      let encoded = "";
      for (const byte of new Uint8Array(digest)) encoded += String.fromCharCode(byte);
      const expectedChallenge = btoa(encoded)
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=+$/, "");
      expect(request.searchParams.get("code_challenge")).toBe(expectedChallenge);
      expect(request.searchParams.has("nonce")).toBe(false);
      expect(request.searchParams.has("client_secret")).toBe(false);
      expect(request.searchParams.has("code_verifier")).toBe(false);
      expect(transaction).toEqual({
        version: 1,
        state: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
        verifier: expect.stringMatching(/^[A-Za-z0-9_-]{43}$/),
        created_at: expect.any(Number),
      });
    });

    it.each(["Bearer", "bearer", "BEARER", "BeArEr"])(
      "scrubs and exchanges a matching callback exactly once in StrictMode for token type %s",
      async (tokenType) => {
        window.history.replaceState(null, "", "/?code=short-code&state=callback-state#old");
        setOidcTransaction();
        vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
        vi.mocked(api.fetchCurrentUser).mockResolvedValue({
          user_id: "oidc-user",
          username: "oidc-user",
          display_name: null,
          email: null,
          groups: [],
        });
        vi.mocked(fetch).mockResolvedValue(
          new Response(JSON.stringify({ token_type: tokenType, access_token: "access-token" }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
        );

        render(
          <StrictMode>
            <LoginPage />
          </StrictMode>,
        );

        expect(window.location.search).toBe("");
        expect(window.location.hash).toBe("");
        expect(sessionStorage.getItem("oidc_transaction")).toBeNull();
        await waitFor(() => expect(useAuthStore.getState().token).toBe("access-token"));
        expect(fetch).toHaveBeenCalledTimes(1);
        const [endpoint, options] = vi.mocked(fetch).mock.calls[0];
        expect(endpoint).toBe(oidcConfig().token_endpoint);
        expect(options).toMatchObject({
          method: "POST",
          credentials: "omit",
          redirect: "error",
          cache: "no-store",
          referrerPolicy: "no-referrer",
        });
        const form = new URLSearchParams(String(options?.body));
        expect(Object.fromEntries(form)).toEqual({
          grant_type: "authorization_code",
          code: "short-code",
          client_id: "public-client-id",
          redirect_uri: window.location.origin + "/",
          code_verifier: "v".repeat(64),
        });
        expect(localStorage.getItem("auth_token")).toBe("access-token");
        expect(localStorage.getItem("refresh_token")).toBeNull();
      },
    );

    it.each([
      ["mismatched state", { state: "different" }, "/?code=code&state=callback-state"],
      ["missing state", {}, "/?code=code"],
      ["missing code", {}, "/?state=callback-state"],
      ["duplicate code", {}, "/?code=one&code=two&state=callback-state"],
      ["mixed error", {}, "/?code=code&state=callback-state&error=denied&error_description=secret"],
      ["expired transaction", { created_at: Date.now() - 300_001 }, "/?code=code&state=callback-state"],
      ["future transaction", { created_at: Date.now() + 60_000 }, "/?code=code&state=callback-state"],
      ["malformed verifier", { verifier: "short" }, "/?code=code&state=callback-state"],
      ["tampered metadata", { token_endpoint: "https://evil.example/token" }, "/?code=code&state=callback-state"],
    ])("fails closed for %s after synchronous cleanup", async (_name, overrides, url) => {
      window.history.replaceState(null, "", url);
      setOidcTransaction(overrides);
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());

      render(<LoginPage />);

      expect(window.location.search).toBe("");
      expect(sessionStorage.getItem("oidc_transaction")).toBeNull();
      expect(await screen.findByRole("alert")).toHaveTextContent("Single sign-on failed");
      expect(fetch).not.toHaveBeenCalled();
      expect(useAuthStore.getState().token).toBeNull();
      expect(screen.getByRole("alert")).not.toHaveTextContent(/secret|evil|code|callback-state/);
    });

    it.each([
      ["redirected", new Response("", { status: 200 }), { redirected: true }],
      ["non-2xx", new Response("provider secret", { status: 400 }), {}],
      ["malformed JSON", new Response("not json", { status: 200 }), {}],
      ["wrong token type", jsonResponse({ token_type: "Basic", access_token: "secret" }), {}],
      ["blank access token", jsonResponse({ token_type: "Bearer", access_token: " " }), {}],
      [
        "oversized access token",
        jsonResponse({ token_type: "Bearer", access_token: "x".repeat(16_385) }),
        {},
      ],
      ["oversized response", new Response("x".repeat(65_537), { status: 200 }), {}],
    ])("rejects a %s token response without adopting secrets", async (_name, response, responseOverrides) => {
      Object.defineProperties(response, {
        ...Object.fromEntries(
          Object.entries(responseOverrides).map(([key, value]) => [key, { value }]),
        ),
      });
      window.history.replaceState(null, "", "/?code=short-code&state=callback-state");
      setOidcTransaction();
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
      vi.mocked(fetch).mockResolvedValue(response);

      render(<LoginPage />);

      expect(await screen.findByRole("alert")).toHaveTextContent("Single sign-on failed");
      expect(useAuthStore.getState().token).toBeNull();
      expect(screen.getByRole("alert")).not.toHaveTextContent(/provider secret|short-code|access-token/);
    });

    it("fails closed on a token endpoint network error", async () => {
      window.history.replaceState(null, "", "/?code=short-code&state=callback-state");
      setOidcTransaction();
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
      vi.mocked(fetch).mockRejectedValue(new Error("network credential"));
      render(<LoginPage />);
      expect(await screen.findByRole("alert")).toHaveTextContent("Single sign-on failed");
      expect(screen.getByRole("alert")).not.toHaveTextContent("credential");
      expect(useAuthStore.getState().token).toBeNull();
    });

    it.each([null, "not-json", JSON.stringify({ version: 1 })])(
      "rejects a missing or malformed transaction",
      async (transaction) => {
        window.history.replaceState(null, "", "/?code=short-code&state=callback-state");
        if (transaction !== null) sessionStorage.setItem("oidc_transaction", transaction);
        vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
        render(<LoginPage />);
        expect(await screen.findByRole("alert")).toHaveTextContent("Single sign-on failed");
        expect(fetch).not.toHaveBeenCalled();
      },
    );

    it("does not exchange when fresh auth configuration fails", async () => {
      window.history.replaceState(null, "", "/?code=short-code&state=callback-state");
      setOidcTransaction();
      vi.mocked(api.fetchAuthConfig).mockRejectedValue(new Error("config secret"));
      render(<LoginPage />);
      expect(await screen.findByRole("alert")).toHaveTextContent("Single sign-on failed");
      expect(fetch).not.toHaveBeenCalled();
    });

    it("removes legacy token URLs without adopting them", async () => {
      window.history.replaceState(null, "", "/?token=legacy-secret&state=callback-state#access_token=fragment-secret");
      setOidcTransaction();
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(oidcConfig());
      render(<LoginPage />);
      expect(window.location.search).toBe("");
      expect(window.location.hash).toBe("");
      expect(fetch).not.toHaveBeenCalled();
      expect(useAuthStore.getState().token).toBeNull();
    });

    it("keeps verify_token on the separate email-verification path", async () => {
      window.history.replaceState(null, "", "/?verify_token=email-token");
      sessionStorage.setItem("oidc_transaction", "unrelated");
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("email_verified"));
      vi.mocked(api.verifyEmail).mockResolvedValue({ access_token: "verified-token" });
      vi.mocked(api.fetchCurrentUser).mockResolvedValue({
        user_id: "verified",
        username: "verified",
        display_name: null,
        email: "verified@example.com",
        groups: [],
      });
      render(<LoginPage />);
      await waitFor(() => expect(api.verifyEmail).toHaveBeenCalledWith("email-token"));
      expect(sessionStorage.getItem("oidc_transaction")).toBe("unrelated");
    });
  });

  it("exposes a single status region while auth configuration is loading", () => {
    const { container } = render(<LoginPage />);

    const statuses = screen.getAllByRole("status");
    expect(statuses).toHaveLength(1);
    expect(statuses[0]).toHaveAccessibleName(
      "Loading authentication configuration",
    );

    const spinner = container.querySelector(".spinner");
    expect(spinner).toHaveAttribute("aria-hidden", "true");
    expect(spinner).not.toHaveAttribute("role");
    expect(spinner).not.toHaveAttribute("aria-label");
  });

  it("renders the local-auth form with labelled inputs and a sign-in button", async () => {
    vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig());
    render(<LoginPage />);
    expect(await screen.findByLabelText("Username")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /sign in/i })).toBeInTheDocument();
  });

  it("keeps the username and clears only the password after a failed sign-in", async () => {
    // WCAG 3.3.7 Redundant Entry (elspeth-d49f8ad511): the app cleared
    // BOTH fields via an AuthGuard remount; only the rejected password
    // may be discarded.
    vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig());
    const user = userEvent.setup();
    render(<LoginPage />);

    const alert = await failOneSignIn(user);

    expect(alert).toHaveTextContent("Invalid username or password.");
    expect(screen.getByLabelText("Username")).toHaveValue("alice");
    expect(screen.getByLabelText("Password")).toHaveValue("");
  });

  it("associates the sign-in error with both credential fields via aria", async () => {
    // The error copy is deliberately generic (never says which field was
    // wrong), so both fields are flagged and described by the banner.
    vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig());
    const user = userEvent.setup();
    render(<LoginPage />);

    const alert = await failOneSignIn(user);
    expect(alert).toHaveAttribute("id", "login-error");

    for (const label of ["Username", "Password"]) {
      const input = screen.getByLabelText(label);
      expect(input).toHaveAttribute("aria-invalid", "true");
      expect(input).toHaveAttribute("aria-describedby", "login-error");
      expect(input).toHaveAccessibleDescription(
        "Invalid username or password.",
      );
    }
  });

  it("stays mounted through a failed sign-in when rendered inside AuthGuard", async () => {
    // Regression for the actual wipe mechanism: authStore.login() used to
    // flip the global isLoading flag, so AuthGuard swapped LoginPage for
    // its boot spinner mid-attempt and remounted a blank form on failure.
    vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig());
    const user = userEvent.setup();
    render(
      <AuthGuard>
        <div data-testid="app-shell" />
      </AuthGuard>,
    );

    await failOneSignIn(user);

    expect(screen.getByLabelText("Username")).toHaveValue("alice");
    expect(screen.queryByTestId("app-shell")).not.toBeInTheDocument();
  });

  describe("registration", () => {
    it("offers Create an account when registration is open", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("open"));
      render(<LoginPage />);
      expect(
        await screen.findByRole("button", { name: "Create an account" }),
      ).toBeInTheDocument();
    });

    it("offers Create an account when email verification is required", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("email_verified"));
      render(<LoginPage />);
      expect(
        await screen.findByRole("button", { name: "Create an account" }),
      ).toBeInTheDocument();
    });

    it.each(["closed"] as const)(
      "renders no registration affordance when registration_mode is %s",
      async (mode) => {
        vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig(mode));
        render(<LoginPage />);
        await screen.findByLabelText("Username");
        expect(
          screen.queryByRole("button", { name: "Create an account" }),
        ).not.toBeInTheDocument();
      },
    );

    it("registers a new account and signs it in with the returned token", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("open"));
      vi.mocked(api.register).mockResolvedValue({ access_token: "tok-new" });
      vi.mocked(api.fetchCurrentUser).mockResolvedValue({
        user_id: "u-new",
        username: "newuser",
        display_name: "newuser",
        email: null,
        groups: [],
      });
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(
        await screen.findByRole("button", { name: "Create an account" }),
      );
      await user.type(screen.getByLabelText("Username"), "newuser");
      await user.type(screen.getByLabelText("Password"), "correct-horse");
      await user.type(
        screen.getByLabelText("Confirm password"),
        "correct-horse",
      );
      await user.click(screen.getByRole("button", { name: "Create account" }));

      await waitFor(() => {
        expect(useAuthStore.getState().token).toBe("tok-new");
      });
      expect(api.register).toHaveBeenCalledWith("newuser", "correct-horse");
      expect(useAuthStore.getState().user).toMatchObject({
        username: "newuser",
      });
      expect(localStorage.getItem("auth_token")).toBe("tok-new");
    });

    it("registers an email-verified account and waits for verification", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("email_verified"));
      vi.mocked(api.register).mockResolvedValue({
        status: "verification_required",
        email: "new@example.com",
      });
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(
        await screen.findByRole("button", { name: "Create an account" }),
      );
      await user.type(screen.getByLabelText("Username"), "newuser");
      await user.type(screen.getByLabelText("Email"), "new@example.com");
      await user.type(screen.getByLabelText("Password"), "correct-horse");
      await user.type(
        screen.getByLabelText("Confirm password"),
        "correct-horse",
      );
      await user.click(screen.getByRole("button", { name: "Create account" }));

      expect(api.register).toHaveBeenCalledWith(
        "newuser",
        "correct-horse",
        "new@example.com",
      );
      expect(
        await screen.findByText(/check new@example.com/i),
      ).toBeInTheDocument();
      expect(useAuthStore.getState().token).toBeNull();
      expect(localStorage.getItem("auth_token")).toBeNull();
    });

    it("rejects mismatched passwords locally with aria-wired feedback", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("open"));
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(
        await screen.findByRole("button", { name: "Create an account" }),
      );
      await user.type(screen.getByLabelText("Username"), "newuser");
      await user.type(screen.getByLabelText("Password"), "one-password");
      await user.type(
        screen.getByLabelText("Confirm password"),
        "another-password",
      );
      await user.click(screen.getByRole("button", { name: "Create account" }));

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("Passwords do not match.");
      expect(alert).toHaveAttribute("id", "register-error");
      expect(api.register).not.toHaveBeenCalled();
      for (const label of ["Password", "Confirm password"]) {
        const input = screen.getByLabelText(label);
        expect(input).toHaveAttribute("aria-invalid", "true");
        expect(input).toHaveAttribute("aria-describedby", "register-error");
      }
      expect(screen.getByLabelText("Username")).not.toHaveAttribute(
        "aria-invalid",
      );
    });

    it("surfaces a username conflict without discarding the attempt", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("open"));
      vi.mocked(api.register).mockRejectedValue({
        status: 409,
        detail: "User already exists: taken",
      });
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(
        await screen.findByRole("button", { name: "Create an account" }),
      );
      await user.type(screen.getByLabelText("Username"), "taken");
      await user.type(screen.getByLabelText("Password"), "correct-horse");
      await user.type(
        screen.getByLabelText("Confirm password"),
        "correct-horse",
      );
      await user.click(screen.getByRole("button", { name: "Create account" }));

      const alert = await screen.findByRole("alert");
      expect(alert).toHaveTextContent("That username is not available.");
      const username = screen.getByLabelText("Username");
      expect(username).toHaveValue("taken");
      expect(username).toHaveAttribute("aria-invalid", "true");
      expect(username).toHaveAttribute("aria-describedby", "register-error");
    });

    it("returns to the sign-in view from the registration form", async () => {
      vi.mocked(api.fetchAuthConfig).mockResolvedValue(localConfig("open"));
      const user = userEvent.setup();
      render(<LoginPage />);

      await user.click(
        await screen.findByRole("button", { name: "Create an account" }),
      );
      expect(screen.getByLabelText("Confirm password")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Sign in" }));
      expect(
        screen.queryByLabelText("Confirm password"),
      ).not.toBeInTheDocument();
      expect(screen.getByLabelText("Username")).toBeInTheDocument();
    });
  });
});

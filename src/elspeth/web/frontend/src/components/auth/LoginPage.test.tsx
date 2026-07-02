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
  };
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

    it.each(["closed", "email_verified"] as const)(
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

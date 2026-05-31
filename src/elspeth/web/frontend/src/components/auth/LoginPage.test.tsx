import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LoginPage } from "./LoginPage";
import * as api from "../../api/client";

vi.mock("../../api/client", () => ({
  fetchAuthConfig: vi.fn(),
}));

vi.mock("../../hooks/useAuth", () => ({
  useAuth: () => ({
    login: vi.fn(),
    loginWithToken: vi.fn(),
    loginError: null,
  }),
}));

describe("LoginPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "/");
    sessionStorage.clear();
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
});

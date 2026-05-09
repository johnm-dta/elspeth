import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "./App";
import * as api from "./api/client";
import type { SystemStatus, UserProfile } from "./types/index";

// ── Sub-component stubs ──────────────────────────────────────────────────────
// App renders many heavy children (Layout, SessionSidebar, ChatPanel, …).
// Stub them out so the test focuses solely on App's own banner DOM.

vi.mock("./components/common/Layout", () => ({
  Layout: () => <div data-testid="layout-stub" />,
}));

vi.mock("./components/sessions/SessionSidebar", () => ({
  SessionSidebar: () => <div data-testid="session-sidebar-stub" />,
}));

vi.mock("./components/chat/ChatPanel", () => ({
  ChatPanel: () => <div data-testid="chat-panel-stub" />,
}));

vi.mock("./components/inspector/InspectorPanel", () => ({
  InspectorPanel: () => <div data-testid="inspector-panel-stub" />,
}));

vi.mock("./components/settings/SecretsPanel", () => ({
  SecretsPanel: () => <div data-testid="secrets-panel-stub" />,
}));

vi.mock("./components/common/CommandPalette", () => ({
  CommandPalette: () => <div data-testid="command-palette-stub" />,
  SWITCH_TAB_EVENT: "elspeth-switch-tab",
}));

vi.mock("./components/common/ShortcutsHelp", () => ({
  ShortcutsHelp: () => <div data-testid="shortcuts-help-stub" />,
}));

vi.mock("./components/common/ConfirmDialog", () => ({
  ConfirmDialog: () => <div data-testid="confirm-dialog-stub" />,
}));

// ── Auth stub ────────────────────────────────────────────────────────────────
// Mock useAuth so AuthGuard passes and renders children immediately.

vi.mock("./hooks/useAuth", () => ({
  useAuth: () => ({
    isAuthenticated: true,
    isLoading: false,
    user: {
      user_id: "test-001",
      username: "test-operator",
      display_name: null,
      email: null,
      groups: [],
    } satisfies UserProfile,
    loginError: null,
    login: vi.fn(),
    loginWithToken: vi.fn(),
    logout: vi.fn(),
  }),
}));

// ── API client stub ──────────────────────────────────────────────────────────
// fetchSystemStatus is called by App's health-check effect; provide a
// default resolved value (backend up, composer available) so the no-banner
// state is the default.  Individual tests override with vi.spyOn.

vi.mock("./api/client", () => ({
  fetchSystemStatus: vi.fn().mockResolvedValue({
    composer_available: true,
    composer_model: "gpt-4o",
    composer_provider: "openai",
    composer_reason: null,
    composer_missing_keys: [],
  } satisfies SystemStatus),
}));

// ── Store subscriptions ──────────────────────────────────────────────────────
// initStoreSubscriptions() runs at module load when App is imported and is
// idempotent (guarded by `initialized` flag), so it is benign here.

describe("App banner roles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Restore the default (backend up, composer available) after any
    // per-test override.
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
  });

  it("uses role=alert for the backend-unavailable banner (hard outage)", async () => {
    vi.spyOn(api, "fetchSystemStatus").mockRejectedValue(new Error("down"));
    // Silence the intentional diagnostic log so test stderr stays clean,
    // AND positively VER that the diagnostic actually fired.
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(<App />);

    const banner = await screen.findByText(/Backend unavailable/i);
    const root = banner.closest(".alert-banner") as HTMLElement | null;
    expect(root).not.toBeNull();
    expect(root!.getAttribute("role")).toBe("alert");

    // Diagnostic must have fired — the role=alert banner is the user-visible
    // signal; console.error is the operator-visible signal in DevTools.
    expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[health-check]"),
      expect.any(Error),
    );
    errorSpy.mockRestore();
  });

  it("uses role=status, not role=alert, for the composer-unavailable banner", async () => {
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: false,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: "No API key configured",
      composer_missing_keys: ["OPENAI_API_KEY"],
    } satisfies SystemStatus);

    render(<App />);

    const banner = await screen.findByText(/Service unavailable/i);
    const root = banner.closest(".alert-banner") as HTMLElement | null;
    expect(root).not.toBeNull();
    expect(root!.getAttribute("role")).toBe("status");
  });
});

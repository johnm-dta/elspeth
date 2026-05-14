import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import * as api from "./api/client";
import { resetStore } from "@/test/store-helpers";
import { useSessionStore } from "./stores/sessionStore";
import type {
  ChatMessage,
  CompositionState,
  SystemStatus,
  UserProfile,
} from "./types/index";

// ── Sub-component stubs ──────────────────────────────────────────────────────
// App renders many heavy children (Layout, SessionSidebar, ChatPanel, …).
// Stub them out so the test focuses solely on App's own banner DOM.

vi.mock("./components/common/Layout", () => ({
  Layout: ({
    sidebar,
    chat,
    inspector,
  }: {
    sidebar: React.ReactNode;
    chat: React.ReactNode;
    inspector: React.ReactNode;
  }) => (
    <div data-testid="layout-stub">
      {sidebar}
      {chat}
      {inspector}
    </div>
  ),
}));

vi.mock("./components/sessions/SessionSidebar", () => ({
  SessionSidebar: () => <div data-testid="session-sidebar-stub" />,
}));

vi.mock("./components/chat/ChatPanel", () => ({
  ChatPanel: () => {
    const sendMessage = useSessionStore((state) => state.sendMessage);
    const error = useSessionStore((state) => state.error);
    return (
      <div data-testid="chat-panel-stub">
        <button type="button" onClick={() => void sendMessage("compose")}>
          Send compose
        </button>
        {error ? <div role="alert">{error}</div> : null}
      </div>
    );
  },
}));

vi.mock("./components/inspector/InspectorPanel", () => ({
  InspectorPanel: () => <div data-testid="inspector-panel-stub" />,
  OPEN_CATALOG_EVENT: "open-catalog",
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
  fetchComposerProgress: vi.fn().mockResolvedValue({ phase: "idle" }),
  fetchRecoveryTranscript: vi.fn().mockResolvedValue([]),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  fetchMessages: vi.fn(),
}));

// ── Store subscriptions ──────────────────────────────────────────────────────
// initStoreSubscriptions() runs at module load when App is imported and is
// idempotent (guarded by `initialized` flag), so it is benign here.

describe("App banner roles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
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

  it("dispatches an open-catalog event on Ctrl+Shift+P", async () => {
    const onOpenCatalog = vi.fn();
    window.addEventListener("open-catalog", onOpenCatalog);

    render(<App />);
    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    fireEvent.keyDown(document, {
      key: "P",
      code: "KeyP",
      ctrlKey: true,
      shiftKey: true,
    });

    expect(onOpenCatalog).toHaveBeenCalledTimes(1);
    window.removeEventListener("open-catalog", onOpenCatalog);
  });
});

function makeState(version: number): CompositionState {
  return {
    id: `state-${version}`,
    version,
    source: null,
    nodes: [],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function makeAssistantMessage(): ChatMessage {
  return {
    id: "assistant-1",
    session_id: "session-1",
    role: "assistant",
    content: "done",
    tool_calls: null,
    created_at: "2026-05-14T00:00:00Z",
  };
}

describe("App composer recovery panel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
    vi.spyOn(api, "fetchRecoveryTranscript").mockResolvedValue([
      {
        id: "assistant-failed",
        session_id: "session-1",
        role: "assistant",
        content: "calling tools",
        raw_content: null,
        tool_calls: null,
        created_at: "2026-05-14T00:00:00Z",
        composition_state_id: null,
        tool_call_id: null,
        parent_assistant_id: null,
        sequence_no: 1,
      },
    ]);
  });

  it("opens the recovery panel for recovery-shaped send failures", async () => {
    const recovered = makeState(2);
    vi.spyOn(api, "sendMessage").mockRejectedValue({
      status: 500,
      detail: "compose failed",
      error_type: "composer_plugin_crash",
      partial_state: recovered,
      failed_turn: {
        assistant_message_id: "assistant-failed",
        tool_calls_attempted: 1,
        tool_responses_persisted: 1,
        transcript_url: null,
      },
    });
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(1),
    });

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Send compose" }));

    expect(
      await screen.findByRole("dialog", { name: "Recover partial composer draft" }),
    ).toBeInTheDocument();
  });

  it("applies and discards recovery locally without compose retries", async () => {
    const user = userEvent.setup();
    const recovered = makeState(3);
    const original = makeState(1);
    vi.spyOn(api, "sendMessage").mockRejectedValue({
      status: 500,
      detail: "compose failed",
      error_type: "composer_plugin_crash",
      partial_state: recovered,
      failed_turn: {
        assistant_message_id: "assistant-failed",
        tool_calls_attempted: 1,
        tool_responses_persisted: 1,
        transcript_url: null,
      },
    });
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: original,
    });

    render(<App />);
    await user.click(screen.getByRole("button", { name: "Send compose" }));
    await screen.findByRole("dialog", { name: "Recover partial composer draft" });
    await user.click(screen.getByRole("button", { name: "Apply partial draft" }));

    expect(useSessionStore.getState().compositionState).toBe(recovered);
    expect(api.sendMessage).toHaveBeenCalledTimes(1);
    expect(api.recompose).not.toHaveBeenCalled();
    expect(api.fetchMessages).not.toHaveBeenCalled();

    vi.mocked(api.sendMessage).mockRejectedValueOnce({
      status: 500,
      detail: "compose failed again",
      error_type: "composer_plugin_crash",
      partial_state: makeState(4),
      failed_turn: {
        assistant_message_id: "assistant-failed",
        tool_calls_attempted: 1,
        tool_responses_persisted: 1,
        transcript_url: null,
      },
    });
    await user.click(screen.getByRole("button", { name: "Send compose" }));
    await screen.findByRole("dialog", { name: "Recover partial composer draft" });
    await user.click(screen.getByRole("button", { name: "Discard recovery" }));

    expect(useSessionStore.getState().compositionState).toBe(recovered);
    expect(api.sendMessage).toHaveBeenCalledTimes(2);
    expect(api.recompose).not.toHaveBeenCalled();
    expect(api.fetchMessages).not.toHaveBeenCalled();
  });

  it("keeps non-recovery convergence errors on the existing chat error path", async () => {
    vi.spyOn(api, "sendMessage").mockRejectedValue({
      status: 422,
      error_type: "convergence",
      detail: "ignored",
    });
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(1),
    });

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Send compose" }));

    expect(
      await screen.findByText(/couldn't complete the composition/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("does not open recovery for successful sends", async () => {
    vi.spyOn(api, "sendMessage").mockResolvedValue({
      message: makeAssistantMessage(),
      state: makeState(2),
      proposals: [],
    });
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(1),
    });

    render(<App />);
    await userEvent.click(screen.getByRole("button", { name: "Send compose" }));

    await waitFor(() => expect(api.sendMessage).toHaveBeenCalledTimes(1));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(useSessionStore.getState().compositionState?.version).toBe(2);
  });
});

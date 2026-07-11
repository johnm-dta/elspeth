import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import * as api from "./api/client";
import { resetStore } from "@/test/store-helpers";
import { useSessionStore } from "./stores/sessionStore";
import { useExecutionStore } from "./stores/executionStore";
import { useAuthStore } from "./stores/authStore";
import {
  OPEN_GRAPH_MODAL_EVENT,
  OPEN_YAML_MODAL_EVENT,
} from "./lib/composer-events";
import type {
  ChatMessage,
  CompositionState,
  Session,
  SystemStatus,
  UserProfile,
} from "./types/index";
import {
  COMPOSE_CLIENT_GRACE_MS,
  getComposeTimeoutMs,
  resetComposeTimeoutForTests,
} from "@/config/composer";

// ── Sub-component stubs ──────────────────────────────────────────────────────
// App renders many heavy children (Layout, ChatPanel, …).
// Stub them out so the test focuses solely on App's own banner DOM.

const tutorialMountSpy = vi.hoisted(() => vi.fn());

vi.mock("./components/common/Layout", () => ({
  Layout: ({
    chat,
    siderail,
  }: {
    chat: React.ReactNode;
    siderail: React.ReactNode;
  }) => (
    <div data-testid="layout-stub">
      {chat}
      {siderail}
    </div>
  ),
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

vi.mock("./components/settings/SecretsPanel", () => ({
  SecretsPanel: () => <div data-testid="secrets-panel-stub" />,
}));

vi.mock("./components/tutorial", async () => {
  const React = await import("react");
  return {
    HelloWorldTutorial: () => {
      React.useEffect(() => {
        tutorialMountSpy();
      }, []);
      return <div data-testid="tutorial-stub" />;
    },
  };
});

vi.mock("./components/audit/AuditReadinessPanel", () => ({
  AuditReadinessPanel: () => <div data-testid="audit-readiness-stub" />,
}));

// GraphMiniView and GraphView render @xyflow flow graphs whenever the
// composition has content; xyflow needs ResizeObserver, which jsdom lacks.
// Stubbed — their behaviour is covered in their own test files.
vi.mock("./components/sidebar/GraphMiniView", () => ({
  GraphMiniView: () => <div data-testid="graph-mini-view-stub" />,
}));

vi.mock("./components/inspector/GraphView", () => ({
  GraphView: () => <div data-testid="graph-view-stub" />,
}));

vi.mock("./components/sidebar/SideRailValidationBanner", () => ({
  SideRailValidationBanner: () => (
    <div data-testid="side-rail-validation-banner-stub" />
  ),
}));

vi.mock("./components/common/CommandPalette", () => ({
  CommandPalette: () => <div data-testid="command-palette-stub" />,
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
  fetchSessions: vi.fn().mockResolvedValue([]),
  fetchRuns: vi.fn().mockResolvedValue([]),
  fetchComposerProgress: vi.fn().mockResolvedValue({ phase: "idle" }),
  fetchRecoveryTranscript: vi.fn().mockResolvedValue([]),
  listSources: vi.fn().mockResolvedValue([]),
  listTransforms: vi.fn().mockResolvedValue([]),
  listSinks: vi.fn().mockResolvedValue([]),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  fetchMessages: vi.fn(),
  // YamlView (inside ExportYamlModal) fetches the rendered YAML when the
  // modal opens — reachable now that the Ctrl+Shift+Y test seeds a
  // non-empty composition.
  fetchYaml: vi.fn().mockResolvedValue({ yaml: "sources: {}" }),
  // refreshAll fans out to refreshInterpretationEventsForSession on session
  // select, so this is called incidentally during App render. Without the mock
  // entry the call throws "no export defined on the mock" as an unhandled error
  // and fails the run even though every assertion passes.
  listInterpretationEvents: vi.fn().mockResolvedValue([]),
  // Phase 1B: account-level composer preferences. The real preferencesStore
  // module imports these from @/api/client and would receive `undefined`
  // (throwing at first call) without the mock entries.
  fetchUserComposerPreferences: vi.fn().mockResolvedValue({
    default_mode: "guided",
    banner_dismissed_at: null,
    tutorial_completed_at: "2026-05-19T00:00:00Z",
    tutorial_stage: null,
    tutorial_session_id: null,
    tutorial_run_id: null,
    tutorial_source_data_hash: null,
    updated_at: "2026-05-15T00:00:00Z",
  }),
  updateUserComposerPreferences: vi.fn(),
}));

// ── Shareable-reviews API stub ───────────────────────────────────────────────
// SharedInspectView (Phase 6B Task 8) calls fetchSharedInspect on mount. The
// shared-route test below never resolves the promise — that leaves the view
// in its "loading" state, which is enough to assert (a) SharedInspectView is
// mounted, (b) the composer Layout is suppressed.

vi.mock("./api/shareableReviews", () => ({
  fetchSharedInspect: vi.fn().mockReturnValue(new Promise(() => {})),
  markReadyForReview: vi.fn(),
  fetchShareableLink: vi.fn(),
}));

// ── Store subscriptions ──────────────────────────────────────────────────────
// initStoreSubscriptions() runs at module load when App is imported and is
// idempotent (guarded by `initialized` flag), so it is benign here.

describe("App banner roles", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
    useExecutionStore.getState().reset();
    // Seed authStore as authenticated. useSessionLifecycle now reads
    // useAuthStore(selectIsAuthenticated) directly and skips loadSessions
    // when not authenticated — without this seed, the post-AuthGuard
    // integration tests below would never trigger session load. The
    // useAuth() mock above only covers consumers of that hook; the
    // lifecycle reads the store directly.
    useAuthStore.setState({
      token: "test-token",
      user: {
        user_id: "test-001",
        username: "test-operator",
        display_name: null,
        email: null,
        groups: [],
      } as never,
    } as never);
    localStorage.clear();
    window.history.replaceState(null, "", "/");
    // Restore the default (backend up, composer available) after any
    // per-test override.
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
    vi.spyOn(api, "fetchSessions").mockResolvedValue([]);
    vi.spyOn(api, "fetchRuns").mockResolvedValue([]);
    tutorialMountSpy.mockClear();
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

  it("silently canonicalizes stale Runs hashes", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    window.history.replaceState(null, "", "#/session-1/runs");

    render(<App />);

    await waitFor(() => {
      expect(window.location.hash).toBe("#/session-1");
    });
  });

  it("silently canonicalizes stale Spec hashes", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    window.history.replaceState(null, "", "#/session-1/spec");

    render(<App />);

    await waitFor(() => {
      expect(window.location.hash).toBe("#/session-1");
    });
  });

  it("does not mount the retired sessions sidebar", async () => {
    render(<App />);

    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });
    expect(screen.queryByLabelText(/sessions sidebar/i)).not.toBeInTheDocument();
  });

  it("mounts audit readiness and validation through side rail slots", async () => {
    // An active session keeps the composer shell mounted — with no sessions
    // at all App now renders the empty landing instead (elspeth-e69642fede).
    useSessionStore.setState({ activeSessionId: "session-1" });
    render(<App />);

    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });
    expect(screen.getByTestId("audit-readiness-stub")).toBeInTheDocument();
    expect(
      screen.getByTestId("side-rail-validation-banner-stub"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("inspector-panel-stub")).toBeNull();
  });

  it("suppresses the SideRail while a guided build is active — the workspace rail inside ChatPanel replaces it", async () => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      // Non-terminal guided session at step_3 with no server turn — the
      // cold-start arm of isGuidedBuildActive, the same predicate ChatPanel's
      // workspace branch renders under. Rendering the SideRail alongside it
      // would put two rails side by side.
      guidedSession: {
        step: "step_3_transforms",
        history: [],
        terminal: null,
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      } as unknown as import("./types/guided").GuidedSession,
      guidedNextTurn: null,
    });
    render(<App />);

    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });
    // The composer shell is still mounted...
    expect(screen.getByTestId("chat-panel-stub")).toBeInTheDocument();
    // ...but App passed siderail={null}: no rail slots render.
    expect(screen.queryByTestId("audit-readiness-stub")).toBeNull();
    expect(screen.queryByTestId("side-rail-validation-banner-stub")).toBeNull();
  });

  it("restores the SideRail when the guided session reaches a terminal (Run/Export live in the rail post-completion)", async () => {
    useSessionStore.setState({
      activeSessionId: "session-1",
      guidedSession: {
        step: "step_4_wire",
        history: [],
        terminal: {
          kind: "completed",
          reason: null,
          pipeline_yaml: "pipeline: {}",
        },
        chat_history: [],
        chat_turn_seq: 0,
        profile: null,
      } as unknown as import("./types/guided").GuidedSession,
      guidedNextTurn: null,
    });
    render(<App />);

    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });
    expect(screen.getByTestId("audit-readiness-stub")).toBeInTheDocument();
    expect(
      screen.getByTestId("side-rail-validation-banner-stub"),
    ).toBeInTheDocument();
  });

  it("loads sessions on startup after SessionSidebar removal", async () => {
    const session: Session = {
      id: "session-loaded",
      title: "Loaded session",
      created_at: "2026-05-17T00:00:00Z",
      updated_at: "2026-05-17T00:00:00Z",
    };
    vi.spyOn(api, "fetchSessions").mockResolvedValue([session]);

    render(<App />);

    await waitFor(() => {
      expect(api.fetchSessions).toHaveBeenCalledTimes(1);
    });
    expect(useSessionStore.getState().sessions).toEqual([session]);
  });

  it("resets execution state and loads runs for the active session on startup", async () => {
    useSessionStore.setState({ activeSessionId: "session-1" });
    useExecutionStore.setState({
      activeRunId: "stale-run",
      runs: [{ id: "stale-run", status: "running" } as never],
      progress: { status: "running" } as never,
    } as never);

    render(<App />);

    await waitFor(() => {
      expect(api.fetchRuns).toHaveBeenCalledWith("session-1");
    });
    expect(useExecutionStore.getState().activeRunId).toBeNull();
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
    expect(screen.getByRole("dialog", { name: "Plugin Catalog" })).toBeInTheDocument();
    window.removeEventListener("open-catalog", onOpenCatalog);
  });

  it("dispatches graph and YAML modal events on Ctrl+Shift shortcuts", async () => {
    const onOpenGraph = vi.fn();
    const onOpenYaml = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
    window.addEventListener(OPEN_YAML_MODAL_EVENT, onOpenYaml);
    // Ctrl+Shift+Y is content-gated (elspeth-bff8043d33 residual): seed a
    // non-empty composition so the YAML dispatch fires.
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: {
        ...makeState(1),
        sources: { source: { plugin: "csv", options: {} } },
      },
    });

    render(<App />);
    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    fireEvent.keyDown(document, {
      key: "G",
      code: "KeyG",
      ctrlKey: true,
      shiftKey: true,
    });
    fireEvent.keyDown(document, {
      key: "Y",
      code: "KeyY",
      metaKey: true,
      shiftKey: true,
    });

    expect(onOpenGraph).toHaveBeenCalledTimes(1);
    expect(onOpenYaml).toHaveBeenCalledTimes(1);
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, onOpenGraph);
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, onOpenYaml);
  });

  it("does not dispatch retired inspector tab shortcuts on Alt+digit", async () => {
    const onSwitchTab = vi.fn();
    window.addEventListener("elspeth-switch-tab", onSwitchTab);

    render(<App />);
    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    fireEvent.keyDown(document, {
      key: "1",
      code: "Digit1",
      altKey: true,
    });

    expect(onSwitchTab).not.toHaveBeenCalled();
    window.removeEventListener("elspeth-switch-tab", onSwitchTab);
  });
});

function makeState(version: number): CompositionState {
  return {
    id: `state-${version}`,
    version,
    sources: {},
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

describe("App compose timeout readiness (bootstrap race)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
    resetComposeTimeoutForTests();
    useExecutionStore.getState().reset();
    useAuthStore.setState({
      token: "test-token",
      user: {
        user_id: "test-001",
        username: "test-operator",
        display_name: null,
        email: null,
        groups: [],
      } as never,
    } as never);
    localStorage.clear();
    window.history.replaceState(null, "", "/");
    vi.spyOn(api, "fetchSessions").mockResolvedValue([]);
    vi.spyOn(api, "fetchRuns").mockResolvedValue([]);
  });

  it("marks the composer ready and adopts the backend ceiling once system status lands", async () => {
    // A deployment configured ABOVE the checked-in default (300s wall clock)
    // is the exact case the stale 295s default would abort early. Readiness
    // must flip only after applyServerComposerTimeout adopts 300s → 325s.
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
      composer_timeout_seconds: 300,
    } satisfies SystemStatus);

    render(<App />);

    await waitFor(() =>
      expect(useSessionStore.getState().composeTimeoutReady).toBe(true),
    );
    expect(getComposeTimeoutMs()).toBe(300_000 + COMPOSE_CLIENT_GRACE_MS);
  });

  it("leaves the composer unready when system status fails, so no send starts against the unsafe default", async () => {
    vi.spyOn(api, "fetchSystemStatus").mockRejectedValue(new Error("down"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(<App />);

    await screen.findByText(/Backend unavailable/i);
    expect(useSessionStore.getState().composeTimeoutReady).toBe(false);
    errorSpy.mockRestore();
  });
});

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

// ── Phase 1B: preferences bootstrap on auth-success ──────────────────────
describe("App preferences bootstrap (Phase 1B)", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    // Reset the preferences store so the bootstrap spy fires on a clean
    // module-singleton state — without this, an earlier test in the same
    // worker that touched usePreferencesStore would leak `loaded: true`
    // and the new bootstrap call would still fire (we assert it does)
    // but the test intent is fragile to ordering. Mirrors the Phase 1A
    // finding-7 pattern noted in the plan.
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    resetStore(usePreferencesStore);
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    });
  });

  it("calls preferencesStore.bootstrap once authenticated", async () => {
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    const bootstrap = vi
      .spyOn(usePreferencesStore.getState(), "bootstrap")
      .mockResolvedValueOnce(undefined);

    render(<App />);

    await waitFor(() => {
      expect(bootstrap).toHaveBeenCalled();
    });
  });

  it("App still renders when preferences bootstrap fails (Panel test #1)", async () => {
    // Resilience claim: a failing prefs bootstrap MUST NOT block app
    // render. The earlier test spied on bootstrap-rejection and asserted
    // the App.tsx caller's .catch(console.error) was hit — but that
    // silent-swallow was the I5 bug (CorruptPreferencesError got
    // logged-and-forgotten, leaving the user with no signal). Bootstrap
    // is now contracted to NEVER reject; failures are surfaced via the
    // store's writeError so the role="alert" region (Phase 1B-round-2)
    // shows the user something is wrong.
    //
    // We exercise the failure path through the lower API mock rather
    // than spying on bootstrap directly, so the test runs through the
    // real bootstrap() implementation including the catch/writeError
    // branch — the part that was previously uncovered.
    const apiClient = await import("@/api/client");
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    (
      apiClient.fetchUserComposerPreferences as ReturnType<typeof vi.fn>
    ).mockRejectedValueOnce(new Error("network down"));

    render(<App />);

    // Wait for bootstrap to settle into the failure-path state.
    await waitFor(() => {
      const state = usePreferencesStore.getState();
      expect(state.loaded).toBe(true);
      expect(state.writeError).not.toBeNull();
    });
    // No-fabrication shape: defaultMode is still null (we don't guess).
    expect(usePreferencesStore.getState().defaultMode).toBeNull();
    // App chrome remains rendered. The Layout/ChatPanel stubs are still
    // present — proves the bootstrap failure didn't unmount the tree.
    expect(screen.getByTestId("chat-panel-stub")).toBeInTheDocument();
    // I5: the failure MUST be surfaced to the user via the always-mounted
    // alert region in App.tsx. Without this assertion the test would
    // pass even if writeError were set in the store but never rendered
    // to the DOM — the silent-failure-one-layer-up regression I5
    // exists to prevent. The role="alert" region carries the
    // writeError text from the store.
    await waitFor(() => {
      const alerts = screen.getAllByRole("alert");
      const surfaced = alerts.some(
        (el) => el.textContent !== null && el.textContent.includes("network down"),
      );
      expect(surfaced).toBe(true);
    });
  });

  it("renders the tutorial instead of the composer layout before completion", async () => {
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
    });
    vi.spyOn(usePreferencesStore.getState(), "bootstrap").mockResolvedValueOnce(
      undefined,
    );

    render(<App />);

    expect(screen.getByTestId("tutorial-stub")).toBeInTheDocument();
    expect(screen.queryByTestId("layout-stub")).not.toBeInTheDocument();
  });

  it("remounts the tutorial shell after Reset tutorial succeeds", async () => {
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      tutorialStage: "guided",
      tutorialSessionId: "sess-in-progress",
      tutorialRunId: null,
      tutorialSourceDataHash: null,
      writeError: null,
    });
    vi.spyOn(usePreferencesStore.getState(), "bootstrap").mockResolvedValueOnce(
      undefined,
    );
    vi.spyOn(api, "updateUserComposerPreferences").mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-10T07:30:00Z",
    });

    render(<App />);

    await waitFor(() => {
      expect(tutorialMountSpy).toHaveBeenCalledTimes(1);
    });
    await userEvent.click(screen.getByRole("button", { name: /account menu/i }));
    await userEvent.click(screen.getByRole("button", { name: /composer preferences/i }));
    await userEvent.click(screen.getByRole("button", { name: /reset tutorial/i }));

    await waitFor(() => {
      expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
        tutorial_completed_at: null,
        tutorial_stage: null,
        tutorial_session_id: null,
        tutorial_run_id: null,
        tutorial_source_data_hash: null,
      });
    });
    await waitFor(() => {
      expect(tutorialMountSpy).toHaveBeenCalledTimes(2);
    });
  });
});

// ── Phase 6B Task 8: shared-route Layout suppression ─────────────────────
//
// Plan §Task 8 test case 9 (`docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`):
// when the URL hash is `#/shared/{token}`, the composer Layout (chat +
// side rail) MUST NOT be rendered. The SharedInspectView mounts in its
// place. A future refactor that accidentally drops the App.tsx
// short-circuit and renders Layout under a shared route would convert
// every shared link into a full composer session for the reviewer — this
// test pins the suppression so that regression is caught.
describe("App shared-route Layout suppression (Phase 6B Task 8)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
    useExecutionStore.getState().reset();
    useAuthStore.setState({
      token: "test-token",
      user: {
        user_id: "test-001",
        username: "test-operator",
        display_name: null,
        email: null,
        groups: [],
      } as never,
    } as never);
    localStorage.clear();
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
  });

  it("renders SharedInspectView and does NOT mount the composer Layout for #/shared/{token}", async () => {
    // Set the hash BEFORE render — App reads it via useSharedToken on first
    // render. Setting after render would defeat the short-circuit branch
    // this test exists to pin.
    window.history.replaceState(null, "", "#/shared/tok-abc");

    render(<App />);

    // SharedInspectView is mounted. fetchSharedInspect is stubbed with a
    // never-resolving promise (see top-level vi.mock), so the view stays
    // in the loading state — which is exactly what we want to assert
    // against. We do NOT depend on the eventual resolution.
    expect(
      await screen.findByTestId("shared-inspect-loading"),
    ).toBeInTheDocument();

    // The composer Layout (mocked above as data-testid="layout-stub") is
    // NOT rendered under a shared route. If a refactor drops the
    // short-circuit in App.tsx, this assertion fails and the regression
    // is caught.
    expect(screen.queryByTestId("layout-stub")).toBeNull();
    // The chat panel is part of Layout; pin its absence too, because the
    // layout-stub testid is one indirection away from the actual chrome.
    expect(screen.queryByTestId("chat-panel-stub")).toBeNull();
  });

  it("renders the composer Layout when the hash is NOT a shared route", async () => {
    // Counterpoint: with no hash, the regular composer flow runs. This
    // protects the test above from a false-positive caused by the
    // Layout mock simply never rendering. An active session keeps the
    // shell mounted (no sessions at all → empty landing instead,
    // elspeth-e69642fede).
    window.history.replaceState(null, "", "/");
    useSessionStore.setState({ activeSessionId: "session-1" });

    render(<App />);

    expect(await screen.findByTestId("layout-stub")).toBeInTheDocument();
    expect(screen.queryByTestId("shared-inspect-loading")).toBeNull();
  });
});

// ── elspeth-e69642fede: returning-user landing ────────────────────────────
describe("App empty landing and auto-resume", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetStore(useSessionStore);
    useExecutionStore.getState().reset();
    useAuthStore.setState({
      token: "test-token",
      user: {
        user_id: "test-001",
        username: "test-operator",
        display_name: null,
        email: null,
        groups: [],
      } as never,
    } as never);
    localStorage.clear();
    window.history.replaceState(null, "", "/");
    vi.spyOn(api, "fetchSystemStatus").mockResolvedValue({
      composer_available: true,
      composer_model: "gpt-4o",
      composer_provider: "openai",
      composer_reason: null,
      composer_missing_keys: [],
    } satisfies SystemStatus);
    vi.spyOn(api, "fetchRuns").mockResolvedValue([]);
  });

  it("renders a real empty state with primary actions when no sessions exist", async () => {
    vi.spyOn(api, "fetchSessions").mockResolvedValue([]);

    render(<App />);

    expect(await screen.findByText(/no sessions yet/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /new session/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /browse the catalog/i }),
    ).toBeInTheDocument();
    // The composer shell is replaced, not layered under.
    expect(screen.queryByTestId("layout-stub")).toBeNull();
  });

  it("auto-resumes the most recently active session for a returning user", async () => {
    const older: Session = {
      id: "older",
      title: "Older",
      created_at: "2026-06-01T00:00:00Z",
      updated_at: "2026-06-01T00:00:00Z",
    };
    const newest: Session = {
      id: "newest",
      title: "Newest",
      created_at: "2026-07-01T00:00:00Z",
      updated_at: "2026-07-01T00:00:00Z",
    };
    vi.spyOn(api, "fetchSessions").mockResolvedValue([older, newest]);
    vi.spyOn(api, "fetchMessages").mockResolvedValue([]);

    render(<App />);

    await waitFor(() => {
      expect(useSessionStore.getState().activeSessionId).toBe("newest");
    });
    // With a session active, the composer shell renders — not the landing.
    expect(await screen.findByTestId("layout-stub")).toBeInTheDocument();
    expect(screen.queryByText(/no sessions yet/i)).toBeNull();
  });

  it("does not open the YAML modal on Ctrl+Shift+Y when the pipeline is empty", async () => {
    const onOpenYaml = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, onOpenYaml);
    vi.spyOn(api, "fetchSessions").mockResolvedValue([]);
    useSessionStore.setState({
      activeSessionId: "session-1",
      compositionState: makeState(1), // no sources/nodes/outputs
    });

    render(<App />);
    await waitFor(() => {
      expect(api.fetchSystemStatus).toHaveBeenCalled();
    });

    fireEvent.keyDown(document, {
      key: "Y",
      code: "KeyY",
      ctrlKey: true,
      shiftKey: true,
    });

    expect(onOpenYaml).not.toHaveBeenCalled();
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, onOpenYaml);
  });
});

import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { resetStore } from "@/test/store-helpers";
import type {
  ChatMessage,
  ComposerRecoveryError,
  ComposerProgressSnapshot,
  CompositionState,
  CompositionProposal,
} from "@/types/api";

const clearValidationMock = vi.hoisted(() => vi.fn());

// Mock the API client — store tests verify state logic, not HTTP calls
vi.mock("@/api/client", () => ({
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
  fetchMessages: vi.fn(),
  fetchCompositionState: vi.fn(),
  fetchCompositionProposals: vi.fn(),
  fetchComposerPreferences: vi.fn(),
  fetchComposerProgress: vi.fn(),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  acceptCompositionProposal: vi.fn(),
  rejectCompositionProposal: vi.fn(),
  forkFromMessage: vi.fn(),
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn(),
  archiveSession: vi.fn(),
  getGuided: vi.fn(),
}));

// Mock the execution store dependency
vi.mock("./executionStore", () => ({
  useExecutionStore: {
    getState: () => ({ clearValidation: clearValidationMock }),
  },
}));

function makeCompositionState(version: number, nodeIds: string[] = []): CompositionState {
  return {
    id: `state-${version}`,
    version,
    source: null,
    nodes: nodeIds.map((id) => ({
      id,
      node_type: "transform",
      plugin: "passthrough",
      input: "source",
      on_success: "out",
      on_error: null,
      options: {},
    })),
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function makeRecoveryError(
  partialState = makeCompositionState(2),
): ComposerRecoveryError {
  return {
    status: 500,
    detail: "compose failed",
    error_type: "composer_plugin_crash",
    partial_state: partialState,
    failed_turn: {
      assistant_message_id: "assistant-1",
      tool_calls_attempted: 2,
      tool_responses_persisted: 1,
      transcript_url: null,
    },
  };
}

function makeCompositionProposal(
  overrides: Partial<CompositionProposal> = {},
): CompositionProposal {
  return {
    id: "proposal-1",
    session_id: "session-1",
    tool_call_id: "tool-call-1",
    tool_name: "set_pipeline",
    status: "pending",
    summary: "Replace the current pipeline.",
    rationale: "The user asked for a new pipeline.",
    affects: ["source", "transforms", "outputs"],
    arguments_redacted_json: { source: { plugin: "csv" } },
    base_state_id: "state-1",
    committed_state_id: null,
    audit_event_id: null,
    created_at: "2026-05-14T00:00:00Z",
    updated_at: "2026-05-14T00:00:00Z",
    ...overrides,
  };
}

describe("sessionStore", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    resetStore(useSessionStore);
  });

  describe("initial state", () => {
    it("starts with empty sessions and no active session", () => {
      const state = useSessionStore.getState();
      expect(state.sessions).toEqual([]);
      expect(state.activeSessionId).toBeNull();
      expect(state.messages).toEqual([]);
      expect(state.compositionState).toBeNull();
      expect(state.isComposing).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe("sendMessage optimistic insert", () => {
    it("appends optimistic user message and sets composing", async () => {
      // Pre-condition: set an active session so sendMessage proceeds
      useSessionStore.setState({ activeSessionId: "session-1" });

      // Start the send — it will await the mocked API call (which
      // returns undefined by default, causing the catch branch).
      // We only care about the intermediate optimistic state here.
      const sendPromise = useSessionStore.getState().sendMessage("hello");

      // After the synchronous part of sendMessage runs, check state
      const state = useSessionStore.getState();
      expect(state.isComposing).toBe(true);
      expect(state.messages).toHaveLength(1);
      expect(state.messages[0].role).toBe("user");
      expect(state.messages[0].content).toBe("hello");
      expect(state.messages[0].local_status).toBe("pending");

      // Let the promise settle (will hit error path since mock returns undefined)
      await sendPromise;
    });

    it("marks message as failed when API call throws", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 500,
        detail: "Server error",
      });

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.isComposing).toBe(false);
      expect(state.error).toBe("Server error");
      expect(state.messages[0].local_status).toBe("failed");
      expect(state.messages[0].local_error).toBe("Server error");
    });

    it("clears local_status on successful response", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      (mockSendMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        message: {
          id: "asst-1",
          session_id: "session-1",
          role: "assistant",
          content: "Hello back",
          tool_calls: null,
          created_at: new Date().toISOString(),
        },
        state: null,
      });

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.isComposing).toBe(false);
      // User message should have local_status cleared
      const userMsg = state.messages.find((m) => m.role === "user");
      expect(userMsg?.local_status).toBeUndefined();
      // Assistant message should be appended
      const asstMsg = state.messages.find((m) => m.role === "assistant");
      expect(asstMsg?.content).toBe("Hello back");
    });

    it("handles convergence error with specific message", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 422,
        error_type: "convergence",
        detail: "ignored",
      });

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.error).toContain("couldn't complete the composition");
    });

    it("maps a client-side AbortError to the compose-timeout copy, not the generic fallback", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      // AbortController.abort() rejects the in-flight fetch with a
      // DOMException whose name is 'AbortError'. The store must
      // distinguish this from a server failure or the user gets the
      // misleading "Failed to send message. Please try again." fallback.
      const abortError = new DOMException(
        "The user aborted a request.",
        "AbortError",
      );
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        abortError,
      );

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.isComposing).toBe(false);
      expect(state.error).toContain("ELSPETH took too long");
      expect(state.error).not.toContain("Failed to send message");
      expect(state.messages[0].local_status).toBe("failed");
      expect(state.messages[0].local_error).toBe(state.error);
    });

    it("includes provider detail when an LLM unavailable response exposes it", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 502,
        error_type: "llm_unavailable",
        detail: "APIError",
        provider_detail:
          "litellm.APIError: OpenRouter upstream rejected request: insufficient credits",
        provider_status_code: 402,
      });

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.error).toContain("The AI service is temporarily unavailable");
      expect(state.error).toContain(
        "litellm.APIError: OpenRouter upstream rejected request: insufficient credits",
      );
      expect(state.error).toContain("Provider status: 402");
      expect(state.messages[0].local_error).toBe(state.error);
    });

    it("opens recovery state for recovery-shaped compose failures", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      const recoveryError = makeRecoveryError();
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        recoveryError,
      );

      useSessionStore.setState({
        activeSessionId: "session-1",
        compositionState: makeCompositionState(5),
      });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.recoveryError).toBe(recoveryError);
      expect(state.recoveryStartedCompositionVersion).toBe(5);
      expect(state.messages[0].local_status).toBe("failed");
      expect(state.messages[0].local_error).toBe("compose failed");
    });

    it("does not open recovery state for convergence errors without recovery fields", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      (mockSendMessage as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 422,
        error_type: "convergence",
        detail: "ignored",
      });

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("hello");

      const state = useSessionStore.getState();
      expect(state.error).toContain("couldn't complete the composition");
      expect(state.recoveryError).toBeNull();
      expect(state.recoveryStartedCompositionVersion).toBeNull();
    });

    it("polls composer progress only while a send is composing", async () => {
      vi.useFakeTimers();
      try {
        const {
          sendMessage: mockSendMessage,
          fetchComposerProgress,
        } = await import("@/api/client");
        const sendDeferred =
          deferred<{ message: ChatMessage; state: null }>();
        const progress: ComposerProgressSnapshot = {
          session_id: "session-1",
          request_id: "message-1",
          phase: "using_tools",
          headline: "The model requested plugin schemas.",
          evidence: ["Checking available source, transform, and sink tools."],
          likely_next: "ELSPETH will use the schemas to choose a pipeline shape.",
          reason: null,
          updated_at: "2026-04-26T10:00:00Z",
        };
        const assistantMessage: ChatMessage = {
          id: "assistant-1",
          session_id: "session-1",
          role: "assistant",
          content: "Done",
          tool_calls: null,
          created_at: "2026-04-26T10:00:02Z",
        };

        (mockSendMessage as ReturnType<typeof vi.fn>).mockReturnValueOnce(
          sendDeferred.promise,
        );
        (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue(
          progress,
        );

        useSessionStore.setState({ activeSessionId: "session-1" });
        const sendPromise = useSessionStore.getState().sendMessage("hello");

        await Promise.resolve();
        expect(fetchComposerProgress).toHaveBeenCalledTimes(1);
        expect(fetchComposerProgress).toHaveBeenLastCalledWith("session-1");

        await vi.advanceTimersByTimeAsync(1500);
        expect(fetchComposerProgress).toHaveBeenCalledTimes(2);
        expect(useSessionStore.getState().composerProgress).toEqual(progress);

        sendDeferred.resolve({ message: assistantMessage, state: null });
        await sendPromise;

        expect(useSessionStore.getState().isComposing).toBe(false);
        expect(useSessionStore.getState().composerProgress).toBeNull();

        await vi.advanceTimersByTimeAsync(3000);
        expect(fetchComposerProgress).toHaveBeenCalledTimes(2);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe("composer proposals", () => {
    it("loads proposals when selecting a session", async () => {
      const apiClient = await import("@/api/client");
      const proposal = makeCompositionProposal();
      (apiClient.fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValue([]);
      (apiClient.fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValue(
        null,
      );
      (
        apiClient.fetchCompositionProposals as ReturnType<typeof vi.fn>
      ).mockResolvedValue([proposal]);
      (
        apiClient.fetchComposerPreferences as ReturnType<typeof vi.fn>
      ).mockResolvedValue({
        session_id: "session-1",
        trust_mode: "explicit_approve",
        density_default: "high",
        updated_at: "2026-05-14T00:00:00Z",
      });
      (apiClient.getGuided as ReturnType<typeof vi.fn>).mockRejectedValue(
        new Error("guided unavailable"),
      );

      await useSessionStore.getState().selectSession("session-1");

      expect(useSessionStore.getState().compositionProposals).toEqual([
        proposal,
      ]);
      expect(useSessionStore.getState().composerPreferences?.trust_mode).toBe(
        "explicit_approve",
      );
    });

    it("appends proposals returned by sendMessage without waiting for a session reload", async () => {
      const apiClient = await import("@/api/client");
      const proposal = makeCompositionProposal();
      (apiClient.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValue({
        message: {
          id: "asst-1",
          session_id: "session-1",
          role: "assistant",
          content: "Review this proposed change.",
          tool_calls: null,
          created_at: "2026-05-14T00:00:01Z",
        },
        state: null,
        proposals: [proposal],
      });

      useSessionStore.setState({ activeSessionId: "session-1", messages: [] });
      await useSessionStore.getState().sendMessage("build it");

      expect(useSessionStore.getState().compositionProposals).toEqual([
        proposal,
      ]);
    });

    it("marks stale proposals after accept returns a stale-state conflict", async () => {
      const apiClient = await import("@/api/client");
      (
        apiClient.acceptCompositionProposal as ReturnType<typeof vi.fn>
      ).mockRejectedValue(Object.assign(new Error("stale"), { status: 409 }));
      (
        apiClient.fetchCompositionProposals as ReturnType<typeof vi.fn>
      ).mockResolvedValue([]);

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().acceptProposal("proposal-1");

      expect(useSessionStore.getState().staleProposalIds).toContain(
        "proposal-1",
      );
    });
  });

  describe("recovery state actions", () => {
    it("applies recovered state locally and clears validation selection and recovery state", () => {
      const recovered = makeCompositionState(2, ["kept"]);
      useSessionStore.setState({
        compositionState: makeCompositionState(1, ["stale"]),
        selectedNodeId: "stale",
        recoveryError: makeRecoveryError(recovered),
        recoveryStartedCompositionVersion: 1,
      });

      const result = useSessionStore.getState().applyRecoveredState();

      const state = useSessionStore.getState();
      expect(result).toEqual({ applied: true, needsConfirmation: false });
      expect(state.compositionState).toBe(recovered);
      expect(state.selectedNodeId).toBeNull();
      expect(state.recoveryError).toBeNull();
      expect(state.recoveryStartedCompositionVersion).toBeNull();
      expect(clearValidationMock).toHaveBeenCalledTimes(1);
    });

    it("requires confirmation when current version differs from compose-start version", () => {
      const recovered = makeCompositionState(3, ["next"]);
      useSessionStore.setState({
        compositionState: makeCompositionState(2, ["current"]),
        recoveryError: makeRecoveryError(recovered),
        recoveryStartedCompositionVersion: 1,
      });

      const first = useSessionStore.getState().applyRecoveredState();
      expect(first).toEqual({ applied: false, needsConfirmation: true });
      expect(useSessionStore.getState().compositionState?.version).toBe(2);

      const confirmed = useSessionStore
        .getState()
        .applyRecoveredState({ confirmed: true });
      expect(confirmed).toEqual({ applied: true, needsConfirmation: false });
      expect(useSessionStore.getState().compositionState).toBe(recovered);
    });

    it("discardRecovery closes local recovery state without API calls or state mutation", async () => {
      const apiClient = await import("@/api/client");
      const current = makeCompositionState(4, ["current"]);
      useSessionStore.setState({
        compositionState: current,
        recoveryError: makeRecoveryError(makeCompositionState(5)),
        recoveryStartedCompositionVersion: 4,
      });

      useSessionStore.getState().discardRecovery();
      useSessionStore.getState().discardRecovery();

      expect(useSessionStore.getState().compositionState).toBe(current);
      expect(useSessionStore.getState().recoveryError).toBeNull();
      expect(useSessionStore.getState().recoveryStartedCompositionVersion).toBeNull();
      expect(apiClient.sendMessage).not.toHaveBeenCalled();
      expect(apiClient.recompose).not.toHaveBeenCalled();
      expect(apiClient.fetchMessages).not.toHaveBeenCalled();
    });

    it("replaces stale recovery state with the newest recovery failure", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      const first = makeRecoveryError(makeCompositionState(2));
      const second = makeRecoveryError(makeCompositionState(3));
      (mockSendMessage as ReturnType<typeof vi.fn>)
        .mockRejectedValueOnce(first)
        .mockRejectedValueOnce(second);

      useSessionStore.setState({
        activeSessionId: "session-1",
        compositionState: makeCompositionState(1),
      });
      await useSessionStore.getState().sendMessage("first");
      useSessionStore.getState().discardRecovery();
      await useSessionStore.getState().sendMessage("second");

      expect(useSessionStore.getState().recoveryError).toBe(second);
      expect(useSessionStore.getState().recoveryStartedCompositionVersion).toBe(1);
    });

    it("successful compose while recovery is open makes later apply require confirmation", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      const recovered = makeCompositionState(2);
      useSessionStore.setState({
        activeSessionId: "session-1",
        compositionState: makeCompositionState(1),
        recoveryError: makeRecoveryError(recovered),
        recoveryStartedCompositionVersion: 1,
      });
      const assistantMessage: ChatMessage = {
        id: "assistant-2",
        session_id: "session-1",
        role: "assistant",
        content: "new success",
        tool_calls: null,
        created_at: new Date().toISOString(),
      };
      (mockSendMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        message: assistantMessage,
        state: makeCompositionState(3),
      });

      await useSessionStore.getState().sendMessage("new work");
      const result = useSessionStore.getState().applyRecoveredState();

      expect(result).toEqual({ applied: false, needsConfirmation: true });
      expect(useSessionStore.getState().compositionState?.version).toBe(3);
    });

    it("clears recovery state on session transitions and reset", async () => {
      const apiClient = await import("@/api/client");
      const session = {
        id: "new-session",
        title: "New",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      };
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockResolvedValue(
        session,
      );
      (apiClient.archiveSession as ReturnType<typeof vi.fn>).mockResolvedValue(
        undefined,
      );
      (apiClient.fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValue(
        [],
      );
      (apiClient.fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValue(
        makeCompositionState(1),
      );
      (apiClient.forkFromMessage as ReturnType<typeof vi.fn>).mockResolvedValue({
        session,
        messages: [],
        composition_state: makeCompositionState(1),
      });
      (apiClient.getGuided as ReturnType<typeof vi.fn>).mockResolvedValue({
        guided_session: null,
        next_turn: null,
        terminal: null,
        composition_state: null,
      });

      const seedRecovery = () =>
        useSessionStore.setState({
          activeSessionId: "session-1",
          recoveryError: makeRecoveryError(),
          recoveryStartedCompositionVersion: 1,
        });

      seedRecovery();
      await useSessionStore.getState().selectSession("session-2");
      expect(useSessionStore.getState().recoveryError).toBeNull();

      seedRecovery();
      await useSessionStore.getState().createSession();
      expect(useSessionStore.getState().recoveryError).toBeNull();

      seedRecovery();
      await useSessionStore.getState().archiveSession("session-1");
      expect(useSessionStore.getState().recoveryError).toBeNull();

      seedRecovery();
      await useSessionStore.getState().forkFromMessage("message-1", "fork");
      expect(useSessionStore.getState().recoveryError).toBeNull();

      seedRecovery();
      useSessionStore.getState().reset();
      expect(useSessionStore.getState().recoveryError).toBeNull();
      expect(useSessionStore.getState().recoveryStartedCompositionVersion).toBeNull();
    });
  });

  describe("retryMessage abort handling", () => {
    it("maps a client-side AbortError to the compose-timeout copy", async () => {
      const { recompose: mockRecompose } = await import("@/api/client");
      const abortError = new DOMException(
        "The user aborted a request.",
        "AbortError",
      );
      (mockRecompose as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        abortError,
      );

      const userMessage: ChatMessage = {
        id: "user-1",
        session_id: "session-1",
        role: "user",
        content: "hello",
        tool_calls: null,
        created_at: new Date().toISOString(),
      };
      useSessionStore.setState({
        activeSessionId: "session-1",
        messages: [userMessage],
      });

      await useSessionStore.getState().retryMessage("user-1");

      const state = useSessionStore.getState();
      expect(state.isComposing).toBe(false);
      expect(state.error).toContain("ELSPETH took too long");
      expect(state.error).not.toContain("Failed to send message");
      expect(state.messages[0].local_status).toBe("failed");
      expect(state.messages[0].local_error).toBe(state.error);
    });
  });

  describe("reset", () => {
    it("restores initial state", () => {
      useSessionStore.setState({
        activeSessionId: "session-1",
        isComposing: true,
        error: "some error",
      });

      useSessionStore.getState().reset();

      const state = useSessionStore.getState();
      expect(state.activeSessionId).toBeNull();
      expect(state.isComposing).toBe(false);
      expect(state.error).toBeNull();
    });
  });
});

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason?: unknown) => void = () => undefined;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

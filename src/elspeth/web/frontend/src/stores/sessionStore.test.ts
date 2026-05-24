import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useInterpretationEventsStore } from "./interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type {
  ChatMessage,
  ComposerRecoveryError,
  ComposerProgressSnapshot,
  CompositionState,
  CompositionProposal,
} from "@/types/api";
import type { InterpretationEvent } from "@/types/interpretation";

const clearValidationMock = vi.hoisted(() => vi.fn());
const validateMock = vi.hoisted(() => vi.fn());

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
  renameSession: vi.fn(),
  getGuided: vi.fn(),
  // Phase 1B — sessionStore.createSession calls resolveDefaultMode() on the
  // preferencesStore, which falls back to fetchUserComposerPreferences()
  // when the prefs store hasn't been bootstrapped. The default mock returns
  // freeform so existing createSession-touching tests (which preceded this
  // change and assert non-guided behaviour) keep passing.
  fetchUserComposerPreferences: vi.fn().mockResolvedValue({
    default_mode: "freeform",
    banner_dismissed_at: null,
    tutorial_completed_at: null,
    updated_at: "2026-05-15T00:00:00Z",
  }),
  updateUserComposerPreferences: vi.fn(),
  // Phase 5b — sessionStore.selectSession fires a fire-and-forget refreshAll
  // on the interpretationEventsStore, which routes through this method.
  // Mocked to resolve with an empty array so the unhandled-rejection path
  // does not trip session-load tests; targeted assertions on the call live
  // in interpretationEventsStore.test.ts.
  listInterpretationEvents: vi.fn().mockResolvedValue([]),
}));

// Mock the execution store dependency
vi.mock("./executionStore", () => ({
  useExecutionStore: {
    getState: () => ({
      clearValidation: clearValidationMock,
      validate: validateMock,
    }),
  },
}));

function makeCompositionState(version: number, nodeIds: string[] = []): CompositionState {
  return {
    id: `state-${version}`,
    version,
    sources: {},
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

function makePendingInterpretationEvent(id: string): InterpretationEvent {
  return {
    id,
    session_id: "session-1",
    composition_state_id: "state-1",
    affected_node_id: "analyze_colors",
    tool_call_id: "call-1",
    user_term: "llm_model_choice:analyze_colors",
    kind: "llm_model_choice",
    llm_draft: "openrouter/openai/gpt-5.4-mini",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-05-29T12:00:00Z",
    resolved_at: null,
    actor: "system:composer",
    interpretation_source: "user_approved",
    model_identifier: null,
    model_version: null,
    provider: null,
    composer_skill_hash: null,
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
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
  beforeEach(async () => {
    vi.resetAllMocks();
    resetStore(useSessionStore);
    // Phase 1B: keep existing createSession-touching tests on the pre-change
    // behaviour by pinning preferences to freeform-loaded. The new
    // "createSession honours default mode" describe overrides this per test
    // to exercise guided / unloaded paths explicitly.
    const { usePreferencesStore } = await import("@/stores/preferencesStore");
    resetStore(usePreferencesStore);
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
      writing: false,
    });
    // Reseed the @/api/client mock that vi.resetAllMocks() cleared so the
    // preferences-bootstrap fallback path in resolveDefaultMode() still
    // resolves under tests that drive an unloaded prefs store.
    const apiMod = await import("@/api/client");
    (apiMod.fetchUserComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValue({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });
    // Phase 5b — sessionStore.selectSession fires a fire-and-forget
    // refreshAll on the interpretationEventsStore which awaits this
    // method.  vi.resetAllMocks() wiped the at-mock-declaration default,
    // so reseed it here; an unmocked or undefined-returning fn surfaces
    // as an unhandled rejection in the per-session refresh path.
    (apiMod.listInterpretationEvents as ReturnType<typeof vi.fn>).mockResolvedValue([]);
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

    it("refreshes pending interpretation events after a successful freeform compose turn", async () => {
      // Regression: a freeform compose turn can create new pending
      // interpretation events (invented_source / llm_prompt_template /
      // llm_model_choice / pipeline_decision). Unlike guided mode (review
      // delivered as a guided turn) and the tutorial (explicit refreshAll),
      // the freeform path had no trigger to pull them into the
      // interpretationEventsStore — so the inline review widgets and their
      // sign-off buttons never rendered mid-session, while the run-gate still
      // blocked execution on the pending rows. selectSession refreshes on
      // reload, which previously masked the gap.
      resetStore(useInterpretationEventsStore);
      const apiMod = await import("@/api/client");
      (apiMod.sendMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        message: {
          id: "asst-1",
          session_id: "session-1",
          role: "assistant",
          content: "Drafted.",
          tool_calls: null,
          created_at: new Date().toISOString(),
        },
        state: null,
      });
      (apiMod.listInterpretationEvents as ReturnType<typeof vi.fn>).mockResolvedValue([
        makePendingInterpretationEvent("evt-1"),
      ]);

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().sendMessage("rate these pages");

      // refreshAll is fire-and-forget inside sendMessage; await the microtask.
      await vi.waitFor(() => {
        const map =
          useInterpretationEventsStore.getState().pendingBySession["session-1"];
        expect(map?.["evt-1"]).toBeDefined();
      });
    });

    it("refreshes pending interpretation events after a successful recompose (retry)", async () => {
      // Same bug class as the freeform sendMessage path: recompose can mint new
      // interpretive decisions, and without a refresh the inline review widgets
      // never surface mid-session.
      resetStore(useInterpretationEventsStore);
      const apiMod = await import("@/api/client");
      (apiMod.recompose as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        message: {
          id: "asst-r",
          session_id: "session-1",
          role: "assistant",
          content: "Redone.",
          tool_calls: null,
          created_at: new Date().toISOString(),
        },
        state: null,
      });
      (apiMod.listInterpretationEvents as ReturnType<typeof vi.fn>).mockResolvedValue([
        makePendingInterpretationEvent("evt-recompose"),
      ]);

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

      await vi.waitFor(() => {
        const map =
          useInterpretationEventsStore.getState().pendingBySession["session-1"];
        expect(map?.["evt-recompose"]).toBeDefined();
      });
    });

    it("refreshes pending interpretation events after accepting a proposal", async () => {
      // Same bug class: accepting a proposal (explicit_approve path) can create
      // interpretation events that must surface their review widgets.
      resetStore(useInterpretationEventsStore);
      const apiMod = await import("@/api/client");
      (apiMod.acceptCompositionProposal as ReturnType<typeof vi.fn>).mockResolvedValue({
        id: "proposal-1",
      });
      (apiMod.fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValue(null);
      (apiMod.fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValue([]);
      (apiMod.listInterpretationEvents as ReturnType<typeof vi.fn>).mockResolvedValue([
        makePendingInterpretationEvent("evt-accept"),
      ]);

      useSessionStore.setState({ activeSessionId: "session-1" });
      await useSessionStore.getState().acceptProposal("proposal-1");

      await vi.waitFor(() => {
        const map =
          useInterpretationEventsStore.getState().pendingBySession["session-1"];
        expect(map?.["evt-accept"]).toBeDefined();
      });
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
        const completeProgress: ComposerProgressSnapshot = {
          ...progress,
          phase: "complete",
          headline: "Composition saved.",
          evidence: ["The pipeline state was updated."],
          likely_next: "Review the pipeline or run it.",
          updated_at: "2026-04-26T10:00:03Z",
        };

        (fetchComposerProgress as ReturnType<typeof vi.fn>)
          .mockResolvedValueOnce(progress)
          .mockResolvedValueOnce(progress)
          .mockResolvedValueOnce(completeProgress);

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
        expect(useSessionStore.getState().composerProgress).toEqual(
          completeProgress,
        );

        await vi.advanceTimersByTimeAsync(3000);
        expect(fetchComposerProgress).toHaveBeenCalledTimes(3);
      } finally {
        vi.useRealTimers();
      }
    });

    it("polls inflight messages while a send is composing and stops on completion", async () => {
      vi.useFakeTimers();
      try {
        const {
          sendMessage: mockSendMessage,
          fetchMessages,
          fetchComposerProgress,
        } = await import("@/api/client");
        const sendDeferred =
          deferred<{ message: ChatMessage; state: null }>();
        const canonicalUser: ChatMessage = {
          id: "msg-canonical-user",
          session_id: "session-1",
          role: "user",
          content: "hello",
          tool_calls: null,
          created_at: "2026-04-26T10:00:00Z",
        };
        const inflightAssistant: ChatMessage = {
          id: "msg-inflight-assistant",
          session_id: "session-1",
          role: "assistant",
          content: "",
          tool_calls: [
            {
              id: "tc-1",
              type: "function",
              function: { name: "list_models", arguments: "{}" },
            },
          ],
          created_at: "2026-04-26T10:00:01Z",
        };
        const finalAssistant: ChatMessage = {
          id: "msg-final-assistant",
          session_id: "session-1",
          role: "assistant",
          content: "Done",
          tool_calls: null,
          created_at: "2026-04-26T10:00:02Z",
        };
        // First poll sees the canonical user + an in-flight tool-call row.
        // Subsequent polls would see more rows; the final post-completion
        // sync sees the full list.
        (fetchMessages as ReturnType<typeof vi.fn>)
          .mockResolvedValueOnce([canonicalUser, inflightAssistant])
          .mockResolvedValueOnce([canonicalUser, inflightAssistant, finalAssistant]);
        (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
          session_id: "session-1",
          request_id: "msg-canonical-user",
          phase: "idle",
          headline: "",
          evidence: [],
          likely_next: null,
          reason: null,
          updated_at: "2026-04-26T10:00:00Z",
        });
        (mockSendMessage as ReturnType<typeof vi.fn>).mockReturnValueOnce(
          sendDeferred.promise,
        );

        useSessionStore.setState({ activeSessionId: "session-1", messages: [] });
        const sendPromise = useSessionStore.getState().sendMessage("hello");

        // Drain the optimistic-append microtask.
        await Promise.resolve();
        // Polling fires on the interval, not at start, so we advance to the
        // first tick.
        await vi.advanceTimersByTimeAsync(1500);
        expect(fetchMessages).toHaveBeenCalledTimes(1);
        const afterFirstPoll = useSessionStore.getState().messages;
        // Canonical user has replaced the local-* optimistic row, and the
        // in-flight assistant is now visible — exactly the "tool calls
        // appear in the same bubble as they come in" behaviour.
        expect(afterFirstPoll.map((m) => m.id)).toEqual([
          "msg-canonical-user",
          "msg-inflight-assistant",
        ]);

        sendDeferred.resolve({ message: finalAssistant, state: null });
        await sendPromise;
        // Post-completion sync brings the final assistant row in.
        expect(useSessionStore.getState().messages.map((m) => m.id)).toEqual([
          "msg-canonical-user",
          "msg-inflight-assistant",
          "msg-final-assistant",
        ]);
        // Polling stops once isComposing flips back to false.
        await vi.advanceTimersByTimeAsync(3000);
        expect(fetchMessages).toHaveBeenCalledTimes(2);
      } finally {
        vi.useRealTimers();
      }
    });

    it("stops inflight message polling when the store is reset", async () => {
      vi.useFakeTimers();
      try {
        const {
          sendMessage: mockSendMessage,
          fetchMessages,
          fetchComposerProgress,
        } = await import("@/api/client");
        const sendDeferred =
          deferred<{ message: ChatMessage; state: null }>();
        (mockSendMessage as ReturnType<typeof vi.fn>).mockReturnValueOnce(
          sendDeferred.promise,
        );
        (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValue([]);
        (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
          session_id: "session-1",
          request_id: "msg-pending",
          phase: "idle",
          headline: "",
          evidence: [],
          likely_next: null,
          reason: null,
          updated_at: "2026-04-26T10:00:00Z",
        });

        useSessionStore.setState({ activeSessionId: "session-1", messages: [] });
        const sendPromise = useSessionStore.getState().sendMessage("hello");
        await Promise.resolve();

        await vi.advanceTimersByTimeAsync(1500);
        expect(fetchMessages).toHaveBeenCalledTimes(1);

        useSessionStore.getState().reset();
        await vi.advanceTimersByTimeAsync(4500);
        expect(fetchMessages).toHaveBeenCalledTimes(1);

        sendDeferred.resolve({
          message: {
            id: "msg-final",
            session_id: "session-1",
            role: "assistant",
            content: "ok",
            tool_calls: null,
            created_at: "2026-04-26T10:00:01Z",
          },
          state: null,
        });
        await sendPromise;
      } finally {
        vi.useRealTimers();
      }
    });

    it("drops a stale send response after the active session changes", async () => {
      const { sendMessage: mockSendMessage } = await import("@/api/client");
      const sendDeferred = deferred<{
        message: ChatMessage;
        state: CompositionState | null;
      }>();
      (mockSendMessage as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        sendDeferred.promise,
      );

      useSessionStore.setState({
        activeSessionId: "session-1",
        messages: [],
        compositionState: makeCompositionState(1),
      });
      const sendPromise = useSessionStore.getState().sendMessage("hello");
      await Promise.resolve();

      const sessionTwoState = makeCompositionState(2);
      useSessionStore.setState({
        activeSessionId: "session-2",
        messages: [],
        compositionState: sessionTwoState,
        isComposing: false,
      });
      sendDeferred.resolve({
        message: {
          id: "stale-assistant",
          session_id: "session-1",
          role: "assistant",
          content: "stale",
          tool_calls: null,
          created_at: "2026-04-26T10:00:01Z",
        },
        state: makeCompositionState(99),
      });
      await sendPromise;

      const state = useSessionStore.getState();
      expect(state.activeSessionId).toBe("session-2");
      expect(state.messages).toEqual([]);
      expect(state.compositionState).toBe(sessionTwoState);
      expect(state.isComposing).toBe(false);
    });

    it("preserves the optimistic user message when the canonical row has not yet appeared", async () => {
      vi.useFakeTimers();
      try {
        const {
          sendMessage: mockSendMessage,
          fetchMessages,
          fetchComposerProgress,
        } = await import("@/api/client");
        const sendDeferred =
          deferred<{ message: ChatMessage; state: null }>();
        // First poll returns an empty list (the canonical user hasn't been
        // persisted yet — should not happen in production, but the merge
        // logic must not silently drop the optimistic row if it does).
        (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
        (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
          session_id: "session-1",
          request_id: "msg-pending",
          phase: "idle",
          headline: "",
          evidence: [],
          likely_next: null,
          reason: null,
          updated_at: "2026-04-26T10:00:00Z",
        });
        (mockSendMessage as ReturnType<typeof vi.fn>).mockReturnValueOnce(
          sendDeferred.promise,
        );

        useSessionStore.setState({ activeSessionId: "session-1", messages: [] });
        const sendPromise = useSessionStore.getState().sendMessage("the user's question");
        await Promise.resolve();
        await vi.advanceTimersByTimeAsync(1500);
        const afterEmptyPoll = useSessionStore.getState().messages;
        // The local-* optimistic row survives because no canonical row in
        // fresh matched its (role, content) tuple.
        expect(afterEmptyPoll).toHaveLength(1);
        expect(afterEmptyPoll[0].id).toMatch(/^local-/);
        expect(afterEmptyPoll[0].role).toBe("user");
        expect(afterEmptyPoll[0].content).toBe("the user's question");

        // Wind down cleanly.
        (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValue([
          {
            id: "msg-final",
            session_id: "session-1",
            role: "assistant",
            content: "ok",
            tool_calls: null,
            created_at: "2026-04-26T10:00:01Z",
          } as ChatMessage,
        ]);
        sendDeferred.resolve({
          message: {
            id: "msg-final",
            session_id: "session-1",
            role: "assistant",
            content: "ok",
            tool_calls: null,
            created_at: "2026-04-26T10:00:01Z",
          } as ChatMessage,
          state: null,
        });
        await sendPromise;
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe("renameSession", () => {
    it("persists a trimmed title and updates the matching session", async () => {
      const apiClient = await import("@/api/client");
      const renamed = {
        id: "session-1",
        title: "Renamed pipeline",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:01:00Z",
      };
      (
        apiClient.renameSession as ReturnType<typeof vi.fn>
      ).mockResolvedValue(renamed);
      useSessionStore.setState({
        activeSessionId: "session-1",
        sessions: [
          {
            id: "session-1",
            title: "Current session",
            created_at: "2026-05-14T00:00:00Z",
            updated_at: "2026-05-14T00:00:00Z",
          },
        ],
      });

      await useSessionStore.getState().renameSession("session-1", "  Renamed pipeline  ");

      expect(apiClient.renameSession).toHaveBeenCalledWith(
        "session-1",
        "Renamed pipeline",
      );
      expect(useSessionStore.getState().sessions[0]).toBe(renamed);
      expect(useSessionStore.getState().error).toBeNull();
    });

    it("does not call the API for a blank title", async () => {
      const apiClient = await import("@/api/client");
      useSessionStore.setState({
        sessions: [
          {
            id: "session-1",
            title: "Current session",
            created_at: "2026-05-14T00:00:00Z",
            updated_at: "2026-05-14T00:00:00Z",
          },
        ],
      });

      await useSessionStore.getState().renameSession("session-1", "   ");

      expect(apiClient.renameSession).not.toHaveBeenCalled();
      expect(useSessionStore.getState().sessions[0].title).toBe("Current session");
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
        interpretation_review_disabled: false,
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

  describe("applyResolvedInterpretation", () => {
    it("applies the patched composition state and re-validates so the run-gate can reopen", () => {
      const newState = makeCompositionState(3, ["analyze_colors"]);
      useSessionStore.setState({
        activeSessionId: "session-1",
        compositionState: makeCompositionState(2, ["analyze_colors"]),
      });

      useSessionStore.getState().applyResolvedInterpretation(newState);

      // Display sync: the resolved interpretation's patched pipeline is shown.
      expect(useSessionStore.getState().compositionState).toBe(newState);
      // Gate clearing: an explicit re-validate runs (the auto-validate
      // subscription only fires on a version bump, which a resolve can't
      // guarantee).
      expect(validateMock).toHaveBeenCalledWith("session-1");
    });

    it("re-validates even when the resolve returns no new state (null)", () => {
      useSessionStore.setState({
        activeSessionId: "session-1",
        compositionState: makeCompositionState(2),
      });

      useSessionStore.getState().applyResolvedInterpretation(null);

      // No state to apply, but the gate must still be re-checked.
      expect(useSessionStore.getState().compositionState?.version).toBe(2);
      expect(validateMock).toHaveBeenCalledWith("session-1");
    });

    it("is a no-op without an active session", () => {
      useSessionStore.setState({ activeSessionId: null });

      useSessionStore
        .getState()
        .applyResolvedInterpretation(makeCompositionState(3));

      expect(validateMock).not.toHaveBeenCalled();
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

  describe("selectSession stale session handling", () => {
    it("clears stale active session when selected session no longer exists", async () => {
      const apiClient = await import("@/api/client");
      (apiClient.fetchMessages as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 404,
        detail: "Session not found",
      });
      (apiClient.fetchCompositionState as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 404,
        detail: "Session not found",
      });
      (
        apiClient.fetchCompositionProposals as ReturnType<typeof vi.fn>
      ).mockRejectedValueOnce({
        status: 404,
        detail: "Session not found",
      });
      (
        apiClient.fetchComposerPreferences as ReturnType<typeof vi.fn>
      ).mockRejectedValueOnce({
        status: 404,
        detail: "Session not found",
      });

      useSessionStore.setState({
        sessions: [],
        activeSessionId: null,
        messages: [
          {
            id: "old-message",
            session_id: "old-session",
            role: "user",
            content: "stale",
            tool_calls: null,
            created_at: "2026-05-14T00:00:00Z",
          },
        ],
        compositionState: makeCompositionState(7),
        error: null,
      });

      await useSessionStore.getState().selectSession("missing-session");

      const state = useSessionStore.getState();
      expect(state.activeSessionId).toBeNull();
      expect(state.messages).toEqual([]);
      expect(state.compositionState).toBeNull();
      expect(state.compositionProposals).toEqual([]);
      expect(state.composerPreferences).toBeNull();
      expect(state.error).toBeNull();
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

  // ── Phase 1B: createSession honours composer default-mode preference ──
  describe("createSession honours default mode", () => {
    it("leaves guidedSession null when default mode is freeform", async () => {
      const apiClient = await import("@/api/client");
      const { usePreferencesStore } = await import("@/stores/preferencesStore");
      usePreferencesStore.setState({
        loaded: true,
        defaultMode: "freeform",
        bannerDismissedAt: null,
        writing: false,
      });
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        id: "sess-1",
        title: "untitled",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      });
      const enterGuided = vi
        .spyOn(useSessionStore.getState(), "enterGuided")
        .mockResolvedValue();

      await useSessionStore.getState().createSession();

      expect(enterGuided).not.toHaveBeenCalled();
      expect(useSessionStore.getState().guidedSession).toBeNull();
      expect(useSessionStore.getState().activeSessionId).toBe("sess-1");
    });

    it("enters guided mode when default mode is guided", async () => {
      const apiClient = await import("@/api/client");
      const { usePreferencesStore } = await import("@/stores/preferencesStore");
      usePreferencesStore.setState({
        loaded: true,
        defaultMode: "guided",
        bannerDismissedAt: null,
        writing: false,
      });
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        id: "sess-2",
        title: "untitled",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      });
      const enterGuided = vi
        .spyOn(useSessionStore.getState(), "enterGuided")
        .mockResolvedValue();

      await useSessionStore.getState().createSession();

      expect(enterGuided).toHaveBeenCalledTimes(1);
    });

    it("prefs-bootstrap failure does NOT mask successful session creation (Panel M1)", async () => {
      // Regression pin for the createSession try-block split. Earlier shape:
      // single try wrapped both api.createSession() and resolveDefaultMode();
      // a prefs-bootstrap rejection was attributed to "Failed to create
      // session" even though the session had already been created and
      // activated. New shape: separate try blocks per concern.
      const apiClient = await import("@/api/client");
      const { usePreferencesStore } = await import("@/stores/preferencesStore");
      // Unloaded prefs, bootstrap rejects with a network error.
      resetStore(usePreferencesStore);
      (apiClient.fetchUserComposerPreferences as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("Network error"),
      );
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        id: "sess-prefs-fail",
        title: "untitled",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      });

      await useSessionStore.getState().createSession();

      const state = useSessionStore.getState();
      // Session was created and is the active session — NOT masked.
      expect(state.activeSessionId).toBe("sess-prefs-fail");
      expect(state.sessions[0]?.id).toBe("sess-prefs-fail");
      // The error message names the *secondary* failure, not the false
      // "Failed to create session" attribution.
      expect(state.error).toMatch(/couldn't apply your default mode/i);
      expect(state.error).not.toMatch(/failed to create session/i);
      // No guided entry attempted because resolveDefaultMode threw before
      // returning a mode value.
      expect(state.guidedSession).toBeNull();
    });

    it("session-create failure still surfaces 'Failed to create session' (no regression)", async () => {
      const apiClient = await import("@/api/client");
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("500"),
      );

      await useSessionStore.getState().createSession();

      const state = useSessionStore.getState();
      expect(state.error).toMatch(/failed to create session/i);
      // No session was added; the early-return prevented the activation
      // set() from running.
      expect(state.activeSessionId).toBeNull();
    });

    it("bootstrap race: createSession before bootstrap resolves still enters guided when prefs resolve to guided", async () => {
      const apiClient = await import("@/api/client");
      const { usePreferencesStore } = await import("@/stores/preferencesStore");
      // Start with unloaded prefs — resolveDefaultMode() must await bootstrap.
      resetStore(usePreferencesStore);
      (apiClient.fetchUserComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        default_mode: "guided",
        banner_dismissed_at: null,
        tutorial_completed_at: null,
        updated_at: "2026-05-15T00:00:00Z",
      });
      (apiClient.createSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        id: "sess-3",
        title: "untitled",
        created_at: "2026-05-14T00:00:00Z",
        updated_at: "2026-05-14T00:00:00Z",
      });
      const enterGuided = vi
        .spyOn(useSessionStore.getState(), "enterGuided")
        .mockResolvedValue();

      await useSessionStore.getState().createSession();

      expect(apiClient.fetchUserComposerPreferences).toHaveBeenCalled();
      expect(enterGuided).toHaveBeenCalledTimes(1);
      expect(usePreferencesStore.getState().loaded).toBe(true);
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

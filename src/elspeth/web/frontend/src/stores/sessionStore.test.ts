import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { ChatMessage, ComposerProgressSnapshot } from "@/types/api";

// Mock the API client — store tests verify state logic, not HTTP calls
vi.mock("@/api/client", () => ({
  fetchSessions: vi.fn(),
  createSession: vi.fn(),
  fetchMessages: vi.fn(),
  fetchCompositionState: vi.fn(),
  fetchComposerProgress: vi.fn(),
  sendMessage: vi.fn(),
  recompose: vi.fn(),
  forkFromMessage: vi.fn(),
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn(),
  archiveSession: vi.fn(),
}));

// Mock the execution store dependency
vi.mock("./executionStore", () => ({
  useExecutionStore: {
    getState: () => ({ clearValidation: vi.fn() }),
  },
}));

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

// src/stores/sessionStore.guided.test.ts
//
// Tests for guided-mode state fields and actions on useSessionStore.
// TDD pass: these import types/actions that don't exist yet. First run
// expected to fail on missing fields/actions.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { GuidedSession, TurnPayload, TerminalState, GetGuidedResponse, GuidedRespondResponse, GuidedChatResponse } from "@/types/guided";

// Mock the API client — store tests verify state logic, not HTTP calls.
// Must include all exports used by sessionStore (not just guided ones).
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
  getGuided: vi.fn(),
  respondGuided: vi.fn(),
  reenterGuided: vi.fn(),
  chatGuided: vi.fn(),
}));

// Mock the execution store dependency (selectSession calls clearValidation)
vi.mock("./executionStore", () => ({
  useExecutionStore: {
    getState: () => ({ clearValidation: vi.fn() }),
  },
}));

// Mock blobStore — selectSession calls useBlobStore.getState().loadBlobs() fire-and-forget.
// Without this, the real blobStore makes HTTP calls against jsdom.
vi.mock("./blobStore", () => ({
  useBlobStore: {
    getState: () => ({ loadBlobs: vi.fn() }),
  },
}));

// ── Sample fixtures ───────────────────────────────────────────────────────────

const sampleGuidedSession: GuidedSession = {
  step: "step_1_source",
  history: [],
  terminal: null,
  chat_history: [],
  chat_turn_seq: 0,
};

const sampleNextTurn: TurnPayload = {
  type: "single_select",
  step_index: 0,
  payload: { options: ["csv", "jsonl"] },
};

const sampleTerminal: TerminalState = {
  kind: "completed",
  reason: null,
  pipeline_yaml: "source:\n  plugin: csv\n",
};

const sampleCompositionState = {
  id: "state-1",
  version: 1,
  nodes: [],
  edges: [],
  source: null,
  outputs: [],
  metadata: { name: null, description: null },
};

const sampleGetGuidedResponse: GetGuidedResponse = {
  guided_session: sampleGuidedSession,
  next_turn: sampleNextTurn,
  terminal: null,
  composition_state: sampleCompositionState,
};

const sampleRespondResponse: GuidedRespondResponse = {
  guided_session: { ...sampleGuidedSession, step: "step_2_sink" },
  next_turn: { type: "single_select", step_index: 1, payload: {} },
  terminal: null,
  composition_state: { ...sampleCompositionState, version: 2 },
};

const sampleChatResponse: GuidedChatResponse = {
  assistant_message: "Try inspecting the CSV header row.",
  guided_session: {
    ...sampleGuidedSession,
    chat_history: [
      {
        role: "user",
        content: "What columns are available?",
        seq: 0,
        step: "step_1_source",
        ts_iso: "2026-05-13T00:00:00+00:00",
      },
      {
        role: "assistant",
        content: "Try inspecting the CSV header row.",
        seq: 1,
        step: "step_1_source",
        ts_iso: "2026-05-13T00:00:00+00:00",
      },
    ],
    chat_turn_seq: 2,
  },
  next_turn: null,
  terminal: null,
  composition_state: null,
};

const sampleSourceResolvingChatResponse: GuidedChatResponse = {
  assistant_message: "I set this up as a CSV source.",
  guided_session: { ...sampleGuidedSession, step: "step_2_sink" },
  next_turn: { type: "single_select", step_index: 1, payload: {} },
  terminal: null,
  composition_state: {
    ...sampleCompositionState,
    version: 2,
    source: {
      plugin: "csv",
      options: {
        path: "/tmp/teal_colours.csv",
        schema: { mode: "observed" },
      },
    },
  },
};

const sampleExitedGuidedSession: GuidedSession = {
  ...sampleGuidedSession,
  terminal: {
    kind: "exited_to_freeform",
    reason: "user_pressed_exit",
    pipeline_yaml: null,
  },
};

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("sessionStore — guided-mode fields and actions", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    resetStore(useSessionStore);
  });

  // ── Test 1: Initial state ─────────────────────────────────────────────────

  it("initial state: guidedSession, guidedNextTurn, guidedTerminal all null", () => {
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
  });

  // ── Test 2: startGuided happy path ────────────────────────────────────────

  it("startGuided: populates all 4 wire fields atomically on success", async () => {
    const { getGuided } = await import("@/api/client");
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    // Pre-seed activeSessionId to match the requested session.  The stale-fetch
    // guard (Codex #3) requires activeSessionId === requestedSessionId after the
    // await; without this pre-seed the guard would drop the response and the
    // assertions below would fail.  Real callers (createSession, selectSession,
    // forkFromMessage) always set activeSessionId synchronously before firing
    // startGuided, so this invariant holds in production.
    useSessionStore.setState({ activeSessionId: "sess-1" });

    await useSessionStore.getState().startGuided("sess-1");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleGetGuidedResponse.terminal);
    expect(state.compositionState).toEqual(
      sampleGetGuidedResponse.composition_state,
    );
  });

  // ── Test 3: startGuided failure ───────────────────────────────────────────

  it("startGuided: sets error string and leaves guided state alone on failure", async () => {
    const { getGuided } = await import("@/api/client");
    (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new Error("network error"),
    );

    // Pre-seed: if previously loaded guided state exists it should be
    // preserved on error (same convention as selectSession lines 207-209:
    // set error, don't clobber fields that already loaded).
    useSessionStore.setState({ guidedSession: sampleGuidedSession });

    await useSessionStore.getState().startGuided("sess-1");

    const state = useSessionStore.getState();
    // Error must be set to a non-empty string
    expect(state.error).toBeTruthy();
    expect(typeof state.error).toBe("string");
    // Pre-seeded guided state is preserved (error path doesn't clear it)
    expect(state.guidedSession).toEqual(sampleGuidedSession);
  });

  // ── Test 4: respondGuided happy path ─────────────────────────────────────

  it("respondGuided: atomically replaces all 4 wire fields on success", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );

    // Pre-seed active session
    useSessionStore.setState({ activeSessionId: "sess-1" });

    await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleRespondResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleRespondResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleRespondResponse.terminal);
    expect(state.compositionState).toEqual(
      sampleRespondResponse.composition_state,
    );
  });

  // ── Test 5: respondGuided invariant violation ─────────────────────────────

  it("respondGuided: throws when activeSessionId is null (offensive guard)", async () => {
    // activeSessionId is null from resetStore
    await expect(
      useSessionStore.getState().respondGuided({
        chosen: null,
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      }),
    ).rejects.toThrow("respondGuided called without active session");
  });

  // ── Test 6: exitToFreeform ────────────────────────────────────────────────

  it("exitToFreeform: delegates to respondGuided with control_signal=exit_to_freeform", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );

    useSessionStore.setState({ activeSessionId: "sess-1" });
    await useSessionStore.getState().exitToFreeform();

    expect(respondGuided).toHaveBeenCalledWith(
      "sess-1",
      expect.objectContaining({ control_signal: "exit_to_freeform" }),
    );
  });

  it("reenterGuided: calls backend and atomically restores active guided fields", async () => {
    const { reenterGuided } = await import("@/api/client");
    (reenterGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({
      activeSessionId: "sess-1",
      guidedSession: sampleExitedGuidedSession,
      guidedNextTurn: null,
      guidedTerminal: sampleExitedGuidedSession.terminal,
      compositionState: sampleCompositionState,
    });

    await useSessionStore.getState().reenterGuided();

    expect(reenterGuided).toHaveBeenCalledWith("sess-1");
    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleGetGuidedResponse.terminal);
    expect(state.compositionState).toEqual(sampleGetGuidedResponse.composition_state);
  });

  it("reenterGuided: throws when activeSessionId is null", async () => {
    await expect(useSessionStore.getState().reenterGuided()).rejects.toThrow(
      "reenterGuided called without active session",
    );
  });

  // ── Test 7: session switch clears guided state (leak regression) ──────────

  it("selectSession: clears guidedSession, guidedNextTurn, guidedTerminal on session switch", async () => {
    const { fetchMessages, fetchCompositionState } = await import("@/api/client");
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

    // Pre-seed non-null guided state from a previous session
    useSessionStore.setState({
      activeSessionId: "sess-1",
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: sampleTerminal,
    });

    // Switch to a different session — guided state must be cleared immediately
    // (the synchronous set() at the top of selectSession, before Promise.all)
    await useSessionStore.getState().selectSession("sess-2");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
  });

  // ── Test 8: reset() clears guided state ──────────────────────────────────

  it("reset: clears guidedSession, guidedNextTurn, guidedTerminal", () => {
    useSessionStore.setState({
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: sampleTerminal,
    });

    useSessionStore.getState().reset();

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
  });

  // ── Test 9: startGuided stale-fetch guard (Codex #3) ─────────────────────
  //
  // If the user switches to a different session while getGuided is in flight
  // for sess-A, the resolved response must be dropped and the new session's
  // guided state must remain null.

  it("startGuided: drops response when active session changes before resolution", async () => {
    const { getGuided } = await import("@/api/client");

    // Controllable promise — we resolve it manually after simulating a session switch.
    let resolveGuided!: (v: GetGuidedResponse) => void;
    (getGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GetGuidedResponse>((resolve) => {
        resolveGuided = resolve;
      }),
    );

    // The caller always sets activeSessionId before firing startGuided.
    useSessionStore.setState({ activeSessionId: "sess-A" });
    const startPromise = useSessionStore.getState().startGuided("sess-A");

    // Simulate a session switch happening while the request is in flight.
    useSessionStore.setState({ activeSessionId: "sess-B" });

    // Now let the request resolve with sess-A's data.
    resolveGuided(sampleGetGuidedResponse);
    await startPromise;

    // The stale response must have been dropped — sess-B's guided state
    // must remain null.
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.compositionState).toBeNull();
  });

  // ── Test 10: respondGuided stale-fetch guard (Codex #4) ──────────────────
  //
  // If the user switches sessions while respondGuided's API call is in flight,
  // the resolved response must be dropped and the new session's state unchanged.

  it("respondGuided: drops response when active session changes before resolution", async () => {
    const { respondGuided } = await import("@/api/client");

    // Controllable promise — we resolve it manually after simulating a session switch.
    let resolveRespond!: (v: GuidedRespondResponse) => void;
    (respondGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GuidedRespondResponse>((resolve) => {
        resolveRespond = resolve;
      }),
    );

    // Pre-seed: sess-A has existing guided state before the respond call.
    useSessionStore.setState({
      activeSessionId: "sess-A",
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    const respondPromise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    // Simulate a session switch before the response arrives.
    useSessionStore.setState({
      activeSessionId: "sess-B",
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
    });

    // Let sess-A's response arrive.
    resolveRespond(sampleRespondResponse);
    await respondPromise;

    // The stale response must have been dropped — sess-B's guided state
    // must remain null (not overwritten by sess-A's respond result).
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
  });

  // ── Test 11: forkFromMessage clears guided state and reloads (Codex #6) ──
  //
  // After a successful fork:
  // 1. The fork's guided state must start null (synchronous clear).
  // 2. startGuided must be called for the new fork session.

  it("forkFromMessage: clears guided state and fires startGuided for fork", async () => {
    const { forkFromMessage, getGuided } = await import("@/api/client");

    const forkResult = {
      session: { id: "sess-fork", title: "Fork", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" },
      messages: [],
      composition_state: null,
    };

    (forkFromMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce(forkResult);

    // Controllable promise for startGuided so we can verify it was called
    // without letting it resolve (keeps test deterministic).
    let resolveGuided!: (v: GetGuidedResponse) => void;
    (getGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GetGuidedResponse>((resolve) => {
        resolveGuided = resolve;
      }),
    );

    // Pre-seed: parent session has guided state that must NOT bleed into fork.
    useSessionStore.setState({
      activeSessionId: "sess-parent",
      sessions: [{ id: "sess-parent", title: "Parent", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }],
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    // Start the fork — this triggers the async forkFromMessage path.
    const forkPromise = useSessionStore.getState().forkFromMessage("msg-1", "new content");
    await forkPromise;

    // Guided state must be null immediately after the set() (synchronous clear).
    const state = useSessionStore.getState();
    expect(state.activeSessionId).toBe("sess-fork");
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();

    // startGuided must have been invoked for the fork session.
    expect(getGuided).toHaveBeenCalledWith("sess-fork");

    // Let startGuided resolve — must write into the fork's guided state
    // since activeSessionId is still "sess-fork".
    resolveGuided(sampleGetGuidedResponse);
    // Give microtask queue a tick to drain.
    await Promise.resolve();

    const stateAfterReload = useSessionStore.getState();
    expect(stateAfterReload.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
  });

  describe("chatGuided", () => {
    it("throws when activeSessionId is null", async () => {
      await expect(
        useSessionStore.getState().chatGuided("What columns are available?"),
      ).rejects.toThrow("chatGuided called without active session");
    });

    it("throws when guidedSession is null", async () => {
      useSessionStore.setState({ activeSessionId: "sess-1" });

      await expect(
        useSessionStore.getState().chatGuided("What columns are available?"),
      ).rejects.toThrow("chatGuided called before guidedSession loaded");
    });

    it("sets pending while in flight and clears it on success", async () => {
      const { chatGuided } = await import("@/api/client");
      let resolveChat!: (v: GuidedChatResponse) => void;
      (chatGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        new Promise<GuidedChatResponse>((resolve) => {
          resolveChat = resolve;
        }),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const chatPromise = useSessionStore.getState().chatGuided("What columns are available?");

      expect(useSessionStore.getState().guidedChatPending).toBe(true);
      expect(chatGuided).toHaveBeenCalledWith("sess-1", {
        message: "What columns are available?",
        step_index: "step_1_source",
      });

      resolveChat(sampleChatResponse);
      await chatPromise;

      const state = useSessionStore.getState();
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedSession).toEqual(sampleChatResponse.guided_session);
    });

    it("applies step-advancing chat response fields when source is resolved", async () => {
      const { chatGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleSourceResolvingChatResponse,
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
        compositionState: sampleCompositionState,
      });

      await useSessionStore.getState().chatGuided("make ten teal colour rows");

      const state = useSessionStore.getState();
      expect(state.guidedSession).toEqual(
        sampleSourceResolvingChatResponse.guided_session,
      );
      expect(state.guidedNextTurn).toEqual(
        sampleSourceResolvingChatResponse.next_turn,
      );
      expect(state.guidedTerminal).toEqual(
        sampleSourceResolvingChatResponse.terminal,
      );
      expect(state.compositionState).toEqual(
        sampleSourceResolvingChatResponse.composition_state,
      );
      expect(state.guidedChatPending).toBe(false);
    });

    it("drops response when active session changes before resolution", async () => {
      const { chatGuided } = await import("@/api/client");
      let resolveChat!: (v: GuidedChatResponse) => void;
      (chatGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        new Promise<GuidedChatResponse>((resolve) => {
          resolveChat = resolve;
        }),
      );
      useSessionStore.setState({
        activeSessionId: "sess-A",
        guidedSession: sampleGuidedSession,
      });

      const chatPromise = useSessionStore.getState().chatGuided("What columns are available?");

      useSessionStore.setState({
        activeSessionId: "sess-B",
        guidedSession: null,
        guidedChatPending: false,
      });
      resolveChat(sampleChatResponse);
      await chatPromise;

      const state = useSessionStore.getState();
      expect(state.guidedSession).toBeNull();
      expect(state.guidedChatPending).toBe(false);
    });

    it("sets error and clears pending on request failure", async () => {
      const { chatGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("network failed"),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      const state = useSessionStore.getState();
      expect(state.error).toBe("Failed to send chat message. Please try again.");
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedSession).toEqual(sampleGuidedSession);
    });
  });
});

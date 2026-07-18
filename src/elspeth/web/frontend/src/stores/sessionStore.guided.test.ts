// src/stores/sessionStore.guided.test.ts
//
// Tests for guided-mode state fields and actions on useSessionStore.
// TDD pass: these import types/actions that don't exist yet. First run
// expected to fail on missing fields/actions.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import type { GuidedSession, TurnPayload, TerminalState, GetGuidedResponse, GuidedRespondRequest, GuidedRespondResponse, GuidedChatResponse } from "@/types/guided";

// Mock the API client — store tests verify state logic, not HTTP calls.
// Must include all exports used by sessionStore (not just guided ones).
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
  forkFromMessage: vi.fn(),
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn(),
  archiveSession: vi.fn(),
  getGuided: vi.fn(),
  respondGuided: vi.fn(),
  reenterGuided: vi.fn(),
  convertToGuided: vi.fn(),
  chatGuided: vi.fn(),
  // Phase 5b — selectSession fires a fire-and-forget refreshAll on the
  // interpretationEventsStore.  Mocked to resolve empty so jsdom does not
  // attempt the real HTTP call; targeted assertions live in
  // interpretationEventsStore.test.ts.
  listInterpretationEvents: vi.fn().mockResolvedValue([]),
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
  profile: null,
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
  sources: {},
  outputs: [],
  metadata: { name: null, description: null },
};

const RETRY_SESSION_ID = "00000000-0000-4000-8000-000000000101";
const RETRY_SESSION_B = "00000000-0000-4000-8000-000000000102";

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
    sources: {
      source: {
        plugin: "csv",
        options: {
          path: "/tmp/teal_colours.csv",
          schema: { mode: "observed" },
        },
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
  beforeEach(async () => {
    vi.resetAllMocks();
    window.sessionStorage.clear();
    resetStore(useSessionStore);
    // Phase 5b — reseed the listInterpretationEvents mock that
    // vi.resetAllMocks() cleared, so selectSession's fire-and-forget
    // refreshAll path does not produce unhandled rejections in tests
    // that exercise it indirectly.
    const apiMod = await import("@/api/client");
    (apiMod.listInterpretationEvents as ReturnType<typeof vi.fn>).mockResolvedValue([]);
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
    useSessionStore.setState({
      activeSessionId: "sess-1",
      guidedSession: sampleGuidedSession,
    });

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

  // ── Test 4b: respondGuided refreshes the interpretation-event store (B1/D12) ─
  //
  // P3.2 (backend) surfaces pending interpretation cards into the store on the
  // commit path; the frontend only sees them after respondGuided refreshes the
  // interpretationEventsStore.  The refresh must complete (be awaited) before
  // guidedResponsePending clears, otherwise the submit button can briefly
  // re-enable before the card-block arrives.

  it("awaits interpretation-event refresh after a successful guided respond", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );
    const refreshAll = vi.fn(async () => {});
    vi.spyOn(useInterpretationEventsStore, "getState").mockReturnValue({
      ...useInterpretationEventsStore.getState(),
      refreshAll,
    });
    // Pre-seed the active session (same as the happy-path test above).
    useSessionStore.setState({ activeSessionId: "sess-1" });

    await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    expect(refreshAll).toHaveBeenCalledWith("sess-1");
    expect(useSessionStore.getState().guidedResponsePending).toBe(false);
  });

  it("keeps submit disabled while the pending-card refresh is deferred", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );
    let releaseRefresh!: () => void;
    const refreshAll = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          releaseRefresh = resolve;
        }),
    );
    vi.spyOn(useInterpretationEventsStore, "getState").mockReturnValue({
      ...useInterpretationEventsStore.getState(),
      refreshAll,
    });
    useSessionStore.setState({ activeSessionId: "sess-1" });

    const promise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    await vi.waitFor(() => expect(refreshAll).toHaveBeenCalledWith("sess-1"));
    expect(useSessionStore.getState().guidedResponsePending).toBe(true);
    releaseRefresh();
    await promise;
    expect(useSessionStore.getState().guidedResponsePending).toBe(false);
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
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleExitedGuidedSession,
      guidedNextTurn: null,
      guidedTerminal: sampleExitedGuidedSession.terminal,
      compositionState: sampleCompositionState,
    });

    await useSessionStore.getState().reenterGuided();

    expect(reenterGuided).toHaveBeenCalledWith(RETRY_SESSION_ID, expect.any(String));
    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleGetGuidedResponse.terminal);
    expect(state.compositionState).toEqual(sampleGetGuidedResponse.composition_state);
  });

  it("reenterGuided: reuses the operation id after an ambiguous 5xx and clears it after success", async () => {
    const { reenterGuided } = await import("@/api/client");
    const reenterMock = reenterGuided as ReturnType<typeof vi.fn>;
    reenterMock
      .mockRejectedValueOnce({ status: 503, detail: "upstream unavailable" })
      .mockResolvedValueOnce(sampleGetGuidedResponse)
      .mockResolvedValueOnce(sampleGetGuidedResponse);

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleExitedGuidedSession,
      guidedTerminal: sampleExitedGuidedSession.terminal,
    });

    await useSessionStore.getState().reenterGuided();
    await useSessionStore.getState().reenterGuided();
    await useSessionStore.getState().reenterGuided();

    const firstOperationId = reenterMock.mock.calls[0]?.[1];
    const retryOperationId = reenterMock.mock.calls[1]?.[1];
    const nextActionOperationId = reenterMock.mock.calls[2]?.[1];
    expect(firstOperationId).toEqual(expect.any(String));
    expect(retryOperationId).toBe(firstOperationId);
    expect(nextActionOperationId).not.toBe(firstOperationId);
  });

  it("reenterGuided: throws when activeSessionId is null", async () => {
    await expect(useSessionStore.getState().reenterGuided()).rejects.toThrow(
      "reenterGuided called without active session",
    );
  });

  it("reenterGuided: drops failure when active session changes before rejection", async () => {
    const { reenterGuided } = await import("@/api/client");

    let rejectReenter!: (reason?: unknown) => void;
    (reenterGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GetGuidedResponse>((_resolve, reject) => {
        rejectReenter = reject;
      }),
    );

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleExitedGuidedSession,
      guidedNextTurn: null,
      guidedTerminal: sampleExitedGuidedSession.terminal,
      error: null,
    });

    const reenterPromise = useSessionStore.getState().reenterGuided();

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_B,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      error: null,
    });

    rejectReenter(new Error("network down for stale session"));
    await reenterPromise;

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.error).toBeNull();
  });

  // ── enterGuided unified entry point (default-freeform switch button) ──────
  //
  // The "Switch to guided" affordance in the freeform ChatPanel header
  // binds to enterGuided().  It branches on the current guidedSession
  // terminal so callers always have a single action regardless of whether
  // the session is fresh or has previously exited.

  it("enterGuided: calls convertToGuided when guidedSession is null (fresh / worked freeform)", async () => {
    // Routing changed with elspeth-e2c3dba6b5: the non-terminal branch now goes
    // through convertToGuided (POST /guided/convert) instead of startGuided
    // (GET /guided). GET 400s for a worked freeform session; convert is the
    // idempotent superset that also does the fresh-wizard conversion. The GET
    // path must NOT be taken.
    const { convertToGuided, getGuided } = await import("@/api/client");
    (convertToGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({
      activeSessionId: "sess-1",
      guidedSession: null,
    });

    await useSessionStore.getState().enterGuided();

    expect(convertToGuided).toHaveBeenCalledWith("sess-1");
    expect(getGuided).not.toHaveBeenCalled();
    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
  });

  it("convertToGuided: populates all 4 wire fields atomically on success", async () => {
    const { convertToGuided } = await import("@/api/client");
    (convertToGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({ activeSessionId: "sess-1" });
    await useSessionStore.getState().convertToGuided("sess-1");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleGetGuidedResponse.terminal);
    expect(state.compositionState).toEqual(
      sampleGetGuidedResponse.composition_state,
    );
    expect(state.error).toBeNull();
  });

  it("convertToGuided: surfaces the backend's typed detail on failure", async () => {
    const { convertToGuided } = await import("@/api/client");
    (convertToGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
      status: 400,
      detail: "You do not own this session.",
    });

    useSessionStore.setState({ activeSessionId: "sess-1" });
    await useSessionStore.getState().convertToGuided("sess-1");

    expect(useSessionStore.getState().error).toBe(
      "You do not own this session.",
    );
  });

  it("startGuided: surfaces the backend's typed detail instead of the generic banner", async () => {
    // elspeth-e2c3dba6b5 secondary fix: startGuided's catch used to hardcode
    // "Failed to load guided session. Please try again." and discard
    // ApiError.detail — asymmetric with respondGuided/chatGuided. A typed 400
    // that names a recoverable mode-state boundary must reach the user.
    const { getGuided } = await import("@/api/client");
    (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
      status: 400,
      detail: "Session is not in guided mode. Use /api/sessions/{id}/messages.",
    });

    useSessionStore.setState({ activeSessionId: "sess-1" });
    await useSessionStore.getState().startGuided("sess-1");

    expect(useSessionStore.getState().error).toBe(
      "Session is not in guided mode. Use /api/sessions/{id}/messages.",
    );
  });

  // ── revertToVersion re-derives the guided surface from the reverted version ──
  //
  // Reverting can cross the guided/freeform boundary — most visibly the
  // recoverability flow behind convertToGuided's "fresh wizard + consent"
  // (elspeth-e2c3dba6b5): convert a worked freeform session to guided, then
  // revert to the prior freeform version to get the pipeline back. Before this
  // fix revertToVersion only patched compositionState, so the stale cached
  // guidedSession kept the guided wizard on screen over restored freeform
  // state — the "can be restored" promise was broken at the surface level.

  it("revertToVersion: reverting to a freeform version clears the stale guided surface", async () => {
    const { revertToVersion, getGuided } = await import("@/api/client");
    // Reverted version is freeform: GET /guided 400s (no guided_session).
    (revertToVersion as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleCompositionState,
      version: 3,
    });
    (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({ status: 400 });

    // Pre-seed a live guided surface (as if we just converted to guided).
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    await useSessionStore.getState().revertToVersion("state-freeform");

    expect(revertToVersion).toHaveBeenCalledWith(
      RETRY_SESSION_ID,
      "state-freeform",
      expect.any(String),
    );

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.compositionState?.version).toBe(3);
  });

  it("revertToVersion: reuses the operation id after an ambiguous network failure and clears it after success", async () => {
    const { revertToVersion, getGuided } = await import("@/api/client");
    const revertMock = revertToVersion as ReturnType<typeof vi.fn>;
    revertMock
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce({ ...sampleCompositionState, version: 2 })
      .mockResolvedValueOnce({ ...sampleCompositionState, version: 3 });
    (getGuided as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce({ status: 400 })
      .mockRejectedValueOnce({ status: 400 });
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().revertToVersion("state-old");
    await useSessionStore.getState().revertToVersion("state-old");
    await useSessionStore.getState().revertToVersion("state-old");

    const firstOperationId = revertMock.mock.calls[0]?.[2];
    const retryOperationId = revertMock.mock.calls[1]?.[2];
    const nextActionOperationId = revertMock.mock.calls[2]?.[2];
    expect(firstOperationId).toEqual(expect.any(String));
    expect(retryOperationId).toBe(firstOperationId);
    expect(nextActionOperationId).not.toBe(firstOperationId);
  });

  it("revertToVersion: reverting to a guided version restores the guided surface", async () => {
    const { revertToVersion, getGuided } = await import("@/api/client");
    (revertToVersion as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleCompositionState,
      version: 3,
    });
    // Reverted version is guided: GET /guided 200 with a real composition_state.
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
    });

    await useSessionStore.getState().revertToVersion("state-guided");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
  });

  it("revertToVersion: uses the probed guided composition state when GET /guided materializes it", async () => {
    const { revertToVersion, getGuided } = await import("@/api/client");
    const revertedCompositionState = {
      ...sampleCompositionState,
      id: "state-reverted",
      version: 3,
    };
    const probedCompositionState = {
      ...sampleCompositionState,
      id: "state-probed",
      version: 4,
    };

    (revertToVersion as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      revertedCompositionState,
    );
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleGetGuidedResponse,
      composition_state: probedCompositionState,
    });

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
    });

    await useSessionStore.getState().revertToVersion("state-guided");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.compositionState).toEqual(probedCompositionState);
  });

  it("enterGuided: calls reenterGuided when terminal.kind === 'exited_to_freeform'", async () => {
    const { reenterGuided, getGuided } = await import("@/api/client");
    (reenterGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleExitedGuidedSession,
      guidedTerminal: sampleExitedGuidedSession.terminal,
    });

    await useSessionStore.getState().enterGuided();

    expect(reenterGuided).toHaveBeenCalledWith(RETRY_SESSION_ID, expect.any(String));
    // startGuided's underlying GET must NOT be called on the reenter path.
    expect(getGuided).not.toHaveBeenCalled();
  });

  // C-4b (composer first-principles review 2026-07-04, elspeth-04d2757bf1):
  // the live bug was that a user_pressed_exit terminal session's "Switch to
  // guided" silently no-op'd (client GET-observed the same terminal and
  // stayed in freeform with zero feedback) instead of actually re-entering.
  // sampleExitedGuidedSession's terminal.reason IS "user_pressed_exit" — the
  // one reason POST /guided/reenter honours (routes/composer/guided.py's
  // post_guided_reenter guard rejects solver_exhausted/protocol_violation
  // with a 409). This test pins the full round-trip: enterGuided() reaches
  // reenterGuided(), and the resulting state is a RESUMED, non-terminal
  // guided session — not just "the right API got called".
  it("enterGuided: a user_pressed_exit terminal session actually resumes guided (C-4b — not a silent no-op)", async () => {
    const { reenterGuided } = await import("@/api/client");
    (reenterGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleExitedGuidedSession,
      guidedTerminal: sampleExitedGuidedSession.terminal,
    });
    expect(useSessionStore.getState().guidedSession?.terminal).not.toBeNull();

    await useSessionStore.getState().enterGuided();

    const state = useSessionStore.getState();
    // Resumed: guided is live again (non-terminal, a next turn is present) —
    // the opposite of the old bug, where the session stayed terminal and
    // freeform kept rendering with no feedback at all.
    expect(state.guidedTerminal).toBeNull();
    expect(state.guidedSession?.terminal).toBeNull();
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.error).toBeNull();
  });

  it("enterGuided: throws when activeSessionId is null", async () => {
    await expect(useSessionStore.getState().enterGuided()).rejects.toThrow(
      "enterGuided called without active session",
    );
  });

  // ── Test 7: session switch clears guided state (leak regression) ──────────

  it("selectSession: clears guidedSession, guidedNextTurn, guidedTerminal on session switch", async () => {
    const {
      fetchMessages,
      fetchCompositionState,
      fetchCompositionProposals,
      fetchComposerPreferences,
    } = await import("@/api/client");
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);

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
    expect(state.error).toBeNull();
  });

  // ── Test 7b: C-4a — selectSession restores a LIVE/persisted guided session ──
  //
  // fp-review 2026-07-04, elspeth-04d2757bf1: a browser reload (which
  // re-runs selectSession for the previously active session) must not
  // strand a mid-guided-build user in freeform. GET /guided returning a
  // response with a non-null composition_state confirms the session's
  // guided_session was genuinely persisted (not the lazy in-memory stub a
  // brand-new, never-touched session gets — see the next test).

  it("selectSession: restores guidedSession/guidedNextTurn/guidedTerminal from a persisted guided session (C-4a)", async () => {
    const {
      fetchMessages,
      fetchCompositionState,
      fetchCompositionProposals,
      fetchComposerPreferences,
      getGuided,
    } = await import("@/api/client");
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleCompositionState,
    );
    (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    await useSessionStore.getState().selectSession("sess-3");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleGetGuidedResponse.terminal);
  });

  it("selectSession: does NOT adopt GET /guided's lazy stub for a brand-new session (composition_state: null)", async () => {
    // get_guided's docstring: a session with no persisted CompositionState
    // yet gets a non-mutating in-memory stub GuidedSession + first turn,
    // returned with composition_state: null — that stub is not evidence the
    // session was ever really in guided mode. Adopting it here would flip
    // every brand-new, freeform-preferring session into the guided surface
    // on its very first load.
    const {
      fetchMessages,
      fetchCompositionState,
      fetchCompositionProposals,
      fetchComposerPreferences,
      getGuided,
    } = await import("@/api/client");
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      guided_session: sampleGuidedSession,
      next_turn: sampleNextTurn,
      terminal: null,
      composition_state: null,
    });

    await useSessionStore.getState().selectSession("sess-4");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
  });

  it("selectSession: tolerates GET /guided's 400 for a plain freeform session (no error surfaced)", async () => {
    const {
      fetchMessages,
      fetchCompositionState,
      fetchCompositionProposals,
      fetchComposerPreferences,
      getGuided,
    } = await import("@/api/client");
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleCompositionState,
    );
    (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
      status: 400,
      detail: "Session is not in guided mode. Use /api/sessions/{id}/messages.",
    });

    await useSessionStore.getState().selectSession("sess-5");

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.compositionState).toEqual(sampleCompositionState);
    // The expected "not guided" outcome must not read as a selectSession
    // failure — the freeform surface renders normally with no error banner.
    expect(state.error).toBeNull();
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

  it("startGuided: drops failure when active session changes before rejection", async () => {
    const { getGuided } = await import("@/api/client");

    let rejectGuided!: (reason?: unknown) => void;
    (getGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GetGuidedResponse>((_resolve, reject) => {
        rejectGuided = reject;
      }),
    );

    useSessionStore.setState({
      activeSessionId: "sess-A",
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
      error: null,
    });

    const startPromise = useSessionStore.getState().startGuided("sess-A");

    useSessionStore.setState({
      activeSessionId: "sess-B",
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      error: null,
    });

    rejectGuided(new Error("network down for stale session"));
    await startPromise;

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.error).toBeNull();
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
      guidedResponsePending: true,
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
    expect(state.guidedResponsePending).toBe(true);
  });

  it("respondGuided: drops failure when active session changes before rejection", async () => {
    const { respondGuided } = await import("@/api/client");

    let rejectRespond!: (reason?: unknown) => void;
    (respondGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise<GuidedRespondResponse>((_resolve, reject) => {
        rejectRespond = reject;
      }),
    );

    useSessionStore.setState({
      activeSessionId: "sess-A",
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
      error: null,
    });

    const respondPromise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    useSessionStore.setState({
      activeSessionId: "sess-B",
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      guidedResponsePending: true,
      error: null,
    });

    rejectRespond(new Error("network down for stale session"));
    await respondPromise;

    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.guidedResponsePending).toBe(true);
    expect(state.error).toBeNull();
  });

  // ── Test 11: forkFromMessage clears guided state (default-freeform) ──
  //
  // Default-freeform contract (post-Phase-A button-switch change):
  // - The fork's guided state must start null (synchronous clear).
  // - The fork must NOT auto-fetch GET /guided.  Forks open into the
  //   freeform surface like create/select; the user opts into guided
  //   by clicking "Switch to guided" in the header.
  //
  // Earlier revisions of this test required getGuided to be called for
  // the fork session.  That invariant was deliberately removed when the
  // operator asked for default-freeform sessions with a button-switch to
  // guided; see _initial_composition_state_with_guided_session docstring
  // and sessionStore.selectSession / .createSession / .forkFromMessage
  // comments for the new contract.

  it("forkFromMessage: clears guided state and does NOT auto-fetch guided for fork (default-freeform)", async () => {
    const { forkFromMessage, getGuided } = await import("@/api/client");

    const forkResult = {
      session: { id: "sess-fork", title: "Fork", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" },
      messages: [],
      composition_state: null,
    };

    (forkFromMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce(forkResult);

    // Pre-seed: parent session has guided state that must NOT bleed into fork.
    useSessionStore.setState({
      activeSessionId: "sess-parent",
      sessions: [{ id: "sess-parent", title: "Parent", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }],
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    const forkPromise = useSessionStore.getState().forkFromMessage("msg-1", "new content");
    await forkPromise;

    // Guided state must be null immediately after the set() (synchronous clear).
    const state = useSessionStore.getState();
    expect(state.activeSessionId).toBe("sess-fork");
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();

    // GET /guided must NOT be auto-called for the fork (default-freeform).
    expect(getGuided).not.toHaveBeenCalled();
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
      expect(chatGuided).toHaveBeenCalledWith(
        "sess-1",
        {
          message: "What columns are available?",
          step_index: "step_1_source",
        },
        undefined,
      );

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

    it("surfaces the backend detail on a structured failure", async () => {
      // A 409 step-mismatch (the wizard advanced under the user) carries an
      // actionable, egress-safe detail. Surface it verbatim rather than the
      // blanket "failed" message, which forced the user to GUESS the cause.
      const { chatGuided } = await import("@/api/client");
      const detail =
        "step_index 'step_1_source' does not match the session's current step " +
        "'step_2_sink'. Re-fetch GET /api/sessions/{id}/guided and retry.";
      (chatGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 409,
        detail,
      });
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      const state = useSessionStore.getState();
      expect(state.error).toBe(detail);
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedSession).toEqual(sampleGuidedSession);
    });

    it("user cancel: abort resets pending and surfaces the cancelled copy (elspeth-fb4464cdf0)", async () => {
      const { chatGuided } = await import("@/api/client");
      // Mirror the fetch abort contract: abort(reason) rejects the in-flight
      // fetch with the RAW reason value (here a bare string), not a
      // DOMException (elspeth-475647c47a).
      (chatGuided as ReturnType<typeof vi.fn>).mockImplementationOnce(
        (_sessionId: string, _body: unknown, signal?: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason));
          }),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const controller = new AbortController();
      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?", controller.signal);
      expect(useSessionStore.getState().guidedChatPending).toBe(true);

      controller.abort("compose_user_cancel");
      await chatPromise;

      const state = useSessionStore.getState();
      expect(state.guidedChatPending).toBe(false);
      expect(state.error).toBe(
        "Composition stopped. You can revise your request and send it again.",
      );
      // The turn is retryable: guided state is untouched.
      expect(state.guidedSession).toEqual(sampleGuidedSession);
    });

    it("resyncs durable guided state after an aborted guided chat (elspeth-b2d9e4d084)", async () => {
      // A client-side abort only rejects the local fetch — the guided turn
      // keeps running server-side until the disconnect watcher cancels it,
      // and whatever it committed before the cancel (chat_history turns,
      // step results, composition-state advances from the step-1 upload
      // commit path) is durable. Guided state is server-authoritative, so
      // the abort branch must refetch GET /guided once the session
      // quiesces, or the client renders the pre-send snapshot until the
      // next successful guided action.
      const { chatGuided, getGuided, fetchComposerProgress } = await import(
        "@/api/client"
      );
      (chatGuided as ReturnType<typeof vi.fn>).mockImplementationOnce(
        (_sessionId: string, _body: unknown, signal?: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason));
          }),
      );
      // The aborted route has fully unwound: quiescent registry.
      (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
        phase: "cancelled",
        inflight_requests: 0,
      });
      const durableGuided = {
        ...sampleGuidedSession,
        chat_turn_seq: 2,
        chat_history: [
          {
            role: "user",
            content: "What columns are available?",
            seq: 0,
            step: "step_1_source",
            ts_iso: "2026-07-11T00:00:00Z",
          },
          {
            role: "assistant",
            content: "Partial reply persisted before the cancel.",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-07-11T00:00:01Z",
          },
        ],
      };
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValue({
        guided_session: durableGuided,
        next_turn: null,
        terminal: null,
        composition_state: null,
      });
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const controller = new AbortController();
      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?", controller.signal);
      controller.abort("compose_user_cancel");
      await chatPromise;

      const state = useSessionStore.getState();
      expect(getGuided).toHaveBeenCalledWith("sess-1");
      expect(state.guidedSession).toEqual(durableGuided);
      expect(state.error).toBe(
        "Composition stopped. You can revise your request and send it again.",
      );
    });

    it("starts composer-progress polling on send and applies the loaded snapshot (elspeth-a8eeebb3aa)", async () => {
      // The REAL production seam, not composerProgress injected via setState:
      // previously chatGuided never called startComposerProgressPolling at
      // all, so composerProgress stayed null for the entire guided compose
      // and the tutorial step-2 substep indicator never advanced. This test
      // fails against the pre-fix chatGuided (fetchComposerProgress would
      // never be called) even though setState-injection tests elsewhere in
      // this suite passed throughout.
      const { chatGuided, fetchComposerProgress } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleChatResponse,
      );
      (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
        session_id: "sess-1",
        request_id: null,
        phase: "calling_model",
        headline: "I'm asking the model to choose the next safe pipeline update.",
        evidence: [],
        likely_next: null,
        reason: null,
        updated_at: "2026-07-10T00:00:00Z",
      });
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      expect(fetchComposerProgress).toHaveBeenCalledWith("sess-1");
      expect(useSessionStore.getState().composerProgress).toEqual(
        expect.objectContaining({ phase: "calling_model" }),
      );
    });

    it("finally block scopes stop+reload to the session captured before the await, not one switched to mid-flight", async () => {
      // Mirrors sendMessage/retryMessage's captured-activeSessionId finally
      // semantics: every fetchComposerProgress call this chatGuided call
      // makes must stay scoped to sess-A even though the store's
      // activeSessionId flips to sess-B before the response resolves.
      const { chatGuided, fetchComposerProgress } = await import("@/api/client");
      let resolveChat!: (v: GuidedChatResponse) => void;
      (chatGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        new Promise<GuidedChatResponse>((resolve) => {
          resolveChat = resolve;
        }),
      );
      (fetchComposerProgress as ReturnType<typeof vi.fn>).mockResolvedValue({
        session_id: "sess-A",
        request_id: null,
        phase: "idle",
        headline: "x",
        evidence: [],
        likely_next: null,
        reason: null,
        updated_at: "2026-07-10T00:00:00Z",
      });
      useSessionStore.setState({
        activeSessionId: "sess-A",
        guidedSession: sampleGuidedSession,
      });

      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?");

      useSessionStore.setState({
        activeSessionId: "sess-B",
        guidedSession: null,
        guidedChatPending: false,
      });
      resolveChat(sampleChatResponse);
      await chatPromise;

      const calledSessions = (
        fetchComposerProgress as ReturnType<typeof vi.fn>
      ).mock.calls.map((call) => call[0]);
      expect(calledSessions.length).toBeGreaterThan(0);
      expect(calledSessions.every((id) => id === "sess-A")).toBe(true);
    });

    it("discards a stale terminal snapshot at poll start; a fresh terminal snapshot after compose settles still surfaces (elspeth-a8eeebb3aa review follow-up)", async () => {
      // Regression for a review-caught race: startComposerProgressPolling's
      // immediate poll can win the race against the backend's own "starting"
      // publish and return the PREVIOUS turn's terminal snapshot still
      // sitting in the registry (e.g. a step-1 send left phase="complete";
      // the step-2 send's immediate GET can beat the POST's guard clauses).
      // Surfacing that stale snapshot flashed the tutorial substep
      // indicator's LAST step as current before dropping back to the first —
      // the exact backward-jump class the calling_model/using_tools remap
      // was meant to prevent, just via a different path.
      const { chatGuided, fetchComposerProgress } = await import(
        "@/api/client"
      );
      const fetchMock = fetchComposerProgress as ReturnType<typeof vi.fn>;

      let resolveChat!: (v: GuidedChatResponse) => void;
      (chatGuided as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        new Promise<GuidedChatResponse>((resolve) => {
          resolveChat = resolve;
        }),
      );

      const staleComplete = {
        session_id: "sess-1",
        request_id: null,
        phase: "complete" as const,
        headline: "stale — from the PREVIOUS turn",
        evidence: [],
        likely_next: null,
        reason: "composer_complete" as const,
        updated_at: "2026-07-09T00:00:00Z",
      };
      const freshComplete = {
        ...staleComplete,
        headline: "fresh — this turn's own completion",
        updated_at: "2026-07-10T00:00:02Z",
      };

      // First call = startComposerProgressPolling's immediate poll (must be
      // discarded). Manually controlled so the test can deterministically
      // wait for its discard/set decision to fully settle before asserting.
      let resolveFirstFetch!: (v: typeof staleComplete) => void;
      fetchMock.mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirstFetch = resolve;
          }),
      );
      // Second call = chatGuided's finally block's explicit final load,
      // which is NOT part of the poll session's discard filter.
      fetchMock.mockResolvedValueOnce(freshComplete);

      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?");

      resolveFirstFetch(staleComplete);
      // Flush every pending microtask (the mock promise's resolution AND
      // loadComposerProgress's post-await discard decision) via a macrotask
      // tick, so the assertion below reflects the settled state rather than
      // a coincidental pass before the continuation has run at all.
      await new Promise((resolve) => setTimeout(resolve, 0));

      expect(useSessionStore.getState().composerProgress).toBeNull();

      resolveChat(sampleChatResponse);
      await chatPromise;

      expect(useSessionStore.getState().composerProgress).toEqual(
        freshComplete,
      );
    });

    it("client timeout: abort surfaces the timeout copy", async () => {
      const { chatGuided } = await import("@/api/client");
      // Raw abort-reason string, matching real fetch semantics
      // (elspeth-475647c47a).
      (chatGuided as ReturnType<typeof vi.fn>).mockImplementationOnce(
        (_sessionId: string, _body: unknown, signal?: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason));
          }),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const controller = new AbortController();
      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?", controller.signal);

      controller.abort("compose_timeout");
      await chatPromise;

      const state = useSessionStore.getState();
      expect(state.guidedChatPending).toBe(false);
      expect(state.error).toMatch(/took too long/i);
    });
  });

  describe("respondGuided rejection surfacing (elspeth-3b35abf148 variant 3)", () => {
    it("surfaces a structured wire_confirm_rejected 409 as error + errorDetails", async () => {
      const { respondGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 409,
        detail:
          "The pipeline can't be confirmed yet - validation found 2 blocking issue(s) at the wiring step. Fix the issues below, then confirm again.",
        error_type: "wire_confirm_rejected",
        validation_errors: [
          { component: "pipeline", message: "No sinks configured.", severity: "high" },
          { component: "node:rater", message: "Missing input.", severity: "high" },
        ],
      });
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["confirm"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });

      const state = useSessionStore.getState();
      expect(state.error).toMatch(/can't be confirmed yet/);
      expect(state.errorDetails).toEqual([
        "pipeline: No sinks configured.",
        "node:rater: Missing input.",
      ]);
      expect(state.guidedResponsePending).toBe(false);
    });

    it("falls back to the generic message when the failure carries no detail", async () => {
      const { respondGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("network down"),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["confirm"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });

      const state = useSessionStore.getState();
      expect(state.error).toBe(
        "Failed to submit guided response. Please try again.",
      );
      expect(state.errorDetails).toBeNull();
    });
  });

  // ── C-3: turn_not_emitted self-heal (composer first-principles review
  // 2026-07-04, elspeth-948eb9c0b8) ────────────────────────────────────────
  describe("respondGuided turn_not_emitted self-heal", () => {
    const turnNotEmittedError = {
      status: 400,
      error_type: "turn_not_emitted",
      detail:
        "Your session's step is out of sync with the server. Refreshing the session will resync this automatically.",
    };

    it("refetches guided state and surfaces a calm notice — never the raw rejection detail", async () => {
      const { respondGuided, getGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        turnNotEmittedError,
      );
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleGetGuidedResponse,
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });

      const state = useSessionStore.getState();
      // The refetched (current) turn re-renders.
      expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
      expect(state.guidedNextTurn).toEqual(sampleGetGuidedResponse.next_turn);
      expect(state.guidedResponsePending).toBe(false);
      // A calm, distinct notice — not the generic alarm-red `error` field,
      // and never the backend's raw rejection text verbatim.
      expect(state.error).toBeNull();
      expect(state.guidedSelfHealNotice).not.toBeNull();
      expect(state.guidedSelfHealNotice).not.toContain(turnNotEmittedError.detail);
    });

    it("falls back to a plain error state when the resync refetch itself fails", async () => {
      const { respondGuided, getGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        turnNotEmittedError,
      );
      (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("network down"),
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });

      const state = useSessionStore.getState();
      expect(state.guidedSelfHealNotice).toBeNull();
      // Falls through to the plain-error path — apiErr.detail here is
      // ALREADY the backend's plain-language "out of sync" copy (not the
      // old raw protocol instruction the pre-fix backend sent), so showing
      // it verbatim is honest, not a regression.
      expect(state.error).toBe(turnNotEmittedError.detail);
    });

    it("no infinite loop: a second consecutive turn_not_emitted for the same session stops self-healing", async () => {
      const { respondGuided, getGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>)
        .mockRejectedValueOnce(turnNotEmittedError)
        .mockRejectedValueOnce(turnNotEmittedError);
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleGetGuidedResponse,
      );
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const body: GuidedRespondRequest = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      };

      // First rejection: self-heals (refetch succeeds, calm notice shown).
      await useSessionStore.getState().respondGuided(body);
      expect(useSessionStore.getState().guidedSelfHealNotice).not.toBeNull();

      // Second consecutive rejection for the SAME session: budget exhausted
      // — falls through to the plain error state instead of refetching
      // again. getGuided is only mocked once above; a second self-heal
      // attempt would throw on the unconfigured mock, which the plain-error
      // assertion below would not match if it had silently swallowed.
      await useSessionStore.getState().respondGuided(body);

      const state = useSessionStore.getState();
      expect(state.guidedSelfHealNotice).toBeNull();
      expect(state.error).toBe(turnNotEmittedError.detail);
    });

    it("a successful respond resets the self-heal budget for the next staleness", async () => {
      const { respondGuided, getGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>)
        .mockRejectedValueOnce(turnNotEmittedError)
        .mockResolvedValueOnce(sampleRespondResponse)
        .mockRejectedValueOnce(turnNotEmittedError);
      (getGuided as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce(sampleGetGuidedResponse)
        .mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        activeSessionId: "sess-1",
        guidedSession: sampleGuidedSession,
      });

      const body: GuidedRespondRequest = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      };

      await useSessionStore.getState().respondGuided(body); // self-heals
      await useSessionStore.getState().respondGuided(body); // succeeds — resets budget
      expect(useSessionStore.getState().guidedSelfHealNotice).toBeNull();

      // A THIRD, later staleness gets its own self-heal attempt rather than
      // inheriting the first cycle's exhausted budget.
      await useSessionStore.getState().respondGuided(body);
      expect(useSessionStore.getState().guidedSelfHealNotice).not.toBeNull();
    });

    it("a successful chatGuided clears a stale self-heal notice (documented lifecycle)", async () => {
      const { respondGuided, getGuided, chatGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(turnNotEmittedError);
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleGetGuidedResponse);
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        guided_session: sampleGuidedSession,
        next_turn: null,
        terminal: null,
        composition_state: null,
      });
      // Fresh session id: the self-heal budget counter is module-scoped and
      // other tests here consume "sess-1"'s budget, which would skip the
      // self-heal and never set the notice this test needs.
      useSessionStore.setState({
        activeSessionId: "sess-heal-clear",
        guidedSession: sampleGuidedSession,
      });

      const body: GuidedRespondRequest = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      };

      await useSessionStore.getState().respondGuided(body); // sets the notice
      expect(useSessionStore.getState().guidedSelfHealNotice).not.toBeNull();

      // The user sends an advisory chat instead of re-submitting the turn; a
      // successful chat must not leave the "we've refreshed — try again" notice
      // pinned above it.
      await useSessionStore.getState().chatGuided("What columns are available?");
      expect(useSessionStore.getState().guidedSelfHealNotice).toBeNull();
    });
  });
});

// src/stores/sessionStore.guided.test.ts
//
// Tests for guided-mode state fields and actions on useSessionStore.
// TDD pass: these import types/actions that don't exist yet. First run
// expected to fail on missing fields/actions.

import { describe, it, expect, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { acquireGuidedRetry, GUIDED_RETRY_STORAGE_KEY } from "./guidedOperationRetry";
import { resetStore } from "@/test/store-helpers";
import type { GuidedSession, TurnPayload, TerminalState, GetGuidedResponse, GuidedRespondAction, GuidedRespondResponse, GuidedChatResponse } from "@/types/guided";

const MockGuidedResponseReceiptError = vi.hoisted(() => class extends Error {
  readonly received = true;
  readonly cause: unknown;

  constructor(cause: unknown) {
    super("guided response was received but unusable");
    this.cause = cause;
  }
});

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
  isForkCommittedResponseError: (error: unknown) =>
    typeof error === "object" && error !== null && "committedSuccessResponse" in error,
  revertToVersion: vi.fn(),
  fetchStateVersions: vi.fn(),
  archiveSession: vi.fn(),
  getGuided: vi.fn(),
  startGuidedSession: vi.fn(),
  reconcileGuidedStartOperation: vi.fn(),
  respondGuided: vi.fn(),
  GuidedResponseReceiptError: MockGuidedResponseReceiptError,
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
  turn_token: "a".repeat(64),
  payload: {
    question: "Choose a source",
    options: [
      { id: "csv", label: "CSV", hint: null },
      { id: "jsonl", label: "JSONL", hint: null },
    ],
    allow_custom: false,
  },
};

const PROPOSAL_ID = "00000000-0000-4000-8000-000000000501";
const PROPOSAL_HASH = "d".repeat(64);
const sampleProposalTurn: TurnPayload = {
  type: "propose_pipeline",
  step_index: 2,
  turn_token: "c".repeat(64),
  payload: {
    proposal_id: PROPOSAL_ID,
    draft_hash: PROPOSAL_HASH,
    supersedes_draft_hash: null,
    summary: "guided.proposal.summary.full_graph.v1",
    rationale: "guided.proposal.rationale.review_required.v1",
    component_counts: { sources: 1, nodes: 0, edges: 2, outputs: 1 },
    blockers: [],
    graph: {
      sources: [{
        stable_id: "00000000-0000-4000-8000-000000000502",
        label: "source-1",
        plugin: { kind: "source", id: "csv" },
      }],
      edges: [
        {
          stable_id: "00000000-0000-4000-8000-000000000503",
          from_endpoint: { kind: "source", stable_id: "00000000-0000-4000-8000-000000000502" },
          to_endpoint: { kind: "output", stable_id: "00000000-0000-4000-8000-000000000504" },
          flow: { kind: "source_success", branch: null },
        },
        {
          stable_id: "00000000-0000-4000-8000-000000000505",
          from_endpoint: { kind: "source", stable_id: "00000000-0000-4000-8000-000000000502" },
          to_endpoint: { kind: "discard" },
          flow: { kind: "source_validation_failure" },
        },
      ],
    },
    nodes: [],
    outputs: [{
      stable_id: "00000000-0000-4000-8000-000000000504",
      label: "output-1",
      plugin: { kind: "sink", id: "json" },
    }],
    edit_targets: [],
  },
};

const sampleTerminal: TerminalState = {
  kind: "completed",
  reason: null,
  pipeline_yaml: "source:\n  plugin: csv\n",
};

const sampleCompositionState = {
  id: "state-1",
  session_id: "00000000-0000-4000-8000-000000000101",
  version: 1,
  nodes: [],
  edges: [],
  sources: {},
  outputs: [],
  metadata: { name: null, description: null },
  is_valid: true,
  validation_errors: null,
  validation_warnings: null,
  validation_suggestions: null,
  derived_from_state_id: null,
  created_at: "2026-07-19T00:00:00Z",
  composer_meta: null,
  plugin_policy_findings: [],
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
  next_turn: {
    type: "single_select",
    step_index: 1,
    turn_token: "b".repeat(64),
    payload: { question: "Choose a sink", options: [], allow_custom: false },
  },
  terminal: null,
  composition_state: { ...sampleCompositionState, version: 2 },
};

const sampleChatResponse: GuidedChatResponse = {
  assistant_message: "Try inspecting the CSV header row.",
  assistant_message_kind: "assistant",
  guided_session: {
    ...sampleGuidedSession,
    chat_history: [
      {
        role: "user",
        content: "What columns are available?",
        seq: 0,
        step: "step_1_source",
        ts_iso: "2026-05-13T00:00:00+00:00",
        assistant_message_kind: null,
        synthetic_failure_reason: null,
      },
      {
        role: "assistant",
        content: "Try inspecting the CSV header row.",
        seq: 1,
        step: "step_1_source",
        ts_iso: "2026-05-13T00:00:00+00:00",
        assistant_message_kind: "assistant",
        synthetic_failure_reason: null,
      },
    ],
    chat_turn_seq: 2,
  },
  next_turn: sampleNextTurn,
  terminal: null,
  composition_state: sampleCompositionState,
};

const sampleTransitioningChatResponse: GuidedChatResponse = {
  assistant_message: "I prepared the CSV source form.",
  assistant_message_kind: "assistant",
  guided_session: sampleGuidedSession,
  next_turn: {
    type: "schema_form",
    step_index: 0,
    turn_token: "b".repeat(64),
    payload: {
      mode: "plugin_options",
      plugin: "csv",
      knobs: { fields: [] },
      prefilled: { schema: { mode: "observed" } },
    },
  },
  terminal: null,
  composition_state: {
    ...sampleCompositionState,
    version: 2,
    sources: {},
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

  it("initial state: guidedSession, guidedNextTurn, guidedTerminal and proposal review all null", () => {
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.guidedProposalReview).toBeNull();
  });

  it("startGuided: binds an authoritative proposal turn to its exact active review state", async () => {
    const { getGuided } = await import("@/api/client");
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleGetGuidedResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: sampleProposalTurn,
    });
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().startGuided(RETRY_SESSION_ID);

    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
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
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });

    await useSessionStore.getState().startGuided(RETRY_SESSION_ID);

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

  it("seedGuided: retries an ambiguous failure with the same operation id and applies the response before clearing", async () => {
    const { startGuidedSession } = await import("@/api/client");
    const start = startGuidedSession as ReturnType<typeof vi.fn>;
    start
      .mockRejectedValueOnce(new TypeError("network response lost"))
      .mockResolvedValueOnce(sampleGetGuidedResponse);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await expect(
      useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial"),
    ).rejects.toThrow("network response lost");
    await useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial");

    expect(start).toHaveBeenCalledTimes(2);
    expect(start.mock.calls[0]?.[1].operationId).toBe(
      start.mock.calls[1]?.[1].operationId,
    );
    expect(useSessionStore.getState().guidedSession).toEqual(
      sampleGetGuidedResponse.guided_session,
    );
    expect(useSessionStore.getState().compositionState).toEqual(
      sampleGetGuidedResponse.composition_state,
    );
  });

  it("seedGuided: clears a typed terminal failure before the next action", async () => {
    const { startGuidedSession } = await import("@/api/client");
    const start = startGuidedSession as ReturnType<typeof vi.fn>;
    start
      .mockRejectedValueOnce({
        status: 500,
        error_type: "guided_operation_terminal_failure",
        detail: "The operation failed.",
      })
      .mockResolvedValueOnce(sampleGetGuidedResponse);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_B });

    await expect(
      useSessionStore.getState().seedGuided(RETRY_SESSION_B, "tutorial"),
    ).rejects.toMatchObject({ error_type: "guided_operation_terminal_failure" });
    await useSessionStore.getState().seedGuided(RETRY_SESSION_B, "tutorial");

    expect(start.mock.calls[0]?.[1].operationId).not.toBe(
      start.mock.calls[1]?.[1].operationId,
    );
  });

  it("seedGuided: retains the operation id when downstream response application fails", async () => {
    const { startGuidedSession } = await import("@/api/client");
    const start = startGuidedSession as ReturnType<typeof vi.fn>;
    start.mockResolvedValue(sampleGetGuidedResponse);
    const refreshAll = vi
      .fn()
      .mockRejectedValueOnce(new Error("interpretation refresh failed"))
      .mockResolvedValueOnce(undefined);
    useInterpretationEventsStore.setState({ refreshAll } as never);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await expect(
      useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial"),
    ).rejects.toThrow("interpretation refresh failed");
    await useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial");

    expect(start.mock.calls[0]?.[1].operationId).toBe(
      start.mock.calls[1]?.[1].operationId,
    );
  });

  it("seedGuided: clears a definitive response dropped by the stale-session guard", async () => {
    const { startGuidedSession } = await import("@/api/client");
    const start = startGuidedSession as ReturnType<typeof vi.fn>;
    let resolveFirst: (response: GetGuidedResponse) => void = () => undefined;
    start
      .mockReturnValueOnce(
        new Promise<GetGuidedResponse>((resolve) => {
          resolveFirst = resolve;
        }),
      )
      .mockResolvedValueOnce(sampleGetGuidedResponse);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    const stale = useSessionStore
      .getState()
      .seedGuided(RETRY_SESSION_ID, "tutorial");
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_B });
    resolveFirst(sampleGetGuidedResponse);
    await stale;
    expect(useSessionStore.getState().activeSessionId).toBe(RETRY_SESSION_B);
    expect(useSessionStore.getState().guidedSession).toBeNull();

    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });
    await useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial");

    expect(start.mock.calls[0]?.[1].operationId).not.toBe(
      start.mock.calls[1]?.[1].operationId,
    );
  });

  // ── Test 4: respondGuided happy path ─────────────────────────────────────

  it("respondGuided: atomically replaces all 4 wire fields on success", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );

    // Pre-seed active session
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });

    const outcome = await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    expect(outcome).toEqual({ status: "applied" });
    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleRespondResponse.guided_session);
    expect(state.guidedNextTurn).toEqual(sampleRespondResponse.next_turn);
    expect(state.guidedTerminal).toEqual(sampleRespondResponse.terminal);
    expect(state.compositionState).toEqual(
      sampleRespondResponse.composition_state,
    );
  });

  it("respondGuided: blocks while a guided chat is in flight (single in-flight-mutation gate)", async () => {
    // Session 93edfc90: a long-running respond (the step-3 replan runs the
    // planner in-request, 3-4 min) plus a second concurrent guided mutation
    // hung the server. The two pending flags must gate EACH OTHER — one
    // in-flight guided mutation per session, whichever endpoint carries it.
    const { respondGuided: apiRespond } = await import("@/api/client");
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedChatPending: true,
    });

    const outcome = await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    expect(outcome.status).toBe("not_applied");
    expect((outcome as { reason?: string }).reason).toBe("pending");
    expect(apiRespond).not.toHaveBeenCalled();
  });

  it("chatGuided: blocks while a guided respond is in flight (single in-flight-mutation gate)", async () => {
    const { chatGuided: apiChat, respondGuided: apiRespond } = await import("@/api/client");
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      compositionState: sampleCompositionState as never,
      guidedResponsePending: true,
    });

    await useSessionStore.getState().chatGuided("add a filter step");

    expect(apiChat).not.toHaveBeenCalled();
    expect(apiRespond).not.toHaveBeenCalled();
    // The gate is a silent no-op (the disabled affordances own the messaging);
    // it must not thrash pending/error state.
    expect(useSessionStore.getState().guidedChatPending).toBe(false);
    expect(useSessionStore.getState().guidedResponsePending).toBe(true);
  });

  it("respondGuided: settles pending and un-pins a submitting review on a same-session generation-stale response", async () => {
    // A same-session generation advance that publishes NOTHING (here: a
    // seedGuided whose start POST fails) leaves the long respond's eventual
    // response generation-stale. The stale drop must still settle the
    // in-flight submit — clearing guidedResponsePending and moving the
    // proposal review off "submitting" — or every primary and the Send stay
    // disabled forever and the review pins "Submitting this proposal
    // decision…".
    const { respondGuided: apiRespond, startGuidedSession } = await import("@/api/client");
    let resolveRespond!: (value: unknown) => void;
    (apiRespond as ReturnType<typeof vi.fn>).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveRespond = resolve;
      }),
    );
    (startGuidedSession as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
      new TypeError("interleaved seed failed"),
    );
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleProposalTurn,
    });

    const inFlight = useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    });
    expect(useSessionStore.getState().guidedProposalReview?.status).toBe("submitting");

    await expect(
      useSessionStore.getState().seedGuided(RETRY_SESSION_ID, "tutorial"),
    ).rejects.toThrow("interleaved seed failed");

    resolveRespond(sampleRespondResponse);
    const outcome = await inFlight;

    expect(outcome).toMatchObject({ status: "not_applied", reason: "stale" });
    const state = useSessionStore.getState();
    expect(state.guidedResponsePending).toBe(false);
    expect(state.guidedProposalReview?.status).not.toBe("submitting");
  });

  it("respondGuided: reuses one operation id for an ambiguous retry of the exact action and turn", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockRejectedValueOnce({ status: 503, detail: "upstream unavailable" })
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const action: GuidedRespondAction = {
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };

    const unsettled = await useSessionStore.getState().respondGuided(action);
    expect(unsettled).toMatchObject({
      status: "not_applied",
      reason: "unsettled",
    });
    await useSessionStore.getState().respondGuided(action);

    const firstRequest = respondMock.mock.calls[0]?.[1];
    const retryRequest = respondMock.mock.calls[1]?.[1];
    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(retryRequest.operation_id).toBe(firstRequest.operation_id);
    expect(retryRequest.turn_token).toBe(firstRequest.turn_token);
    expect(retryRequest.turn_token).toBe(sampleNextTurn.turn_token);
  });

  it("respondGuided: suppresses a concurrent duplicate component action", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    let resolveRespond!: (response: GuidedRespondResponse) => void;
    respondMock.mockReturnValueOnce(
      new Promise<GuidedRespondResponse>((resolve) => {
        resolveRespond = resolve;
      }),
    );
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const action: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
      component_action: { action: "add", component_kind: "source" },
    };

    const first = useSessionStore.getState().respondGuided(action);
    expect(useSessionStore.getState().guidedResponsePending).toBe(true);
    const duplicate = await useSessionStore.getState().respondGuided(action);
    expect(duplicate).toMatchObject({ status: "not_applied", reason: "pending" });
    expect(respondMock).toHaveBeenCalledTimes(1);

    resolveRespond(sampleRespondResponse);
    await first;
    expect(useSessionStore.getState().guidedResponsePending).toBe(false);
  });

  it("respondGuided: retries an ambiguous component action with the exact body and operation id", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const action: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
      component_action: {
        action: "reorder",
        component_kind: "source",
        stable_ids: [
          "00000000-0000-4000-8000-000000000102",
          "00000000-0000-4000-8000-000000000101",
        ],
      },
    };

    const unsettled = await useSessionStore.getState().respondGuided(action);
    expect(unsettled).toMatchObject({ status: "not_applied", reason: "unsettled" });
    await useSessionStore.getState().respondGuided(action);

    const firstRequest = respondMock.mock.calls[0]?.[1];
    const retryRequest = respondMock.mock.calls[1]?.[1];
    expect(retryRequest).toEqual(firstRequest);
    expect(retryRequest.operation_id).toBe(firstRequest.operation_id);
    expect(retryRequest.component_action).toEqual(action.component_action);
  });

  it.each([
    ["malformed JSON", new SyntaxError("unexpected end of JSON input")],
    ["schema-invalid JSON", new Error("guided respond response has unexpected fields")],
  ])("respondGuided: retains the exact request after a received but unusable 2xx (%s)", async (_label, cause) => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockRejectedValueOnce(new MockGuidedResponseReceiptError(cause))
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const action: GuidedRespondAction = {
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };

    const unsettled = await useSessionStore.getState().respondGuided(action);
    expect(unsettled).toMatchObject({ status: "not_applied", reason: "unsettled" });
    await useSessionStore.getState().respondGuided(action);

    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(respondMock.mock.calls[1]?.[1]).toEqual(respondMock.mock.calls[0]?.[1]);
  });

  it("respondGuided: rejects a different action while an ambiguous component operation is unsettled", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const original: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
      component_action: { action: "add", component_kind: "source" },
    };
    const conflicting: GuidedRespondAction = {
      ...original,
      component_action: { action: "finish", component_kind: "source" },
    };

    await useSessionStore.getState().respondGuided(original);
    const originalRequest = respondMock.mock.calls[0]?.[1];
    const conflict = await useSessionStore.getState().respondGuided(conflicting);

    expect(conflict).toMatchObject({ status: "not_applied", reason: "custody_conflict" });
    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(useSessionStore.getState().error).toMatch(/unsettled.*same action/i);

    await useSessionStore.getState().respondGuided(original);
    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(respondMock.mock.calls[1]?.[1]).toEqual(originalRequest);
  });

  it("respondGuided: resyncs decoded success before accepting an action for the authoritative turn", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const getMock = getGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockResolvedValueOnce(sampleRespondResponse)
      .mockResolvedValueOnce(sampleRespondResponse);
    getMock.mockResolvedValueOnce(sampleRespondResponse);
    const refreshAll = vi
      .fn()
      .mockRejectedValueOnce(new Error("local projection failed"))
      .mockResolvedValueOnce(undefined);
    useInterpretationEventsStore.setState({ refreshAll } as never);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const original: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
      component_action: {
        action: "reorder",
        component_kind: "source",
        stable_ids: [
          "00000000-0000-4000-8000-000000000102",
          "00000000-0000-4000-8000-000000000101",
        ],
      },
    };
    const resynced = await useSessionStore.getState().respondGuided(original);
    const originalRequest = respondMock.mock.calls[0]?.[1];

    expect(resynced).toEqual({ status: "applied" });
    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(getMock).toHaveBeenCalledWith(RETRY_SESSION_ID);
    expect(useSessionStore.getState().guidedNextTurn).toEqual(
      sampleRespondResponse.next_turn,
    );

    await useSessionStore.getState().respondGuided({
      chosen: ["json"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(respondMock.mock.calls[1]?.[1].operation_id).not.toBe(
      originalRequest.operation_id,
    );
    expect(respondMock.mock.calls[1]?.[1].turn_token).toBe(
      sampleRespondResponse.next_turn?.turn_token,
    );
    expect(refreshAll).toHaveBeenCalledTimes(3);
  });

  it("respondGuided: allocates a new component operation after a definitive rejection", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    respondMock
      .mockRejectedValueOnce({ status: 400, detail: "component action rejected" })
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const action: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
      component_action: { action: "finish", component_kind: "source" },
    };

    const rejected = await useSessionStore.getState().respondGuided(action);
    expect(rejected).toMatchObject({ status: "not_applied", reason: "rejected" });
    await useSessionStore.getState().respondGuided(action);

    expect(respondMock.mock.calls[1]?.[1].operation_id).not.toBe(
      respondMock.mock.calls[0]?.[1].operation_id,
    );
  });

  it.each(["authoritative GET", "authoritative publication"])(
    "respondGuided: blocks the stale turn when %s fails after decoded settlement",
    async (failurePoint) => {
      const { getGuided, respondGuided } = await import("@/api/client");
      const respondMock = respondGuided as ReturnType<typeof vi.fn>;
      const getMock = getGuided as ReturnType<typeof vi.fn>;
      respondMock.mockResolvedValueOnce(sampleRespondResponse);
      const refreshAll = vi.fn().mockRejectedValue(
        new TypeError("interpretation refresh interrupted"),
      );
      useInterpretationEventsStore.setState({ refreshAll } as never);
      if (failurePoint === "authoritative GET") {
        getMock.mockRejectedValueOnce(new Error("guided resync failed"));
      } else {
        getMock.mockResolvedValueOnce(sampleRespondResponse);
      }
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
        guidedTerminal: null,
        compositionState: sampleCompositionState,
      });
      const action: GuidedRespondAction = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
        control_signal: null,
      };

      const outcome = await useSessionStore.getState().respondGuided(action);

      expect(outcome).toMatchObject({ status: "not_applied", reason: "refresh_required" });
      expect(respondMock).toHaveBeenCalledTimes(1);
      expect(getMock).toHaveBeenCalledWith(RETRY_SESSION_ID);
      expect(useSessionStore.getState().guidedNextTurn).toBeNull();
      expect(useSessionStore.getState().guidedResponsePending).toBe(false);
      expect(useSessionStore.getState().error).toMatch(/accepted.*refresh.*re-enter/i);
      await expect(useSessionStore.getState().respondGuided(action)).resolves.toMatchObject({
        status: "not_applied",
        reason: "no_current_turn",
      });
      expect(respondMock).toHaveBeenCalledTimes(1);
    },
  );

  it("respondGuided: resyncs a proposal-binding 409 without replaying the rejected action", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const getMock = getGuided as ReturnType<typeof vi.fn>;
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    respondMock.mockRejectedValueOnce({ status: 409, detail });
    let resolveReload!: (response: GetGuidedResponse) => void;
    getMock.mockReturnValueOnce(new Promise<GetGuidedResponse>((resolve) => {
      resolveReload = resolve;
    }));
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });
    const action: GuidedRespondAction = {
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    };

    const response = useSessionStore.getState().respondGuided(action);

    await vi.waitFor(() => expect(getMock).toHaveBeenCalledWith(RETRY_SESSION_ID));
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "reloading",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
    resolveReload({
      ...sampleGetGuidedResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: sampleProposalTurn,
    });
    const outcome = await response;

    expect(outcome).toMatchObject({ status: "not_applied", reason: "rejected" });
    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(useSessionStore.getState().guidedNextTurn).toEqual(sampleProposalTurn);
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "stale",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
    expect(useSessionStore.getState().error).toBe(detail);
    expect(window.sessionStorage.length).toBe(0);
  });

  it("respondGuided: activates a different authoritative proposal after a proposal-binding 409", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const getMock = getGuided as ReturnType<typeof vi.fn>;
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    const successorId = "00000000-0000-4000-8000-000000000506";
    const successorHash = "e".repeat(64);
    const successorTurn: TurnPayload = {
      ...sampleProposalTurn,
      turn_token: "e".repeat(64),
      payload: {
        ...sampleProposalTurn.payload,
        proposal_id: successorId,
        draft_hash: successorHash,
      },
    };
    respondMock.mockRejectedValueOnce({ status: 409, detail });
    getMock.mockResolvedValueOnce({
      ...sampleGetGuidedResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: successorTurn,
    });
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });

    await useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    });

    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(useSessionStore.getState().guidedNextTurn).toEqual(successorTurn);
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "active",
      proposal_id: successorId,
      draft_hash: successorHash,
    });
    expect(window.sessionStorage.length).toBe(0);
  });

  it("respondGuided: keeps the old proposal stale when a proposal-binding 409 reload has no proposal", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({ status: 409, detail });
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleGetGuidedResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: null,
    });
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });

    await useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    });

    expect(useSessionStore.getState().guidedNextTurn).toBeNull();
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "stale",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
    expect(window.sessionStorage.length).toBe(0);
  });

  it("respondGuided: cannot clobber a newly selected session during a proposal 409 reload", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const getMock = getGuided as ReturnType<typeof vi.fn>;
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({ status: 409, detail });
    let resolveReload!: (response: GetGuidedResponse) => void;
    getMock.mockReturnValueOnce(new Promise<GetGuidedResponse>((resolve) => {
      resolveReload = resolve;
    }));
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });

    const pending = useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    });
    await vi.waitFor(() => expect(getMock).toHaveBeenCalledWith(RETRY_SESSION_ID));

    const newSessionId = "00000000-0000-4000-8000-000000000799";
    const newProposalId = "00000000-0000-4000-8000-000000000798";
    useSessionStore.setState({
      activeSessionId: newSessionId,
      guidedNextTurn: null,
      guidedProposalReview: {
        status: "active",
        proposal_id: newProposalId,
        draft_hash: "9".repeat(64),
      },
      guidedResponsePending: false,
      error: null,
    });
    resolveReload({
      ...sampleGetGuidedResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: sampleProposalTurn,
    });
    await pending;

    expect(useSessionStore.getState().activeSessionId).toBe(newSessionId);
    expect(useSessionStore.getState().guidedNextTurn).toBeNull();
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "active",
      proposal_id: newProposalId,
      draft_hash: "9".repeat(64),
    });
    expect(useSessionStore.getState().error).toBeNull();
  });

  it("respondGuided: locks the old proposal when its authoritative reload fails after a 409", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    respondMock.mockRejectedValueOnce({ status: 409, detail });
    (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new TypeError("reload unavailable"));
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });

    await useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    });

    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(useSessionStore.getState().guidedNextTurn).toEqual(sampleProposalTurn);
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "error",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      message: expect.stringMatching(/authoritative replacement could not be loaded/i),
      retryable: false,
      retry_action: null,
    });
    expect(useSessionStore.getState().guidedResponsePending).toBe(false);
    expect(window.sessionStorage.length).toBe(0);
  });

  it("respondGuided: marks an ambiguous proposal transport failure as retryable error and reuses its operation id", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const successful = {
      ...sampleRespondResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: sampleProposalTurn,
    };
    respondMock
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(successful);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });
    const action: GuidedRespondAction = {
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: null,
      control_signal: null,
    };

    await useSessionStore.getState().respondGuided(action);
    expect(useSessionStore.getState().guidedProposalReview).toMatchObject({
      status: "error",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      retryable: true,
      retry_action: { kind: "review_wiring" },
    });
    await useSessionStore.getState().respondGuided(action);

    expect(respondMock.mock.calls[1]?.[1].operation_id).toBe(
      respondMock.mock.calls[0]?.[1].operation_id,
    );
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
  });

  it("respondGuided: resyncs a decoded proposal operation when local apply fails", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    const response = {
      ...sampleRespondResponse,
      guided_session: { ...sampleGuidedSession, step: "step_3_transforms" },
      next_turn: sampleProposalTurn,
    };
    respondMock.mockResolvedValueOnce(response);
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(response);
    const refreshAll = vi
      .fn()
      .mockRejectedValueOnce(new Error("local interpretation refresh failed"))
      .mockResolvedValueOnce(undefined);
    useInterpretationEventsStore.setState({ refreshAll } as never);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
      guidedNextTurn: sampleProposalTurn,
      guidedProposalReview: {
        status: "active",
        proposal_id: PROPOSAL_ID,
        draft_hash: PROPOSAL_HASH,
      },
    });
    const action: GuidedRespondAction = {
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
      edit_target: { kind: "source", stable_id: sampleProposalTurn.payload.graph.sources[0].stable_id },
      control_signal: null,
    };

    await useSessionStore.getState().respondGuided(action);

    expect(respondMock).toHaveBeenCalledTimes(1);
    expect(getGuided).toHaveBeenCalledWith(RETRY_SESSION_ID);
    expect(useSessionStore.getState().guidedNextTurn).toEqual(sampleProposalTurn);
    expect(useSessionStore.getState().guidedProposalReview).toEqual({
      status: "active",
      proposal_id: PROPOSAL_ID,
      draft_hash: PROPOSAL_HASH,
    });
  });

  it("respondGuided: does not publish proposal resync when interpretation refresh fails", async () => {
    const { getGuided, respondGuided } = await import("@/api/client");
    const detail = "proposal_id and draft_hash do not identify the active guided proposal";
    (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({ status: 409, detail });
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      ...sampleGetGuidedResponse,
      next_turn: { ...sampleNextTurn, turn_token: "c".repeat(64) },
    });
    const refreshAll = vi.fn().mockRejectedValueOnce(new Error("review refresh failed"));
    useInterpretationEventsStore.setState({ refreshAll } as never);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      compositionState: sampleCompositionState,
    });

    await useSessionStore.getState().respondGuided({
      chosen: ["review_wiring"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: "00000000-0000-4000-8000-000000000501",
      draft_hash: "d".repeat(64),
      edit_target: null,
      control_signal: null,
    });

    const state = useSessionStore.getState();
    expect(refreshAll).toHaveBeenCalledWith(RETRY_SESSION_ID);
    expect(state.guidedNextTurn).toEqual(sampleNextTurn);
    expect(state.compositionState).toEqual(sampleCompositionState);
    expect(state.error).toBe(detail);
    expect(state.guidedResponsePending).toBe(false);
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
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });

    await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    expect(refreshAll).toHaveBeenCalledWith(RETRY_SESSION_ID);
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
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });

    const promise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    await vi.waitFor(() => expect(refreshAll).toHaveBeenCalledWith(RETRY_SESSION_ID));
    expect(useSessionStore.getState().guidedResponsePending).toBe(true);
    releaseRefresh();
    await promise;
    expect(useSessionStore.getState().guidedResponsePending).toBe(false);
  });

  // ── Test 5: respondGuided invariant violation ─────────────────────────────

  it("respondGuided: reports that no active session can receive the response", async () => {
    // activeSessionId is null from resetStore
    await expect(
      useSessionStore.getState().respondGuided({
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
        control_signal: null,
      }),
    ).resolves.toMatchObject({
      status: "not_applied",
      reason: "no_active_session",
    });
  });

  // ── Test 6: exitToFreeform ────────────────────────────────────────────────

  it("exitToFreeform: delegates to respondGuided with control_signal=exit_to_freeform", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
    });
    const outcome = await useSessionStore.getState().exitToFreeform();

    expect(outcome).toEqual({ status: "applied" });
    expect(respondGuided).toHaveBeenCalledWith(
      RETRY_SESSION_ID,
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
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: null,
    });

    await useSessionStore.getState().enterGuided();

    expect(convertToGuided).toHaveBeenCalledWith(RETRY_SESSION_ID, expect.any(String));
    expect(getGuided).not.toHaveBeenCalled();
    const state = useSessionStore.getState();
    expect(state.guidedSession).toEqual(sampleGetGuidedResponse.guided_session);
  });

  it("convertToGuided: populates all 4 wire fields atomically on success", async () => {
    const { convertToGuided } = await import("@/api/client");
    (convertToGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleGetGuidedResponse,
    );

    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });
    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);

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

    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });
    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);

    expect(useSessionStore.getState().error).toBe(
      "You do not own this session.",
    );
  });

  it("convertToGuided: reuses the operation id after an ambiguous failure and clears it after success", async () => {
    const { convertToGuided } = await import("@/api/client");
    const convertMock = convertToGuided as ReturnType<typeof vi.fn>;
    convertMock
      .mockRejectedValueOnce({ status: 503 })
      .mockResolvedValueOnce(sampleGetGuidedResponse)
      .mockResolvedValueOnce(sampleGetGuidedResponse);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);
    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);
    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);

    const firstOperationId = convertMock.mock.calls[0]?.[1];
    const retryOperationId = convertMock.mock.calls[1]?.[1];
    const nextActionOperationId = convertMock.mock.calls[2]?.[1];
    expect(firstOperationId).toEqual(expect.any(String));
    expect(retryOperationId).toBe(firstOperationId);
    expect(nextActionOperationId).not.toBe(firstOperationId);
  });

  it("convertToGuided: clears a terminal 500 marker so an explicit retry gets a new operation id", async () => {
    const { convertToGuided } = await import("@/api/client");
    const convertMock = convertToGuided as ReturnType<typeof vi.fn>;
    convertMock
      .mockRejectedValueOnce({
        status: 500,
        error_type: "guided_operation_terminal_failure",
        detail: "The operation failed.",
      })
      .mockResolvedValueOnce(sampleGetGuidedResponse);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);
    await useSessionStore.getState().convertToGuided(RETRY_SESSION_ID);

    expect(convertMock.mock.calls[0]?.[1]).toEqual(expect.any(String));
    expect(convertMock.mock.calls[1]?.[1]).toEqual(expect.any(String));
    expect(convertMock.mock.calls[1]?.[1]).not.toBe(convertMock.mock.calls[0]?.[1]);
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

  it("revertToVersion: reports a conflicting target without rejecting or sending another request", async () => {
    const { revertToVersion } = await import("@/api/client");
    const revertMock = revertToVersion as ReturnType<typeof vi.fn>;
    revertMock.mockRejectedValue(new TypeError("response lost"));
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().revertToVersion("state-old");
    await expect(
      useSessionStore.getState().revertToVersion("state-different"),
    ).resolves.toBeUndefined();

    expect(revertMock).toHaveBeenCalledTimes(1);
    expect(useSessionStore.getState().error).toMatch(/unsettled/i);

    await useSessionStore.getState().revertToVersion("state-old");
    expect(revertMock).toHaveBeenCalledTimes(2);
    expect(revertMock.mock.calls[1]?.[2]).toBe(revertMock.mock.calls[0]?.[2]);
  });

  it("revertToVersion: retains the operation id when the POST succeeds but the guided probe is ambiguous", async () => {
    const { revertToVersion, getGuided } = await import("@/api/client");
    const revertMock = revertToVersion as ReturnType<typeof vi.fn>;
    revertMock
      .mockResolvedValueOnce({ ...sampleCompositionState, version: 2 })
      .mockResolvedValueOnce({ ...sampleCompositionState, version: 2 })
      .mockResolvedValueOnce({ ...sampleCompositionState, version: 3 });
    (getGuided as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce({ status: 503 })
      .mockRejectedValueOnce({ status: 400 })
      .mockRejectedValueOnce({ status: 400 });
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

    await useSessionStore.getState().revertToVersion("state-old");
    await useSessionStore.getState().revertToVersion("state-old");
    await useSessionStore.getState().revertToVersion("state-old");

    const firstOperationId = revertMock.mock.calls[0]?.[2];
    const retryOperationId = revertMock.mock.calls[1]?.[2];
    const nextActionOperationId = revertMock.mock.calls[2]?.[2];
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
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
      error: null,
    });

    const startPromise = useSessionStore.getState().startGuided("sess-A");

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_B,
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
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    const respondPromise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    // Simulate a session switch before the response arrives.
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_B,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      guidedResponsePending: true,
    });

    // Let sess-A's response arrive.
    resolveRespond(sampleRespondResponse);
    const outcome = await respondPromise;

    expect(outcome).toMatchObject({ status: "not_applied", reason: "stale" });
    // The stale response must have been dropped — sess-B's guided state
    // must remain null (not overwritten by sess-A's respond result).
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.guidedResponsePending).toBe(true);
  });

  it("respondGuided: clears successful stale-session custody before a new authoritative turn is submitted", async () => {
    const { respondGuided } = await import("@/api/client");
    const respondMock = respondGuided as ReturnType<typeof vi.fn>;
    let resolveRespond!: (response: GuidedRespondResponse) => void;
    respondMock
      .mockReturnValueOnce(new Promise<GuidedRespondResponse>((resolve) => {
        resolveRespond = resolve;
      }))
      .mockResolvedValueOnce(sampleRespondResponse);
    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });
    const firstAction: GuidedRespondAction = {
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    };

    const staleRequest = useSessionStore.getState().respondGuided(firstAction);
    useSessionStore.setState({ activeSessionId: RETRY_SESSION_B });
    resolveRespond(sampleRespondResponse);
    const staleOutcome = await staleRequest;

    expect(staleOutcome).toMatchObject({
      status: "not_applied",
      reason: "stale",
    });

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleRespondResponse.guided_session,
      guidedNextTurn: sampleRespondResponse.next_turn,
      guidedTerminal: sampleRespondResponse.terminal,
      guidedResponsePending: false,
    });
    await useSessionStore.getState().respondGuided({
      ...firstAction,
      chosen: ["json"],
    });

    expect(respondMock).toHaveBeenCalledTimes(2);
    expect(respondMock.mock.calls[1]?.[1].operation_id).not.toBe(
      respondMock.mock.calls[0]?.[1].operation_id,
    );
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
      activeSessionId: RETRY_SESSION_ID,
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
      error: null,
    });

    const respondPromise = useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });

    useSessionStore.setState({
      activeSessionId: RETRY_SESSION_B,
      guidedSession: null,
      guidedNextTurn: null,
      guidedTerminal: null,
      guidedResponsePending: true,
      error: null,
    });

    rejectRespond(new Error("network down for stale session"));
    const outcome = await respondPromise;

    expect(outcome).toMatchObject({ status: "not_applied", reason: "stale" });
    const state = useSessionStore.getState();
    expect(state.guidedSession).toBeNull();
    expect(state.guidedNextTurn).toBeNull();
    expect(state.guidedTerminal).toBeNull();
    expect(state.guidedResponsePending).toBe(true);
    expect(state.error).toBeNull();
  });

  it("forkFromMessage: loads the locator child authoritatively, including guided state", async () => {
    const {
      forkFromMessage,
      fetchSessions,
      fetchMessages,
      fetchCompositionState,
      fetchCompositionProposals,
      fetchComposerPreferences,
      getGuided,
    } = await import("@/api/client");
    const parentId = "00000000-0000-4000-8000-000000000701";
    const childId = "00000000-0000-4000-8000-000000000702";
    const child = { id: childId, title: "Fork", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" };

    (forkFromMessage as ReturnType<typeof vi.fn>).mockResolvedValueOnce({ session_id: childId });
    (fetchSessions as ReturnType<typeof vi.fn>).mockResolvedValueOnce([child]);
    (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleCompositionState);
    (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
    (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
    (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      guided_session: sampleGuidedSession,
      next_turn: sampleNextTurn,
      terminal: null,
      composition_state: sampleCompositionState,
    });

    // Pre-seed: parent session has guided state that must NOT bleed into fork.
    useSessionStore.setState({
      activeSessionId: parentId,
      sessions: [{ id: parentId, title: "Parent", created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z" }],
      guidedSession: sampleGuidedSession,
      guidedNextTurn: sampleNextTurn,
      guidedTerminal: null,
    });

    const forkPromise = useSessionStore.getState().forkFromMessage("msg-1", "new content");
    await forkPromise;

    const state = useSessionStore.getState();
    expect(state.activeSessionId).toBe(childId);
    expect(state.guidedSession).toEqual(sampleGuidedSession);
    expect(state.guidedNextTurn).toEqual(sampleNextTurn);
    expect(state.guidedTerminal).toBeNull();
    expect(getGuided).toHaveBeenCalledWith(childId);
  });

  describe("chatGuided", () => {
    beforeEach(() => {
      useSessionStore.setState({ compositionState: sampleCompositionState });
    });

    it("throws when activeSessionId is null", async () => {
      await expect(
        useSessionStore.getState().chatGuided("What columns are available?"),
      ).rejects.toThrow("chatGuided called without active session");
    });

    it("throws when guidedSession is null", async () => {
      useSessionStore.setState({ activeSessionId: RETRY_SESSION_ID });

      await expect(
        useSessionStore.getState().chatGuided("What columns are available?"),
      ).rejects.toThrow("chatGuided called before guidedSession loaded");
    });

    it("throws before transport when there is no current unanswered turn token", async () => {
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
      });

      await expect(
        useSessionStore.getState().chatGuided("What columns are available?"),
      ).rejects.toThrow("chatGuided called without a current unanswered turn");
    });

    it("starts a cold live session with the exact first ordinary message before any guided chat", async () => {
      const { chatGuided, getGuided, startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockResolvedValueOnce(sampleGetGuidedResponse);
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleGetGuidedResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");

      expect(start).toHaveBeenCalledWith(
        RETRY_SESSION_ID,
        {
          profile: "live",
          intent: "Build the exact graph",
          operationId: expect.any(String),
        },
        undefined,
      );
      expect(getGuided).not.toHaveBeenCalled();
      expect(chatGuided).not.toHaveBeenCalled();
      expect(useSessionStore.getState().compositionState).toEqual(
        sampleCompositionState,
      );
      expect(useSessionStore.getState().guidedChatPending).toBe(false);
    });

    it("cold cancel reconciles with a fresh signal, clears a failed attempt, and permits revised text", async () => {
      const { reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      const reconcile = reconcileGuidedStartOperation as ReturnType<typeof vi.fn>;
      start
        .mockImplementationOnce(
          (_sessionId: string, _body: unknown, signal?: AbortSignal) =>
            new Promise((_resolve, reject) => {
              signal?.addEventListener("abort", () => reject(signal.reason));
            }),
        )
        .mockResolvedValueOnce(sampleGetGuidedResponse);
      reconcile.mockResolvedValueOnce({ status: "failed", failure_code: "request_cancelled" });
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });
      const controller = new AbortController();

      const pending = useSessionStore.getState().chatGuided("Original prompt", controller.signal);
      controller.abort("compose_user_cancel");
      await pending;

      expect(reconcile).toHaveBeenCalledTimes(1);
      const reconcileSignal = reconcile.mock.calls[0]?.[2] as AbortSignal;
      expect(reconcileSignal).not.toBe(controller.signal);
      expect(reconcileSignal.aborted).toBe(false);
      expect(useSessionStore.getState().error).toMatch(/stopped.*revise.*send it again/i);
      expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "").not.toContain(
        '"kind":"guided_start"',
      );

      await useSessionStore.getState().chatGuided("Revised prompt");
      expect(start).toHaveBeenCalledTimes(2);
      expect(start.mock.calls[1]?.[1]).toEqual(
        expect.objectContaining({ intent: "Revised prompt" }),
      );
    });

    it("cold cancel applies an authoritatively completed operation and clears custody", async () => {
      const { getGuided, reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const controller = new AbortController();
      (startGuidedSession as ReturnType<typeof vi.fn>).mockImplementationOnce(
        (_sessionId: string, _body: unknown, signal?: AbortSignal) =>
          new Promise((_resolve, reject) => {
            signal?.addEventListener("abort", () => reject(signal.reason));
          }),
      );
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "completed",
        composition_state_id: sampleCompositionState.id,
      });
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      const pending = useSessionStore.getState().chatGuided("Original prompt", controller.signal);
      controller.abort("compose_user_cancel");
      await pending;

      expect(getGuided).toHaveBeenCalledWith(RETRY_SESSION_ID, expect.any(AbortSignal));
      expect(useSessionStore.getState().compositionState).toEqual(sampleCompositionState);
      expect(useSessionStore.getState().error).toBeNull();
      expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "").not.toContain(
        '"kind":"guided_start"',
      );
    });

    it.each([
      ["in_progress", { status: "in_progress" }],
      ["unknown network state", new TypeError("reconciliation unavailable")],
    ])("retains %s custody and sends no conflicting revised POST", async (_label, reconciliation) => {
      const { reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const reconcile = reconcileGuidedStartOperation as ReturnType<typeof vi.fn>;
      (startGuidedSession as ReturnType<typeof vi.fn>).mockRejectedValueOnce(new TypeError("response lost"));
      if (reconciliation instanceof Error) {
        reconcile.mockRejectedValue(reconciliation);
      } else {
        reconcile.mockResolvedValue(reconciliation);
      }
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Original prompt");
      await useSessionStore.getState().chatGuided("Revised prompt");

      expect(startGuidedSession).toHaveBeenCalledTimes(1);
      expect(reconcile).toHaveBeenCalledTimes(2);
      expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain(
        '"kind":"guided_start"',
      );
      expect(useSessionStore.getState().error).toMatch(/confirm|running/i);
    });

    it("selectSession clears a failed cold-start descriptor without needing the lost prompt", async () => {
      const {
        fetchComposerPreferences,
        fetchCompositionProposals,
        fetchCompositionState,
        fetchMessages,
        getGuided,
        reconcileGuidedStartOperation,
      } = await import("@/api/client");
      acquireGuidedRetry("guided_start", RETRY_SESSION_ID, ["live", "prompt no longer in memory"]);
      (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
      (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
      (getGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({ status: 400 });
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "failed",
        failure_code: "request_cancelled",
      });

      await useSessionStore.getState().selectSession(RETRY_SESSION_ID);

      expect(reconcileGuidedStartOperation).toHaveBeenCalledWith(
        RETRY_SESSION_ID,
        expect.any(String),
        expect.any(AbortSignal),
      );
      expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "").not.toContain(
        '"kind":"guided_start"',
      );
      expect(useSessionStore.getState().activeSessionId).toBe(RETRY_SESSION_ID);
    });

    it("recovers an ambiguous cold start through authoritative completion without another POST", async () => {
      const { getGuided, reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockRejectedValueOnce(new TypeError("response lost"));
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "completed",
        composition_state_id: sampleCompositionState.id,
      });
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleGetGuidedResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");

      expect(start).toHaveBeenCalledTimes(1);
      expect(getGuided).toHaveBeenCalledTimes(1);
      expect(useSessionStore.getState().compositionState).toEqual(sampleCompositionState);
    });

    it("selectSession retires cold-start custody after its fenced authoritative reload", async () => {
      const {
        fetchMessages,
        fetchCompositionState,
        fetchCompositionProposals,
        fetchComposerPreferences,
        getGuided,
        reconcileGuidedStartOperation,
        startGuidedSession,
      } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockRejectedValueOnce(new TypeError("response lost"));
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");
      expect(window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toContain(
        '"kind":"guided_start"',
      );

      (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleCompositionState,
      );
      (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValueOnce([]);
      (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValueOnce(null);
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValue(sampleGetGuidedResponse);
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "completed",
        composition_state_id: sampleCompositionState.id,
      });

      await useSessionStore.getState().selectSession(RETRY_SESSION_ID);

      expect(
        window.sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY) ?? "",
      ).not.toContain('"kind":"guided_start"');

      start.mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        compositionState: null,
        guidedNextTurn: null,
      });
      await useSessionStore.getState().chatGuided("Build a different graph");

      expect(start).toHaveBeenCalledTimes(2);
      expect(start.mock.calls[1]?.[1]).toEqual(
        expect.objectContaining({ intent: "Build a different graph" }),
      );
      expect(useSessionStore.getState().error).toBeNull();
    });

    it("passes the caller abort signal through the cold guided-start POST", async () => {
      const { startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });
      const controller = new AbortController();

      await useSessionStore
        .getState()
        .chatGuided("Build the exact graph", controller.signal);

      expect(start).toHaveBeenCalledWith(
        RETRY_SESSION_ID,
        expect.objectContaining({ intent: "Build the exact graph" }),
        controller.signal,
      );
    });

    it("recovers a malformed committed response through authoritative reconciliation", async () => {
      const { getGuided, reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockRejectedValueOnce(
        new MockGuidedResponseReceiptError(new Error("invalid response")),
      );
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        status: "completed",
        composition_state_id: sampleCompositionState.id,
      });
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");

      expect(start).toHaveBeenCalledTimes(1);
      expect(useSessionStore.getState().compositionState).toEqual(sampleCompositionState);
    });

    it("retains cold start custody when canonical response application fails after commit", async () => {
      const { getGuided, reconcileGuidedStartOperation, startGuidedSession } = await import("@/api/client");
      const start = startGuidedSession as ReturnType<typeof vi.fn>;
      start.mockResolvedValue(sampleGetGuidedResponse);
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValue(
        sampleGetGuidedResponse,
      );
      (reconcileGuidedStartOperation as ReturnType<typeof vi.fn>).mockResolvedValue({
        status: "completed",
        composition_state_id: sampleCompositionState.id,
      });
      const refreshAll = vi.fn().mockRejectedValue(
        new Error("interpretation refresh failed"),
      );
      useInterpretationEventsStore.setState({ refreshAll } as never);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");
      expect(start).toHaveBeenCalledTimes(1);
      expect(useSessionStore.getState().compositionState).toBeNull();
      expect(useSessionStore.getState().guidedChatPending).toBe(false);
      refreshAll.mockResolvedValue(undefined);
      await useSessionStore.getState().chatGuided("Build the exact graph");

      expect(start).toHaveBeenCalledTimes(1);
      expect(useSessionStore.getState().compositionState).toEqual(sampleCompositionState);
    });

    it("does not publish an old cold-start response after selecting A to B to A", async () => {
      const {
        fetchComposerPreferences,
        fetchCompositionProposals,
        fetchCompositionState,
        fetchMessages,
        getGuided,
        startGuidedSession,
      } = await import("@/api/client");
      const latestSession = { ...sampleGuidedSession, chat_turn_seq: 42 };
      const latestState = { ...sampleCompositionState, version: 42 };
      const latestA = {
        ...sampleGetGuidedResponse,
        guided_session: latestSession,
        composition_state: latestState,
      };
      (fetchMessages as ReturnType<typeof vi.fn>).mockResolvedValue([]);
      (fetchCompositionState as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ...sampleCompositionState,
        session_id: RETRY_SESSION_B,
      }).mockResolvedValueOnce(latestState);
      (fetchCompositionProposals as ReturnType<typeof vi.fn>).mockResolvedValue([]);
      (fetchComposerPreferences as ReturnType<typeof vi.fn>).mockResolvedValue(null);
      (getGuided as ReturnType<typeof vi.fn>)
        .mockResolvedValueOnce({
          ...sampleGetGuidedResponse,
          composition_state: {
            ...sampleCompositionState,
            session_id: RETRY_SESSION_B,
          },
        })
        .mockResolvedValueOnce(latestA)
        .mockResolvedValueOnce(sampleGetGuidedResponse);
      let resolveStart!: (response: GetGuidedResponse) => void;
      (startGuidedSession as ReturnType<typeof vi.fn>).mockReturnValueOnce(
        new Promise<GetGuidedResponse>((resolve) => {
          resolveStart = resolve;
        }),
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
      });

      const coldStart = useSessionStore.getState().chatGuided("Build the exact graph");
      await useSessionStore.getState().selectSession(RETRY_SESSION_B);
      await useSessionStore.getState().selectSession(RETRY_SESSION_ID);
      resolveStart(sampleGetGuidedResponse);
      await coldStart;

      expect(useSessionStore.getState().activeSessionId).toBe(RETRY_SESSION_ID);
      expect(useSessionStore.getState().guidedSession).toEqual(latestSession);
      expect(useSessionStore.getState().compositionState).toEqual(latestState);
    });

    it("clears stale error state after a successful cold start retry", async () => {
      const { startGuidedSession } = await import("@/api/client");
      (startGuidedSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleGetGuidedResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: null,
        compositionState: null,
        error: "previous failure",
        errorDetails: ["previous detail"],
      });

      await useSessionStore.getState().chatGuided("Build the exact graph");

      expect(useSessionStore.getState().error).toBeNull();
      expect(useSessionStore.getState().errorDetails).toBeNull();
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const chatPromise = useSessionStore.getState().chatGuided("What columns are available?");

      expect(useSessionStore.getState().guidedChatPending).toBe(true);
      expect(chatGuided).toHaveBeenCalledWith(
        RETRY_SESSION_ID,
        {
          operation_id: expect.any(String),
          turn_token: sampleNextTurn.turn_token,
          message: "What columns are available?",
        },
        undefined,
      );

      resolveChat(sampleChatResponse);
      await chatPromise;

      const state = useSessionStore.getState();
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedSession).toEqual(sampleChatResponse.guided_session);
    });

    it("routes a step-3 proposal-turn send through /guided/respond as a prose revision", async () => {
      const { chatGuided, respondGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleRespondResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
        guidedNextTurn: sampleProposalTurn,
        compositionState: sampleCompositionState,
      });

      await useSessionStore
        .getState()
        .chatGuided("Add a deduplication transform before the output.");

      // The instruction is a proposal revision, not a chat message: it goes to
      // /guided/respond with the proposal binding + the closed revision bag, and
      // never touches /guided/chat.
      expect(chatGuided).not.toHaveBeenCalled();
      expect(respondGuided).toHaveBeenCalledTimes(1);
      expect(respondGuided).toHaveBeenCalledWith(
        RETRY_SESSION_ID,
        expect.objectContaining({
          proposal_id: PROPOSAL_ID,
          draft_hash: PROPOSAL_HASH,
          chosen: null,
          edited_values: {
            revision_instruction: "Add a deduplication transform before the output.",
          },
          custom_inputs: null,
          edit_target: null,
          control_signal: null,
          turn_token: sampleProposalTurn.turn_token,
          operation_id: expect.any(String),
        }),
      );
    });

    it("keeps a non-proposal step-3 send on /guided/chat", async () => {
      const { chatGuided, respondGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleChatResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        // Step 3, but the current turn is an ordinary chat-eligible turn (e.g. a
        // deferred-intent management surface), not the proposal review — the
        // revision gate must NOT intercept it.
        guidedSession: { ...sampleGuidedSession, step: "step_3_transforms" },
        guidedNextTurn: sampleNextTurn,
        compositionState: sampleCompositionState,
      });

      await useSessionStore.getState().chatGuided("Which columns exist?");

      expect(respondGuided).not.toHaveBeenCalled();
      expect(chatGuided).toHaveBeenCalledTimes(1);
    });

    it("applies all authoritative fields from a pure chat transition", async () => {
      const { chatGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
        sampleTransitioningChatResponse,
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
        compositionState: sampleCompositionState,
      });

      await useSessionStore.getState().chatGuided("make ten teal colour rows");

      const state = useSessionStore.getState();
      expect(state.guidedSession).toEqual(
        sampleTransitioningChatResponse.guided_session,
      );
      expect(state.guidedNextTurn).toEqual(
        sampleTransitioningChatResponse.next_turn,
      );
      expect(state.guidedTerminal).toEqual(
        sampleTransitioningChatResponse.terminal,
      );
      expect(state.compositionState).toEqual(
        sampleTransitioningChatResponse.composition_state,
      );
      expect(state.guidedChatPending).toBe(false);
    });

    it("replaces nullable authoritative fields instead of retaining stale local values", async () => {
      const { chatGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ...sampleChatResponse,
        next_turn: null,
        terminal: null,
        composition_state: null,
      });
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
        guidedTerminal: sampleTerminal,
        compositionState: sampleCompositionState,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      const state = useSessionStore.getState();
      expect(state.guidedNextTurn).toBeNull();
      expect(state.guidedTerminal).toBeNull();
      expect(state.compositionState).toBeNull();
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const chatPromise = useSessionStore.getState().chatGuided("What columns are available?");

      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_B,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      const state = useSessionStore.getState();
      expect(state.error).toBe("Failed to send chat message. Please try again.");
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedSession).toEqual(sampleGuidedSession);
    });

    it("reuses the operation id for an ambiguous retry of the same token and message", async () => {
      const { chatGuided } = await import("@/api/client");
      (chatGuided as ReturnType<typeof vi.fn>)
        .mockRejectedValueOnce(new TypeError("network failed after send"))
        .mockResolvedValueOnce(sampleChatResponse);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");
      await useSessionStore.getState().chatGuided("What columns are available?");

      const requests = (chatGuided as ReturnType<typeof vi.fn>).mock.calls;
      expect(requests).toHaveLength(2);
      expect(requests[1][1].operation_id).toBe(requests[0][1].operation_id);
      expect(requests[1][1].turn_token).toBe(sampleNextTurn.turn_token);
    });

    it("reports a conflicting chat message without rejecting or sending another request", async () => {
      const { chatGuided } = await import("@/api/client");
      const chatMock = chatGuided as ReturnType<typeof vi.fn>;
      chatMock.mockRejectedValue(new TypeError("response lost"));
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().chatGuided("inspect columns");
      await expect(
        useSessionStore.getState().chatGuided("change the source"),
      ).resolves.toBeUndefined();

      expect(chatMock).toHaveBeenCalledTimes(1);
      expect(useSessionStore.getState().guidedChatPending).toBe(false);
      expect(useSessionStore.getState().error).toMatch(/unsettled/i);

      await useSessionStore.getState().chatGuided("inspect columns");
      expect(chatMock).toHaveBeenCalledTimes(2);
      expect(chatMock.mock.calls[1]?.[1]).toEqual(chatMock.mock.calls[0]?.[1]);
    });

    it("reloads current guided state on stale 409 without retrying the old token", async () => {
      const { chatGuided, getGuided } = await import("@/api/client");
      const detail = "turn_token does not identify the current unanswered turn.";
      (chatGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce({
        status: 409,
        detail,
      });
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
        ...sampleGetGuidedResponse,
        next_turn: { ...sampleNextTurn, turn_token: "c".repeat(64) },
      });
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      const state = useSessionStore.getState();
      expect(chatGuided).toHaveBeenCalledTimes(1);
      expect(getGuided).toHaveBeenCalledWith(RETRY_SESSION_ID);
      expect(state.error).toBe(detail);
      expect(state.guidedChatPending).toBe(false);
      expect(state.guidedNextTurn?.turn_token).toBe("c".repeat(64));
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
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
            assistant_message_kind: null,
            synthetic_failure_reason: null,
          },
          {
            role: "assistant",
            content: "Partial reply persisted before the cancel.",
            seq: 1,
            step: "step_1_source",
            ts_iso: "2026-07-11T00:00:01Z",
            assistant_message_kind: "assistant",
            synthetic_failure_reason: null,
          },
        ],
      };
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValue({
        guided_session: durableGuided,
        next_turn: sampleNextTurn,
        terminal: null,
        composition_state: sampleCompositionState,
      });
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const controller = new AbortController();
      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?", controller.signal);
      controller.abort("compose_user_cancel");
      await chatPromise;

      const state = useSessionStore.getState();
      expect(getGuided).toHaveBeenCalledWith(RETRY_SESSION_ID);
      expect(state.guidedSession).toEqual(durableGuided);
      expect(state.guidedNextTurn).toEqual(sampleNextTurn);
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().chatGuided("What columns are available?");

      expect(fetchComposerProgress).toHaveBeenCalledWith(RETRY_SESSION_ID);
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const chatPromise = useSessionStore
        .getState()
        .chatGuided("What columns are available?");

      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_B,
        guidedSession: null,
        guidedChatPending: false,
      });
      resolveChat(sampleChatResponse);
      await chatPromise;

      const calledSessions = (
        fetchComposerProgress as ReturnType<typeof vi.fn>
      ).mock.calls.map((call) => call[0]);
      expect(calledSessions.length).toBeGreaterThan(0);
      expect(calledSessions.every((id) => id === RETRY_SESSION_ID)).toBe(true);
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
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
      const { getGuided, respondGuided } = await import("@/api/client");
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
      (getGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleGetGuidedResponse);
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["confirm"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
        control_signal: null,
      });

      const state = useSessionStore.getState();
      expect(state.error).toMatch(/can't be confirmed yet/);
      expect(state.errorDetails).toEqual([
        "pipeline: No sinks configured.",
        "node:rater: Missing input.",
      ]);
      expect(getGuided).not.toHaveBeenCalled();
      expect(state.guidedResponsePending).toBe(false);
    });

    it("falls back to the generic message when the failure carries no detail", async () => {
      const { respondGuided } = await import("@/api/client");
      (respondGuided as ReturnType<typeof vi.fn>).mockRejectedValueOnce(
        new Error("network down"),
      );
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["confirm"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      await useSessionStore.getState().respondGuided({
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const body: GuidedRespondAction = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
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
        activeSessionId: RETRY_SESSION_ID,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const body: GuidedRespondAction = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
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
      (chatGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(sampleChatResponse);
      // Fresh session id: the self-heal budget counter is module-scoped and
      // other tests here consume "sess-1"'s budget, which would skip the
      // self-heal and never set the notice this test needs.
      useSessionStore.setState({
        activeSessionId: RETRY_SESSION_B,
        guidedSession: sampleGuidedSession,
        guidedNextTurn: sampleNextTurn,
      });

      const body: GuidedRespondAction = {
        chosen: ["csv"],
        edited_values: null,
        custom_inputs: null,
        proposal_id: null,
        draft_hash: null,
        edit_target: null,
        control_signal: null,
      };

      await useSessionStore.getState().respondGuided(body); // sets the notice
      expect(useSessionStore.getState().guidedSelfHealNotice).not.toBeNull();

      // The user sends chat instead of re-submitting the turn; a
      // successful chat must not leave the "we've refreshed — try again" notice
      // pinned above it.
      await useSessionStore.getState().chatGuided("What columns are available?");
      expect(useSessionStore.getState().guidedSelfHealNotice).toBeNull();
    });
  });
});

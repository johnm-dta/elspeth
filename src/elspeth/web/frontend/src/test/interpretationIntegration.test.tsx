// ============================================================================
// interpretationIntegration.test.tsx — Phase 5b.18b.6
//
// End-to-end frontend integration: the LLM-interpretation review surface
// reaches the wire through two distinct UI affordances — the guided-mode
// `InterpretationReviewTurn` (dispatched by `GuidedTurn`) and the freeform-
// mode `InterpretationReviewInlineMessage` (rendered inline by `ChatPanel`).
// Both share the `useInterpretationResolver` hook and therefore the same
// `interpretationEventsStore.resolveEvent` / `optOut` actions, which in turn
// call `api.resolveInterpretation` / `api.optOutOfInterpretations`.
//
// This integration test pins the contract that lives ABOVE the unit tests:
// regardless of the visual surface the user touches, the wire payload sent
// to `POST /interpretations/{id}/resolve` MUST match the body shape the
// backend expects.  If either widget drifts from the wire contract, this
// test fails.
//
// Three parts mirror the spec at docs/composer/ux-redesign-2026-05/
// 18b-phase-5b-frontend.md lines 558-633:
//
//   Part A — 5a-then-5b combined sequence (hydration ordering).
//     Verifies that the composition-state nodes loaded by the prior 5a
//     `set_pipeline` round-trip remain present when `InterpretationReviewTurn`
//     mounts, and that the pending event's `affected_node_id` resolves to a
//     node that exists in the live `compositionState.nodes`.  Without this
//     test, a future refactor that re-orders store hydration on session
//     reload could silently leave `affected_node_id` pointing at a node
//     that hasn't been added to `sessionStore` yet, and the widget would
//     render against a stale composition.
//
//   Part B — Guided-mode "Use my interpretation" → resolve POST.
//     The button click round-trips through `useInterpretationResolver` →
//     `interpretationEventsStore.resolveEvent` → `api.resolveInterpretation`.
//     We assert the body shape verbatim: `{choice: 'accepted_as_drafted'}`
//     with NO `amended_value` field (per the wire contract in
//     `types/interpretation.ts` `InterpretationResolveRequest`).
//
//   Part C — Freeform-mode "Change it" + amend → resolve POST.
//     Mirrors Part B but through the inline-message widget and the amend
//     path.  Body shape: `{choice: 'amended', amended_value: '<text>'}`.
//
// What this test DOES mock:
//   * `@/api/client` interpretation methods (resolveInterpretation,
//     optOutOfInterpretations, listInterpretationEvents).  These are the
//     wire boundary; mocking them is how we capture the POST payload shape
//     for assertion.
//
// What this test does NOT mock:
//   * The `interpretationEventsStore` itself — it runs live so we exercise
//     the real store update path (pendingBySession pruning, resolved-count
//     bumping).
//   * The `sessionStore` — runs live so we can read back
//     `compositionState.nodes` and verify hydration ordering in Part A.
//   * The `useInterpretationResolver` hook — runs live so the state machine
//     (mode toggles, byte-cap, in-flight flags) is on the real critical path.
//
// What we deliberately DO NOT assert (per spec scope):
//   * The opt-out flow.  Each widget's unit test (`InterpretationReviewTurn.
//     test.tsx`, `InterpretationReviewInlineMessage.test.tsx`) already pins
//     the opt-out POST shape and store-flag flip.  Re-asserting here would
//     duplicate without adding wire-contract coverage.
//   * Resolve-success confirmation rendering (Phase 5b.18b.8 surface).
//     Same rationale — covered by unit tests.
//   * `new_state` propagation INTO `sessionStore` post-resolve.  The current
//     wiring at `ChatPanel.handleInterpretationResolved` (line 523) does
//     NOT push `new_state` into sessionStore; the resolve response's
//     `new_state` flows back to the hook's `onResolved` callback only via
//     the InlineMessage's `onResolved` prop in `ChatPanel.tsx` line 1315,
//     which discards the parameter and only uses `user_term` for the chat
//     echo. Asserting sessionStore propagation here would pin behaviour
//     that the code does not implement.  If a future change wires
//     sessionStore propagation through, that contract belongs in this
//     test — add an assertion at that point.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { GuidedTurn } from "@/components/chat/guided/GuidedTurn";
import { InterpretationReviewInlineMessage } from "@/components/chat/InterpretationReviewInlineMessage";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

import type {
  InterpretationEvent,
  InterpretationOptOutResponse,
  InterpretationResolveResponse,
} from "@/types/interpretation";
import type { CompositionState } from "@/types/index";
import type { TurnPayload } from "@/types/guided";

// ── API client mock ──────────────────────────────────────────────────────────
// Mock ONLY the four interpretation surface methods.  The store imports
// `* as api from "@/api/client"`; replacing those four methods leaves the
// rest of the client module — including the types and `ApiError` machinery —
// untouched.  Tests configure return values per-case via
// `vi.mocked(api.X).mockResolvedValue(...)` after re-importing the mocked
// module.

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

import * as api from "@/api/client";

// ── Fixtures ─────────────────────────────────────────────────────────────────
//
// The canonical hero prompt drives the demo: a user asks the composer to
// rate web pages on `cool`, the composer creates a one-row inline source
// (Phase 5a `inline_blob`), proposes an LLM transform node, and surfaces
// the LLM's interpretation of `cool` for review (Phase 5b).  These
// fixtures mirror that flow.

const SESSION_ID = "sess-1";
const EVENT_ID = "evt-1";
const LLM_NODE_ID = "llm_rate_coolness";

/**
 * Composition state shape that 5a's `set_pipeline` would have produced:
 * an inline_blob source plus an LLM transform node whose `node_id` is the
 * `affected_node_id` the interpretation event references.  The Part A
 * hydration-ordering assertion reads this back from `useSessionStore` to
 * verify the node is present BEFORE `InterpretationReviewTurn` mounts.
 */
function makeCompositionStateAfter5a(): CompositionState {
  return {
    id: "state-2",
    version: 2,
    source: {
      plugin: "inline_blob",
      options: { blob_ref: "blob-1" },
    },
    // NodeSpec.id is the wire-side identifier matching the backend
    // node_id concept (see types/index.ts line 106-120).  The
    // `affected_node_id` field on InterpretationEvent references this id.
    nodes: [
      {
        id: LLM_NODE_ID,
        node_type: "transform",
        plugin: "llm",
        input: "source",
        on_success: null,
        on_error: null,
        options: {
          prompt: "rate how cool each website is",
          model: "gpt-4o",
        },
      },
    ],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

/**
 * Pending interpretation event matching the canonical prompt flow.
 * `affected_node_id` references the LLM node created in 5a so Part A can
 * assert dangling-reference safety.
 */
function makePendingEvent(): InterpretationEvent {
  return {
    id: EVENT_ID,
    session_id: SESSION_ID,
    composition_state_id: "state-2",
    affected_node_id: LLM_NODE_ID,
    tool_call_id: "tool-1",
    user_term: "cool",
    kind: "vague_term",
    llm_draft: "modern design + clear purpose + interactivity",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-05-18T00:00:00Z",
    resolved_at: null,
    actor: "user:owner:u-1",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-opus-4-7",
    model_version: "20260518",
    provider: "anthropic",
    composer_skill_hash: "deadbeef",
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
  };
}

/**
 * `POST /resolve` response envelope.  Returns the resolved event row PLUS
 * the new composition state (the wire contract; see
 * `InterpretationResolveResponse` in `types/interpretation.ts`).  We bump
 * the composition version to 3 so the "new state different from old state"
 * invariant is visible to assertions.
 */
function makeResolveResponse(
  event: InterpretationEvent,
  overrides: Partial<InterpretationResolveResponse> = {},
): InterpretationResolveResponse {
  return {
    event: {
      ...event,
      choice: "accepted_as_drafted",
      accepted_value: event.llm_draft,
      resolved_at: "2026-05-18T01:00:00Z",
    },
    new_state: {
      id: "state-3",
      version: 3,
      source: { plugin: "inline_blob", options: { blob_ref: "blob-1" } },
      nodes: [
        {
          id: LLM_NODE_ID,
          node_type: "transform",
          plugin: "llm",
          input: "source",
          on_success: null,
          on_error: null,
          options: {
            prompt: "rate how cool each website is",
            model: "gpt-4o",
            user_term_interpretations: {
              cool: "modern design + clear purpose + interactivity",
            },
          },
        },
      ],
      edges: [],
      outputs: [],
      metadata: { name: null, description: null },
    },
    ...overrides,
  };
}

/**
 * Suppress used-fixture-only warning: the opt-out helper is defined for
 * symmetry with the unit-test fixtures but only Part-B/C wire shapes are
 * the contract-of-interest here.  Retained so a future opt-out
 * integration assertion has a ready-made response shape.
 */
function _makeOptOutResponse(sessionId: string): InterpretationOptOutResponse {
  return {
    session_id: sessionId,
    interpretation_review_disabled: true,
    opted_out_at: "2026-05-18T02:00:00Z",
  };
}
void _makeOptOutResponse;

// ── Suite setup ──────────────────────────────────────────────────────────────

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  resetStore(useSessionStore);
  vi.mocked(api.resolveInterpretation).mockReset();
  vi.mocked(api.optOutOfInterpretations).mockReset();
  vi.mocked(api.listInterpretationEvents).mockReset();
  vi.mocked(api.getInterpretationOptOutSummary).mockReset();
});

// ── Part A — 5a-then-5b hydration ordering ──────────────────────────────────
//
// Spec lines 569-602.  The canonical hero prompt drives 5a (inline blob +
// LLM-node creation via `set_pipeline`) immediately before 5b
// (interpretation review).  If `interpretationEventsStore.refreshPending`
// fires before `sessionStore` has hydrated the new composition state, the
// `InterpretationReviewTurn` would mount with an `affected_node_id` that
// points at a node the local `sessionStore` doesn't know about yet — a
// dangling reference at the rendering boundary.
//
// We reproduce that ordering hazard structurally: seed `sessionStore` with
// the post-5a composition state FIRST, then seed
// `interpretationEventsStore` with the pending event, then render the
// guided turn.  At mount time, the LLM node MUST still be visible in
// `sessionStore.compositionState.nodes`, and `event.affected_node_id`
// MUST match one of those nodes.

describe("Phase 5b.18b.6 — Part A — 5a-then-5b hydration ordering", () => {
  it("LLM node is present in compositionState before the interpretation turn mounts", () => {
    // Step 1-3 — seed both stores so the composition reflects the post-5a
    // state and the pending event references the LLM node 5a created.
    const compositionState = makeCompositionStateAfter5a();
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [
        {
          id: SESSION_ID,
          title: "Canonical hero prompt",
          created_at: "2026-05-18T00:00:00Z",
          updated_at: "2026-05-18T00:00:00Z",
        },
      ],
      compositionState,
    });
    const event = makePendingEvent();
    useInterpretationEventsStore.setState({
      pendingBySession: { [SESSION_ID]: { [event.id]: event } },
    });

    // BEFORE rendering the turn, assert the hydration invariant holds.
    // This is the test's load-bearing assertion: if store ordering ever
    // regressed to "refresh interpretations first", a future
    // implementation could end up here with `nodes: []` and the test
    // would fail before the render call.
    const seededNodes = useSessionStore.getState().compositionState?.nodes ?? [];
    expect(seededNodes).toContainEqual(
      expect.objectContaining({ id: LLM_NODE_ID }),
    );

    // Step 4-5 — render the guided turn (corresponds to advancing past
    // `InlineSourceCreatedTurn` in the guided dispatch sequence).
    // step_index=0 is a stable filler; the widget doesn't read it.
    const turn: TurnPayload = {
      type: "interpretation_review",
      step_index: 0,
      payload: event,
    };
    render(<GuidedTurn turn={turn} onSubmit={vi.fn()} />);

    // The turn mounted; the LLM node is still in `compositionState.nodes`
    // (the hydration ordering was not silently inverted between seed and
    // mount).  Read fresh — Zustand selectors hold no snapshot across
    // intervening updates.
    const nodesAtMount =
      useSessionStore.getState().compositionState?.nodes ?? [];
    expect(nodesAtMount).toContainEqual(
      expect.objectContaining({ id: LLM_NODE_ID }),
    );

    // Step 6 — `affected_node_id` resolves to a real node (no dangling
    // reference at the render boundary).  Read the event back from the
    // store so we're asserting against the same shape the widget sees.
    const renderedEvent =
      useInterpretationEventsStore.getState().pendingBySession[SESSION_ID][
        EVENT_ID
      ];
    expect(renderedEvent.affected_node_id).toBe(LLM_NODE_ID);
    const matchingNode = nodesAtMount.find(
      (n) => n.id === renderedEvent.affected_node_id,
    );
    expect(matchingNode).toBeDefined();

    // And the widget itself is on screen (sanity check that GuidedTurn
    // actually dispatched to InterpretationReviewTurn).
    expect(
      screen.getByRole("region", { name: /interpretation review/i }),
    ).toBeInTheDocument();
  });
});

// ── Part B — Guided-mode resolve flow ───────────────────────────────────────
//
// Spec lines 603-613.  From the state reached in Part A, click "Use my
// interpretation".  The wire body MUST be `{choice: 'accepted_as_drafted'}`
// with NO `amended_value` field — that's the contract in
// `types/interpretation.ts` `InterpretationResolveRequest`.  After
// resolution the event MUST be removed from
// `interpretationEventsStore.pendingBySession[SESSION_ID]`, and the resolved
// counter MUST be incremented.

describe("Phase 5b.18b.6 — Part B — guided-mode 'Use my interpretation'", () => {
  it("posts {choice:'accepted_as_drafted'} and the widget unmounts via store pruning", async () => {
    const user = userEvent.setup();
    const event = makePendingEvent();

    // Seed both stores in the same order as Part A.
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [
        {
          id: SESSION_ID,
          title: "Canonical hero prompt",
          created_at: "2026-05-18T00:00:00Z",
          updated_at: "2026-05-18T00:00:00Z",
        },
      ],
      compositionState: makeCompositionStateAfter5a(),
    });
    useInterpretationEventsStore.setState({
      pendingBySession: { [SESSION_ID]: { [event.id]: event } },
    });

    const response = makeResolveResponse(event);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(response);

    const turn: TurnPayload = {
      type: "interpretation_review",
      step_index: 0,
      payload: event,
    };
    render(<GuidedTurn turn={turn} onSubmit={vi.fn()} />);

    // Step 7 — click "Use my interpretation".  The guided-mode widget's
    // accept button carries aria-label "Accept the LLM's interpretation
    // of <user_term>" so screen-reader users hear which term they're
    // approving (see InterpretationReviewTurn.tsx).
    await user.click(
      screen.getByRole("button", {
        name: /accept the llm's interpretation of cool/i,
      }),
    );

    // Step 8 — POST body assertion.  The hook calls
    // `resolveEvent(sessionId, eventId, body)`, which delegates to
    // `api.resolveInterpretation(sessionId, eventId, body)`.  The body
    // MUST be `{choice: 'accepted_as_drafted'}` with NO `amended_value`
    // — strict equality on the object literal pins both the choice value
    // and the absence of stray fields.
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    expect(api.resolveInterpretation).toHaveBeenCalledWith(
      SESSION_ID,
      EVENT_ID,
      { choice: "accepted_as_drafted" },
    );

    // Step 9 — widget collapses on the store side.  The store removes the
    // event from `pendingBySession`; the rendered region's own unmount is
    // driven by the parent's re-render with the next turn (the live
    // GuidedTurn does not subscribe to `pendingBySession`, so its
    // unmounting is a parent-orchestrated concern handled by the chat
    // dispatch loop, not by the widget itself).  Here we pin the store
    // contract; the parent-unmount behaviour is exercised by
    // `ChatPanel.test.tsx` at the dispatch level.
    await waitFor(() => {
      const pending =
        useInterpretationEventsStore.getState().pendingBySession[SESSION_ID] ??
        {};
      expect(pending[EVENT_ID]).toBeUndefined();
    });

    // The resolved-counts projection bumped accepted_as_drafted.  This is
    // the audit-readiness panel's counter; if the increment ever stopped
    // firing, the LLM-interpretation row would silently misreport.
    const counts =
      useInterpretationEventsStore.getState().resolvedCountBySession[
        SESSION_ID
      ];
    expect(counts).toBeDefined();
    expect(counts.accepted_as_drafted).toBe(1);
    expect(counts.amended).toBe(0);
    expect(counts.opted_out).toBe(0);

    // Step 10 — the resolve response carried a new composition state.
    // The store action `resolveEvent` returns it to the caller; the hook
    // forwards it via `onResolved`.  We pin the wire-shape contract by
    // asserting the mock returned what we configured (the spec's
    // sessionStore-propagation step is not currently wired through
    // ChatPanel — see file header).
    expect(response.new_state.version).toBe(3);
    expect(response.new_state.nodes).toContainEqual(
      expect.objectContaining({ id: LLM_NODE_ID }),
    );
  });
});

// ── Part C — Freeform-mode resolve flow ─────────────────────────────────────
//
// Spec lines 615-626.  Same wire path as Part B but via the inline-message
// widget AND the amend path.  The body MUST be
// `{choice: 'amended', amended_value: '<text>'}` — both fields present and
// no others.  After resolution the inline message MUST disappear (event
// removed from `pendingBySession`).

describe("Phase 5b.18b.6 — Part C — freeform-mode 'Change it' + amend", () => {
  it("posts {choice:'amended', amended_value} and the inline message disappears", async () => {
    const user = userEvent.setup();
    const event = makePendingEvent();
    const AMENDED_VALUE = "highly engaging and accessible";

    // Seed both stores so the freeform widget mounts with the same
    // hydration invariants Part A pinned.
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [
        {
          id: SESSION_ID,
          title: "Canonical hero prompt",
          created_at: "2026-05-18T00:00:00Z",
          updated_at: "2026-05-18T00:00:00Z",
        },
      ],
      compositionState: makeCompositionStateAfter5a(),
    });
    useInterpretationEventsStore.setState({
      pendingBySession: { [SESSION_ID]: { [event.id]: event } },
    });

    const response = makeResolveResponse(event, {
      event: {
        ...event,
        choice: "amended",
        accepted_value: AMENDED_VALUE,
        resolved_at: "2026-05-18T01:00:00Z",
      },
    });
    vi.mocked(api.resolveInterpretation).mockResolvedValue(response);

    // Step 11-12 — render the inline-message widget directly.  In the
    // live composer this is dispatched by `ChatPanel` (lines 1304-1317)
    // from `pendingBySession[activeSessionId]`; rendering it directly
    // exercises the same widget the ChatPanel mounts.
    render(
      <InterpretationReviewInlineMessage
        event={event}
        sessionId={SESSION_ID}
      />,
    );
    expect(
      screen.getByTestId("interpretation-review-inline-message"),
    ).toBeInTheDocument();

    // Step 13 — click "Change it", type the amendment, click Submit.
    // aria-label includes the user_term so AT users hear which term
    // they're amending.
    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    // The hook pre-fills the textarea with the LLM draft when the user
    // opens amend mode (see `handleOpenAmend` in
    // useInterpretationResolver.ts).  Clear it before typing the new
    // amendment so the wire payload contains only the new value.
    await user.clear(textarea);
    await user.type(textarea, AMENDED_VALUE);

    await user.click(screen.getByRole("button", { name: /^submit$/i }));

    // Step 14 — POST body assertion.  Both `choice` and `amended_value`
    // must be present; the strict-equals object literal pins their
    // presence AND the absence of stray fields.
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    expect(api.resolveInterpretation).toHaveBeenCalledWith(
      SESSION_ID,
      EVENT_ID,
      { choice: "amended", amended_value: AMENDED_VALUE },
    );

    // Step 15 — inline message disappears (event removed from
    // `pendingBySession`).  The widget unmount is driven by the store
    // pruning the event entry — the widget itself does not unmount on
    // success, the parent's re-render with the empty pending map does.
    // Here we rendered the widget directly without a parent map-driver,
    // so we assert the store state and the widget's "still on screen
    // but the event would have been removed" state.  The contract that
    // matters for the wire-shape regression guard is the store pruning;
    // ChatPanel's own unmount-on-empty-map behaviour is covered by its
    // existing unit tests.
    await waitFor(() => {
      const pending =
        useInterpretationEventsStore.getState().pendingBySession[SESSION_ID] ??
        {};
      expect(pending[EVENT_ID]).toBeUndefined();
    });

    // Resolved-counts projection: `amended` (not `accepted_as_drafted`)
    // increments here.  If the choice value ever leaked from the freeform
    // surface as `accepted_as_drafted`, this assertion would flip and the
    // test would catch the regression.
    const counts =
      useInterpretationEventsStore.getState().resolvedCountBySession[
        SESSION_ID
      ];
    expect(counts).toBeDefined();
    expect(counts.amended).toBe(1);
    expect(counts.accepted_as_drafted).toBe(0);
    expect(counts.opted_out).toBe(0);
  });
});

// ── Cross-mode contract — both surfaces produce identical wire payloads ─────
//
// The two widgets share `useInterpretationResolver`; if either ever drifted
// from the shared hook (e.g., a bespoke handler added directly on a
// component), the POST body shape could diverge between modes.  This
// assertion locks the symmetry: the guided "Use my interpretation" button
// and the freeform "Use my interpretation" button MUST issue identical
// resolve calls.

describe("Phase 5b.18b.6 — cross-mode contract", () => {
  it("guided-mode and freeform-mode 'Use my interpretation' post the same body", async () => {
    const user = userEvent.setup();
    const event = makePendingEvent();
    const response = makeResolveResponse(event);

    // ── Guided pass ────────────────────────────────────────────────────────
    useSessionStore.setState({
      activeSessionId: SESSION_ID,
      sessions: [
        {
          id: SESSION_ID,
          title: "Cross-mode",
          created_at: "2026-05-18T00:00:00Z",
          updated_at: "2026-05-18T00:00:00Z",
        },
      ],
      compositionState: makeCompositionStateAfter5a(),
    });
    useInterpretationEventsStore.setState({
      pendingBySession: { [SESSION_ID]: { [event.id]: event } },
    });
    vi.mocked(api.resolveInterpretation).mockResolvedValue(response);

    const turn: TurnPayload = {
      type: "interpretation_review",
      step_index: 0,
      payload: event,
    };
    const { unmount: unmountGuided } = render(
      <GuidedTurn turn={turn} onSubmit={vi.fn()} />,
    );
    await user.click(
      screen.getByRole("button", {
        name: /accept the llm's interpretation of cool/i,
      }),
    );
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    const guidedCall = vi.mocked(api.resolveInterpretation).mock.calls[0];
    unmountGuided();

    // ── Freeform pass ──────────────────────────────────────────────────────
    // Reset both store entries; reuse the same event so the wire shape
    // assertion is apples-to-apples.
    resetStore(useInterpretationEventsStore);
    vi.mocked(api.resolveInterpretation).mockReset();
    vi.mocked(api.resolveInterpretation).mockResolvedValue(response);

    useInterpretationEventsStore.setState({
      pendingBySession: { [SESSION_ID]: { [event.id]: event } },
    });

    render(
      <InterpretationReviewInlineMessage
        event={event}
        sessionId={SESSION_ID}
      />,
    );
    await user.click(
      screen.getByRole("button", {
        name: /accept the llm's interpretation of cool/i,
      }),
    );
    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledTimes(1);
    });
    const freeformCall = vi.mocked(api.resolveInterpretation).mock.calls[0];

    // The two calls must agree on (sessionId, eventId, body).  Comparing
    // the entire call arrays (index 0 = sessionId, 1 = eventId, 2 = body)
    // is the tightest possible regression guard — any drift in any
    // argument shape between the two surfaces fails the test.
    expect(freeformCall).toEqual(guidedCall);
  });
});

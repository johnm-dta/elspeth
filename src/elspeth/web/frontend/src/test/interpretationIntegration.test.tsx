// ============================================================================
// interpretationIntegration.test.tsx — Phase 5b.18b.6 (acknowledge-stack era)
//
// End-to-end frontend integration: the LLM-interpretation review surface
// reaches the wire through ONE unified affordance — the AcknowledgementStack
// (pinned at the top of the chat column in both guided and freeform modes),
// which renders an AcknowledgementCard per pending event.  Both modes consume
// the same `useInterpretationResolver` hook and therefore the same
// `interpretationEventsStore.resolveEvent` / `optOut` actions, which call
// `api.resolveInterpretation` / `api.optOutOfInterpretations`.
//
// This integration test pins the contract ABOVE the unit tests: whatever the
// operator touches, the wire payload sent to `POST /interpretations/{id}/
// resolve` MUST match the body shape the backend expects.
//
//   Part A — hydration ordering: the post-5a composition nodes remain present
//     when the card mounts, and the event's affected_node_id resolves to a
//     live node (no dangling reference; humanised step label, not the raw id).
//   Part B — Acknowledge → `{choice: 'accepted_as_drafted'}` (no amended_value).
//   Part C — Change… + amend → `{choice: 'amended', amended_value: '<text>'}`.
//
// Mocked: the four `@/api/client` interpretation methods (the wire boundary).
// Live: interpretationEventsStore, sessionStore, useInterpretationResolver.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AcknowledgementStack } from "@/components/chat/AcknowledgementStack";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";

import type {
  InterpretationEvent,
  InterpretationResolveResponse,
} from "@/types/interpretation";
import type { CompositionState } from "@/types/index";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

import * as api from "@/api/client";

// ── Fixtures ─────────────────────────────────────────────────────────────────

const SESSION_ID = "sess-1";
const EVENT_ID = "evt-1";
const LLM_NODE_ID = "llm_rate_coolness";

function makeCompositionStateAfter5a(): CompositionState {
  return {
    id: "state-2",
    ...compositionStateAuthorityFields,
    version: 2,
    sources: {
      source: { plugin: "inline_blob", options: { blob_ref: "blob-1" } },
    },
    nodes: [
      {
        id: LLM_NODE_ID,
        node_type: "transform",
        plugin: "llm",
        input: "source",
        on_success: null,
        on_error: null,
        options: { prompt: "rate how cool each website is", model: "gpt-4o" },
      },
    ],
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

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
      ...makeCompositionStateAfter5a(),
      id: "state-3",
      version: 3,
    },
    ...overrides,
  };
}

function seed(event: InterpretationEvent): void {
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
}

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  resetStore(useSessionStore);
  vi.mocked(api.resolveInterpretation).mockReset();
  vi.mocked(api.optOutOfInterpretations).mockReset();
  vi.mocked(api.listInterpretationEvents).mockReset();
  vi.mocked(api.getInterpretationOptOutSummary).mockReset();
});

// ── Part A — hydration ordering ─────────────────────────────────────────────

describe("Phase 5b.18b.6 — Part A — 5a-then-5b hydration ordering", () => {
  it("the LLM node is present and resolvable when the acknowledgement card mounts", () => {
    const event = makePendingEvent();
    seed(event);

    const seededNodes =
      useSessionStore.getState().compositionState?.nodes ?? [];
    expect(seededNodes).toContainEqual(
      expect.objectContaining({ id: LLM_NODE_ID }),
    );

    render(<AcknowledgementStack sessionId={SESSION_ID} />);

    const nodesAtMount =
      useSessionStore.getState().compositionState?.nodes ?? [];
    const renderedEvent =
      useInterpretationEventsStore.getState().pendingBySession[SESSION_ID][
        EVENT_ID
      ];
    expect(renderedEvent.affected_node_id).toBe(LLM_NODE_ID);
    expect(
      nodesAtMount.find((n) => n.id === renderedEvent.affected_node_id),
    ).toBeDefined();

    // The card mounted (one decision in the stack).
    expect(screen.getByTestId("acknowledgement-card")).toBeInTheDocument();
  });

  it("humanises affected_node_id to the node's plugin label, not the raw id", () => {
    const event = makePendingEvent();
    // A model-choice card surfaces the step label in its title row.
    seed({ ...event, kind: "llm_model_choice", llm_draft: "gpt-4o" });
    render(<AcknowledgementStack sessionId={SESSION_ID} />);
    // llm plugin → "Summarise"; the raw node id must NOT leak into copy.
    expect(screen.getByText("Summarise step · model")).toBeInTheDocument();
    expect(screen.queryByText(new RegExp(LLM_NODE_ID))).toBeNull();
  });
});

// ── Part B — Acknowledge resolve flow ───────────────────────────────────────

describe("Phase 5b.18b.6 — Part B — Acknowledge", () => {
  it("posts {choice:'accepted_as_drafted'} and the store prunes the pending event", async () => {
    const user = userEvent.setup();
    const event = makePendingEvent();
    seed(event);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event),
    );

    render(<AcknowledgementStack sessionId={SESSION_ID} />);

    await user.click(
      screen.getByRole("button", {
        name: /acknowledge the llm's interpretation of cool/i,
      }),
    );

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledWith(
        SESSION_ID,
        EVENT_ID,
        { choice: "accepted_as_drafted" },
      );
    });

    await waitFor(() => {
      const pending =
        useInterpretationEventsStore.getState().pendingBySession[SESSION_ID] ??
        {};
      expect(pending[EVENT_ID]).toBeUndefined();
    });

    const counts =
      useInterpretationEventsStore.getState().resolvedCountBySession[
        SESSION_ID
      ];
    expect(counts.accepted_as_drafted).toBe(1);
    expect(counts.amended).toBe(0);
    expect(counts.opted_out).toBe(0);
  });
});

// ── Part C — amend resolve flow ─────────────────────────────────────────────

describe("Phase 5b.18b.6 — Part C — Change… + amend", () => {
  it("posts {choice:'amended', amended_value} and the store prunes the pending event", async () => {
    const user = userEvent.setup();
    const event = makePendingEvent();
    const AMENDED_VALUE = "highly engaging and accessible";
    seed(event);
    vi.mocked(api.resolveInterpretation).mockResolvedValue(
      makeResolveResponse(event, {
        event: {
          ...event,
          choice: "amended",
          accepted_value: AMENDED_VALUE,
          resolved_at: "2026-05-18T01:00:00Z",
        },
      }),
    );

    render(<AcknowledgementStack sessionId={SESSION_ID} />);

    await user.click(
      screen.getByRole("button", { name: /edit the interpretation of cool/i }),
    );
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);
    await user.type(textarea, AMENDED_VALUE);
    await user.click(screen.getByRole("button", { name: /^submit$/i }));

    await waitFor(() => {
      expect(api.resolveInterpretation).toHaveBeenCalledWith(
        SESSION_ID,
        EVENT_ID,
        { choice: "amended", amended_value: AMENDED_VALUE },
      );
    });

    await waitFor(() => {
      const pending =
        useInterpretationEventsStore.getState().pendingBySession[SESSION_ID] ??
        {};
      expect(pending[EVENT_ID]).toBeUndefined();
    });

    const counts =
      useInterpretationEventsStore.getState().resolvedCountBySession[
        SESSION_ID
      ];
    expect(counts.amended).toBe(1);
    expect(counts.accepted_as_drafted).toBe(0);
    expect(counts.opted_out).toBe(0);
  });
});

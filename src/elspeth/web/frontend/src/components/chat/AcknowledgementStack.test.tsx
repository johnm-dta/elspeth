// ============================================================================
// AcknowledgementStack — stack-level behavioural coverage.
//
// Ports the retired GuidedInterpretationReviews tests (one card per pending
// event; renders nothing when empty) and adds: pipeline-step ordering, the
// count announce, no-focus-steal on mount, the foot-of-stack opt-out flow +
// error mapping, tutorial suppression, and a jest-axe a11y assertion.
// ============================================================================

import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe, toHaveNoViolations } from "jest-axe";
import { AcknowledgementLiveRegion, AcknowledgementStack } from "./AcknowledgementStack";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { useSessionStore } from "@/stores/sessionStore";
import { resetStore } from "@/test/store-helpers";
import type { InterpretationEvent } from "@/types/interpretation";
import type { CompositionState, NodeSpec } from "@/types/index";

expect.extend(toHaveNoViolations);

vi.mock("@/api/client", () => ({
  listInterpretationEvents: vi.fn(),
  resolveInterpretation: vi.fn(),
  optOutOfInterpretations: vi.fn(),
  getInterpretationOptOutSummary: vi.fn(),
}));

import * as api from "@/api/client";

const SID = "11111111-1111-1111-1111-111111111111";

function makeEvent(
  id: string,
  overrides: Partial<InterpretationEvent> = {},
): InterpretationEvent {
  return {
    id,
    session_id: SID,
    composition_state_id: "22222222-2222-2222-2222-222222222222",
    affected_node_id: "node-1",
    tool_call_id: "tool-1",
    user_term: "cool",
    kind: "vague_term",
    llm_draft: "interesting and engaging",
    accepted_value: null,
    choice: "pending",
    created_at: "2026-06-22T00:00:00Z",
    resolved_at: null,
    actor: "system:composer",
    interpretation_source: "user_approved",
    model_identifier: "anthropic/claude-opus-4-7",
    model_version: "20260518",
    provider: "anthropic",
    composer_skill_hash: "0".repeat(64),
    arguments_hash: null,
    hash_domain_version: null,
    runtime_model_identifier_at_resolve: null,
    runtime_model_version_at_resolve: null,
    resolved_prompt_template_hash: null,
    ...overrides,
  };
}

function makeNode(id: string, plugin: string): NodeSpec {
  return {
    id,
    node_type: "transform",
    plugin,
    input: "rows",
    on_success: null,
    on_error: null,
    options: {},
  };
}

function makeCompositionState(nodes: NodeSpec[]): CompositionState {
  return {
    id: "state-1",
    version: 1,
    sources: {},
    nodes,
    edges: [],
    outputs: [],
    metadata: { name: null, description: null },
  };
}

function seedPending(events: InterpretationEvent[]): void {
  const map: Record<string, InterpretationEvent> = {};
  for (const event of events) map[event.id] = event;
  useInterpretationEventsStore.setState({ pendingBySession: { [SID]: map } });
}

beforeEach(() => {
  resetStore(useInterpretationEventsStore);
  vi.clearAllMocks();
  useSessionStore.setState({ compositionState: null });
});

describe("AcknowledgementStack — projection", () => {
  it("renders one card per pending user_approved event with a count header", () => {
    seedPending([makeEvent("e1"), makeEvent("e2")]);
    render(<AcknowledgementStack sessionId={SID} />);

    expect(screen.getAllByTestId("acknowledgement-card")).toHaveLength(2);
    expect(
      screen.getByText("2 decisions the LLM made — acknowledge each"),
    ).toBeTruthy();
  });

  it("renders nothing when there are no pending events", () => {
    useInterpretationEventsStore.setState({ pendingBySession: { [SID]: {} } });
    const { container } = render(<AcknowledgementStack sessionId={SID} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("ignores opt-out / no-surface rows (only user_approved pending)", () => {
    seedPending([
      makeEvent("e1"),
      makeEvent("e2", { interpretation_source: "auto_interpreted_opt_out" }),
    ]);
    render(<AcknowledgementStack sessionId={SID} />);
    expect(screen.getAllByTestId("acknowledgement-card")).toHaveLength(1);
  });
});

describe("AcknowledgementStack — ordering", () => {
  it("orders cards by pipeline step, then created_at", () => {
    useSessionStore.setState({
      compositionState: makeCompositionState([
        makeNode("a", "llm"),
        makeNode("b", "web_scrape"),
      ]),
    });
    // e1 targets node b (later step) but was created earlier; e2 targets node a
    // (earlier step). Step order must win → a (Summarise) before b (Fetch).
    seedPending([
      makeEvent("e1", {
        affected_node_id: "b",
        kind: "llm_model_choice",
        created_at: "2026-06-20T00:00:00Z",
      }),
      makeEvent("e2", {
        affected_node_id: "a",
        kind: "llm_model_choice",
        created_at: "2026-06-22T00:00:00Z",
      }),
    ]);
    render(<AcknowledgementStack sessionId={SID} />);

    const titles = screen
      .getAllByTestId("acknowledgement-card")
      .map((card) => card.querySelector(".ack-card-title")?.textContent);
    expect(titles).toEqual(["Summarise step · model", "Fetch step · model"]);
  });
});

describe("AcknowledgementStack — accessibility", () => {
  it("does not host its own role='status' announce (the persistent AcknowledgementLiveRegion owns it)", () => {
    // The stack returns null when empty, so a live region mounted INSIDE it
    // would be inserted with its content already present on the 0->1
    // transition — the documented-unreliable polite-announce pattern. The
    // count announce is therefore the persistent AcknowledgementLiveRegion's
    // job (covered below). This guards against regressing to an in-stack status.
    seedPending([makeEvent("e1"), makeEvent("e2")]);
    render(<AcknowledgementStack sessionId={SID} />);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("does NOT move focus on mount (persistent stack must not yank focus)", () => {
    seedPending([makeEvent("e1")]);
    render(<AcknowledgementStack sessionId={SID} />);
    expect(document.activeElement).toBe(document.body);
  });

  it("has no axe violations", async () => {
    seedPending([makeEvent("e1")]);
    const { container } = render(<AcknowledgementStack sessionId={SID} />);
    expect(await axe(container)).toHaveNoViolations();
  });
});

describe("AcknowledgementLiveRegion — count announce", () => {
  it("carries the pending count in a role='status' region", () => {
    seedPending([makeEvent("e1"), makeEvent("e2")]);
    render(<AcknowledgementLiveRegion sessionId={SID} />);
    expect(screen.getByRole("status").textContent).toMatch(
      /2 decisions to acknowledge/i,
    );
  });

  it("uses the singular form for a single decision", () => {
    seedPending([makeEvent("e1")]);
    render(<AcknowledgementLiveRegion sessionId={SID} />);
    expect(screen.getByRole("status").textContent).toMatch(
      /1 decision to acknowledge/i,
    );
  });

  it("stays mounted with empty text when nothing is pending (so 0->1 announces)", () => {
    useInterpretationEventsStore.setState({ pendingBySession: { [SID]: {} } });
    render(<AcknowledgementLiveRegion sessionId={SID} />);
    const status = screen.getByRole("status");
    expect(status).toBeTruthy();
    expect((status.textContent ?? "").trim()).toBe("");
  });
});

describe("AcknowledgementStack — foot-of-stack opt-out", () => {
  it("renders exactly one opt-out link", () => {
    seedPending([makeEvent("e1"), makeEvent("e2")]);
    render(<AcknowledgementStack sessionId={SID} />);
    expect(
      screen.getAllByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    ).toHaveLength(1);
  });

  it("opens a session-scope confirm, then calls optOut and fires onResolved(null, null)", async () => {
    const user = userEvent.setup();
    vi.mocked(api.optOutOfInterpretations).mockResolvedValue({
      session_id: SID,
      interpretation_review_disabled: true,
      opted_out_at: "2026-06-22T01:00:00Z",
    });
    const onResolved = vi.fn();
    seedPending([makeEvent("e1")]);
    render(<AcknowledgementStack sessionId={SID} onResolved={onResolved} />);

    await user.click(
      screen.getByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    );
    const dialog = screen.getByRole("alertdialog");
    expect(dialog.textContent).toMatch(/this session/i);
    await user.click(
      screen.getByRole("button", { name: /stop reviewing for this session/i }),
    );

    await waitFor(() => {
      expect(api.optOutOfInterpretations).toHaveBeenCalledWith(SID);
    });
    expect(onResolved).toHaveBeenCalledWith(null, null);
  });

  it("maps an opt-out failure to an error banner", async () => {
    const user = userEvent.setup();
    vi.mocked(api.optOutOfInterpretations).mockRejectedValue({
      status: 500,
      detail: "boom",
    });
    seedPending([makeEvent("e1")]);
    render(<AcknowledgementStack sessionId={SID} />);

    await user.click(
      screen.getByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    );
    await user.click(
      screen.getByRole("button", { name: /stop reviewing for this session/i }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/could not resolve interpretation/i);
  });
});

describe("AcknowledgementStack — tutorial mode", () => {
  it("hides the amend escape hatch and the opt-out link", () => {
    seedPending([makeEvent("e1")]);
    render(<AcknowledgementStack sessionId={SID} isTutorial />);
    expect(
      screen.queryByRole("button", { name: /edit the interpretation/i }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", {
        name: /stop reviewing interpretations this session/i,
      }),
    ).toBeNull();
  });
});

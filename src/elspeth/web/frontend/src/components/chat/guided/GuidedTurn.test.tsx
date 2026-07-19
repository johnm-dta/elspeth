// ============================================================================
// GuidedTurn dispatcher -- routing contract regression coverage.
//
// Pins THREE contracts:
//   1. Turn-type routing correctness: each TurnPayload.type routes to
//      exactly the matching leaf widget, identifiable by a widget-specific
//      rendered element.
//   2. onSubmit forwarding: the dispatcher passes onSubmit through unchanged;
//      a widget event fires the SAME fn reference the dispatcher received.
//   3. Exhaustiveness compile-time check: the `const _exhaustive: never`
//      pattern in the default case fails to compile if a new TurnType is
//      added to guided.ts without a matching switch arm -- verified by the
//      type-system, no runtime test needed (the default throw tests the
//      runtime guard path in isolation).
//
// Widget-specific rendered identifiers used per routing test:
//   single_select         -- payload.question text (legend text)
//   inspect_and_confirm   -- "Looks right" button (inspect-view action)
//   multi_select_with_custom -- payload.question text (legend text)
//   schema_form           -- "Continue" button (submit action, present when canSubmit)
//   propose_pipeline      -- ProposePipelineTurn review heading + accept action
//   confirm_wiring        -- WireStageTurn review heading + "Confirm wiring" button
// ============================================================================

import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GuidedTurn } from "./GuidedTurn";
import { nullResponse } from "@/test/guided-fixtures";
import type {
  TurnPayload,
  SingleSelectPayload,
  InspectAndConfirmPayload,
  MultiSelectWithCustomPayload,
  ProposePipelinePayload,
  SchemaFormPayload,
  WireStageData,
} from "@/types/guided";

// ── Fixtures ──────────────────────────────────────────────────────────────────

const SINGLE_SELECT_PAYLOAD: SingleSelectPayload = {
  question: "Which data source should we use?",
  options: [{ id: "csv", label: "CSV File", hint: null }],
  allow_custom: false,
};

const INSPECT_AND_CONFIRM_PAYLOAD: InspectAndConfirmPayload = {
  observed: {
    columns: ["name", "value"],
    samples: [{ name: "foo", value: 42 }],
    warnings: [],
  },
};

const MULTI_SELECT_PAYLOAD: MultiSelectWithCustomPayload = {
  question: "Which output formats do you need?",
  options: [
    { id: "csv", label: "CSV", hint: null },
    { id: "json", label: "JSON", hint: null },
  ],
  default_chosen: ["csv"],
  escape_label: null,
};

// SchemaFormTurn renders a "Continue" button (enabled when canSubmit=true).
// A required string field with a non-empty prefilled value satisfies canSubmit.
const SCHEMA_FORM_PAYLOAD: SchemaFormPayload = {
  mode: "plugin_options",
  plugin: "csv",
  knobs: {
    fields: [
      {
        name: "path",
        label: "Path",
        kind: "text",
        required: true,
        nullable: false,
      },
    ],
  },
  prefilled: { path: "/data/file.csv" },
};

const WIRE_STAGE_PAYLOAD: WireStageData = {
  topology: {
    sources: {
      source: {
        id: "source",
        plugin: "inline_blob",
        on_success: "source_records",
        on_validation_failure: "discard",
      },
    },
    nodes: [
      {
        id: "scrape",
        node_type: "transform",
        plugin: "web_scrape",
        input: "source_records",
        on_success: "scraped",
        on_error: "scrape_error",
        routes: null,
        fork_to: null,
        branches: null,
      },
      {
        id: "mapper",
        node_type: "transform",
        plugin: "field_mapper",
        input: "scraped",
        on_success: "jsonl_out",
        on_error: null,
        routes: null,
        fork_to: null,
        branches: null,
      },
      {
        id: "error_handler",
        node_type: "transform",
        plugin: "dead_letter",
        input: "scrape_error",
        on_success: null,
        on_error: null,
        routes: null,
        fork_to: null,
        branches: null,
      },
    ],
    outputs: [
      {
        id: "output:jsonl_out",
        sink_name: "jsonl_out",
        plugin: "json",
        on_write_failure: "discard",
      },
    ],
  },
  edge_contracts: [
    {
      from: "scrape",
      to: "mapper",
      producer_guarantees: ["body", "status"],
      consumer_requires: ["body"],
      missing_fields: [],
      satisfied: true,
    },
    {
      from: "mapper",
      to: "output:jsonl_out",
      producer_guarantees: ["mapped"],
      consumer_requires: ["mapped"],
      missing_fields: [],
      satisfied: true,
    },
  ],
  semantic_contracts: [],
  warnings: [],
  // Initial confirm_wiring turn shape: no advisor pass has run, so signoff_outcome
  // is absent and the dispatcher routes to the actionable "Confirm wiring" action.
};

const PROPOSAL_PAYLOAD: ProposePipelinePayload = {
  proposal_id: "00000000-0000-4000-8000-000000000401",
  draft_hash: "d".repeat(64),
  summary: "guided.proposal.summary.full_graph.v1",
  rationale: "guided.proposal.rationale.review_required.v1",
  component_counts: { sources: 1, nodes: 0, edges: 2, outputs: 1 },
  blockers: [],
  graph: {
    sources: [{
      stable_id: "00000000-0000-4000-8000-000000000402",
      label: "source-1",
      plugin: { kind: "source", id: "csv" },
    }],
    edges: [
      {
        stable_id: "00000000-0000-4000-8000-000000000403",
        from_endpoint: { kind: "source", stable_id: "00000000-0000-4000-8000-000000000402" },
        to_endpoint: { kind: "output", stable_id: "00000000-0000-4000-8000-000000000404" },
        flow: { kind: "source_success", branch: null },
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000405",
        from_endpoint: { kind: "source", stable_id: "00000000-0000-4000-8000-000000000402" },
        to_endpoint: { kind: "discard" },
        flow: { kind: "source_validation_failure" },
      },
    ],
  },
  nodes: [],
  outputs: [{
    stable_id: "00000000-0000-4000-8000-000000000404",
    label: "output-1",
    plugin: { kind: "sink", id: "json" },
  }],
  edit_targets: [],
};

/** Build a TurnPayload with the given type and typed payload. */
function makeTurn(
  type: "single_select",
  payload: SingleSelectPayload,
): TurnPayload;
function makeTurn(
  type: "inspect_and_confirm",
  payload: InspectAndConfirmPayload,
): TurnPayload;
function makeTurn(
  type: "multi_select_with_custom",
  payload: MultiSelectWithCustomPayload,
): TurnPayload;
function makeTurn(
  type: "schema_form",
  payload: SchemaFormPayload,
): TurnPayload;
function makeTurn(
  type: "propose_pipeline",
  payload: ProposePipelinePayload,
): TurnPayload;
function makeTurn(
  type: "confirm_wiring",
  payload: WireStageData,
): TurnPayload;
function makeTurn(type: TurnPayload["type"], payload: unknown): TurnPayload {
  return { type, step_index: 0, turn_token: "a".repeat(64), payload } as TurnPayload;
}

// ── Suite 1: Turn-type routing correctness ───────────────────────────────────

describe("GuidedTurn dispatcher — routing", () => {
  it("single_select: renders SingleSelectTurn (question legend)", () => {
    render(
      <GuidedTurn
        turn={makeTurn("single_select", SINGLE_SELECT_PAYLOAD)}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Which data source should we use?"),
    ).toBeTruthy();
  });

  it("single_select + isTutorial: suppresses the pick widget entirely", () => {
    // The chip menu is a live, submit-on-click RIVAL to the one action a passive
    // learner has (Send). Its options don't even include the scripted source, so
    // clicking any chip derails the tutorial into an unscripted build. In
    // tutorial mode the pick widget is omitted; the decision collapses to its
    // heading + "press Send" caption (rendered by ChatPanel, not here).
    const { container } = render(
      <GuidedTurn
        turn={makeTurn("single_select", SINGLE_SELECT_PAYLOAD)}
        onSubmit={vi.fn()}
        isTutorial
      />,
    );
    expect(screen.queryByText("Which data source should we use?")).toBeNull();
    expect(screen.queryByRole("button", { name: "CSV File" })).toBeNull();
    expect(container).toBeEmptyDOMElement();
  });

  it("inspect_and_confirm: renders InspectAndConfirmTurn ('Looks right' button)", () => {
    render(
      <GuidedTurn
        turn={makeTurn("inspect_and_confirm", INSPECT_AND_CONFIRM_PAYLOAD)}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("button", { name: "Looks right" }),
    ).toBeTruthy();
  });

  it("multi_select_with_custom: renders MultiSelectWithCustomTurn (question legend)", () => {
    render(
      <GuidedTurn
        turn={makeTurn("multi_select_with_custom", MULTI_SELECT_PAYLOAD)}
        onSubmit={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Which output formats do you need?"),
    ).toBeTruthy();
  });

  it("schema_form: renders SchemaFormTurn ('Continue' button)", () => {
    render(
      <GuidedTurn
        turn={makeTurn("schema_form", SCHEMA_FORM_PAYLOAD)}
        onSubmit={vi.fn()}
      />,
    );
    // "Continue" button is present when the required field is prefilled.
    expect(screen.getByRole("button", { name: "Continue" })).toBeTruthy();
  });

  it("propose_pipeline: renders the current durable proposal renderer", () => {
    render(
      <GuidedTurn
        turn={makeTurn("propose_pipeline", PROPOSAL_PAYLOAD)}
        proposalReviewState={{
          status: "active",
          proposal_id: PROPOSAL_PAYLOAD.proposal_id,
          draft_hash: PROPOSAL_PAYLOAD.draft_hash,
        }}
        onSubmit={vi.fn()}
      />,
    );
    expect(screen.getByRole("heading", { name: "Review pipeline proposal" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Accept pipeline" })).toBeEnabled();
  });

  it("confirm_wiring: renders WireStageTurn UI", () => {
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", WIRE_STAGE_PAYLOAD)}
        onSubmit={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: "Review wiring" })).toBeTruthy();
    // Wiring rows carry human step names, not internal ids (elspeth-016f463ff0).
    expect(screen.getByRole("listitem", { name: /Source to Fetch step/ })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Confirm wiring" }),
    ).toBeTruthy();
  });
});

// ── Suite 2: onSubmit forwarding ──────────────────────────────────────────────

describe("GuidedTurn dispatcher — onSubmit forwarding", () => {
  it("click on option chip forwards onSubmit with the correct GuidedRespondRequest body", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("single_select", SINGLE_SELECT_PAYLOAD)}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "CSV File" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      ...nullResponse(),
      chosen: ["csv"],
      custom_inputs: null,
    });
  });

  it("confirm_wiring click forwards the exact confirm body", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", WIRE_STAGE_PAYLOAD)}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Confirm wiring" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      chosen: ["confirm"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });
  });

  it("confirm_wiring Ask advisor forwards the request_advisor body", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", {
          ...WIRE_STAGE_PAYLOAD,
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED: review",
          passes_remaining: 2,
        })}
        onSubmit={onSubmit}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 2)" }),
    );

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: "request_advisor",
    });
  });

  it("confirm_wiring Exit to freeform forwards the exit_to_freeform body", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", {
          ...WIRE_STAGE_PAYLOAD,
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED: review",
          passes_remaining: 2,
        })}
        onSubmit={onSubmit}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Exit to freeform" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      chosen: null,
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: "exit_to_freeform",
    });
  });

  it("confirm_wiring Complete without sign-off forwards chosen=['complete_without_signoff']", async () => {
    // Governance-critical: chosen is string[], so this literal is NOT
    // type-checked. The forwarded body must match the backend escape guard
    // (_helpers.py: chosen in (['confirm'], ['complete_without_signoff'])); a
    // typo here would silently 400 the only sanctioned advisor-unreachable exit.
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", {
          ...WIRE_STAGE_PAYLOAD,
          signoff_outcome: "escape_unavailable",
          advisor_findings: "Advisor unreachable.",
          passes_remaining: 0,
        })}
        onSubmit={onSubmit}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "Complete without sign-off" }),
    );

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      chosen: ["complete_without_signoff"],
      edited_values: null,
      custom_inputs: null,
      proposal_id: null,
      draft_hash: null,
      edit_target: null,
      control_signal: null,
    });
  });

  it("confirm_wiring disabled mode does not submit", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <GuidedTurn
        turn={makeTurn("confirm_wiring", WIRE_STAGE_PAYLOAD)}
        onSubmit={onSubmit}
        disabled
      />,
    );

    const button = screen.getByRole("button", { name: "Confirm wiring" });
    expect(button).toBeDisabled();
    await user.click(button);

    expect(onSubmit).not.toHaveBeenCalled();
  });
});

// ── Suite 3: Distinctness / independence pin ──────────────────────────────────

describe("GuidedTurn dispatcher — widget instance independence", () => {
  it("remounts a same-type widget when the live turn payload changes", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    const firstPayload: SingleSelectPayload = {
      question: "First custom question",
      options: [{ id: "first", label: "First", hint: null }],
      allow_custom: true,
    };
    const secondPayload: SingleSelectPayload = {
      question: "Second custom question",
      options: [{ id: "second", label: "Second", hint: null }],
      allow_custom: true,
    };

    const { rerender } = render(
      <GuidedTurn
        turn={makeTurn("single_select", firstPayload)}
        onSubmit={onSubmit}
      />,
    );

    await user.type(screen.getByRole("textbox", { name: "Custom" }), "stale");
    expect(
      screen.getByRole("button", { name: "Submit custom" }),
    ).not.toBeDisabled();

    rerender(
      <GuidedTurn
        turn={makeTurn("single_select", secondPayload)}
        onSubmit={onSubmit}
      />,
    );

    expect(screen.getByText("Second custom question")).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "Custom" })).toHaveValue("");
    expect(
      screen.getByRole("button", { name: "Submit custom" }),
    ).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("two simultaneous single_select turns render independently without state bleed", () => {
    const onSubmit1 = vi.fn();
    const onSubmit2 = vi.fn();

    const { container } = render(
      <div>
        <GuidedTurn
          turn={makeTurn("single_select", {
            question: "First turn question",
            options: [{ id: "opt_a", label: "Option A", hint: null }],
            allow_custom: false,
          })}
          onSubmit={onSubmit1}
        />
        <GuidedTurn
          turn={makeTurn("single_select", {
            question: "Second turn question",
            options: [{ id: "opt_b", label: "Option B", hint: null }],
            allow_custom: false,
          })}
          onSubmit={onSubmit2}
        />
      </div>,
    );

    // Both question texts rendered separately.
    expect(screen.getByText("First turn question")).toBeTruthy();
    expect(screen.getByText("Second turn question")).toBeTruthy();

    // Both option buttons present.
    expect(screen.getByRole("button", { name: "Option A" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Option B" })).toBeTruthy();

    // No state bleed: container has two distinct guided-turn roots.
    const turnRoots = container.querySelectorAll(".guided-turn");
    expect(turnRoots.length).toBe(2);
  });
});

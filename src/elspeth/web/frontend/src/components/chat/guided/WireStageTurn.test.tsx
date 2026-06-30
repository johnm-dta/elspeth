import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { WireStageData } from "@/types/guided";
import {
  reconstructWireEdges,
  WireStageTurn,
} from "./WireStageTurn";

function canonicalData(): WireStageData {
  return {
    topology: {
      sources: {
        source: {
          id: "source",
          plugin: "inline_blob",
          on_success: "chain_in",
          on_validation_failure: "quarantine",
        },
      },
      nodes: [
        {
          id: "scrape",
          node_type: "transform",
          plugin: "web_scrape",
          input: "chain_in",
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
        {
          id: "merge_paths",
          node_type: "coalesce",
          plugin: null,
          input: "branches",
          on_success: "jsonl_out",
          on_error: null,
          routes: null,
          fork_to: null,
          branches: { a: "path_a_done", b: "path_b_done" },
        },
        {
          id: "path_a_transform",
          node_type: "transform",
          plugin: "field_mapper",
          input: "path_a",
          on_success: "path_a_done",
          on_error: null,
          routes: null,
          fork_to: null,
          branches: null,
        },
        {
          id: "path_b_transform",
          node_type: "transform",
          plugin: "field_mapper",
          input: "path_b",
          on_success: "path_b_done",
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
          on_write_failure: "failed_writes",
        },
        {
          id: "output:quarantine",
          sink_name: "quarantine",
          plugin: "json",
          on_write_failure: "discard",
        },
        {
          id: "output:failed_writes",
          sink_name: "failed_writes",
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
    // Initial turn shape: no advisor pass has run, so signoff_outcome /
    // advisor_findings are absent and the action area is the actionable
    // "Confirm wiring". Per-outcome cases below override these explicitly.
  };
}

describe("reconstructWireEdges", () => {
  it("builds edges from connection labels", () => {
    const edges = reconstructWireEdges(canonicalData());

    expect(edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ from: "source", to: "scrape" }),
        expect.objectContaining({ from: "scrape", to: "mapper" }),
        expect.objectContaining({ from: "scrape", to: "error_handler" }),
        expect.objectContaining({ from: "mapper", to: "output:jsonl_out" }),
        expect.objectContaining({ from: "source", to: "output:quarantine" }),
        expect.objectContaining({
          from: "merge_paths",
          to: "output:jsonl_out",
        }),
        expect.objectContaining({
          from: "output:jsonl_out",
          to: "output:failed_writes",
        }),
      ]),
    );
  });

  it("builds coalesce fan-in edges from branches", () => {
    const edges = reconstructWireEdges(canonicalData());

    expect(edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          from: "path_a_transform",
          to: "merge_paths",
          label: "path_a_done",
        }),
        expect.objectContaining({
          from: "path_b_transform",
          to: "merge_paths",
          label: "path_b_done",
        }),
      ]),
    );
  });

  it("builds coalesce fan-in edges from array-form branches", () => {
    // Backend coalesce branches are `Sequence[str] | Mapping[str, str]`
    // (state.py CoalesceBranches); the Record form is covered above. This
    // pins that the alternate array (`string[]`) shape also produces correct
    // fan-in edges (regression coverage for the real backend shape).
    const data = canonicalData();
    const merge = data.topology.nodes.find((node) => node.id === "merge_paths");
    merge!.branches = ["path_a_done", "path_b_done"];

    const edges = reconstructWireEdges(data);

    expect(edges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          from: "path_a_transform",
          to: "merge_paths",
          label: "path_a_done",
        }),
        expect.objectContaining({
          from: "path_b_transform",
          to: "merge_paths",
          label: "path_b_done",
        }),
      ]),
    );
  });

  it("overlays edge_contracts keyed by from/to for scrape to mapper", () => {
    const edges = reconstructWireEdges(canonicalData());

    expect(
      edges.find((edge) => edge.from === "scrape" && edge.to === "mapper"),
    ).toMatchObject({
      label: "scraped",
      satisfied: true,
      missing_fields: [],
    });
  });

  it("overlays output sink contracts using output:<sink_name> ids", () => {
    const edges = reconstructWireEdges(canonicalData());

    expect(
      edges.find(
        (edge) =>
          edge.from === "mapper" && edge.to === "output:jsonl_out",
      ),
    ).toMatchObject({
      label: "jsonl_out",
      satisfied: true,
      missing_fields: [],
    });
  });

  it("uses named source contract ids for non-default sources", () => {
    const data = canonicalData();
    data.topology.sources = {
      refunds: {
        id: "source:refunds",
        plugin: "inline_blob",
        on_success: "chain_in",
        on_validation_failure: "discard",
      },
    };
    data.edge_contracts = [
      {
        from: "source:refunds",
        to: "scrape",
        producer_guarantees: ["url"],
        consumer_requires: ["url"],
        missing_fields: [],
        satisfied: true,
      },
    ];

    expect(
      reconstructWireEdges(data).find(
        (edge) =>
          edge.from === "source:refunds" && edge.to === "scrape",
      ),
    ).toMatchObject({
      satisfied: true,
    });
  });

  it("marks edge with no contract row as honest-gap satisfied null", () => {
    const edges = reconstructWireEdges(canonicalData());

    expect(
      edges.find((edge) => edge.from === "source" && edge.to === "scrape"),
    ).toMatchObject({
      satisfied: null,
      missing_fields: [],
    });
  });
});

describe("WireStageTurn", () => {
  it("renders prompt-shield advisory warning when present", () => {
    const data = canonicalData();
    data.warnings = [
      {
        type: "prompt_shield",
        message: "Prompt shield advisory: source text was reviewed.",
      },
    ];

    render(
      <WireStageTurn
        data={data}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );

    expect(screen.getByText(/Prompt shield advisory/)).toBeTruthy();
  });

  it("disables confirm when confirmDisabled is true", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled
      />,
    );

    expect(
      screen.getByRole("button", { name: "Confirm wiring" }),
    ).toBeDisabled();
  });

  it("conveys edge status as text not colour alone", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );

    expect(screen.getAllByText("(connected)").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText("(contract unchecked)").length,
    ).toBeGreaterThan(0);
  });

  it("renders missing field text for not satisfied contracts", () => {
    const data = canonicalData();
    data.edge_contracts = [
      ...data.edge_contracts,
      {
        from: "source",
        to: "scrape",
        producer_guarantees: ["url"],
        consumer_requires: ["url", "body"],
        missing_fields: ["body"],
        satisfied: false,
      },
    ];

    render(
      <WireStageTurn
        data={data}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );

    const edge = screen.getByRole("listitem", { name: /source to scrape/ });
    expect(within(edge).getByText("(not satisfied)")).toBeTruthy();
    expect(within(edge).getByText("Missing fields: body")).toBeTruthy();
  });

  it("lets keyboard focus reach confirm and activate it", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={onConfirm}
        confirmDisabled={false}
      />,
    );

    const button = screen.getByRole("button", { name: "Confirm wiring" });
    await user.tab();
    expect(button).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(onConfirm).toHaveBeenCalledTimes(1);
  });
});

// ── Outcome → affordance contract (slice B) ──────────────────────────────────
//
// One row of the table per case: the action area switches on signoff_outcome,
// the findings always show on a flag/block, the cost copy gates on
// passes_remaining, and "Complete without sign-off" appears ONLY on
// escape_unavailable. The default branch is a safe escape (never empty, never a
// bypassing confirm).

function outcomeData(overrides: Partial<WireStageData>): WireStageData {
  return { ...canonicalData(), ...overrides };
}

const ALL_HANDLERS = {
  onConfirm: () => {},
  confirmDisabled: false,
  onAskAdvisor: () => {},
  onExitToFreeform: () => {},
  onCompleteWithoutSignoff: () => {},
};

describe("WireStageTurn — outcome → affordance", () => {
  it("revise: findings + Ask advisor with disclosed cost + Exit, no bare Confirm", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED: tighten the source allowlist.",
          passes_remaining: 2,
        })}
      />,
    );

    expect(screen.getByText(/tighten the source allowlist/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 2)" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeTruthy();
    // The silent repeat-burn fix: no bare Confirm after a flag.
    expect(
      screen.queryByRole("button", { name: "Confirm wiring" }),
    ).toBeNull();
  });

  it("blocked_flagged: findings + Exit, no budget-burning or bypass button", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "blocked_flagged",
          advisor_findings: "Advisor sign-off budget exhausted.",
          passes_remaining: 0,
        })}
      />,
    );

    expect(screen.getByText(/budget exhausted/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Ask advisor/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Confirm wiring" })).toBeNull();
  });

  it("blocked_unavailable: explanation + Exit", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "blocked_unavailable",
          advisor_findings: "Advisor sign-off service is not configured.",
        })}
      />,
    );

    expect(screen.getByText(/cannot be completed here/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Confirm wiring" })).toBeNull();
  });

  it("escape_unavailable: Complete without sign-off is offered (only here) + Exit", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "escape_unavailable",
          advisor_findings: "Advisor unreachable.",
          passes_remaining: 0,
        })}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Complete without sign-off" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeTruthy();
  });

  it("complete: re-emitted clean verdict still offers an actionable Confirm", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "complete",
          advisor_findings: "No sign-off concerns.",
        })}
      />,
    );

    // Backend does not auto-complete a clean re-emit, so Confirm must remain
    // actionable (not a dead-end).
    const confirm = screen.getByRole("button", { name: "Confirm wiring" });
    expect(confirm).toBeTruthy();
    expect(confirm).not.toBeDisabled();
  });

  it("unknown outcome: safe default — explanation + Exit, never an empty area, never Confirm", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({ signoff_outcome: "some_future_status" })}
      />,
    );

    expect(screen.getByText(/does not recognise/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Exit to freeform" }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Confirm wiring" })).toBeNull();
  });

  it("governance: Complete without sign-off is ABSENT on revise and blocked_flagged", () => {
    const { rerender } = render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED",
          passes_remaining: 1,
        })}
      />,
    );
    expect(
      screen.queryByRole("button", { name: "Complete without sign-off" }),
    ).toBeNull();

    rerender(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "blocked_flagged",
          advisor_findings: "exhausted",
          passes_remaining: 0,
        })}
      />,
    );
    expect(
      screen.queryByRole("button", { name: "Complete without sign-off" }),
    ).toBeNull();
  });

  it("cost copy: present when passes_remaining is defined, absent when undefined (tutorial-honesty)", () => {
    const { rerender } = render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED",
          passes_remaining: 3,
        })}
      />,
    );
    expect(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 3)" }),
    ).toBeTruthy();

    // Advisor-off / no-budget snapshot: the button must NOT advertise a cost.
    rerender(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({ signoff_outcome: "revise", advisor_findings: "FLAGGED" })}
      />,
    );
    expect(screen.getByRole("button", { name: "Ask advisor" })).toBeTruthy();
    expect(screen.queryByText(/spends 1 of/)).toBeNull();
  });

  it("disable-at-0: Ask advisor is disabled when passes_remaining is 0 (defensive guard)", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED",
          passes_remaining: 0,
        })}
      />,
    );

    expect(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 0)" }),
    ).toBeDisabled();
  });

  it("forwards the exact control-signal bodies for each affordance", async () => {
    const user = userEvent.setup();
    const onAskAdvisor = vi.fn();
    const onExitToFreeform = vi.fn();
    const onCompleteWithoutSignoff = vi.fn();

    const { rerender } = render(
      <WireStageTurn
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: "FLAGGED",
          passes_remaining: 2,
        })}
        onConfirm={vi.fn()}
        confirmDisabled={false}
        onAskAdvisor={onAskAdvisor}
        onExitToFreeform={onExitToFreeform}
        onCompleteWithoutSignoff={onCompleteWithoutSignoff}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 2)" }),
    );
    expect(onAskAdvisor).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "Exit to freeform" }));
    expect(onExitToFreeform).toHaveBeenCalledTimes(1);

    rerender(
      <WireStageTurn
        data={outcomeData({
          signoff_outcome: "escape_unavailable",
          advisor_findings: "Advisor unreachable.",
          passes_remaining: 0,
        })}
        onConfirm={vi.fn()}
        confirmDisabled={false}
        onAskAdvisor={onAskAdvisor}
        onExitToFreeform={onExitToFreeform}
        onCompleteWithoutSignoff={onCompleteWithoutSignoff}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Complete without sign-off" }),
    );
    expect(onCompleteWithoutSignoff).toHaveBeenCalledTimes(1);
  });
});

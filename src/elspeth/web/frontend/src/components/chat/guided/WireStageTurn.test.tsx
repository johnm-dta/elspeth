import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { WireStageData } from "@/types/guided";
import {
  buildEntityNames,
  reconstructWireEdges,
  WireStageTurn,
} from "./WireStageTurn";

function canonicalData(): WireStageData {
  return {
    proposal_id: "00000000-0000-4000-8000-000000000001",
    draft_hash: "d".repeat(64),
    topology: {
      sources: {
        source: {
          id: "source",
          plugin: "inline_blob",
          on_success: "source_records",
          on_validation_failure: "quarantine",
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
        on_success: "source_records",
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

// ── Structural queue fan-in (elspeth-a5b86149d4 / elspeth-6421ffa028) ─────────
//
// A declared queue node accepts many producers publishing one connection name
// and feeds exactly one ordinary downstream consumer. The wire stage must
// reconstruct every producer -> queue edge plus the single queue -> consumer
// edge, in either source order, with no dishonest producer -> consumer bypass
// and no queue self-loop.

function queueData(reverseSources = false): WireStageData {
  const orders = {
    id: "source:orders",
    plugin: "csv",
    on_success: "inbound",
    on_validation_failure: "discard",
  };
  const refunds = {
    id: "source:refunds",
    plugin: "csv",
    on_success: "inbound",
    on_validation_failure: "discard",
  };
  return {
    proposal_id: "00000000-0000-4000-8000-000000000001",
    draft_hash: "d".repeat(64),
    topology: {
      sources: reverseSources ? { refunds, orders } : { orders, refunds },
      nodes: [
        {
          id: "inbound",
          node_type: "queue",
          plugin: null,
          input: "inbound",
          on_success: null,
          on_error: null,
          routes: null,
          fork_to: null,
          branches: null,
        },
        {
          id: "normalize",
          node_type: "transform",
          plugin: "passthrough",
          input: "inbound",
          on_success: "combined",
          on_error: null,
          routes: null,
          fork_to: null,
          branches: null,
        },
      ],
      outputs: [
        {
          id: "output:combined",
          sink_name: "combined",
          plugin: "json",
          on_write_failure: "discard",
        },
      ],
    },
    edge_contracts: [],
    semantic_contracts: [],
    warnings: [],
  };
}

describe("reconstructWireEdges — queue fan-in", () => {
  it("draws every producer->queue edge plus one queue->consumer edge, no bypass or self-loop", () => {
    const pairs = reconstructWireEdges(queueData()).map(
      (edge) => `${edge.from}->${edge.to}`,
    );

    expect(pairs).toContain("source:orders->inbound");
    expect(pairs).toContain("source:refunds->inbound");
    expect(pairs).toContain("inbound->normalize");
    expect(pairs).toContain("normalize->output:combined");
    // No dishonest producer -> consumer bypass edge.
    expect(pairs).not.toContain("source:orders->normalize");
    expect(pairs).not.toContain("source:refunds->normalize");
    // The queue draws exactly one edge to its downstream consumer.
    expect(pairs.filter((pair) => pair.endsWith("->normalize"))).toEqual([
      "inbound->normalize",
    ]);
    // No queue self-loop (the implicit output uses the queue's own id).
    expect(pairs).not.toContain("inbound->inbound");
  });

  it("produces the same edge set when source insertion order is reversed", () => {
    const forward = reconstructWireEdges(queueData(false))
      .map((edge) => `${edge.from}->${edge.to}`)
      .sort();
    const reversed = reconstructWireEdges(queueData(true))
      .map((edge) => `${edge.from}->${edge.to}`)
      .sort();

    expect(reversed).toEqual(forward);
  });
});

describe("WireStageTurn — queue labels", () => {
  it("names a queue node '<id> queue step', never merge/coalesce", () => {
    const names = buildEntityNames(queueData());
    expect(names.get("inbound")).toBe("inbound queue step");
  });

  it("renders the plain-language queue step label in the wiring rows", () => {
    render(
      <WireStageTurn
        data={queueData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );
    const list = document.querySelector(".wire-stage__edges");
    expect(list?.textContent).toContain("inbound queue step");
    // Uncorrelated interleave, never merge/coalesce language.
    expect(list?.textContent).not.toMatch(/merge|coalesce/i);
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

  it("conveys edge status as plain-language text not colour alone", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );

    // Plain language (elspeth-016f463ff0): "connected" / "not yet checked",
    // never the engineer-register "(contract unchecked)".
    expect(screen.getAllByText("connected").length).toBeGreaterThan(0);
    expect(screen.getAllByText("not yet checked").length).toBeGreaterThan(0);
    expect(screen.queryByText("(contract unchecked)")).toBeNull();
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

    const edge = screen.getByRole("listitem", { name: /Source to Fetch step/ });
    expect(within(edge).getByText("not connected correctly")).toBeTruthy();
    expect(within(edge).getByText("Missing fields: body")).toBeTruthy();
  });

  // ── elspeth-016f463ff0: internal ids never reach the wiring rows ────────────
  it("renders human step names in the wiring rows, raw ids only behind Technical details", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );

    // Human names: acknowledge-card step labels for transforms, "Source" for
    // the single source, "<sink_name> output" for outputs.
    expect(
      screen.getByRole("listitem", { name: "Source to Fetch step" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("listitem", { name: "Fetch step to Output step" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("listitem", { name: "Output step to jsonl_out output" }),
    ).toBeTruthy();

    // The raw rows (ids + connection labels) survive verbatim behind the
    // expander for operators.
    expect(screen.getByText("Technical details")).toBeTruthy();
    const raw = document.querySelector(".wire-stage__raw-text");
    expect(raw?.textContent).toContain("source -> scrape via source_records (contract unchecked)");
    expect(raw?.textContent).toContain("scrape -> mapper via scraped (connected)");

    // Internal ids appear ONLY inside the raw expander — never in the list.
    const list = document.querySelector(".wire-stage__edges");
    expect(list?.textContent).not.toContain("scrape");
    expect(list?.textContent).not.toContain("mapper");
    expect(list?.textContent).not.toContain("output:jsonl_out");
    expect(list?.textContent).not.toContain("source_records");
  });

  it("buildEntityNames: single source is 'Source', named sources keep their name, plugin-less nodes fall back to node_type", () => {
    const names = buildEntityNames(canonicalData());
    expect(names.get("source")).toBe("Source");
    expect(names.get("scrape")).toBe("Fetch step"); // web_scrape → ack-card label
    expect(names.get("mapper")).toBe("Output step"); // field_mapper → ack-card label
    expect(names.get("merge_paths")).toBe("Coalesce step"); // plugin null → node_type
    expect(names.get("output:jsonl_out")).toBe("jsonl_out output");

    const multi = canonicalData();
    multi.topology.sources = {
      refunds: {
        id: "source:refunds",
        plugin: "inline_blob",
        on_success: "source_records",
        on_validation_failure: "discard",
      },
      orders: {
        id: "source:orders",
        plugin: "inline_blob",
        on_success: "source_records",
        on_validation_failure: "discard",
      },
    };
    const multiNames = buildEntityNames(multi);
    expect(multiNames.get("source:refunds")).toBe("refunds source");
    expect(multiNames.get("source:orders")).toBe("orders source");
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

  it("revise: formats fenced multiline advisor findings without exposing the fence", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "revise",
          advisor_findings: [
            "BEGIN_UNTRUSTED_ADVISOR_FINDINGS",
            "FLAGGED: check the field contract.",
            "",
            "- Read `page_content`.",
            "- Preserve `summary`.",
            "END_UNTRUSTED_ADVISOR_FINDINGS",
          ].join("\n"),
          passes_remaining: 1,
        })}
      />,
    );

    const review = screen.getByRole("heading", { name: "Advisor review" })
      .parentElement;
    expect(review).not.toBeNull();
    expect(review!.textContent).not.toContain(
      "BEGIN_UNTRUSTED_ADVISOR_FINDINGS",
    );
    expect(review!.textContent).not.toContain("END_UNTRUSTED_ADVISOR_FINDINGS");
    expect(within(review!).getByRole("list")).toBeTruthy();
    expect(
      within(review!).getByText("page_content", { selector: "code" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Ask advisor (spends 1 of 1)" }),
    ).toBeTruthy();
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

  it("blocked_flagged: formats embedded fenced findings and preserves the failure prefix", () => {
    render(
      <WireStageTurn
        {...ALL_HANDLERS}
        data={outcomeData({
          signoff_outcome: "blocked_flagged",
          advisor_findings: [
            "The advisor sign-off did not pass (exhausted); the pipeline cannot complete.",
            "",
            "BEGIN_UNTRUSTED_ADVISOR_FINDINGS",
            "FLAGGED: repair the field contract.",
            "",
            "1. Set `content_field` to `page_content`.",
            "2. Keep the final mapper selective.",
            "END_UNTRUSTED_ADVISOR_FINDINGS",
          ].join("\n"),
          passes_remaining: 0,
        })}
      />,
    );

    const review = screen.getByRole("heading", { name: "Advisor review" })
      .parentElement;
    expect(review).not.toBeNull();
    expect(review!.textContent).toContain("pipeline cannot complete");
    expect(review!.textContent).not.toContain(
      "BEGIN_UNTRUSTED_ADVISOR_FINDINGS",
    );
    expect(review!.textContent).not.toContain("END_UNTRUSTED_ADVISOR_FINDINGS");
    expect(within(review!).getByRole("list")).toBeTruthy();
    expect(
      within(review!).getByText("content_field", { selector: "code" }),
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Ask advisor/ })).toBeNull();
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

describe("named blockers under the confirm button (elspeth-3b35abf148 variant 1)", () => {
  it("renders each pending acknowledgement as a jump link that focuses the card", async () => {
    const user = userEvent.setup();
    // Fake blocking card in the other column: same DOM contract the real
    // AcknowledgementCard renders (id=ack-card-<eventId>, tabIndex=-1).
    const card = document.createElement("section");
    card.id = "ack-card-evt-1";
    card.tabIndex = -1;
    card.scrollIntoView = vi.fn();
    document.body.appendChild(card);
    try {
      render(
        <WireStageTurn
          data={canonicalData()}
          onConfirm={vi.fn()}
          confirmDisabled={true}
          pendingAcknowledgements={[
            { id: "evt-1", label: "Summarise step · prompt" },
          ]}
        />,
      );

      expect(
        screen.getByText(/1 acknowledgement pending — resolve it/i),
      ).toBeTruthy();
      const link = screen.getByRole("button", {
        name: "Summarise step · prompt",
      });
      // The blocker list is wired to the disabled confirm via aria-describedby.
      const confirm = screen.getByRole("button", {
        name: "Confirm wiring",
      }) as HTMLButtonElement;
      expect(confirm.disabled).toBe(true);
      const describedBy = confirm.getAttribute("aria-describedby");
      expect(describedBy).toBeTruthy();
      expect(document.getElementById(describedBy as string)).toBeTruthy();

      await user.click(link);
      expect(card.scrollIntoView).toHaveBeenCalled();
      expect(document.activeElement).toBe(card);
    } finally {
      card.remove();
    }
  });

  it("pluralises the heading for multiple pending acknowledgements", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={true}
        pendingAcknowledgements={[
          { id: "evt-1", label: "Summarise step · prompt" },
          { id: "evt-2", label: "Summarise step · model" },
        ]}
      />,
    );
    expect(
      screen.getByText(/2 acknowledgements pending — resolve each/i),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Summarise step · model" }),
    ).toBeTruthy();
  });

  it("renders no blocker panel when nothing is pending", () => {
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
      />,
    );
    expect(screen.queryByText(/acknowledgements? pending/i)).toBeNull();
    const confirm = screen.getByRole("button", {
      name: "Confirm wiring",
    }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(false);
    expect(confirm.getAttribute("aria-describedby")).toBeNull();
  });
});

describe("client-known invalid chain (elspeth-3b35abf148 variant 3)", () => {
  it("disables confirm and names the validation issues", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={onConfirm}
        confirmDisabled={false}
        validationIssues={[
          "Pipeline has no outputs configured.",
          "Node 'mapper' input label is not produced by any step.",
        ]}
      />,
    );

    const confirm = screen.getByRole("button", {
      name: "Confirm wiring",
    }) as HTMLButtonElement;
    expect(confirm.disabled).toBe(true);
    expect(
      screen.getByText(/isn't ready to confirm/i),
    ).toBeTruthy();
    expect(
      screen.getByText("Pipeline has no outputs configured."),
    ).toBeTruthy();

    // A disabled confirm never fires — no silent no-op click path.
    await user.click(confirm);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("keeps an escape control visible when validation issues disable confirm", async () => {
    const user = userEvent.setup();
    const onExitToFreeform = vi.fn();
    render(
      <WireStageTurn
        data={canonicalData()}
        onConfirm={vi.fn()}
        confirmDisabled={false}
        onExitToFreeform={onExitToFreeform}
        validationIssues={["Source -> Output is missing field line."]}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: "Exit to freeform" }),
    );

    expect(onExitToFreeform).toHaveBeenCalledTimes(1);
  });
});

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import type { WireStageData } from "@/types/guided";
import { buildEntityNames, reconstructWireEdges, WireStageTurn } from "./WireStageTurn";

const SOURCE_ID = "00000000-0000-4000-8000-000000000010";
const NODE_ID = "00000000-0000-4000-8000-000000000020";
const OUTPUT_ID = "00000000-0000-4000-8000-000000000030";
const EDGE_ID = "00000000-0000-4000-8000-000000000040";

function canonicalData(overrides: Partial<WireStageData> = {}): WireStageData {
  return {
    proposal_id: "00000000-0000-4000-8000-000000000001",
    draft_hash: "d".repeat(64),
    sources: [{
      stable_id: SOURCE_ID,
      label: "source-1",
      plugin: "inline_blob",
      on_validation_failure: "discard",
      guaranteed_fields: ["body"],
      row_cardinality: { input: "none", output: "zero_or_many", expected_output_count: null },
    }],
    nodes: [{
      stable_id: NODE_ID,
      label: "node-1",
      node_type: "transform",
      plugin: "field_mapper",
      behavior: { kind: "transform" },
      required_fields: ["body"],
      guaranteed_fields: ["mapped"],
      row_cardinality: { input: "one", output: "one", expected_output_count: null },
      structured_output_fields: [],
    }],
    outputs: [{
      stable_id: OUTPUT_ID,
      label: "output-1",
      plugin: "json",
      on_write_failure: "discard",
      required_fields: ["mapped"],
      business_schema: { mode: "observed", fields: [], guaranteed_fields: [], required_fields: ["mapped"] },
    }],
    connections: [{
      stable_id: EDGE_ID,
      from_endpoint: { kind: "source", stable_id: SOURCE_ID },
      to_endpoint: { kind: "node", stable_id: NODE_ID },
      flow: { kind: "source_success", branch: null },
      schema_contract: {
        from: "source",
        to: "mapper",
        producer_guarantees: ["body"],
        consumer_requires: ["body"],
        missing_fields: [],
        satisfied: true,
      },
    }, {
      stable_id: "00000000-0000-4000-8000-000000000041",
      from_endpoint: { kind: "node", stable_id: NODE_ID },
      to_endpoint: { kind: "output", stable_id: OUTPUT_ID },
      flow: { kind: "node_success", branch: null },
      schema_contract: null,
    }],
    semantic_contracts: [],
    warnings: [],
    blockers: [],
    can_confirm: true,
    ...overrides,
  };
}

describe("WireStageTurn", () => {
  it("renders candidate-authored connections without reconstructing a spine", () => {
    const data = canonicalData();
    expect(reconstructWireEdges(data)).toEqual([
      expect.objectContaining({ from: SOURCE_ID, to: NODE_ID, label: "source_success", satisfied: true }),
      expect.objectContaining({ from: NODE_ID, to: OUTPUT_ID, label: "node_success", satisfied: null }),
    ]);
    expect(buildEntityNames(data)).toEqual(new Map([
      [SOURCE_ID, "source-1"],
      [NODE_ID, "node-1 (Output)"],
      [OUTPUT_ID, "output-1"],
      ["discard", "Discard"],
    ]));
  });

  it("confirms only when server-authored blockers permit it", async () => {
    const onConfirm = vi.fn();
    const { rerender } = render(<WireStageTurn data={canonicalData()} onConfirm={onConfirm} confirmDisabled={false} />);
    await userEvent.click(screen.getByRole("button", { name: "Confirm wiring" }));
    expect(onConfirm).toHaveBeenCalledOnce();

    rerender(<WireStageTurn data={canonicalData({ blockers: [{ message: "invalid route" }], can_confirm: false })} onConfirm={onConfirm} confirmDisabled={false} />);
    expect(screen.getByRole("button", { name: "Confirm wiring" })).toBeDisabled();
    expect(screen.getByText("invalid route")).toBeInTheDocument();
  });

  it("submits bounded correction feedback against the selected stable target", async () => {
    const onCorrect = vi.fn();
    render(<WireStageTurn data={canonicalData()} onConfirm={vi.fn()} confirmDisabled={false} onCorrect={onCorrect} />);
    await userEvent.selectOptions(screen.getByLabelText("Component"), NODE_ID);
    await userEvent.type(screen.getByLabelText("What should change?"), "Add the reviewed mapping.");
    await userEvent.click(screen.getByRole("button", { name: "Re-plan wiring" }));
    expect(onCorrect).toHaveBeenCalledWith({ kind: "node", stable_id: NODE_ID }, "Add the reviewed mapping.");
  });

  it("summarises route status once and renders per-route chips, not trailing prose", () => {
    // canonicalData: one satisfied contract (connected) + one null contract
    // (not yet checked). The per-line "— not yet checked" dangling clause was
    // the operator-reported debug-dump read; status renders as a compact chip
    // with the count summary above the list.
    render(<WireStageTurn data={canonicalData()} onConfirm={vi.fn()} confirmDisabled={false} />);

    expect(screen.getByText("2 routes — 1 connected, 1 not yet checked")).toBeInTheDocument();
    expect(screen.getByText("connected")).toBeInTheDocument();
    expect(screen.getByText("not yet checked")).toBeInTheDocument();
    // Screen readers keep the status even though it left the visible prose:
    // the row's accessible name carries it (aria-label overrides li content).
    expect(
      screen.getByRole("listitem", { name: "source-1 to node-1 (Output) — connected" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("listitem", { name: "node-1 (Output) to output-1 — not yet checked" }),
    ).toBeInTheDocument();
  });

  it("labels the correction controls and styles them as the app's form idiom", () => {
    render(<WireStageTurn data={canonicalData()} onConfirm={vi.fn()} confirmDisabled={false} onCorrect={vi.fn()} />);
    const select = screen.getByLabelText("Component");
    expect(select.tagName).toBe("SELECT");
    expect(select).toHaveClass("guided-schema-select");
    const feedback = screen.getByLabelText("What should change?");
    expect(feedback.tagName).toBe("TEXTAREA");
    // Explicit for/id association — the old wrapping-label markup overlapped
    // the bare native select with its own label text at some widths.
    expect(select).toHaveAttribute("id");
    expect(feedback).toHaveAttribute("id");
  });

  it("shows warnings, contract gaps, and technical stable ids", () => {
    const data = canonicalData({
      warnings: [{ message: "Review expansion cardinality." }],
      connections: canonicalData().connections.map((connection, index) => index === 0
        ? { ...connection, schema_contract: { ...connection.schema_contract!, missing_fields: ["body"], satisfied: false } }
        : connection),
    });
    render(<WireStageTurn data={data} onConfirm={vi.fn()} confirmDisabled={false} />);
    expect(screen.getByText("Review expansion cardinality.")).toBeInTheDocument();
    expect(screen.getByText("Missing fields: body")).toBeInTheDocument();
    expect(screen.getByText("Technical details")).toBeInTheDocument();
  });

  it("renders the authoritative node policies, cardinality, fields, structured outputs, and business schema", () => {
    const nodes: WireStageData["nodes"] = [
      {
        ...canonicalData().nodes[0],
        plugin: "llm",
        required_fields: ["body"],
        guaranteed_fields: ["summary"],
        row_cardinality: { input: "one", output: "zero_or_many", expected_output_count: null },
        structured_output_fields: [{
          query: "classify",
          field: "classification",
          type: "str",
          enum_values: ["safe", "unsafe"],
        }],
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000021",
        label: "decision",
        node_type: "gate",
        plugin: null,
        behavior: {
          kind: "gate",
          route_aliases: ["route-1", "route-2"],
          fork_branches: [{ routes: ["route-2"], branch: "branch-1" }],
        },
        required_fields: ["classification"],
        guaranteed_fields: [],
        row_cardinality: { input: "one", output: "one", expected_output_count: null },
        structured_output_fields: [],
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000022",
        label: "batch",
        node_type: "aggregation",
        plugin: "batch_stats",
        behavior: {
          kind: "aggregation",
          trigger_kinds: ["count", "timeout"],
          count: "25",
          timeout_seconds: 12.5,
          output_mode: "transform",
          expected_output_count: "1",
        },
        required_fields: ["classification"],
        guaranteed_fields: ["count"],
        row_cardinality: { input: "batch", output: "expected_count", expected_output_count: "1" },
        structured_output_fields: [],
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000023",
        label: "merge",
        node_type: "coalesce",
        plugin: null,
        behavior: {
          kind: "coalesce",
          branch_aliases: ["branch-1", "branch-2"],
          policy: "require_all",
          merge: "union",
        },
        required_fields: [],
        guaranteed_fields: ["count"],
        row_cardinality: { input: "branches", output: "one_per_branch_set", expected_output_count: null },
        structured_output_fields: [],
      },
    ];
    const outputs: WireStageData["outputs"] = [{
      ...canonicalData().outputs[0],
      on_write_failure: "quarantine",
      business_schema: {
        mode: "fixed",
        fields: [
          { name: "id", type: "int", required: true, nullable: false },
          { name: "email", type: "str", required: false, nullable: true },
        ],
        guaranteed_fields: ["id"],
        required_fields: ["email"],
      },
    }];

    render(<WireStageTurn data={canonicalData({ nodes, outputs })} onConfirm={vi.fn()} confirmDisabled={false} />);

    expect(screen.getByText("Cardinality: one → zero or many")).toBeInTheDocument();
    expect(screen.getAllByText("Required fields: body")).toHaveLength(1);
    expect(screen.getByText("Guaranteed fields: summary")).toBeInTheDocument();
    expect(screen.getByText("classification (str) from classify; values: safe, unsafe")).toBeInTheDocument();
    expect(screen.getByText("Routes: route-1, route-2")).toBeInTheDocument();
    expect(screen.getByText("Fork branch branch-1: route-2")).toBeInTheDocument();
    expect(screen.getByText("Triggers: count, timeout")).toBeInTheDocument();
    expect(screen.getByText("Count: 25")).toBeInTheDocument();
    expect(screen.getByText("Timeout: 12.5 seconds")).toBeInTheDocument();
    expect(screen.getByText("Output mode: transform")).toBeInTheDocument();
    expect(screen.getByText("Branches: branch-1, branch-2")).toBeInTheDocument();
    expect(screen.getByText("Policy: require all")).toBeInTheDocument();
    expect(screen.getByText("Merge: union")).toBeInTheDocument();
    expect(screen.getByText("Schema mode: fixed")).toBeInTheDocument();
    expect(screen.getByText("id: int — required, non-null")).toBeInTheDocument();
    expect(screen.getByText("email: str — optional, nullable")).toBeInTheDocument();
    expect(screen.getByText("Write failure: quarantine")).toBeInTheDocument();
    expect(screen.queryByText(/\/private\//)).not.toBeInTheDocument();
  });

  it("renders detailed success, route, branch, and failure semantics with stable connection ids", () => {
    const connections: WireStageData["connections"] = [
      ...canonicalData().connections,
      {
        stable_id: "00000000-0000-4000-8000-000000000042",
        from_endpoint: { kind: "node", stable_id: NODE_ID },
        to_endpoint: { kind: "output", stable_id: OUTPUT_ID },
        flow: { kind: "gate_route", route: "route-1", branch: null },
        schema_contract: null,
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000043",
        from_endpoint: { kind: "node", stable_id: NODE_ID },
        to_endpoint: { kind: "output", stable_id: OUTPUT_ID },
        flow: { kind: "gate_fork", routes: ["route-2"], branch: "branch-1" },
        schema_contract: null,
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000044",
        from_endpoint: { kind: "node", stable_id: NODE_ID },
        to_endpoint: { kind: "discard" },
        flow: { kind: "node_error" },
        schema_contract: null,
      },
      {
        stable_id: "00000000-0000-4000-8000-000000000045",
        from_endpoint: { kind: "output", stable_id: OUTPUT_ID },
        to_endpoint: { kind: "discard" },
        flow: { kind: "output_write_failure" },
        schema_contract: null,
      },
    ];

    render(<WireStageTurn data={canonicalData({ connections })} onConfirm={vi.fn()} confirmDisabled={false} />);

    // Flow semantics stay per-row; status moved out of the prose into chips
    // (operator-reported "— not yet checked" dump) with a single count line.
    expect(screen.getByText("Source success")).toBeInTheDocument();
    expect(screen.getByText("Gate route route-1")).toBeInTheDocument();
    expect(screen.getByText("Gate fork route-2 as branch-1")).toBeInTheDocument();
    expect(screen.getByText("Node failure")).toBeInTheDocument();
    expect(screen.getByText("Output write failure")).toBeInTheDocument();
    expect(screen.getByText("6 routes — 1 connected, 5 not yet checked")).toBeInTheDocument();
    expect(screen.getAllByText("not yet checked")).toHaveLength(5);
    expect(screen.getByText(new RegExp(EDGE_ID))).toBeInTheDocument();
  });
});

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
    advisor_findings: "No prompt-shield warnings.",
    signoff_outcome: "approved",
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

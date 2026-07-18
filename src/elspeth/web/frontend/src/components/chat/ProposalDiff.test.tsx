import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  ArgumentFields,
  buildProposalDiff,
  ProposalChanges,
} from "./ProposalDiff";
import type { CompositionState } from "@/types/api";
import { compositionStateAuthorityFields } from "@/test/composerFixtures";

function makeState(overrides: Partial<CompositionState> = {}): CompositionState {
  return {
    id: "state-1",
    ...compositionStateAuthorityFields,
    version: 3,
    sources: {
      source: {
        plugin: "csv",
        options: { path: "input.csv" },
        on_success: "rows",
        on_validation_failure: "discard",
      },
    },
    nodes: [
      {
        id: "extract",
        node_type: "transform",
        plugin: "field_mapper",
        input: "rows",
        on_success: "mapped",
        on_error: null,
        options: { mappings: { a: "b" } },
      },
    ],
    edges: [
      {
        id: "e1",
        from_node: "source",
        to_node: "extract",
        edge_type: "on_success",
        label: null,
      },
    ],
    outputs: [{ name: "results", plugin: "json", options: { path: "out.json" } }],
    metadata: { name: "My pipeline", description: null },
    ...overrides,
  };
}

describe("buildProposalDiff", () => {
  it("returns null with no current state — no honest before side exists", () => {
    expect(buildProposalDiff("upsert_node", { id: "x", node_type: "transform" }, null)).toBeNull();
  });

  it("returns null for tools with no state-fragment projection", () => {
    expect(buildProposalDiff("save_session", { name: "s" }, makeState())).toBeNull();
    expect(buildProposalDiff("create_blob", { filename: "f" }, makeState())).toBeNull();
  });

  it("projects upsert_node on an existing id as a changed node", () => {
    const entries = buildProposalDiff(
      "upsert_node",
      { id: "extract", node_type: "transform", plugin: "html_extract", input: "rows" },
      makeState(),
    );
    expect(entries).toEqual([
      expect.objectContaining({
        kind: "changed",
        section: "node",
        identity: "extract",
        beforeSummary: "transform field_mapper",
        afterSummary: "transform html_extract",
      }),
    ]);
  });

  it("projects upsert_node on a new id as an added node", () => {
    const entries = buildProposalDiff(
      "upsert_node",
      { id: "score", node_type: "transform", plugin: "llm_judge", input: "mapped" },
      makeState(),
    );
    expect(entries).toEqual([
      expect.objectContaining({
        kind: "added",
        section: "node",
        identity: "score",
        afterSummary: "transform llm_judge",
      }),
    ]);
  });

  it("projects remove_node against the current fragment", () => {
    const entries = buildProposalDiff("remove_node", { id: "extract" }, makeState());
    expect(entries).toEqual([
      expect.objectContaining({
        kind: "removed",
        section: "node",
        identity: "extract",
        beforeSummary: "transform field_mapper",
      }),
    ]);
  });

  it("projects set_source over the existing source as changed", () => {
    const entries = buildProposalDiff(
      "set_source",
      { plugin: "json", on_success: "rows", options: {}, on_validation_failure: "discard" },
      makeState(),
    );
    expect(entries).toEqual([
      expect.objectContaining({
        kind: "changed",
        section: "source",
        identity: "source",
        beforeSummary: "csv",
        afterSummary: "json",
      }),
    ]);
  });

  it("projects set_output and remove_output by sink name", () => {
    const setEntries = buildProposalDiff(
      "set_output",
      { sink_name: "errors", plugin: "csv", options: {} },
      makeState(),
    );
    expect(setEntries).toEqual([
      expect.objectContaining({ kind: "added", section: "output", identity: "errors" }),
    ]);

    const removeEntries = buildProposalDiff(
      "remove_output",
      { sink_name: "results" },
      makeState(),
    );
    expect(removeEntries).toEqual([
      expect.objectContaining({
        kind: "removed",
        section: "output",
        identity: "results",
        beforeSummary: "results (json)",
      }),
    ]);
  });

  it("projects patch_node_options as per-key option rows honouring patch semantics", () => {
    const entries = buildProposalDiff(
      "patch_node_options",
      {
        node_id: "extract",
        patch: {
          mappings: { a: "c" }, // changed
          model: "anthropic/claude-haiku-4.5", // added
          unused: null, // delete of a key that is not set → no row
        },
      },
      makeState(),
    );
    expect(entries).toHaveLength(2);
    expect(entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          kind: "changed",
          section: "option",
          identity: "extract.mappings",
          beforeSummary: '{"a":"b"}',
          afterSummary: '{"a":"c"}',
        }),
        expect.objectContaining({
          kind: "added",
          section: "option",
          identity: "extract.model",
        }),
      ]),
    );
  });

  it("projects a null patch value on an existing key as a removed option", () => {
    const entries = buildProposalDiff(
      "patch_node_options",
      { node_id: "extract", patch: { mappings: null } },
      makeState(),
    );
    expect(entries).toEqual([
      expect.objectContaining({
        kind: "removed",
        section: "option",
        identity: "extract.mappings",
      }),
    ]);
  });

  it("returns an empty projection (not null) when a patch is all no-ops", () => {
    const entries = buildProposalDiff(
      "patch_node_options",
      { node_id: "extract", patch: { mappings: { a: "b" } } },
      makeState(),
    );
    expect(entries).toEqual([]);
  });

  it("returns null when a patch targets a node missing from the current state", () => {
    expect(
      buildProposalDiff("patch_node_options", { node_id: "ghost", patch: { x: 1 } }, makeState()),
    ).toBeNull();
  });

  it("projects set_metadata field changes", () => {
    const entries = buildProposalDiff(
      "set_metadata",
      { patch: { name: "Renamed", description: "Now described" } },
      makeState(),
    );
    expect(entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "changed", section: "metadata", identity: "name" }),
        expect.objectContaining({ kind: "added", section: "metadata", identity: "description" }),
      ]),
    );
  });

  it("projects set_pipeline as added/removed/changed rows across collections", () => {
    const entries = buildProposalDiff(
      "set_pipeline",
      {
        source: { plugin: "json", on_success: "rows" },
        nodes: [
          // extract kept but plugin changed
          { id: "extract", node_type: "transform", plugin: "html_extract", input: "rows" },
          // score added
          { id: "score", node_type: "transform", plugin: "llm_judge", input: "mapped" },
        ],
        edges: [
          { id: "e1", from_node: "source", to_node: "extract", edge_type: "on_success" },
        ],
        outputs: [
          // results dropped, errors added
          { sink_name: "errors", plugin: "csv", options: {} },
        ],
      },
      makeState(),
    );

    expect(entries).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ kind: "changed", section: "source", identity: "source" }),
        expect.objectContaining({ kind: "changed", section: "node", identity: "extract" }),
        expect.objectContaining({ kind: "added", section: "node", identity: "score" }),
        expect.objectContaining({ kind: "added", section: "output", identity: "errors" }),
        expect.objectContaining({ kind: "removed", section: "output", identity: "results" }),
      ]),
    );
    // e1's provided keys match the current edge — must NOT be flagged.
    expect(entries?.some((entry) => entry.section === "edge")).toBe(false);
  });
});

describe("ProposalChanges", () => {
  it("renders diff rows through the shared recovery-diff row rendering", () => {
    const entries = buildProposalDiff(
      "upsert_node",
      { id: "extract", node_type: "transform", plugin: "html_extract", input: "rows" },
      makeState(),
    );
    render(<ProposalChanges entries={entries ?? []} />);

    expect(screen.getByText("Proposed changes")).toBeInTheDocument();
    expect(screen.getByText("Changed node")).toBeInTheDocument();
    expect(screen.getByText("extract")).toBeInTheDocument();
    expect(screen.getByText("transform field_mapper")).toBeInTheDocument();
    expect(screen.getByText("transform html_extract")).toBeInTheDocument();
  });

  it("says so plainly when the projection finds no difference", () => {
    render(<ProposalChanges entries={[]} />);
    expect(
      screen.getByText("No difference from the current pipeline."),
    ).toBeInTheDocument();
  });
});

describe("ArgumentFields", () => {
  it("renders one labelled row per top-level argument", () => {
    render(
      <ArgumentFields
        args={{ sink_name: "results", plugin: "json", options: { path: "out.json" } }}
      />,
    );

    expect(screen.getByTestId("proposal-arg-fields")).toBeInTheDocument();
    expect(screen.getByText("sink_name")).toBeInTheDocument();
    expect(screen.getByText('"results"')).toBeInTheDocument();
    expect(screen.getByText("plugin")).toBeInTheDocument();
    // Nested objects render as formatted JSON, not a flat stringify of the
    // whole argument payload.
    expect(screen.getByText(/"path": "out\.json"/)).toBeInTheDocument();
  });

  it("handles zero-argument tools without rendering an empty list", () => {
    render(<ArgumentFields args={{}} />);
    expect(screen.getByText("This tool call takes no arguments.")).toBeInTheDocument();
  });
});

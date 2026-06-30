import { describe, expect, it } from "vitest";

import {
  buildPlainPhraseMap,
  GLOSS_FALLBACK,
  pipelineGloss,
  UNKNOWN_COMPONENT_PHRASE,
} from "./pipelineGloss";
import { makeComposition } from "@/test/composerFixtures";

// A source→llm→csv composition exercised by several cases: deterministic
// plugins so the phrase map is stable (text→"read your data", llm→"rate each
// row", csv→"write a CSV").
function sourceLlmCsv() {
  return makeComposition(1, {
    sources: { source: { plugin: "text", options: {} } },
    nodes: [
      {
        id: "rater",
        node_type: "transform",
        plugin: "llm",
        input: "source",
        on_success: null,
        on_error: null,
        options: {},
      },
    ],
    outputs: [{ name: "out", plugin: "csv", options: {} }],
  });
}

describe("pipelineGloss", () => {
  it("derives a one-sentence gloss from an ordered source→llm→csv pipeline", () => {
    expect(pipelineGloss(sourceLlmCsv())).toBe(
      "This pipeline will read your data, rate each row, and write a CSV.",
    );
  });

  it("derives a two-clause gloss with an Oxford 'and' for a source→sink pipeline", () => {
    const state = makeComposition(1, {
      sources: { source: { plugin: "text", options: {} } },
      nodes: [],
      outputs: [{ name: "out", plugin: "csv", options: {} }],
    });
    expect(pipelineGloss(state)).toBe(
      "This pipeline will read your data and write a CSV.",
    );
  });

  it("falls back to a safe phrase for an empty or null composition", () => {
    const empty = makeComposition(1, { sources: {}, nodes: [], outputs: [] });
    expect(pipelineGloss(empty)).toBe(GLOSS_FALLBACK);
    expect(pipelineGloss(null)).toBe(GLOSS_FALLBACK);
    expect(pipelineGloss(undefined)).toBe(GLOSS_FALLBACK);
  });

  it("handles a partial composition (source only) without crashing", () => {
    const partial = makeComposition(1, {
      sources: { source: { plugin: "text", options: {} } },
      nodes: [],
      outputs: [],
    });
    expect(pipelineGloss(partial)).toBe("This pipeline will read your data.");
  });
});

describe("buildPlainPhraseMap", () => {
  it("keys phrases by the GraphView component_id scheme (source / node.id / output.name)", () => {
    const map = buildPlainPhraseMap(sourceLlmCsv());
    // Default source name "source" → component_id "source".
    expect(map.get("source")).toBe("read your data");
    // Node keyed by node.id.
    expect(map.get("rater")).toBe("rate each row");
    // Output keyed by output.name.
    expect(map.get("out")).toBe("write a CSV");
  });

  it("keys a non-default source name via sourceComponentId (source:<name>)", () => {
    const state = makeComposition(1, {
      sources: { feed: { plugin: "api", options: {} } },
      nodes: [],
      outputs: [],
    });
    const map = buildPlainPhraseMap(state);
    expect(map.get("source:feed")).toBe("read from an API");
  });

  it("returns an empty map for a null/undefined composition", () => {
    expect(buildPlainPhraseMap(null).size).toBe(0);
    expect(buildPlainPhraseMap(undefined).size).toBe(0);
  });

  it("exports a non-empty generic fallback phrase for unmappable component ids", () => {
    expect(UNKNOWN_COMPONENT_PHRASE).toMatch(/\S/);
  });
});

import { describe, expect, it } from "vitest";

import {
  formatFindingBody,
  humaniseValidationMessage,
  makePhraseFor,
} from "./validationHumaniser";
import { UNKNOWN_COMPONENT_PHRASE } from "@/components/chat/guided/pipelineGloss";
import { makeComposition } from "@/test/composerFixtures";

// ── humaniseValidationMessage ───────────────────────────────────────────────

describe("humaniseValidationMessage", () => {
  const identityPhraseFor = (id: string | null): string => id ?? "(null)";

  it("passes an unrecognised message through untouched", () => {
    const finding = humaniseValidationMessage("Some other error", identityPhraseFor);
    expect(finding).toEqual({ headline: "Some other error", raw: null });
  });

  it("humanises a two-sided contract violation with both producer and consumer phrases", () => {
    const finding = humaniseValidationMessage(
      "Schema contract violation: 'rater' -> 'out'. Consumer requires: [score]",
      identityPhraseFor,
    );
    expect(finding.headline).toBe(
      "Two steps aren't connected correctly: the \"rater\" step's output doesn't match what \"out\" expects.",
    );
    expect(finding.raw).toContain("Schema contract violation");
  });

  it("humanises a one-sided contract violation (no consumer capture) without a second phrase", () => {
    const finding = humaniseValidationMessage(
      "Semantic contract violation: 'rater'. Declares output fields that don't match downstream.",
      identityPhraseFor,
    );
    expect(finding.headline).toBe(
      "A step isn't connected correctly: \"rater\" doesn't match what the next step expects.",
    );
  });

  it("humanises the edge-contract preflight dump format", () => {
    const finding = humaniseValidationMessage(
      "Edge contract violation between producer node 'rater' (schema 'A') and consumer node 'out' (schema 'B'):\nMissing: score",
      identityPhraseFor,
    );
    expect(finding.headline).toContain("aren't connected correctly");
  });

  it("humanises an interpretation-review-pending dump via stepLabelFor", () => {
    const finding = humaniseValidationMessage(
      "pipeline_decision review pending for transform 'rater': drop_raw_html_fields",
      identityPhraseFor,
      () => "Summarise",
    );
    expect(finding.headline).toBe("The Summarise step is waiting for your review.");
    expect(finding.raw).toContain("pipeline_decision");
  });

  it("falls back to a generic review-pending headline when stepLabelFor cannot resolve the id", () => {
    const finding = humaniseValidationMessage(
      "pipeline_decision review pending for transform 'ghost': drop_raw_html_fields",
      identityPhraseFor,
      () => null,
    );
    expect(finding.headline).toBe("A step is waiting for your review.");
  });

  it("does not special-case review-pending dumps when no stepLabelFor is supplied", () => {
    // No stepLabelFor → falls through to the generic contract-violation /
    // passthrough path rather than crashing.
    const finding = humaniseValidationMessage(
      "pipeline_decision review pending for transform 'rater': drop_raw_html_fields",
      identityPhraseFor,
    );
    expect(finding.raw).toBeNull();
    expect(finding.headline).toContain("review pending");
  });
});

// ── makePhraseFor — direct / stripped / fuzzy / fallback / unknown ─────────

describe("makePhraseFor", () => {
  it("returns the neutral phrase for a null component id", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor(null)).toBe(UNKNOWN_COMPONENT_PHRASE);
  });

  it("resolves a direct component_id hit from the composition", () => {
    const state = makeComposition(1, {
      sources: { source: { plugin: "text", options: {} } },
      nodes: [],
      outputs: [{ name: "out", plugin: "csv", options: {} }],
    });
    const phraseFor = makePhraseFor(state);
    expect(phraseFor("out")).toBe("write a CSV");
  });

  it("resolves a role-prefixed id by stripping the node:/source:/output: prefix", () => {
    const state = makeComposition(1, {
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
      outputs: [],
    });
    const phraseFor = makePhraseFor(state);
    expect(phraseFor("node:rater")).toBe("rate each row");
  });

  it("falls back to the neutral phrase for an id with no direct, stripped, fuzzy, or generated match", () => {
    const state = makeComposition(1, { sources: {}, nodes: [], outputs: [] });
    const phraseFor = makePhraseFor(state);
    expect(phraseFor("xyz")).toBe(UNKNOWN_COMPONENT_PHRASE);
  });

  it("guesses a phrase for a role+format-bearing generated id absent from the composition", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("sink_guided_output_csv_abcd1234")).toBe("write a CSV");
    expect(phraseFor("transform_guided_xform_0_abcd1234")).toBe("process each row");
  });

  // ── elspeth-66f50ba810: fuzzy known-component match must win over the ────
  // generic role guess when the two diverge (a specific user phrase exists
  // for a *different* format than the generic guess would produce).
  it("prefers a specific fuzzy-matched component phrase over the generic role fallback (elspeth-66f50ba810)", () => {
    const state = makeComposition(1, {
      sources: {},
      nodes: [],
      outputs: [{ name: "report", plugin: "json", options: {} }],
    });
    const phraseFor = makePhraseFor(state);
    // Generic role-only guessing (role="output", no format token in the id
    // itself) would produce the DEFAULT "write the results" — but "report"
    // is a known output whose real phrase is "write a JSON file". The fuzzy
    // match on the shared 'report' token must be tried before the generic
    // guess and win.
    expect(phraseFor("output_report_a1b2c3")).toBe("write a JSON file");
    expect(phraseFor("output_report_a1b2c3")).not.toBe("write the results");
  });

  // ── elspeth-8f89b0ba34: fuzzy match must prefer the candidate with the ───
  // most matched meaningful tokens, not the first entries()-order hit.
  it("prefers the more specific (more-tokens-matched) fuzzy candidate over an earlier, less-specific one (elspeth-8f89b0ba34)", () => {
    const state = makeComposition(1, {
      sources: {},
      // Array order controls Map insertion order here (unlike sources, which
      // sort alphabetically) — "refunds_raw" is inserted BEFORE
      // "refunds_clean" so a first-match-wins bug would surface here.
      nodes: [
        {
          id: "refunds_raw",
          node_type: "transform",
          plugin: "field_mapper",
          input: "source",
          on_success: null,
          on_error: null,
          options: {},
        },
        {
          id: "refunds_clean",
          node_type: "transform",
          plugin: "llm",
          input: "source",
          on_success: null,
          on_error: null,
          options: {},
        },
      ],
      outputs: [],
    });
    const phraseFor = makePhraseFor(state);
    // "refunds_clean_v2" shares 1 meaningful token with "refunds_raw"
    // ('raw' is <4 chars and filtered) but 2 meaningful tokens with
    // "refunds_clean" ('refunds' + 'clean') — the more specific candidate
    // must win regardless of map iteration order.
    expect(phraseFor("refunds_clean_v2")).toBe("rate each row");
    expect(phraseFor("refunds_clean_v2")).not.toBe("reshape each row");
  });

  // ── elspeth-ede84df6b3: a role-less generated id must not default to a ───
  // write-direction phrase; a structured component_type hint should inform
  // (and can override) the role guess when the id itself carries no role
  // token.
  it("does not guess a write-direction phrase for a role-less CSV id with no component_type hint (elspeth-ede84df6b3)", () => {
    const phraseFor = makePhraseFor(null);
    const phrase = phraseFor("csv_refunds_a1b2");
    expect(phrase).not.toBe("write a CSV");
    expect(phrase).toBe(UNKNOWN_COMPONENT_PHRASE);
  });

  it("uses the component_type hint to resolve a role-less CSV id to the read-direction phrase (elspeth-ede84df6b3)", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("csv_refunds_a1b2", "source")).toBe("read your CSV");
    expect(phraseFor("csv_refunds_a1b2", "source")).not.toBe("write a CSV");
  });

  it("uses the component_type hint for a role-less JSON output id", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("json_export_a1b2", "sink")).toBe("write a JSON file");
  });

  it("uses the component_type hint for a role-less transform id with no format token", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("select_cols", "transform")).toBe("process each row");
  });

  it("prefers the id's own role token over a conflicting component_type hint", () => {
    // The id itself says "output"; a (hypothetically wrong) "source" hint
    // must not override an explicit role token present in the id.
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("output_csv_a1b2", "source")).toBe("write a CSV");
  });

  it("ignores an unrecognised component_type value and falls through to the id-substring guess", () => {
    const phraseFor = makePhraseFor(null);
    expect(phraseFor("transform_csv_normalize_a1b2c3", "graph")).toBe("process each row");
  });

  it("does not let a bare 'source' or 'output' component match everything via fuzzy overreach", () => {
    const state = makeComposition(1, {
      sources: {
        source: { plugin: "text", options: {} },
        refunds: { plugin: "csv", options: {} },
      },
      nodes: [],
      outputs: [],
    });
    const phraseFor = makePhraseFor(state);
    expect(phraseFor("source_csv_refunds_a1b2c3")).toBe("read your CSV");
    expect(phraseFor("source_csv_refunds_a1b2c3")).not.toBe("read your data");
  });
});

// ── formatFindingBody ───────────────────────────────────────────────────────

describe("formatFindingBody", () => {
  it("prefixes the possessive step phrase when the finding is attributed and not raw-humanised", () => {
    const body = formatFindingBody(
      1,
      "problem to fix",
      { headline: "Prompt is empty", raw: null },
      "rater",
      "transform",
      (id) => (id === "rater" ? "rate each row" : UNKNOWN_COMPONENT_PHRASE),
    );
    expect(body).toBe("1 problem to fix — 'rate each row': Prompt is empty");
  });

  it("omits the possessive prefix for a null component_id (settings-level finding)", () => {
    const body = formatFindingBody(
      1,
      "problem to fix",
      { headline: "Pipeline has no sink", raw: null },
      null,
      null,
      () => UNKNOWN_COMPONENT_PHRASE,
    );
    expect(body).toBe("1 problem to fix — Pipeline has no sink");
  });

  it("omits the possessive prefix when the finding already carries a raw dump (already humanised)", () => {
    const body = formatFindingBody(
      2,
      "problems to fix",
      { headline: "Two steps aren't connected correctly: …", raw: "Schema contract violation: …" },
      "rater",
      "transform",
      () => "rate each row",
    );
    expect(body).toBe("2 problems to fix — Two steps aren't connected correctly: …");
  });

  it("threads component_type into the phraseFor call so a role-less id resolves correctly", () => {
    const calls: Array<[string | null, string | null | undefined]> = [];
    const phraseFor = (id: string | null, componentType?: string | null): string => {
      calls.push([id, componentType]);
      return "read your CSV";
    };
    formatFindingBody(
      1,
      "problem to fix",
      { headline: "Missing field", raw: null },
      "csv_refunds_a1b2",
      "source",
      phraseFor,
    );
    expect(calls).toEqual([["csv_refunds_a1b2", "source"]]);
  });
});

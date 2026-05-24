import { describe, expect, it } from "vitest";
import type { CompositionState } from "@/types/index";
import {
  CANONICAL_TUTORIAL_PROMPT,
  initialTutorialState,
  previousStep,
  summariseCompositionState,
  tutorialReducer,
} from "./tutorialMachine";

describe("tutorialMachine", () => {
  it("pins the canonical tutorial prompt verbatim", () => {
    expect(CANONICAL_TUTORIAL_PROMPT).toBe(
      "Create a data source with URLs for five public government agency web pages " +
        "that you choose. Use abuse contact noreply@dta.gov.au and " +
        "scraping reason 'DTA technical demonstration'. Read the HTML for each " +
        "page, have an LLM identify the primary colours for each government agency. " +
        "Remove the HTML and save the rest to a json file.",
    );
  });

  it("walks the main tutorial sequence", () => {
    const described = tutorialReducer(initialTutorialState, { type: "start" });
    expect(described.step).toBe("describe");

    const built = tutorialReducer(described, {
      type: "built",
      result: {
        sessionId: "session-1",
        prompt: CANONICAL_TUTORIAL_PROMPT,
        summary: {
          sourceLabel: "inline_blob",
          urls: ["australia.gov.au"],
          transforms: ["web_scrape", "llm_rate"],
          sinkLabel: "jsonl",
        },
      },
    });
    expect(built.step).toBe("showBuilt");
    expect(built.sessionId).toBe("session-1");

    const graph = tutorialReducer(built, { type: "showGraph" });
    expect(graph.step).toBe("graph");
    const run = tutorialReducer(graph, { type: "startRun" });
    expect(run.step).toBe("run");
    const audit = tutorialReducer(run, {
      type: "runCompleted",
      result: {
        runId: "run-1",
        sourceDataHash: "abc123",
        rows: [{ url: "australia.gov.au", score: 6 }],
        seededFromCache: false,
        cacheKey: null,
        discardedRowCount: 0,
      },
    });
    expect(audit.step).toBe("audit");
    expect(audit.runId).toBe("run-1");
    expect(tutorialReducer(audit, { type: "continueToMode" }).step).toBe("mode");
  });

  it("moves from mode choice to graduation and back to mode", () => {
    const modeState = {
      ...initialTutorialState,
      step: "mode" as const,
    };
    const graduation = tutorialReducer(modeState, { type: "finishMode" });

    expect(graduation.step).toBe("graduation");
    expect(previousStep(graduation)).toBe("mode");
    expect(tutorialReducer(graduation, { type: "back" }).step).toBe("mode");
  });

  it("extracts URLs and plugin labels from a composition state", () => {
    const state: CompositionState = {
      id: "state-1",
      version: 1,
      sources: {
        source: {
          plugin: "inline_blob",
          options: {
            rows: [
              { url: "https://www.australia.gov.au" },
              { url: "dta.gov.au" },
            ],
          },
        },
      },
      nodes: [
        {
          id: "scrape",
          node_type: "transform",
          plugin: "web_scrape",
          input: "source",
          on_success: "rate",
          on_error: null,
          options: {},
        },
        {
          id: "rate",
          node_type: "transform",
          plugin: "llm_rate",
          input: "scrape",
          on_success: "sink",
          on_error: null,
          options: {},
        },
      ],
      edges: [],
      outputs: [{ name: "ratings", plugin: "jsonl", options: {} }],
      metadata: { name: null, description: null },
    };

    expect(summariseCompositionState(state)).toEqual({
      sourceLabel: "inline_blob",
      urls: ["https://www.australia.gov.au", "dta.gov.au"],
      transforms: ["web_scrape", "llm_rate"],
      sinkLabel: "jsonl",
    });
  });
});

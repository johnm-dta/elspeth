import { describe, expect, it } from "vitest";
import {
  CANONICAL_TUTORIAL_PROMPT,
  initialTutorialState,
  tutorialReducer,
  type TutorialState,
} from "./tutorialMachine";

describe("tutorialMachine", () => {
  it("pins the canonical tutorial prompt verbatim", () => {
    expect(CANONICAL_TUTORIAL_PROMPT).toBe(
      "Create a data source from these five Australian government pages: " +
        "https://www.naa.gov.au, https://my.gov.au, https://www.aec.gov.au, " +
        "https://www.oaic.gov.au, and https://www.dta.gov.au. Use abuse contact " +
        "noreply@dta.gov.au and scraping reason 'DTA technical demonstration'. " +
        "Read the HTML for each page, have an LLM return a single fact about each " +
        "government agency based on the page HTML. Remove the HTML and save the " +
        "rest to a json file.",
    );
  });
});

describe("tutorialReducer staged flow", () => {
  it("start advances welcome -> guided", () => {
    const next = tutorialReducer(initialTutorialState, { type: "start" });
    expect(next.step).toBe("guided");
  });

  it("guidedCompleted advances guided -> run and records the session", () => {
    const guided: TutorialState = { ...initialTutorialState, step: "guided" };
    const next = tutorialReducer(guided, {
      type: "guidedCompleted",
      sessionId: "sess-123",
    });
    expect(next.step).toBe("run");
    expect(next.sessionId).toBe("sess-123");
  });

  it("runCompleted advances run -> audit", () => {
    const run: TutorialState = {
      ...initialTutorialState,
      step: "run",
      sessionId: "sess-123",
    };
    const next = tutorialReducer(run, {
      type: "runCompleted",
      result: {
        runId: "run-1",
        sourceDataHash: "hash",
        rows: [],
        seededFromCache: true,
        cacheKey: null,
        discardedRowCount: 0,
      },
    });
    expect(next.step).toBe("audit");
  });

  it("continueToGraduation advances audit -> graduation", () => {
    const audit: TutorialState = { ...initialTutorialState, step: "audit" };
    const next = tutorialReducer(audit, { type: "continueToGraduation" });
    expect(next.step).toBe("graduation");
  });

  it("back from guided returns to welcome", () => {
    const guided: TutorialState = { ...initialTutorialState, step: "guided" };
    const next = tutorialReducer(guided, { type: "back" });
    expect(next.step).toBe("welcome");
  });
});

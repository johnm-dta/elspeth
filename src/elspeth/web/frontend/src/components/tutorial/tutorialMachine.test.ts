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
      "Scrape these three synthetic project-brief pages and, for each page, " +
        "have an LLM read the tables and return one JSON row with the project " +
        "name, the top risk (the highest-impact risk and its mitigation), the " +
        "go-live date, and the total cost (the sum of the cost line items). " +
        "Remove the raw HTML and write the rows to a json file.",
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

  it("back from run is a no-op (consumed guided wizard is non-returnable)", () => {
    // previousStep(run) is null: once the guided wizard completes it is terminal
    // and re-mounting it would only re-fire completion. So Back from run does
    // not navigate (and HelloWorldTutorial renders no Back affordance there).
    const run: TutorialState = {
      ...initialTutorialState,
      step: "run",
      sessionId: "sess-123",
    };
    const next = tutorialReducer(run, { type: "back" });
    expect(next.step).toBe("run");
    expect(next).toEqual(run);
  });

  it("back from audit returns to run, never guided", () => {
    // previousStep(audit) is "run": the run result is cache-backed and
    // re-viewable, but Back must NOT route into the consumed guided wizard.
    const audit: TutorialState = {
      ...initialTutorialState,
      step: "audit",
      sessionId: "sess-123",
      runId: "run-1",
      sourceDataHash: "hash",
    };
    const next = tutorialReducer(audit, { type: "back" });
    expect(next.step).toBe("run");
  });

  it("back from graduation returns to audit", () => {
    const graduation: TutorialState = {
      ...initialTutorialState,
      step: "graduation",
    };
    const next = tutorialReducer(graduation, { type: "back" });
    expect(next.step).toBe("audit");
  });
});

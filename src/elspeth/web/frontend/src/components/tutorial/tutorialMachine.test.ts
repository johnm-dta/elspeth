import { describe, expect, it } from "vitest";
import {
  CANONICAL_TUTORIAL_PROMPT,
  initialTutorialState,
  isAbandonOnPageHide,
  tutorialReducer,
  type TutorialState,
} from "./tutorialMachine";

describe("tutorialMachine", () => {
  it("pins the canonical tutorial prompt verbatim", () => {
    expect(CANONICAL_TUTORIAL_PROMPT).toBe(
      "Scrape these three synthetic project-brief pages and, for each page, " +
        "have an LLM write a short summary of the page. Remove the raw HTML and " +
        "write the rows to a json file.",
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

describe("isAbandonOnPageHide", () => {
  it("never counts the welcome bookend as an abandon", () => {
    expect(isAbandonOnPageHide("welcome", false)).toBe(false);
  });

  it("counts teardown mid-tutorial as an abandon", () => {
    expect(isAbandonOnPageHide("guided", false)).toBe(true);
    expect(isAbandonOnPageHide("run", false)).toBe(true);
    expect(isAbandonOnPageHide("audit", false)).toBe(true);
  });

  it("never counts teardown at graduation as an abandon", () => {
    expect(isAbandonOnPageHide("graduation", false)).toBe(false);
    expect(isAbandonOnPageHide("graduation", true)).toBe(false);
  });

  it("graduation latches: Back re-views after graduating are not abandons", () => {
    // graduation -> Back -> audit (or audit -> Back -> run), then tab close:
    // the learner finished the tutorial; the re-view must not overcount
    // composer.tutorial.abandon_total.
    expect(isAbandonOnPageHide("audit", true)).toBe(false);
    expect(isAbandonOnPageHide("run", true)).toBe(false);
  });
});

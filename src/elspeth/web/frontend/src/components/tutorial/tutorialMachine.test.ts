import { describe, expect, it } from "vitest";
import {
  CANONICAL_TUTORIAL_PROMPT,
  initialTutorialState,
  isAbandonOnPageHide,
  progressForTutorialState,
  resumeTutorialState,
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

describe("tutorialReducer runResultReady", () => {
  it("records the run identity without leaving the run step", () => {
    const run: TutorialState = {
      ...initialTutorialState,
      step: "run",
      sessionId: "sess-123",
    };
    const next = tutorialReducer(run, {
      type: "runResultReady",
      result: {
        runId: "run-7",
        sourceDataHash: "hash-7",
        rows: [{ url: "dta.gov.au" }],
        discardedRowCount: 0,
      },
    });
    expect(next.step).toBe("run");
    expect(next.runId).toBe("run-7");
    expect(next.sourceDataHash).toBe("hash-7");
  });
});

describe("resumeTutorialState (elspeth-918f4434b3)", () => {
  it("returns the fresh Welcome state when nothing is persisted", () => {
    expect(
      resumeTutorialState({
        stage: null,
        sessionId: null,
        runId: null,
        sourceDataHash: null,
      }),
    ).toEqual(initialTutorialState);
  });

  it("treats omitted persisted fields as a fresh Welcome state", () => {
    expect(
      resumeTutorialState({} as Parameters<typeof resumeTutorialState>[0]),
    ).toEqual(initialTutorialState);
  });

  it("refuses to resume a stage without its session (incoherent row)", () => {
    expect(
      resumeTutorialState({
        stage: "guided",
        sessionId: null,
        runId: null,
        sourceDataHash: null,
      }),
    ).toEqual(initialTutorialState);
  });

  it("resumes guided on the same session (idempotent guided start = conversation resumes)", () => {
    const state = resumeTutorialState({
      stage: "guided",
      sessionId: "sess-1",
      runId: null,
      sourceDataHash: null,
    });
    expect(state.step).toBe("guided");
    expect(state.sessionId).toBe("sess-1");
    expect(state.resumed).toBe(true);
  });

  it("resumes run-in-flight at the run step", () => {
    const state = resumeTutorialState({
      stage: "run",
      sessionId: "sess-1",
      runId: null,
      sourceDataHash: null,
    });
    expect(state.step).toBe("run");
    expect(state.sessionId).toBe("sess-1");
  });

  it("resumes a completed run forward at audit — zero re-execution", () => {
    const state = resumeTutorialState({
      stage: "run",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    });
    expect(state.step).toBe("audit");
    expect(state.runId).toBe("run-1");
    expect(state.sourceDataHash).toBe("hash-1");
    expect(state.resumed).toBe(true);
  });

  it("resumes audit at audit when the run identity is recorded", () => {
    const state = resumeTutorialState({
      stage: "audit",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    });
    expect(state.step).toBe("audit");
  });

  it("degrades audit to run when the run identity is missing", () => {
    const state = resumeTutorialState({
      stage: "audit",
      sessionId: "sess-1",
      runId: null,
      sourceDataHash: null,
    });
    expect(state.step).toBe("run");
  });

  it("graduation counts as reached once shown: resumes at graduation, never restarts", () => {
    const state = resumeTutorialState({
      stage: "graduation",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    });
    expect(state.step).toBe("graduation");
    expect(state.resumed).toBe(true);
  });
});

describe("progressForTutorialState (elspeth-918f4434b3)", () => {
  it("welcome projects to all-null (nothing to resume)", () => {
    expect(progressForTutorialState(initialTutorialState, null)).toEqual({
      stage: null,
      sessionId: null,
      runId: null,
      sourceDataHash: null,
    });
  });

  it("backing out to welcome clears even when a session exists", () => {
    expect(
      progressForTutorialState(initialTutorialState, "sess-1"),
    ).toEqual({ stage: null, sessionId: null, runId: null, sourceDataHash: null });
  });

  it("guided projects stage + session", () => {
    const state: TutorialState = { ...initialTutorialState, step: "guided" };
    expect(progressForTutorialState(state, "sess-1")).toEqual({
      stage: "guided",
      sessionId: "sess-1",
      runId: null,
      sourceDataHash: null,
    });
  });

  it("run with an arrived result carries the run identity", () => {
    const state: TutorialState = {
      ...initialTutorialState,
      step: "run",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    };
    expect(progressForTutorialState(state, state.sessionId)).toEqual({
      stage: "run",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    });
  });

  it("round-trips through resumeTutorialState onto a resumable state", () => {
    const audit: TutorialState = {
      ...initialTutorialState,
      step: "audit",
      sessionId: "sess-1",
      runId: "run-1",
      sourceDataHash: "hash-1",
    };
    const resumed = resumeTutorialState(
      progressForTutorialState(audit, audit.sessionId),
    );
    expect(resumed.step).toBe("audit");
    expect(resumed.sessionId).toBe("sess-1");
    expect(resumed.runId).toBe("run-1");
  });
});

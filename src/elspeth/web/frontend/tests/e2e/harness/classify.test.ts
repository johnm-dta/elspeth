// Unit tests for the pure run-outcome classifier (harness de-conflation).
//
// Root cause these tests lock down (notes/tutorial-harness-infra-timeout-
// rootcause-2026-06-07.md): the old classifier matched the Playwright
// "Timeout" string and stamped EVERY render-timeout `infra_fault`, conflating
// (1) genuine provider latency, (2) compose/run validation failures (real
// composer-quality signal), and (3) gateway 5xx. These tests prove we now
// classify on the BACKEND outcome (status + detail + whether the request even
// responded) — so the same Playwright timeout string yields DIFFERENT outcomes
// depending on what the backend actually did.

import { describe, expect, it } from "vitest";

import { classifyOutcome, type ClassifySignals, type StepSignal } from "./classify";

const OK_STEP: StepSignal = { fired: true, responded: true, status: 200, bodyText: null, elapsedMs: 1000 };
const NO_STEP: StepSignal = { fired: false, responded: false, status: null, bodyText: null, elapsedMs: null };

function signals(over: Partial<ClassifySignals> = {}): ClassifySignals {
  return {
    graduated: true,
    turnReached: 7,
    compose: OK_STEP,
    run: OK_STEP,
    composedNodeCount: 3,
    normalized: false,
    underFlaggedCount: 0,
    overFlaggedCount: 0,
    reachable: 5,
    minReachable: 1,
    discardedRowCount: 0,
    maxDiscarded: 1,
    substantive: 5,
    minSubstantive: 1,
    outputRowCount: 5,
    hardError: null,
    ...over,
  };
}

const TIMEOUT = "Timeout: 360000ms\nwaiting for getByRole('button', { name: 'Continue' })";

describe("classifyOutcome — clean flow", () => {
  it("a graduated, clean run is a pass", () => {
    expect(classifyOutcome(signals())).toEqual({ outcome: "pass", sub: null, fix: null });
  });
});

describe("classifyOutcome — de-conflation (the whole point)", () => {
  it("run returned 500 (validation failure) → composer-quality tutorial_fault, NOT infra", () => {
    const r = classifyOutcome(
      signals({
        graduated: false,
        turnReached: 3,
        hardError: TIMEOUT, // identical Playwright string...
        run: { fired: true, responded: true, status: 500, bodyText: '{"detail":{"error_type":"tutorial_live_run_failed","status":"failed"}}', elapsedMs: 90_000 },
      }),
    );
    expect(r.outcome).toBe("tutorial_fault");
    expect(r.sub).toBe("run-validation-failure");
  });

  it("run request fired but never responded by the deadline → infra latency-hang", () => {
    const r = classifyOutcome(
      signals({
        graduated: false,
        turnReached: 3,
        hardError: TIMEOUT, // ...same string, opposite verdict
        run: { fired: true, responded: false, status: null, bodyText: null, elapsedMs: 360_000 },
      }),
    );
    expect(r.outcome).toBe("infra_fault");
    expect(r.sub).toBe("latency-hang");
  });

  it("run returned 504 (server run-timeout) → infra", () => {
    const r = classifyOutcome(
      signals({ graduated: false, turnReached: 3, hardError: TIMEOUT, run: { ...OK_STEP, status: 504, bodyText: '{"detail":{"error_type":"tutorial_run_timeout"}}' } }),
    );
    expect(r.outcome).toBe("infra_fault");
  });

  it("run returned 429 (rate limit) → infra", () => {
    const r = classifyOutcome(signals({ graduated: false, turnReached: 3, hardError: TIMEOUT, run: { ...OK_STEP, status: 429 } }));
    expect(r.outcome).toBe("infra_fault");
  });
});

describe("classifyOutcome — compose step (turn 2)", () => {
  it("compose 502 gateway → infra", () => {
    const r = classifyOutcome(signals({ graduated: false, turnReached: 2, hardError: "Timeout: 300000ms", compose: { ...OK_STEP, status: 502 } }));
    expect(r.outcome).toBe("infra_fault");
  });

  it("compose request never responded → infra latency-hang", () => {
    const r = classifyOutcome(
      signals({ graduated: false, turnReached: 2, hardError: "Timeout: 300000ms", compose: { fired: true, responded: false, status: null, bodyText: null, elapsedMs: 300_000 } }),
    );
    expect(r.outcome).toBe("infra_fault");
    expect(r.sub).toBe("latency-hang");
  });

  it("compose returned 200 but produced NO pipeline → composer-quality, NOT frontend", () => {
    const r = classifyOutcome(
      signals({ graduated: false, turnReached: 2, hardError: "Timeout: 300000ms", compose: { ...OK_STEP }, composedNodeCount: 0 }),
    );
    expect(r.outcome).toBe("tutorial_fault");
    expect(r.sub).toBe("composer-no-pipeline");
  });

  it("compose returned 200 WITH a pipeline but the UI never advanced → genuine frontend fault", () => {
    const r = classifyOutcome(
      signals({ graduated: false, turnReached: 2, hardError: "Timeout: 300000ms", compose: { ...OK_STEP }, composedNodeCount: 4 }),
    );
    expect(r.outcome).toBe("tutorial_fault");
    expect(r.sub).toBe("frontend-state-machine");
  });

  it("compose returned 422 (invalid pipeline at compose) → composer-quality", () => {
    const r = classifyOutcome(signals({ graduated: false, turnReached: 2, hardError: "Timeout: 300000ms", compose: { ...OK_STEP, status: 422 } }));
    expect(r.outcome).toBe("tutorial_fault");
    expect(r.sub).toBe("compose-validation-failure");
  });
});

describe("classifyOutcome — frontend never issued the request", () => {
  it("turn 3 reached but the run POST never fired (a click hung) → frontend fault", () => {
    const r = classifyOutcome(signals({ graduated: false, turnReached: 3, hardError: "locator.click: Timeout 15000ms", run: NO_STEP }));
    expect(r.outcome).toBe("tutorial_fault");
    expect(r.sub).toBe("frontend-state-machine");
  });
});

describe("classifyOutcome — quality precedence (unchanged behaviour)", () => {
  it("graduated + normalization fired → normalization-gap", () => {
    expect(classifyOutcome(signals({ normalized: true })).sub).toBe("normalization-gap");
  });
  it("graduated + assumption under-flag", () => {
    expect(classifyOutcome(signals({ underFlaggedCount: 1 })).sub).toBe("assumption-under-flag");
  });
  it("graduated + over-flag", () => {
    expect(classifyOutcome(signals({ overFlaggedCount: 1 })).sub).toBe("assumption-over-flag");
  });
  it("graduated + unreachable invented source", () => {
    expect(classifyOutcome(signals({ reachable: 0, minReachable: 1 })).sub).toBe("invented-source-unreachable");
  });
  it("graduated + degenerate output", () => {
    expect(classifyOutcome(signals({ substantive: 0, minSubstantive: 1 })).sub).toBe("degenerate-output");
  });
});

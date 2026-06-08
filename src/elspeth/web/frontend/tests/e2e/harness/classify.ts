// Pure run-outcome classifier for the tutorial-reliability battery.
//
// WHY THIS EXISTS (notes/tutorial-harness-infra-timeout-rootcause-2026-06-07.md):
// the original classifier matched the Playwright error string against an
// `INFRA_ERROR` regex. Because a `toBeVisible` failure ALWAYS contains
// "Timeout: NNNms", every render-timeout was stamped `infra_fault`, conflating
// three distinct backend outcomes into one undifferentiated bucket:
//   1. genuine provider/gateway latency (real infra),
//   2. compose/run VALIDATION failures (real composer-quality signal — the
//      harness's entire reason to exist), and
//   3. gateway 5xx / rate limits (real infra).
// Their proportions vary by the composer model under test, which is exactly why
// the rate looked "consistent across batches regardless of code-under-test".
//
// This classifier keys on the BACKEND outcome instead: the HTTP status of the
// blocking step's POST, whether that request even responded before the deadline,
// and (for 2xx-but-stuck) whether a pipeline was actually composed. The
// Playwright `hardError` string is recorded but NEVER used to decide infra.

import type { FaultSubclass, Outcome } from "./types";

/** A captured observation of one compose/run POST.
 *
 * `fired`     — the frontend issued the request at all.
 * `responded` — a response arrived before the test gave up. `fired && !responded`
 *               means the backend was still working at the deadline => latency.
 * `status`    — HTTP status when `responded`.
 * `bodyText`  — response body (FastAPI `detail` JSON) when `!ok`, used only to
 *               refine an infra-vs-quality call, never as the primary signal.
 */
export interface StepSignal {
  fired: boolean;
  responded: boolean;
  status: number | null;
  bodyText: string | null;
  elapsedMs: number | null;
}

export interface ClassifySignals {
  graduated: boolean;
  turnReached: number;
  /** POST /api/sessions/{id}/messages — the compose (turn 2). */
  compose: StepSignal;
  /** POST /api/tutorial/run — the run (turn 3). */
  run: StepSignal;
  /** Node count from the composition state; 0 => compose produced no pipeline. */
  composedNodeCount: number;
  normalized: boolean;
  underFlaggedCount: number;
  overFlaggedCount: number;
  reachable: number;
  minReachable: number;
  discardedRowCount: number;
  maxDiscarded: number;
  substantive: number;
  minSubstantive: number;
  outputRowCount: number;
  /** Playwright/exception text — kept in the record, NOT used to decide infra. */
  hardError: string | null;
}

export interface ClassifyResult {
  outcome: Outcome;
  sub: FaultSubclass;
  fix: string | null;
}

// Provider/transport faults = genuine infra. NB: 500 is deliberately NOT here —
// a plain 500 from the compose/run path is the app failing to handle the
// composed pipeline (a quality/app fault to investigate), not provider infra.
const INFRA_STATUS = new Set([429, 502, 503, 504]);
const INFRA_BODY =
  /rate.?limit|throttl|bad gateway|gateway timeout|service unavailable|econn|socket hang|connection reset|temporarily unavailable/i;

function classifyBlockingStep(
  step: StepSignal,
  which: "compose" | "run",
  composedNodeCount: number,
): ClassifyResult {
  // The frontend never even issued the request => a turn that never advanced.
  if (!step.fired) {
    return { outcome: "tutorial_fault", sub: "frontend-state-machine", fix: "frontend / timing" };
  }
  // Request issued, no response by the deadline => backend still working =>
  // genuine latency/hang. This is the cleanest "provider too slow" signal.
  if (!step.responded) {
    return { outcome: "infra_fault", sub: "latency-hang", fix: null };
  }
  const status = step.status ?? 0;
  const body = step.bodyText ?? "";
  // Transport/provider faults.
  if (INFRA_STATUS.has(status) || INFRA_BODY.test(body)) {
    return { outcome: "infra_fault", sub: "llm-5xx-or-ratelimit", fix: null };
  }
  // Backend answered OK, but the expected UI never rendered.
  if (status >= 200 && status < 300) {
    if (which === "compose" && composedNodeCount === 0) {
      // Compose returned but produced no pipeline => composer-quality fault.
      return {
        outcome: "tutorial_fault",
        sub: "composer-no-pipeline",
        fix: "composer: produced no pipeline draft",
      };
    }
    // A pipeline existed (or this is the run step) but the UI hung => the real
    // frontend state-machine fault this harness exists to catch.
    return { outcome: "tutorial_fault", sub: "frontend-state-machine", fix: "frontend / timing" };
  }
  // 4xx/5xx app error (422 validation, 500 app-failed-on-pipeline) => a
  // composer-quality fault. Defaulting app errors to QUALITY (not infra) is the
  // whole point: never launder a composed-pipeline failure into infra noise.
  if (status >= 400 && status < 600) {
    return which === "compose"
      ? {
          outcome: "tutorial_fault",
          sub: "compose-validation-failure",
          fix: "composer: invalid pipeline at compose (field-contract / schema)",
        }
      : {
          outcome: "tutorial_fault",
          sub: "run-validation-failure",
          fix: "composer: composed pipeline failed execution validation",
        };
  }
  // Unknown shape — conservatively infra, but flagged distinctly.
  return { outcome: "infra_fault", sub: "staging-hiccup", fix: null };
}

export function classifyOutcome(s: ClassifySignals): ClassifyResult {
  // A hard error means a step blocked. Classify on what the BACKEND did at the
  // step the test was waiting on, NOT on the Playwright timeout string.
  if (s.hardError !== null) {
    const onRun = s.turnReached >= 3;
    return classifyBlockingStep(onRun ? s.run : s.compose, onRun ? "run" : "compose", s.composedNodeCount);
  }

  // No hard error => the flow ran to completion; apply the quality precedence
  // (unchanged from the original spec §7 ordering).
  if (s.graduated && s.normalized) {
    return {
      outcome: "tutorial_fault",
      sub: "normalization-gap",
      fix: "remove tutorial-only normalization: make tutorial == regular run",
    };
  }
  if (s.underFlaggedCount > 0) {
    return { outcome: "tutorial_fault", sub: "assumption-under-flag", fix: "composer-skill-prompt: pipeline_composer.md review rules" };
  }
  if (s.overFlaggedCount > 0) {
    return { outcome: "tutorial_fault", sub: "assumption-over-flag", fix: "composer-skill-prompt: pipeline_composer.md review rules" };
  }
  if (s.reachable < s.minReachable) {
    return { outcome: "tutorial_fault", sub: "invented-source-unreachable", fix: "composer-skill-prompt: generated-source discipline" };
  }
  if (s.discardedRowCount > s.maxDiscarded || s.substantive < s.minSubstantive) {
    return { outcome: "tutorial_fault", sub: "degenerate-output", fix: "composer-skill-prompt: extraction discipline" };
  }
  return { outcome: "pass", sub: null, fix: null };
}

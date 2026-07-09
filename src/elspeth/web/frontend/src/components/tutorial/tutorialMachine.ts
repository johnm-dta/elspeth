// Display-only framing copy — NOT sent to the backend and NOT a cache key.
// ``TutorialRunRequest`` (types/api.ts) carries only ``session_id``; every
// tutorial run executes the canonical scrape/summarize/write scenario live,
// every time. The run-cache mechanism this constant used to key into
// (``src/elspeth/web/preferences/tutorial_cache.py``) and its parity-guard
// test (``test_canonical_seed_matches_frontend_constant``) have both been
// deleted — see the "Tutorial run now LIVE" project note. This constant is
// kept only as the descriptive seed for ``TutorialState.prompt``.
// The guided BUILD is now driven PER STAGE by the prompts below — the composer
// is a STAGED orchestrator (source -> sink -> transforms -> wire), one focused
// prompt per phase, NOT one whole-problem prompt at every phase.
export const CANONICAL_TUTORIAL_PROMPT =
  "Scrape these three synthetic project-brief pages and, for each page, " +
  "have an LLM write a short summary of the page. Remove the raw HTML and " +
  "write the rows to a json file.";

// Per-stage prelocked prompts — each phase gets ONLY its stage's intent so the
// light composer model can focus on one task. Verified live against the
// per-stage solvers (resolve_source / resolve_sink / solve_chain). The SOURCE
// prompt names the `url` column so the source declares it as a guaranteed field
// (surface-or-record); the runtime-resolved sample URLs are appended to it.
export const TUTORIAL_SOURCE_PROMPT =
  "Create the source for this pipeline. The rows are these three project-brief " +
  "pages; each row carries the page's address in a `url` column:";

export const TUTORIAL_SINK_PROMPT =
  "Save the pipeline's results to a JSON file.";

export const TUTORIAL_TRANSFORMS_PROMPT =
  "For each row, fetch the page at its URL, then have an LLM write a short " +
  "summary of the page. Finally drop the raw HTML and fingerprint columns so " +
  "the saved rows keep only the summary. These are our own demo pages, so use " +
  "noreply@dta.gov.au as the scraping abuse contact.";

export type TutorialStep =
  | "welcome"
  | "guided"
  | "run"
  | "audit"
  | "graduation";

export type RunResultRow = Record<string, unknown>;

export interface TutorialRunResult {
  runId: string;
  sourceDataHash: string;
  rows: RunResultRow[];
  discardedRowCount: number;
}

export interface TutorialState {
  step: TutorialStep;
  prompt: string;
  sessionId: string | null;
  runId: string | null;
  sourceDataHash: string | null;
  rows: RunResultRow[];
  skipped: boolean;
  /**
   * Set when the user explicitly cancelled the run via the cancel button.
   * Skips the audit step (no audit story is available — the run was aborted)
   * and lands on graduation so the close-out turn can acknowledge the cancel
   * instead of silently swallowing it. Cleared by `reset`.
   */
  cancelled: boolean;
  /**
   * True when this state was reconstructed from the server-persisted resume
   * fields after a reload (elspeth-918f4434b3) rather than walked in-page.
   * A resumed audit/graduation state has no in-memory run cache, so Back
   * into the run turn would silently re-fire the tutorial pipeline (real
   * LLM spend) — the resumed flow suppresses that Back affordance instead.
   */
  resumed: boolean;
}

export type TutorialAction =
  | { type: "start" }
  | { type: "guidedCompleted"; sessionId: string }
  /**
   * The run's result arrived (rendered on the run turn, before the user
   * clicks Continue). Records the run identity WITHOUT changing step so
   * the persisted resume state can skip straight to audit after a reload
   * instead of re-executing the pipeline.
   */
  | { type: "runResultReady"; result: TutorialRunResult }
  | { type: "runCompleted"; result: TutorialRunResult }
  | { type: "continueToGraduation" }
  | { type: "skipToGraduation" }
  | { type: "cancelRun" }
  | { type: "back" }
  | { type: "reset" };

export const initialTutorialState: TutorialState = {
  step: "welcome",
  prompt: CANONICAL_TUTORIAL_PROMPT,
  sessionId: null,
  runId: null,
  sourceDataHash: null,
  rows: [],
  skipped: false,
  cancelled: false,
  resumed: false,
};

/**
 * Explicit back-navigation parent map. With the staged guided walk the flow is
 * a straight line: welcome -> guided -> run -> audit -> graduation. The guided
 * surface owns its own internal stages (source/sink/transform/wire), but once
 * completed it is TERMINAL: the persisted guided session is `terminal=completed`
 * server-side, and re-mounting TutorialGuidedShell onto it cannot re-walk the
 * stages — it would only re-fire completion. So a consumed guided wizard is
 * NON-RETURNABLE: `previousStep(run)` is null (the run turn drops its Back
 * affordance) and `previousStep(audit)` is `run` (the run result stays cache-
 * backed and re-viewable). Neither routes back into `guided`. welcome<->guided
 * and graduation->audit remain navigable.
 */
export function previousStep(state: TutorialState): TutorialStep | null {
  switch (state.step) {
    case "welcome":
      return null;
    case "guided":
      return "welcome";
    case "run":
      return null;
    case "audit":
      return "run";
    case "graduation":
      return "audit";
  }
}

/**
 * Whether a page teardown (pagehide) at `step` counts as abandoning the
 * tutorial. `welcome` means nothing started; `graduation` means the learner
 * finished (every finishing path — completed, skipped, cancelled — lands
 * there). `hasGraduated` is a LATCH: once graduation has been reached, a
 * Back re-view of the audit story or run results followed by tab close is
 * still a completed tutorial, not an abandon.
 */
export function isAbandonOnPageHide(
  step: TutorialStep,
  hasGraduated: boolean,
): boolean {
  return step !== "welcome" && step !== "graduation" && !hasGraduated;
}

export function tutorialReducer(
  state: TutorialState,
  action: TutorialAction,
): TutorialState {
  switch (action.type) {
    case "start":
      return { ...state, step: "guided" };
    case "guidedCompleted":
      return { ...state, step: "run", sessionId: action.sessionId };
    case "runResultReady":
      // Result rendered on the run turn; record the run identity so the
      // persisted resume state carries it, but stay on `run` — the user
      // has not clicked Continue yet.
      return {
        ...state,
        runId: action.result.runId,
        sourceDataHash: action.result.sourceDataHash,
        rows: action.result.rows,
      };
    case "runCompleted":
      return {
        ...state,
        step: "audit",
        runId: action.result.runId,
        sourceDataHash: action.result.sourceDataHash,
        rows: action.result.rows,
      };
    case "continueToGraduation":
      return { ...state, step: "graduation" };
    case "skipToGraduation":
      return { ...initialTutorialState, step: "graduation", skipped: true };
    case "cancelRun":
      // The user cancelled mid-run. Skip the audit step (no audit story is
      // available — the run was aborted) and land on graduation with the
      // `cancelled` flag set so the close-out turn renders an
      // acknowledgement note instead of silently swallowing the cancel.
      // Session metadata (sessionId, prompt) is preserved so the user can
      // rerun the same prompt later from the chat panel.
      return { ...state, step: "graduation", cancelled: true };
    case "back": {
      const previous = previousStep(state);
      if (previous === null) {
        return state;
      }
      return { ...state, step: previous };
    }
    case "reset":
      return initialTutorialState;
    default: {
      const _exhaustive: never = action;
      return _exhaustive;
    }
  }
}

// ── Server-persisted resume state (elspeth-918f4434b3) ─────────────────────
// The tutorial stage is persisted to composer-preferences on every stage
// transition so a reload/close-tab resumes at the persisted stage with the
// SAME session — instead of restarting at Welcome, abandoning the session,
// and silently re-spending LLM budget.

/** Mirror of the four `tutorial_*` resume fields on composer-preferences. */
export interface PersistedTutorialProgress {
  stage: "guided" | "run" | "audit" | "graduation" | null;
  sessionId: string | null;
  runId: string | null;
  sourceDataHash: string | null;
}

/**
 * Reconstruct the tutorial state to mount from the server-persisted resume
 * fields. Fresh start (no persisted stage, or a stage without its session —
 * an incoherent row we refuse to guess from) returns the initial Welcome
 * state.
 *
 * Stage mapping:
 *  - `guided` — remount the guided shell on the same session. The backend
 *    `POST /guided/start` is idempotent (D16): it re-attaches to the
 *    persisted GuidedSession, so the conversation RESUMES; no LLM restart.
 *  - `run` with a recorded run identity — the run had already completed
 *    before the reload (the identity is recorded when the result renders),
 *    so resume forward at `audit`: zero re-execution.
 *  - `run` without a run identity — the reload interrupted the run itself;
 *    resume at `run` (the run turn re-fires; if the pre-reload run is still
 *    active server-side the one-active-run invariant surfaces the friendly
 *    still-finishing message).
 *  - `audit` — requires the recorded run identity; degrades to `run` when
 *    missing (audit cannot render without it).
 *  - `graduation` — graduation counts as reached once SHOWN; resume there,
 *    never restart. (The skipped path never persists a stage — skip
 *    persists the completion opt-out immediately instead. The cancelled
 *    flag is in-memory only, so a resumed graduation omits the cancel note.)
 */
export function resumeTutorialState(
  progress: PersistedTutorialProgress,
): TutorialState {
  const stage = progress.stage ?? null;
  const sessionId = progress.sessionId ?? null;
  const runId = progress.runId ?? null;
  const sourceDataHash = progress.sourceDataHash ?? null;
  if (stage === null || sessionId === null) {
    return initialTutorialState;
  }
  const base: TutorialState = {
    ...initialTutorialState,
    sessionId,
    runId,
    sourceDataHash,
    resumed: true,
  };
  const hasRunIdentity = runId !== null && sourceDataHash !== null;
  switch (stage) {
    case "guided":
      return { ...base, step: "guided", runId: null, sourceDataHash: null };
    case "run":
      return hasRunIdentity
        ? { ...base, step: "audit" }
        : { ...base, step: "run", runId: null, sourceDataHash: null };
    case "audit":
      return hasRunIdentity
        ? { ...base, step: "audit" }
        : { ...base, step: "run", runId: null, sourceDataHash: null };
    case "graduation":
      return { ...base, step: "graduation" };
  }
}

/**
 * Project the persistable resume fields from a live tutorial state (the
 * inverse of `resumeTutorialState`). `welcome` maps to all-null — nothing
 * has started (or the user backed out), so there is nothing to resume.
 *
 * The skipped path is handled by the caller: skip persists the completion
 * opt-out immediately (which clears these fields server-side), so no stage
 * write happens for it.
 */
export function progressForTutorialState(
  state: TutorialState,
  sessionId: string | null,
): PersistedTutorialProgress {
  if (state.step === "welcome" || sessionId === null) {
    return { stage: null, sessionId: null, runId: null, sourceDataHash: null };
  }
  return {
    stage: state.step,
    sessionId,
    runId: state.runId,
    sourceDataHash: state.sourceDataHash,
  };
}

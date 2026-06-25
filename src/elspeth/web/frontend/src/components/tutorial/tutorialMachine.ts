// Kept BYTE-IDENTICAL to the backend cache-key constant ``CANONICAL_SEED_PROMPT``
// in ``src/elspeth/web/preferences/tutorial_cache.py``. Turn 4 posts this string
// to ``/api/tutorial/run``; the backend only engages the tutorial cache when
// ``effective_prompt == CANONICAL_SEED_PROMPT``. If the two drift apart the cache
// silently never hits and every tutorial run goes live. The Python test
// ``test_canonical_seed_matches_frontend_constant`` fails CI if they diverge.
export const CANONICAL_TUTORIAL_PROMPT =
  "Scrape these three synthetic project-brief pages and, for each page, " +
  "have an LLM read the tables and return one JSON row with the project " +
  "name, the top risk (the highest-impact risk and its mitigation), the " +
  "go-live date, and the total cost (the sum of the cost line items). " +
  "Remove the raw HTML and write the rows to a json file.";

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
  seededFromCache: boolean;
  cacheKey: string | null;
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
}

export type TutorialAction =
  | { type: "start" }
  | { type: "guidedCompleted"; sessionId: string }
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

export function tutorialReducer(
  state: TutorialState,
  action: TutorialAction,
): TutorialState {
  switch (action.type) {
    case "start":
      return { ...state, step: "guided" };
    case "guidedCompleted":
      return { ...state, step: "run", sessionId: action.sessionId };
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

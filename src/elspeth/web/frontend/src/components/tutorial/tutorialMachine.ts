import type { CompositionState, NodeSpec, OutputSpec, SourceSpec } from "@/types/index";

export const CANONICAL_TUTORIAL_PROMPT =
  "Create a data source with URLs for five public government agency web pages " +
  "that you choose. Use abuse contact noreply@dta.gov.au and " +
  "scraping reason 'DTA technical demonstration'. Read the HTML for each " +
  "page, have an LLM identify the primary colours for each government agency. " +
  "Remove the HTML and save the rest to a json file.";

export type TutorialStep =
  | "welcome"
  | "describe"
  | "showBuilt"
  | "graph"
  | "run"
  | "audit"
  | "mode"
  | "graduation";

export type RunResultRow = Record<string, unknown>;

export interface TutorialBuiltSummary {
  sourceLabel: string;
  urls: string[];
  transforms: string[];
  sinkLabel: string;
}

export interface TutorialBuildResult {
  sessionId: string;
  prompt: string;
  summary: TutorialBuiltSummary;
}

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
  builtSummary: TutorialBuiltSummary | null;
  skipped: boolean;
  /**
   * Set when the user explicitly cancelled the Turn 4 run via the cancel
   * button. Consumed by Turn 6 to render an acknowledgement note in place
   * of the audit-story summary the user skipped. Cleared by `reset` and
   * by any back navigation that returns to `describe` (i.e. the user is
   * starting over).
   */
  cancelled: boolean;
}

export type TutorialAction =
  | { type: "start" }
  | { type: "built"; result: TutorialBuildResult }
  | { type: "showGraph" }
  | { type: "startRun" }
  | { type: "runCompleted"; result: TutorialRunResult }
  | { type: "continueToMode" }
  | { type: "skipToMode" }
  | { type: "finishMode" }
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
  builtSummary: null,
  skipped: false,
  cancelled: false,
};

/**
 * Explicit back-navigation parent map. Defined as a function (not a static
 * object) because the previous step depends on how the user reached the
 * current one — Turn 6 has three arrival paths (continueToMode, skipToMode,
 * cancelRun) and each goes back to a different turn.
 *
 * For run/audit, back lands on `describe` rather than the mechanically
 * previous step. The user's actual intent at that point is "I want to edit
 * the prompt and start over", not "show me the graph card again".
 */
export function previousStep(state: TutorialState): TutorialStep | null {
  switch (state.step) {
    case "welcome":
      return null;
    case "describe":
      return "welcome";
    case "showBuilt":
      return "describe";
    case "graph":
      return "showBuilt";
    case "run":
      return "describe";
    case "audit":
      return "describe";
    case "mode":
      if (state.skipped) return "welcome";
      if (state.cancelled) return "describe";
      return "audit";
    case "graduation":
      return "mode";
  }
}

export function tutorialReducer(
  state: TutorialState,
  action: TutorialAction,
): TutorialState {
  switch (action.type) {
    case "start":
      return { ...state, step: "describe" };
    case "built":
      return {
        ...state,
        step: "showBuilt",
        prompt: action.result.prompt,
        sessionId: action.result.sessionId,
        builtSummary: action.result.summary,
        skipped: false,
      };
    case "showGraph":
      if (state.sessionId === null) {
        throw new Error("tutorialReducer: graph step requires a session");
      }
      return { ...state, step: "graph" };
    case "startRun":
      if (state.sessionId === null) {
        throw new Error("tutorialReducer: run step requires a session");
      }
      return { ...state, step: "run" };
    case "runCompleted":
      return {
        ...state,
        step: "audit",
        runId: action.result.runId,
        sourceDataHash: action.result.sourceDataHash,
        rows: action.result.rows,
      };
    case "continueToMode":
      return { ...state, step: "mode" };
    case "skipToMode":
      return {
        ...initialTutorialState,
        step: "mode",
        skipped: true,
      };
    case "finishMode":
      if (state.step !== "mode") {
        throw new Error("tutorialReducer: finishMode requires the mode step");
      }
      return { ...state, step: "graduation" };
    case "cancelRun":
      // The user cancelled mid-run. Skip Turn 5 (no audit story is
      // available — the run was aborted) and land on Turn 6 with the
      // `cancelled` flag set so the mode-choice turn renders an
      // acknowledgement note instead of silently swallowing the cancel.
      // Session metadata (sessionId, prompt, builtSummary) is preserved
      // so the user can rerun the same prompt later from the chat panel.
      return { ...state, step: "mode", cancelled: true };
    case "back": {
      const previous = previousStep(state);
      if (previous === null) {
        return state;
      }
      // Going back to `describe` is "I want to edit the prompt and start
      // over": clear the session-derived state so the next build starts
      // fresh. Preserve the prompt itself so the user keeps their edits.
      if (previous === "describe" && state.step !== "welcome") {
        return {
          ...initialTutorialState,
          step: "describe",
          prompt: state.prompt,
        };
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

export function summariseCompositionState(
  state: CompositionState,
): TutorialBuiltSummary {
  return {
    sourceLabel: summariseSource(state.source),
    urls: collectUrls(state.source?.options ?? {}).slice(0, 10),
    transforms: state.nodes.map(summariseNode),
    sinkLabel: summariseOutputs(state.outputs),
  };
}

function summariseSource(source: SourceSpec | null): string {
  if (source === null) {
    return "No source was returned";
  }
  return source.plugin;
}

function summariseNode(node: NodeSpec): string {
  if (node.plugin !== null) {
    return node.plugin;
  }
  return node.node_type;
}

function summariseOutputs(outputs: OutputSpec[]): string {
  if (outputs.length === 0) {
    return "No sink was returned";
  }
  return outputs.map((output) => output.plugin).join(", ");
}

function collectUrls(value: unknown): string[] {
  const found: string[] = [];
  const seen = new Set<string>();

  function add(candidate: string): void {
    const trimmed = candidate.trim().replace(/[),.;\]]+$/, "");
    if (!trimmed || seen.has(trimmed)) {
      return;
    }
    seen.add(trimmed);
    found.push(trimmed);
  }

  function walk(current: unknown): void {
    if (typeof current === "string") {
      const matches = current.match(
        /https?:\/\/[^\s"',)]+|[a-z0-9-]+(?:\.[a-z0-9-]+)*\.gov\.au/gi,
      );
      matches?.forEach(add);
      return;
    }
    if (Array.isArray(current)) {
      current.forEach(walk);
      return;
    }
    if (typeof current === "object" && current !== null) {
      Object.values(current).forEach(walk);
    }
  }

  walk(value);
  return found;
}

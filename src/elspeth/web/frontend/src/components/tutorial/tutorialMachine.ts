import type { CompositionState, NodeSpec, OutputSpec, SourceSpec } from "@/types/index";

export const CANONICAL_TUTORIAL_PROMPT =
  "create a list of 5 government web pages and use an LLM to rate how cool they are";

export type TutorialStep =
  | "welcome"
  | "describe"
  | "showBuilt"
  | "graph"
  | "run"
  | "audit"
  | "mode";

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
}

export type TutorialAction =
  | { type: "start" }
  | { type: "built"; result: TutorialBuildResult }
  | { type: "showGraph" }
  | { type: "startRun" }
  | { type: "runCompleted"; result: TutorialRunResult }
  | { type: "continueToMode" }
  | { type: "skipToMode" }
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
};

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

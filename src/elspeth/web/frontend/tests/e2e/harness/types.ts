export type Outcome = "pass" | "tutorial_fault" | "infra_fault";

export type FaultSubclass =
  | "composer-skill-prompt"
  | "specific-tool"
  | "plugin"
  | "frontend-state-machine"
  | "normalization-gap"
  | "assumption-under-flag"
  | "assumption-over-flag"
  | "degenerate-output"
  | "weak-extraction"
  | "invented-source-unreachable"
  | "wrong-dag-shape"
  | "llm-5xx-or-ratelimit"
  | "scrape-target-down-or-throttled"
  | "staging-hiccup"
  | "timeout"
  // De-conflation subclasses (notes/tutorial-harness-infra-timeout-rootcause-2026-06-07.md):
  // backend-grounded causes the old string-match classifier laundered into `timeout`.
  | "latency-hang" // request fired, no response by deadline (genuine provider latency)
  | "composer-no-pipeline" // compose returned 2xx but produced no pipeline draft
  | "compose-validation-failure" // compose POST 4xx/5xx (invalid pipeline at compose)
  | "run-validation-failure" // run POST 4xx/5xx (composed pipeline failed execution)
  | null;

export interface RunRecord {
  batch_id: string;
  run_index: number;
  outcome: Outcome;
  fault_subclass: FaultSubclass;
  fix_target: string | null;
  turn_reached: number; // 1..7
  tutorial_run_id: string | null;
  realsystem_run_id: string | null;
  seeded_from_cache: boolean; // MUST be false for a fresh run
  dim_a_tutorial_completed: boolean;
  dim_b_realsystem_passed: boolean;
  dim_c_assumptions_ok: boolean;
  dim_d_solution_quality: {
    status: "pending_judge" | "pass" | "fail";
    judge_score: number | null;
    source_reachable: string; // e.g. "5/5"
    discarded_row_count: number;
    substantive_rows: string; // e.g. "4/5"
  };
  assumptions: {
    raised: Array<{ kind: string | null; term: string | null }>;
    under_flagged: string[];
    over_flagged: string[];
  };
  output_rows: Array<Record<string, unknown>>; // raw material for the judge
  landscape: {
    tutorial_failure: string | null;
    realsystem_failure: string | null;
    normalization_fired: boolean;
  };
  stamp: {
    composer_skill_hash: string | null;
    model_identifier: string | null;
  };
  timing_s: Record<string, number>;
  // Backend step evidence for the de-conflated classifier (compose = POST
  // /api/sessions/{id}/messages, run = POST /api/tutorial/run). Optional so
  // older records remain valid.
  steps?: {
    compose: StepEvidence;
    run: StepEvidence;
  };
  error: string | null; // exception text on hard fault
}

export interface StepEvidence {
  fired: boolean;
  responded: boolean;
  status: number | null;
  elapsed_s: number | null;
  // Error body (FastAPI `detail`) when the POST was !ok — truncated. Makes a
  // failed run self-diagnostic (e.g. which validation error) without a server
  // round-trip. null on a 2xx / in-flight step.
  body: string | null;
}

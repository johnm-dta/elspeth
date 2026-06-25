// The coupled triple: prompt + assumption rubric (dim c) + judge rubric (dim d).
// If the prompt changes, BOTH rubrics must change in lockstep (spec §11).

export const HARNESS_VERSION = "2.0.0";

// Semantically equal to CANONICAL_TUTORIAL_PROMPT (the synthetic-scrape
// EXTRACTION task) but a DIFFERENT string, so SHA-256(prompt+model_id) misses
// the tutorial cache (is_canonical_prompt=false in tutorial_service.py) ->
// fresh live composition every run.
//
// EXTRACTION (not scoring): the composer reads authored tables and derives
// facts (name / top risk / go-live date / summed total cost). Extraction is
// not subjective, so vague_term is NOT expected here (contrast the old scoring
// prompt). The always-on prompt-shield (p3) fires every run regardless, so
// prompt_injection_shield_recommendation IS expected.
//
// The prompt MUST carry the 3 concrete scrape targets (mirroring the old
// prompt's 5 literal gov URLs): driveGuidedWalk seeds FIXED_PROMPT as the
// sole driving message (:106/:115), and the Tier-1 no-fabrication source
// driver (p1, contract §2.2) builds rows ONLY from concrete URLs present IN
// the message — it never invents them. A URL-less prompt yields zero scrape
// targets, so the run produces no rows and JUDGE_RUBRIC.minReachableSources:3
// / minSubstantiveRows:3 become UNSATISFIABLE (dim-c/dim-d permanently fail).
// These are staging-absolute URLs because this is a .staging.spec.ts driving
// the live staging origin; there is no harness base-URL constant to resolve
// {base} from. If a deployment serves the tutorial pages elsewhere, change
// these three literals to that origin.
export const FIXED_PROMPT =
  "Fetch each of these three synthetic project-brief pages and use an LLM " +
  "to read the tables and produce, per page, a JSON row containing the " +
  "project name, the single highest-impact risk together with its " +
  "mitigation, the go-live milestone date, and the total cost computed by " +
  "summing the cost line items. Drop the raw HTML and write the rows to a " +
  "JSON file.\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-1.html\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-2.html\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-3.html";

// Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
// Graded on kind, not exact wording. InterpretationKind values come from the
// backend enum (kind field on InterpretationEventResponse):
// vague_term | invented_source | llm_prompt_template | pipeline_decision | llm_model_choice.
export const ASSUMPTION_RUBRIC = {
  // EXTRACTION, not scoring: there is no subjective rating criterion, so
  // vague_term must NOT be expected. The always-on prompt-shield (p3) fires
  // for every LLM node over fetched content, so the pipeline_decision review
  // with user_term=prompt_injection_shield_recommendation IS expected — this
  // is how the harness verifies Component 2 (the 3-state shield) end-to-end.
  expectVerify: ["prompt_injection_shield_recommendation"] as const,
  // The composer MAY also stage a prompt-template review for the LLM node;
  // acceptable-but-not-required (neither under- nor over-flagging).
  allowOptional: ["llm_prompt_template"] as const,
  // Over-flagging: the USER named the 3 pages and the 4 fields explicitly, so
  // raising an invented_source review, or a review on those stated targets, is
  // an over-flag. MIXED targets: "invented_source" is an InterpretationKind
  // (matched on the event KIND); "project_name"/"total_cost" are user_terms
  // (matched on the review's user_term). Step 4b below makes the overFlagged
  // grader match EITHER kind OR user_term so the kind-valued entry can fire —
  // without it, the user_term-only grader at :380-383 can NEVER match
  // invented_source (a dead check, the exact class :374-379 was written to
  // eliminate).
  overFlagTerms: ["invented_source", "project_name", "total_cost"] as const,
  // Patterns are ANCHORED so they cannot spuriously fail dim-c on a legitimate
  // review. /\bproject[_\s-]?name\b/i matches only the field name "project_name"
  // (not e.g. "the project's name as written"), and /\btotal[_\s-]?cost\b/i drops
  // the bare "sum" alternative which would substring-match "assume"/"consume"/
  // "summary"/"summarize" in legitimate llm_prompt_template / allowOptional
  // reviews. Confirm the expected prompt_injection_shield_recommendation and the
  // allowOptional llm_prompt_template user_terms match NO over-flag pattern.
  overFlagTermPatterns: [/invent|fabricat/i, /\bproject[_\s-]?name\b/i, /\btotal[_\s-]?cost\b/i] as const,
};

// Dimension (d): the judge rubric the agent applies to recorded output rows.
export const JUDGE_RUBRIC = {
  // Mechanical (computed in the spec/aggregator, NOT by the judge):
  minReachableSources: 3, // all 3 synthetic pages must have fetched (scrape op status ok)
  maxDiscardedRows: 0, // discarded_row_count from TutorialRunOutput
  minSubstantiveRows: 3, // all 3 rows carry the derived fields
  // Judge-scored (0..1), pass threshold:
  judgePassThreshold: 0.7,
  // The judge question (structured), applied per batch over recorded rows:
  judgePrompt:
    "Given the task (per synthetic project-brief page, extract project_name, " +
    "top_risk = the highest-impact risk and its mitigation, key_date = the " +
    "go-live milestone, and total_cost = the SUM of the cost line items) and " +
    "these output rows, score 0..1: does each row carry a non-empty " +
    "project_name, a top_risk that names a real risk + mitigation, a plausible " +
    "go-live date, and a total_cost that equals the sum of the page's cost " +
    "lines? Penalise degenerate output — missing/null fields, a total_cost " +
    "that is a single line item rather than the sum, or a top_risk that is not " +
    "the highest-impact row.",
};

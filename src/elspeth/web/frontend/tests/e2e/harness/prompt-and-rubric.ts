// The coupled triple: prompt + assumption rubric (dim c) + judge rubric (dim d).
// If the prompt changes, BOTH rubrics must change in lockstep (spec §11).

export const HARNESS_VERSION = "2.0.0";

// Semantically equal to CANONICAL_TUTORIAL_PROMPT (the synthetic-scrape SUMMARY
// task) but a DIFFERENT string, so SHA-256(prompt+model_id) misses the tutorial
// cache (is_canonical_prompt=false in tutorial_service.py) -> fresh live
// composition every run.
//
// SUMMARISATION — one value per page (NOT multi-field extraction): the composer
// builds source(url rows) -> web_scrape -> llm(single response_field) ->
// field_mapper(drop raw HTML) -> json. One value per row, no structured
// multi-query, no joining. The always-on prompt-shield (p3) fires every run
// (LLM over fetched content), so prompt_injection_shield_recommendation IS
// expected. "Summarise" is mildly subjective, so a vague_term review MAY fire —
// it is tolerated (neither expected nor an over-flag).
//
// The prompt MUST carry the 3 concrete scrape targets: driveGuidedWalk seeds
// FIXED_PROMPT as the sole driving message (:106/:115), and the Tier-1
// no-fabrication source driver (p1, contract §2.2) builds rows ONLY from
// concrete URLs present IN the message — it never invents them. A URL-less
// prompt yields zero scrape targets, so the run produces no rows and
// JUDGE_RUBRIC.minReachableSources:3 / minSubstantiveRows:3 become
// UNSATISFIABLE (dim-c/dim-d permanently fail). These are staging-absolute URLs
// because this is a .staging.spec.ts driving the live staging origin; there is
// no harness base-URL constant to resolve {base} from. If a deployment serves
// the tutorial pages elsewhere, change these three literals to that origin.
export const FIXED_PROMPT =
  "Fetch each of these three synthetic project-brief pages and have an LLM " +
  "write a short summary of each page. Drop the raw HTML and write the rows " +
  "to a JSON file.\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-1.html\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-2.html\n" +
  "https://elspeth.foundryside.dev/tutorial-site/project-3.html";

// Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
// Graded on kind, not exact wording. InterpretationKind values come from the
// backend enum (kind field on InterpretationEventResponse):
// vague_term | invented_source | llm_prompt_template | pipeline_decision | llm_model_choice.
export const ASSUMPTION_RUBRIC = {
  // SUMMARISATION: "summarise" is mildly subjective, so a vague_term review is
  // TOLERATED (it appears in neither list — neither required nor an over-flag).
  // The always-on prompt-shield (p3) fires for every LLM node over fetched
  // content, so the pipeline_decision review with
  // user_term=prompt_injection_shield_recommendation IS expected — this is how
  // the harness verifies Component 2 (the 3-state shield) end-to-end.
  expectVerify: ["prompt_injection_shield_recommendation"] as const,
  // The composer MAY also stage a prompt-template review for the LLM node;
  // acceptable-but-not-required (neither under- nor over-flagging).
  allowOptional: ["llm_prompt_template"] as const,
  // Over-flagging: the USER named the 3 pages explicitly, so raising an
  // invented_source review on those stated targets is an over-flag.
  // "invented_source" is an InterpretationKind (matched on the event KIND); the
  // overFlagged grader (spec step 4b) matches EITHER kind OR user_term so the
  // kind-valued entry fires. (The old project_name/total_cost field over-flags
  // were pruned with the move from 4-field extraction to a single summary.)
  overFlagTerms: ["invented_source"] as const,
  // Pattern is index-aligned to overFlagTerms. /invent|fabricat/i matches the
  // invented_source KIND. Confirm the expected
  // prompt_injection_shield_recommendation and the allowOptional
  // llm_prompt_template user_terms match NO over-flag pattern.
  overFlagTermPatterns: [/invent|fabricat/i] as const,
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
    "Given the task (per synthetic project-brief page, write a short summary " +
    "of that page) and these output rows, score 0..1: does each row carry a " +
    "non-empty summary that plausibly describes a project brief (naming real " +
    "content from the page such as the project, its risks, dates, or costs)? " +
    "Penalise degenerate output — missing/null/empty summaries, a bare URL or " +
    "title echoed back instead of a summary, or an error/refusal string " +
    "('cannot', 'unknown', 'no content') in place of a real summary.",
};

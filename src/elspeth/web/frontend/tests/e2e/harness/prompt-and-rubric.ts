// The coupled triple: prompt + assumption rubric (dim c) + judge rubric (dim d).
// If the prompt changes, BOTH rubrics must change in lockstep (spec §11).

export const HARNESS_VERSION = "1.0.0";

// Semantically equal to CANONICAL_TUTORIAL_PROMPT but a DIFFERENT string, so
// SHA-256(prompt+model_id) misses the tutorial cache (is_canonical_prompt=false
// in tutorial_service.py) → fresh live composition every run.
export const FIXED_PROMPT =
  "Build a source containing the URLs of five Australian government agency " +
  "websites of your choosing. Set the abuse contact to noreply@dta.gov.au and " +
  "the scraping reason to 'DTA technical demonstration'. For each page, fetch " +
  "the HTML and use an LLM to determine that agency's primary brand colours. " +
  "Then drop the HTML field and write the remaining results to a JSON file.";

// Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
// Graded on kind, not exact wording. InterpretationKind values come from the
// backend enum (kind field on InterpretationEventResponse).
export const ASSUMPTION_RUBRIC = {
  // Composer invented the 5 URLs and the subjective colour criterion → verify.
  expectVerify: ["invented_source", "vague_term"] as const,
  // The composer MAY also stage a prompt-template review for the LLM node; this
  // is acceptable-but-not-required, so it is neither under- nor over-flagging.
  allowOptional: ["llm_prompt_template"] as const,
  // Explicit values the user stated → must NOT be raised for review.
  expectWaive: ["abuse_contact", "scraping_reason"] as const,
};

// Dimension (d): the judge rubric the agent applies to recorded output rows.
export const JUDGE_RUBRIC = {
  // Mechanical (computed in the spec/aggregator, NOT by the judge):
  minReachableSources: 5, // all 5 invented URLs must have fetched (scrape op status ok)
  maxDiscardedRows: 0, // discarded_row_count from TutorialRunOutput
  minSubstantiveRows: 4, // >= 4 of 5 rows carry a meaningful colour value
  // Judge-scored (0..1), pass threshold:
  judgePassThreshold: 0.7,
  // The judge question (structured), applied per batch over recorded rows:
  judgePrompt:
    "Given the task (extract each agency's primary brand colours from its " +
    "homepage) and these output rows, score 0..1: are the colour values real, " +
    "specific, and plausibly correct (not nulls, not 'cannot determine', not " +
    "fabricated when the page shows no clear palette)? Penalise degenerate or " +
    "hallucinated answers.",
};

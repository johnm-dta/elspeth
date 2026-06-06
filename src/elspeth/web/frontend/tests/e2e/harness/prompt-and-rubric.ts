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
  // These are InterpretationKind values (the `kind` field on an interpretation
  // event): contracts/composer_interpretation.py enum =
  // vague_term | invented_source | llm_prompt_template | pipeline_decision | llm_model_choice.
  expectVerify: ["invented_source", "vague_term"] as const,
  // The composer MAY also stage a prompt-template review for the LLM node; this
  // is acceptable-but-not-required, so it is neither under- nor over-flagging.
  allowOptional: ["llm_prompt_template"] as const,
  // Over-flagging (spec §5): the composer raised a review on a value the USER
  // STATED EXPLICITLY (the abuse contact + the scraping reason in FIXED_PROMPT).
  //
  // IMPORTANT — these are NOT InterpretationKind values. "abuse_contact" and
  // "scraping_reason" are web_scrape.http field paths surfaced through the
  // implicit-decisions mechanism (composer_meta.implicit_decisions), a DIFFERENT
  // surface from interpretation events. The earlier rubric compared these
  // strings against interpretation-event `kind`, which can NEVER match, so the
  // over-flag check was dead.
  //
  // The correct over-flag signal is: did the composer raise an interpretation
  // review whose semantic TARGET is one of these explicitly-stated values?
  // We detect that by matching the review's `user_term` (LLM phrasing varies, so
  // we match on substrings/synonyms, not exact kind). `overFlagTermPatterns`
  // drives that match in the spec classifier.
  overFlagTerms: ["abuse_contact", "scraping_reason"] as const,
  overFlagTermPatterns: [/abuse|contact|noreply|email/i, /scrap\w*\s*reason|reason|purpose|demonstration/i] as const,
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

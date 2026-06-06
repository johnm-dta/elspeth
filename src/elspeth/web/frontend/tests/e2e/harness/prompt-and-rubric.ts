// The coupled triple: prompt + assumption rubric (dim c) + judge rubric (dim d).
// If the prompt changes, BOTH rubrics must change in lockstep (spec §11).

export const HARNESS_VERSION = "1.0.0";

// Semantically equal to CANONICAL_TUTORIAL_PROMPT (a SCORING task, like the
// canonical "rate how cool they are") but a DIFFERENT string, so
// SHA-256(prompt+model_id) misses the tutorial cache (is_canonical_prompt=false
// in tutorial_service.py) → fresh live composition every run.
//
// SCORING (not extraction): the composer authors a rating CRITERION and we supply
// the 1-10 SCALE, so per the skill doctrine (pipeline_composer.md:466-492 — score/
// rate/rank/rubric semantics) a `vague_term` review is UNAMBIGUOUSLY required.
// (The earlier "primary brand colours" prompt was extraction, where vague_term is
// debatable — it made dim-c's vague_term expectation contestable. This restores
// the canonical scoring shape so dim-c tests assumption-handling cleanly.)
export const FIXED_PROMPT =
  "Build a source containing the URLs of five Australian government agency " +
  "websites of your choosing. Set the abuse contact to noreply@dta.gov.au and " +
  "the scraping reason to 'DTA technical demonstration'. For each page, fetch " +
  "the HTML and use an LLM to rate, on a scale of 1 to 10, how visually " +
  "impressive the homepage is, with a one-sentence justification. Then drop the " +
  "HTML field and write the remaining results to a JSON file.";

// Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
// Graded on kind, not exact wording. InterpretationKind values come from the
// backend enum (kind field on InterpretationEventResponse).
export const ASSUMPTION_RUBRIC = {
  // Composer invented the 5 URLs and the subjective rating criterion → verify.
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
  minSubstantiveRows: 4, // >= 4 of 5 rows carry a real rating + justification
  // Judge-scored (0..1), pass threshold:
  judgePassThreshold: 0.7,
  // The judge question (structured), applied per batch over recorded rows:
  judgePrompt:
    "Given the task (rate each agency homepage 1-10 for how visually impressive " +
    "it is, with a one-sentence justification) and these output rows, score " +
    "0..1: does each row carry a real numeric rating in range AND a justification " +
    "that actually refers to the page? Penalise degenerate output — all-identical " +
    "ratings, missing/null scores, 'cannot determine', or boilerplate " +
    "justifications that ignore the specific page.",
};

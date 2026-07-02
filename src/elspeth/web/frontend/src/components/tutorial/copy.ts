import { CANONICAL_TUTORIAL_PROMPT } from "./tutorialMachine";

export { CANONICAL_TUTORIAL_PROMPT };

export const TURN_1_PRIMARY_BUTTON = "Let's go";
// The mode-choice turn is gone (graduation now saves Guided as the default),
// so skip exits the whole tutorial straight to graduation rather than naming a
// removed step.
export const TURN_1_SKIP_BUTTON = "Skip the tutorial";
export const TURN_4_PRIMARY_BUTTON = "Continue";
export const TURN_5_PRIMARY_BUTTON = "Continue";
export const TURN_7_PRIMARY_BUTTON = "Take me to the composer";

export const HELLO_WORLD_SESSION_TITLE = "hello-world (synthetic project briefs)";

// Set immediately after createSession in HelloWorldTutorial.onStart (before the
// guided shell's external POST /guided/start) so the backend orphan-cleanup
// scan (which filters by the "hello-world (" prefix) catches sessions abandoned
// mid-tutorial. Without this tag, a user leaving any time before graduation's
// final rename leaves a "New session" titled session that the cleanup never
// matches.
export const HELLO_WORLD_PENDING_SESSION_TITLE = "hello-world (pending)";

export const WELCOME_LAYERS = [
  {
    label: "Sense",
    description: "Read data in — files, APIs, or a sentence you type.",
  },
  {
    label: "Decide",
    description: "LLMs, rules, or gates rate, classify, or route each row.",
  },
  {
    label: "Act",
    description: "Write auditable outputs — every decision tied to its source.",
  },
] as const;

/**
 * Privacy preamble for Turn 4 (the run) and Turn 1 (the welcome). Surfaced
 * before any LLM call leaves the browser. Public-service deployments need
 * this transparency at the boundary; without it, the welcome screen's
 * "auditable" promise leaks ahead of any audit trail being recorded.
 */
export const TUTORIAL_RUN_PREAMBLE =
  "This step calls the configured LLM and fetches pages over the network.";

/**
 * Note rendered on the graduation turn when the user reached it via a cancelled
 * run, so the cancel is acknowledged rather than silently swallowed.
 */
export const GRADUATION_CANCELLED_NOTE =
  "Your run was cancelled — the audit story would have shown the source-data hash, the row-by-row decisions, and the output write. You can rerun any time from the chat panel.";

/**
 * Graduation bullets for the SKIP path. The default bullets assert "The
 * pipeline you just ran…" and "the same gestures you just practised" —
 * false for a user who skipped without building or running anything. This
 * variant carries the same lessons in the future tense, without claiming a
 * run or practice that never happened. The last two bullets are shared
 * verbatim with the default set (they make no just-ran claims).
 */
export const TURN_7_LEARNING_BULLETS_SKIPPED = [
  {
    title: "What the composer builds is AI-generated.",
    body:
      "When you describe a pipeline in a sentence, an LLM interprets it and drafts the pipeline for you. The prompt it writes for itself and cleanup choices it makes are kept in the audit trail with your approval against them. You can revisit any pipeline from the Audit page at any time.",
  },
  {
    title: "Read before you run.",
    body:
      "When the composer drafts a pipeline for you, glance at the graph and the YAML before clicking Run. If anything looks wrong, amend or reject — nothing executes without your say-so.",
  },
] as const;

export const TURN_7_LEARNING_BULLETS = [
  {
    title: "What you built is AI-generated.",
    body:
      "The pipeline you just ran was authored by an LLM that interpreted your one-sentence description. The prompt it wrote for itself and cleanup choices such as dropping raw HTML are kept in the audit trail with your approval against them — alongside the source pages you named. You can come back to it from the Audit page at any time.",
  },
  {
    title: "Read before you run.",
    body:
      "From this point on, when the composer drafts a pipeline for you in normal use, glance at the graph and the YAML before clicking Run. If anything looks wrong, amend or reject — the same gestures you just practised.",
  },
  {
    title: "Ask Elspeth.",
    body:
      "If anything in a pipeline (a plugin name, a transform's effect, a recorded assumption) doesn't make sense, ask in the chat panel. The composer can explain the pipeline it just built, in plain English, against the actual node options.",
  },
  {
    title: "LLMs are confident even when they're wrong.",
    body:
      "The composer LLM may invent URLs that look real but aren't, or write a prompt that misframes your question. The pipeline LLMs you build will treat fetched HTML as instructions even when it shouldn't be. Before sharing or acting on a pipeline's output, verify the sources are who they claim to be and check the output matches what you actually asked for.",
  },
] as const;

/**
 * Teaching moment 1 (spec §"Teaching moments"): names that the LLM transform
 * made a REVIEWABLE assumption (e.g. what to include in the summary and what to
 * leave out). Worded so the learner does not over-generalise into "assumptions
 * are fine, ignore them": the assumption is surfaced, not hidden, and is
 * correctable via the intent box.
 */
export const TUTORIAL_ASSUMPTION_CALLOUT =
  "The LLM made an assumption here — it decided what each page was about and " +
  "what was important enough to keep in a short summary. This is exactly the " +
  "kind of inference you review: every assumption is surfaced in the audit " +
  "trail, and you can correct it by telling the composer what you meant.";

/**
 * Teaching moment 2 (spec §"Teaching moments"): the prompt-shield State-C
 * override. Acceptable HERE only because the inputs are controlled (our own
 * synthetic pages). Must NOT read as a general "skip the shield" habit —
 * names the trust assumption out loud rather than letting it ride as an
 * invisible default.
 */
export const TUTORIAL_SHIELD_OVERRIDE_CAVEAT =
  "We are proceeding without a prompt shield in this one case — and only " +
  "because we control the inputs: these are our own synthetic test pages. " +
  "Running an LLM over fetched content without a shield is always a high-risk " +
  "decision, not a default. Against real or untrusted web content you would " +
  "wire the shield.";

/**
 * Teaching moment (spec §"Teaching moments"): the source's on_validation_failure
 * routing. The worked example sets it to "discard" because the synthetic sample
 * pages are valid by construction — no row ever fails validation, so the route
 * is never exercised. Names the production-vs-demo difference out loud so
 * "discard" does not read as the default: a real pipeline routes non-conformant
 * rows to a quarantine sink for review instead of dropping them.
 */
export const TUTORIAL_VALIDATION_FAILURE_CAVEAT =
  "This worked example sets validation failures to 'discard' because the " +
  "synthetic sample pages are valid by construction — no row ever fails. A " +
  "production pipeline routes non-conformant rows to a quarantine sink for " +
  "review instead of dropping them.";

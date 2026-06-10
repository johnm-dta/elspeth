import { CANONICAL_TUTORIAL_PROMPT } from "./tutorialMachine";

export { CANONICAL_TUTORIAL_PROMPT };

export const TURN_1_PRIMARY_BUTTON = "Let's go";
export const TURN_1_SKIP_BUTTON = "Skip to mode choice";
export const TURN_2_PRIMARY_BUTTON = "Build it";
export const TURN_2_RESTORE_BUTTON = "Restore the starter prompt";
export const TURN_2B_PRIMARY_BUTTON = "Looks good";
export const TURN_2B_ASSUMPTIONS_BLOCKED_STATUS =
  "Approve the assumptions above first";
export const TURN_2B_ASSUMPTIONS_READY_STATUS =
  "All assumptions approved; ready to run.";
export const TURN_3_PRIMARY_BUTTON = "Looks good, run it";
export const TURN_4_PRIMARY_BUTTON = "Continue";
export const TURN_5_PRIMARY_BUTTON = "Continue";
export const TURN_6_PRIMARY_BUTTON = "Save and go";
export const TURN_7_PRIMARY_BUTTON = "Take me to the composer";

export const HELLO_WORLD_SESSION_TITLE = "hello-world (cool government pages)";

// Set immediately after createSession in buildTutorialDraft so the backend
// orphan-cleanup scan (which filters by the "hello-world (" prefix) catches
// sessions abandoned mid-tutorial. Without this tag, a user leaving any
// time before Turn 6's rename leaves a "New session" titled session that
// the cleanup never matches.
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
  "This is calling the configured LLM and fetching the URLs the composer chose.";

/** Note rendered on Turn 6 when the user reached it via a cancelled run. */
export const TURN_6_CANCELLED_NOTE =
  "Your run was cancelled — the audit story would have shown the source-data hash, the row-by-row decisions, and the output write. You can rerun any time from the chat panel.";

/** Body copy for Turn 6 that names the demonstrated shape (freeform) and
 * explains why Guided is the default. Without this the user is asked to
 * pick between two modes after experiencing only one. */
export const TURN_6_INTRO_BODY =
  "That was freeform — you described it, the composer built it, you ran it. Guided breaks the same kind of prompt into checkpointed steps you can review before each one runs. The default is Guided; you can change it any time from Settings.";

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

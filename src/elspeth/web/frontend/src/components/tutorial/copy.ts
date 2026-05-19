import { CANONICAL_TUTORIAL_PROMPT } from "./tutorialMachine";

export { CANONICAL_TUTORIAL_PROMPT };

export const TURN_1_PRIMARY_BUTTON = "Let's go";
export const TURN_1_SKIP_BUTTON = "I've used ELSPETH before";
export const TURN_2_PRIMARY_BUTTON = "Build it";
export const TURN_2_RESTORE_BUTTON = "Restore the starter prompt";
export const TURN_2B_PRIMARY_BUTTON = "Show me the graph";
export const TURN_3_PRIMARY_BUTTON = "Looks good, run it";
export const TURN_4_PRIMARY_BUTTON = "Continue";
export const TURN_5_PRIMARY_BUTTON = "Continue";
export const TURN_6_PRIMARY_BUTTON = "Save and go";

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
    description: "where data comes from",
  },
  {
    label: "Decide",
    description: "what happens to each row",
  },
  {
    label: "Act",
    description: "where results go",
  },
] as const;

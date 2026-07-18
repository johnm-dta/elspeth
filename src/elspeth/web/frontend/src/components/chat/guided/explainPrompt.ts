// src/components/chat/guided/explainPrompt.ts

/**
 * Canned chat message sent by the decision card's "Explain this step" button.
 * It rides the NORMAL guided-chat path (a real user turn + assistant reply in
 * the transcript, pending strip while it runs) — the button is one-click
 * sugar for typing the question, which matters most in the tutorial where the
 * locked prompt box leaves learners no way to ask "why?".
 *
 * ChatPanel's tutorialStepBuilt check EXCLUDES turns with exactly this
 * content: on the confirm-only tutorial wire step an Explain send must
 * not read as "the step's prompt was sent" and prematurely swap the locked
 * box for the Sent line. Exact string identity is the filter — change the
 * copy here and nowhere else.
 */
export const GUIDED_EXPLAIN_MESSAGE =
  "Explain what I'm seeing on this step: what has been set up so far, why you chose it, and what the settings mean.";

import type { GuidedSession } from "@/types/guided";

/**
 * A rationale is a HEADLINE (rendered as the decision card's h2, plain text,
 * no markdown). Long multi-paragraph replies belong in the transcript bubble
 * — using them verbatim here produced a wall of bold text when a model wrote
 * an essay (or leaked its tool scratchpad) into assistant_message. Beyond
 * this length the caller falls back to the static step purpose.
 */
const RATIONALE_MAX_LENGTH = 200;

/**
 * The current step's latest assistant rationale (the LLM's "what I built"
 * summary), used as the prominent decision headline. Highest-seq assistant
 * turn whose step matches the active step; null when none OR when the turn
 * doesn't read as a headline (the caller falls back to the static step
 * purpose):
 *  - only the FIRST non-empty line is considered (the full reply lives in
 *    the conversation bubble; the headline must not duplicate it), and
 *  - a line that is over-long or carries tool-call scaffolding is rejected.
 */
export function latestAssistantRationale(session: GuidedSession): string | null {
  let best: { seq: number; content: string } | null = null;
  for (const turn of session.chat_history) {
    if (turn.role !== "assistant" || turn.step !== session.step) continue;
    if (best === null || turn.seq > best.seq) best = { seq: turn.seq, content: turn.content };
  }
  if (best === null) return null;
  const firstLine = best.content
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line !== "");
  if (firstLine === undefined || firstLine.length > RATIONALE_MAX_LENGTH) return null;
  if (/<\/?tool_(call|response)/i.test(firstLine)) return null;
  return firstLine;
}

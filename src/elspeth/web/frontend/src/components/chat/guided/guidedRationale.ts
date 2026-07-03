import type { GuidedSession } from "@/types/guided";
import { GUIDED_EXPLAIN_MESSAGE } from "./explainPrompt";

/**
 * A rationale is a HEADLINE (rendered as the decision card's h2, plain text,
 * no markdown). Long multi-paragraph replies belong in the transcript bubble
 * — using them verbatim here produced a wall of bold text when a model wrote
 * an essay (or leaked its tool scratchpad) into assistant_message. Beyond
 * this length the caller falls back to the static step purpose.
 */
const RATIONALE_MAX_LENGTH = 200;

/**
 * The headline renders as PLAIN TEXT in an h2, so markdown emphasis arrives
 * as literal asterisks ("You're at **Step 1: Source**" — operator-observed).
 * Unwrap the common inline tokens; this is a display sanitiser, not a parser
 * — anything it doesn't recognise passes through unchanged.
 */
function stripInlineMarkdown(line: string): string {
  return line
    .replace(/^#{1,6}\s+/, "")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/__([^_]+)__/g, "$1")
    .replace(/\*([^*]+)\*/g, "$1")
    .replace(/`([^`]+)`/g, "$1");
}

/**
 * The current step's latest assistant rationale (the LLM's "what I built"
 * summary), used as the prominent decision headline. Highest-seq assistant
 * turn whose step matches the active step; null when none OR when the turn
 * doesn't read as a headline (the caller falls back to the static step
 * purpose):
 *  - replies to the Explain button's canned question are SKIPPED — an
 *    explanation of the current state is not a build rationale, and letting
 *    it hijack the headline replaced "what I built" with "You're at Step 1…"
 *    (identified by the immediately preceding user turn being the exact
 *    canned message),
 *  - only the FIRST non-empty line is considered (the full reply lives in
 *    the conversation bubble; the headline must not duplicate it), and
 *  - a line that is over-long or carries tool-call scaffolding is rejected;
 *    inline markdown emphasis is unwrapped (the h2 is plain text).
 */
export function latestAssistantRationale(session: GuidedSession): string | null {
  const bySeq = new Map(session.chat_history.map((turn) => [turn.seq, turn]));
  let best: { seq: number; content: string } | null = null;
  for (const turn of session.chat_history) {
    if (turn.role !== "assistant" || turn.step !== session.step) continue;
    const preceding = bySeq.get(turn.seq - 1);
    if (
      preceding !== undefined &&
      preceding.role === "user" &&
      preceding.content === GUIDED_EXPLAIN_MESSAGE
    ) {
      continue;
    }
    if (best === null || turn.seq > best.seq) best = { seq: turn.seq, content: turn.content };
  }
  if (best === null) return null;
  const firstLine = best.content
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line !== "");
  if (firstLine === undefined || firstLine.length > RATIONALE_MAX_LENGTH) return null;
  if (/<\/?tool_(call|response)/i.test(firstLine)) return null;
  return stripInlineMarkdown(firstLine);
}

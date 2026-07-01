import type { GuidedSession } from "@/types/guided";

/**
 * The current step's latest assistant rationale (the LLM's "what I built"
 * summary), used as the prominent decision headline. Highest-seq assistant
 * turn whose step matches the active step; null when none (the caller falls
 * back to the static step purpose).
 */
export function latestAssistantRationale(session: GuidedSession): string | null {
  let best: { seq: number; content: string } | null = null;
  for (const turn of session.chat_history) {
    if (turn.role !== "assistant" || turn.step !== session.step) continue;
    if (best === null || turn.seq > best.seq) best = { seq: turn.seq, content: turn.content };
  }
  return best === null ? null : best.content;
}

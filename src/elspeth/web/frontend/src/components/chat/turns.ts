// Render-time coalescing of audit-grade chat messages into user-facing turns.
//
// The backend persists one chat_messages row per LLM round-trip — required by
// Tier-1 audit doctrine. A single user prompt that triggers multiple sequential
// tool-call rounds therefore lands as one user row + N assistant rows in the DB
// (with role="tool" rows interleaved when audit views opt in). Rendering those
// N rows as N separate bubbles leaks audit granularity into the chat UI; the
// user thinks of "what the agent did in response to my message" as a single
// turn.
//
// This module projects the audit stream into user-visible turns without
// mutating the underlying messages. Each turn carries the underlying rows so
// the rendering layer can still attribute, key, and link back to audit
// artifacts when needed.
import type { ChatMessage, ToolCall } from "@/types/api";

export type ChatTurnKind = "user" | "agent" | "system";

export interface ChatTurn {
  /** Stable key — first underlying message's id. */
  id: string;
  kind: ChatTurnKind;
  /** Underlying audit-grade rows in the turn, in original order. */
  messages: ChatMessage[];
  /** Union of tool_calls across every message in the turn, in emission order. */
  aggregatedToolCalls: ToolCall[];
  /**
   * Last non-empty `content` produced by any assistant row in the turn, or ""
   * when every row in the turn is empty (e.g. a turn that is still mid-flight
   * with only tool-call rows emitted so far).
   */
  finalContent: string;
  /**
   * The message to attribute the bubble to for actions like copy/retry/fork:
   * the last assistant row with non-empty content if one exists, else the last
   * message in the turn. For user/system turns this is the sole message.
   */
  primaryMessage: ChatMessage;
  /**
   * Atomic-reveal contract: true once the turn is safe to render in the chat.
   *
   * - `user` / `system` turns are standalone audit rows and always complete.
   * - `agent` turns become complete the moment any assistant row in the turn
   *   carries non-empty content (i.e. the LLM's text reply has landed). While
   *   the turn contains only tool-call rows, it stays incomplete and the
   *   rendering layer hides the bubble (a placeholder "thinking" affordance
   *   occupies the slot instead, painted by ChatPanel based on isComposing).
   *
   * The contract is purely client-side and derived from already-present audit
   * rows — no backend "turn_end" signal is required. This intentionally
   * reverses the prior "stream tool calls live" behaviour: the assistant turn
   * is presented atomically once assembled, rather than progressively as
   * audit rows trickle in.
   */
  isComplete: boolean;
}

/**
 * Group an ordered list of audit-grade chat messages into user-visible turns.
 *
 * Rules:
 * - role="user" → emits a `user` turn containing only that message.
 * - role="system" → emits a `system` turn containing only that message.
 *   (System messages render as a centred banner and stay visually distinct;
 *   absorbing them into a surrounding agent turn would hide audit markers
 *   like "Pipeline reverted to version N.")
 * - role="assistant" or role="tool" → extends the current `agent` turn, or
 *   opens a new one if the previous turn was not `agent`. `tool` rows are
 *   normally filtered server-side but are absorbed defensively here so an
 *   accidental leak doesn't produce orphan bubbles.
 */
export function groupIntoTurns(messages: ChatMessage[]): ChatTurn[] {
  const turns: ChatTurn[] = [];
  let current: MutableTurn | null = null;

  for (const message of messages) {
    if (message.role === "user") {
      if (current) turns.push(freeze(current));
      current = null;
      turns.push(freeze(makeStandaloneTurn("user", message)));
      continue;
    }
    if (message.role === "system") {
      if (current) turns.push(freeze(current));
      current = null;
      turns.push(freeze(makeStandaloneTurn("system", message)));
      continue;
    }
    // assistant or tool
    if (current === null) {
      current = makeAgentTurn(message);
    } else {
      extendAgentTurn(current, message);
    }
  }
  if (current) turns.push(freeze(current));
  return turns;
}

interface MutableTurn {
  id: string;
  kind: ChatTurnKind;
  messages: ChatMessage[];
  aggregatedToolCalls: ToolCall[];
  finalContent: string;
  primaryMessage: ChatMessage;
  isComplete: boolean;
}

function makeStandaloneTurn(kind: ChatTurnKind, message: ChatMessage): MutableTurn {
  return {
    id: message.id,
    kind,
    messages: [message],
    aggregatedToolCalls: [],
    finalContent: message.content ?? "",
    primaryMessage: message,
    // user / system turns are standalone audit rows: always complete.
    isComplete: true,
  };
}

function makeAgentTurn(message: ChatMessage): MutableTurn {
  const turn: MutableTurn = {
    id: message.id,
    kind: "agent",
    messages: [message],
    aggregatedToolCalls: [],
    finalContent: "",
    primaryMessage: message,
    // Provisional — flipped to true by absorb() the moment any assistant row
    // in the turn carries non-empty content. See ChatTurn.isComplete docs.
    isComplete: false,
  };
  absorb(turn, message);
  return turn;
}

function extendAgentTurn(turn: MutableTurn, message: ChatMessage): void {
  turn.messages.push(message);
  absorb(turn, message);
}

function absorb(turn: MutableTurn, message: ChatMessage): void {
  if (message.tool_calls && message.tool_calls.length > 0) {
    for (const call of message.tool_calls) turn.aggregatedToolCalls.push(call);
  }
  // Track the most recent message overall (acts as the fallback primary when
  // no row in the turn has content).
  turn.primaryMessage = message;
  if (message.content && message.content.length > 0) {
    turn.finalContent = message.content;
    // Atomic-reveal contract: an agent turn is complete the moment any
    // assistant row carries non-empty content. Standalone (user/system) turns
    // are initialised complete by makeStandaloneTurn — this branch is a no-op
    // for them, but it preserves the invariant uniformly.
    turn.isComplete = true;
  }
}

function freeze(turn: MutableTurn): ChatTurn {
  // Re-derive primaryMessage now that the turn is closed: prefer the last
  // message with non-empty content, otherwise keep the current pointer (which
  // already tracks the last message in the turn).
  if (turn.kind === "agent") {
    const lastWithContent = lastIndex(turn.messages, (m) => Boolean(m.content));
    if (lastWithContent !== -1) {
      turn.primaryMessage = turn.messages[lastWithContent];
    }
  }
  return turn;
}

function lastIndex<T>(items: T[], pred: (item: T) => boolean): number {
  for (let i = items.length - 1; i >= 0; i--) {
    if (pred(items[i])) return i;
  }
  return -1;
}

/**
 * Build a single representative ChatMessage for the turn, suitable for passing
 * into MessageBubble as `message`. For user/system turns this is the sole
 * underlying message. For agent turns it is the primary message with
 * `tool_calls` overridden to the aggregated set and `content` overridden to
 * the final non-empty content (or "" while the turn is still mid-flight).
 *
 * The underlying audit rows remain untouched on the turn; this is rendering
 * synthesis, not persistence.
 */
export function turnRepresentativeMessage(turn: ChatTurn): ChatMessage {
  if (turn.kind !== "agent") return turn.primaryMessage;
  return {
    ...turn.primaryMessage,
    tool_calls: turn.aggregatedToolCalls.length > 0 ? turn.aggregatedToolCalls : null,
    content: turn.finalContent,
  };
}

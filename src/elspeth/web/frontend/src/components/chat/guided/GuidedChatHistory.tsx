// src/components/chat/guided/GuidedChatHistory.tsx
//
// Per-step chat history rendered inline above the wizard turn widget
// (Phase A slice 6).  Reads ``guidedSession.chat_history`` directly —
// slice 5 made the field server-authoritative; the frontend is a pure
// view layer over it.
//
// One rendering idiom (the tutorial-workspace bubbles, promoted to the
// only guided transcript when the workspace became the one guided
// layout): freeform's bubble CSS classes (.message-row*, .bubble*) on
// lightweight markup, deliberately NOT MessageBubble itself (it requires
// a full ChatMessage and drags copy/edit-fork/proposal machinery the
// guided transcript must not have).  Step attribution is one stage
// divider per step-change boundary — the chat may span multiple wizard
// steps, and the audit story stays visible in the UI: "this turn was at
// Source, that one at Output."
//
// Conventions inherited from the 7.x widget family + GuidedHistory:
//   - role="log" + aria-live="polite": new chat turns are announced to
//     screen readers when they enter the DOM.  This is a SEPARATE log
//     region from the one wrapping <GuidedTurn> in ChatPanel — each
//     surfaces a different kind of content event (wizard step change
//     vs conversational turn).  Two live regions is intentional;
//     announcements don't compete because they fire when each region's
//     own content mutates.
//   - aria-relevant="additions": only appended content announces, not
//     replacements (matches GuidedTurn's region semantics).
//   - Empty state returns null — the parent should not gate render
//     behind explicit checks; if there's no chat yet, this widget
//     contributes no DOM.
//
// Wire shape: ChatTurn (types/guided.ts) — role / content / seq /
// step / ts_iso.  Ordering driven by ``seq`` because two turns
// produced in the same request share a wall-clock second (slice 5
// guarantee).

import type { ChatTurn } from "@/types/guided";
import { MarkdownRenderer } from "../MarkdownRenderer";
import { GUIDED_STEP_LABELS } from "./stepLabels";

interface Props {
  /** Server-authoritative chat history from GuidedSession.chat_history. */
  chatHistory: ChatTurn[];
}

/**
 * Guided transcript: freeform bubble idiom on guided wire turns.
 *
 * User content renders as plain text: `.bubble` carries
 * `white-space: pre-wrap`, which preserves the tutorial's locked prompt's
 * newline-separated URLs exactly as authored. Assistant content goes
 * through MarkdownRenderer — its `.markdown-body` wrapper carries the
 * `white-space: normal` reset (chat.css) that stops inter-element HTML
 * whitespace rendering as blank lines. Do not swap these: markdown-ising
 * user turns collapses the authored newlines; plain-texting assistant
 * turns shows literal `**`/`#` markup.
 *
 * Returns null when chat_history is empty — the parent should not check;
 * the widget is its own empty-state contract.
 *
 * Ordering: turns are rendered in ``seq`` order rather than array order
 * to defend against the (unlikely) case where the backend returns
 * out-of-order entries.  ``seq`` is monotonic per session (slice 5
 * invariant); sort is stable.
 */
export function GuidedChatHistory({
  chatHistory,
}: Props): React.ReactElement | null {
  if (chatHistory.length === 0) {
    return null;
  }

  const sorted = [...chatHistory].sort((a, b) => a.seq - b.seq);

  const rows: React.ReactNode[] = [];
  let previousStep: ChatTurn["step"] | null = null;

  for (const turn of sorted) {
    if (turn.step !== previousStep) {
      // Stage divider: one per step-change boundary in seq order, INCLUDING
      // the transcript start (there are no per-turn step badges, so the
      // divider is the only stage attribution the opening turns get).
      // Deliberately NOT aria-hidden and NOT its own live region: it is
      // announced exactly once through the parent log's
      // aria-relevant="additions" semantics when the stage's first turn
      // lands — a nested role=status here would double-announce.
      rows.push(
        <div
          key={`stage-${turn.seq}`}
          className="message-row message-row--system"
        >
          <div className="bubble bubble-system bubble-system--stage">
            {GUIDED_STEP_LABELS[turn.step]}
            <span className="sr-only"> stage</span>
          </div>
        </div>,
      );
      previousStep = turn.step;
    }

    const isUser = turn.role === "user";
    rows.push(
      <div
        key={`turn-${turn.seq}`}
        className={`message-row ${isUser ? "message-row--user" : "message-row--assistant"}`}
        data-seq={turn.seq}
      >
        <div
          className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} message-bubble-content${isUser ? " message-bubble-content--user" : ""}`}
        >
          {/* Author attribution for assistive tech, read first (DOM order) —
              freeform's register (MessageBubble): the bubble distinguishes
              who spoke visually, but a screen reader hears a flat run of
              messages otherwise. */}
          <span className="sr-only">{isUser ? "You said:" : "ELSPETH said:"}</span>
          {isUser ? turn.content : <MarkdownRenderer content={turn.content} />}
        </div>
      </div>,
    );
  }

  return (
    <div
      className="guided-chat-bubbles"
      role="log"
      aria-label="Step chat history"
      aria-live="polite"
      aria-relevant="additions"
    >
      {rows}
    </div>
  );
}

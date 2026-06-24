// src/components/chat/guided/GuidedChatHistory.tsx
//
// Per-step chat history rendered inline above the wizard turn widget
// (Phase A slice 6).  Reads ``guidedSession.chat_history`` directly —
// slice 5 made the field server-authoritative; the frontend is a pure
// view layer over it.
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
//   - Per-step badge on each entry: the chat may span multiple wizard
//     steps if the user back-buttons (Phase A.5 makes this concrete
//     with proactive openers).  The step badge makes the audit
//     story visible in the UI: "this turn was at Sources, that one at
//     Sinks."
//
// Wire shape: ChatTurn (types/guided.ts) — role / content / seq /
// step / ts_iso.  Ordering driven by ``seq`` because two turns
// produced in the same request share a wall-clock second (slice 5
// guarantee).

import type { ChatTurn, GuidedStep } from "@/types/guided";

/**
 * Human-readable step labels — mirror of GuidedHistory.STEP_LABELS.
 * Duplicated rather than imported to avoid a circular dependency
 * between two widgets in the same directory; the table is short and
 * the wire shape is closed.
 *
 * CLOSED LIST — must cover every GuidedStep member.  Adding a new step
 * here without updating the wizard playbook (or vice versa) breaks the
 * audit/UX correspondence.
 */
const STEP_LABELS: Record<GuidedStep, string> = {
  step_1_source: "Source",
  step_2_sink: "Sink",
  step_2_5_recipe_match: "Recipe match",
  step_3_transforms: "Transforms",
  step_4_wire: "Wire",
};

interface Props {
  /** Server-authoritative chat history from GuidedSession.chat_history. */
  chatHistory: ChatTurn[];
}

/**
 * Per-step chat log.  Returns null when chat_history is empty — the
 * parent should not check; the widget is its own empty-state contract.
 *
 * Ordering: turns are rendered in ``seq`` order rather than array order
 * to defend against the (unlikely) case where the backend returns
 * out-of-order entries.  ``seq`` is monotonic per session (slice 5
 * invariant); sort is stable.
 */
export function GuidedChatHistory({ chatHistory }: Props): React.ReactElement | null {
  if (chatHistory.length === 0) {
    return null;
  }

  const sorted = [...chatHistory].sort((a, b) => a.seq - b.seq);

  return (
    <div
      className="guided-chat-history"
      role="log"
      aria-label="Step chat history"
      aria-live="polite"
      aria-relevant="additions"
    >
      <ol className="guided-chat-history-list">
        {sorted.map((turn) => (
          <li
            key={turn.seq}
            className={`guided-chat-history-item guided-chat-history-item--${turn.role}`}
            data-seq={turn.seq}
          >
            <span className="guided-chat-history-role" aria-hidden="true">
              {turn.role === "user" ? "You" : "Assistant"}
            </span>
            <span className="guided-chat-history-step" aria-hidden="true">
              {STEP_LABELS[turn.step]}
            </span>
            <p className="guided-chat-history-content">
              {/* Screen-reader prefix that makes the role explicit even
                  though the visual badge is aria-hidden.  Without this
                  the SR would announce just the content; with it, the
                  conversation structure is preserved in the audio
                  rendering. */}
              <span className="visually-hidden">
                {turn.role === "user" ? "You said: " : "Assistant said: "}
              </span>
              {turn.content}
            </p>
          </li>
        ))}
      </ol>
    </div>
  );
}

// ============================================================================
// InlineSourceFallbackPrompt.tsx — Phase 5a Task 5
//
// LLM-skip safety net for the inline-source-from-chat path. The composer
// system prompt (Task 8) nudges the LLM to propose `set_pipeline` with an
// `inline_blob` source when the user's typed input looks like source data
// (URLs, comma-separated lists, short phrases). If the LLM ignores the
// nudge and never proposes a source, the user could end up stuck in a chat
// loop with no path forward. This widget is the floor: when the predicate
// in ChatPanel determines the user has typed source-shaped data and no
// source has been proposed, a small affordance surfaces above the chat
// input offering to create a source from the typed text directly.
//
// Cousin of InlineSourceCreatedTurn (Task 3, post-success informational)
// and InlineSourceDisambiguationTurn (Task 4, pre-success interactive).
// KEY DIFFERENCE from those: this widget renders OUTSIDE the chat message
// stream — it lives above the ChatInput. The other two widgets live
// inside the role="log" messages region; this one is an interactive
// affordance attached to the input control surface.
//
// Design constraints (load-bearing — do not change without re-reading the
// Phase 5a Task 5 spec):
//
//   * Root element is `<section role="region" aria-label="Inline source
//     fallback prompt">`. AT users navigate to the affordance by region
//     name; the ChatPanel wiring tests query by that aria-label.
//
//   * Accept button accessible name MUST match /create source/i. The
//     dismiss button accessible name MUST match /dismiss/i. These two
//     patterns are the contract between the widget, the ChatPanel
//     wiring, and the widget test.
//
//   * F-3 "no API jargon in user-visible chat message": the widget
//     itself emits NO API string (no "set_pipeline", no "inline_blob");
//     the natural-language framing of the chat turn that `onAccept`
//     ultimately sends is the ChatPanel handler's responsibility. This
//     widget passes the candidate text back through `onAccept` so the
//     handler can compose the conversational prelude.
//
//   * F-20 dismiss persistence is enforced by the CALLER (ChatPanel
//     gates `shouldRender` on `!isDismissed(sessionId)`). The widget
//     is intentionally state-less so it doesn't accidentally surface
//     a duplicate dismissal source-of-truth.
// ============================================================================

export interface InlineSourceFallbackPromptProps {
  /**
   * Predicate gate computed by the caller (ChatPanel). When false the
   * widget renders nothing — the gate handles ALL of:
   *   - F-20 dismiss persistence (caller checks inlineSourceStore.isDismissed).
   *   - Source-not-yet-bound check (compositionState.source).
   *   - Inflight source-tool-call check (last assistant message's tool_calls).
   *   - Source-shaped-text detection (looksLikeData on latest user message).
   * Encapsulating the gate in the caller keeps the widget a dumb render.
   */
  shouldRender: boolean;
  /**
   * The user-typed text the predicate identified as source-shaped. Echoed
   * back through `onAccept` so the caller's natural-language prelude can
   * embed it verbatim (F-3 — no API jargon).
   */
  candidateText: string;
  /**
   * Invoked when the user clicks "Create source". The
   * handler receives the candidate text so the chat-turn prelude can
   * embed it verbatim. F-3: the chat message dispatched downstream
   * MUST read as natural language, NOT as a tool invocation string.
   */
  onAccept: (candidateText: string) => void;
  /**
   * Invoked when the user clicks "Dismiss". F-20: the caller persists
   * the dismissal to inlineSourceStore so the prompt does not re-fire
   * within the same session.
   */
  onDismiss: () => void;
}

export function InlineSourceFallbackPrompt({
  shouldRender,
  candidateText,
  onAccept,
  onDismiss,
}: InlineSourceFallbackPromptProps): JSX.Element | null {
  if (!shouldRender) return null;
  return (
    <section
      role="region"
      aria-label="Inline source fallback prompt"
      data-testid="inline-source-fallback-prompt"
      className="inline-source-fallback-prompt"
    >
      <div className="inline-source-fallback-prompt-copy">
        <p className="inline-source-fallback-prompt-title">
          This looks like source data
        </p>
        <p className="inline-source-fallback-prompt-detail">
          Create a source so ELSPETH can use this text in the pipeline.
        </p>
      </div>
      <div className="inline-source-fallback-prompt-actions">
        <button
          type="button"
          className="btn btn-compact btn-primary inline-source-fallback-prompt-accept"
          onClick={() => onAccept(candidateText)}
        >
          Create source
        </button>
        <button
          type="button"
          className="btn-compact inline-source-fallback-prompt-dismiss"
          onClick={onDismiss}
        >
          Dismiss
        </button>
      </div>
    </section>
  );
}

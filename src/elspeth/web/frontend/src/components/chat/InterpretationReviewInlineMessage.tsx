// ============================================================================
// InterpretationReviewInlineMessage.tsx — Phase 5b Task 5 of
// 18b-phase-5b-frontend.md.
//
// Freeform-mode counterpart to InterpretationReviewTurn (the guided-mode
// turn-card widget).  Renders the interpretation-review affordance as an
// inline chat message — speech-bubble styled like an assistant turn, with a
// coloured side-bar to signal "action required" — rather than as a separate
// card.  The wire contract, error mapping, 8 KB amendment cap, and opt-out
// flow are IDENTICAL to the guided-mode widget; both consume the shared
// `useInterpretationResolver` hook so a fix to one path automatically
// covers the other.
//
// Surface differences from the guided-mode widget:
//
//   * Dispatched by ChatPanel from `pendingBySession[activeSessionId]`,
//     rendered inline in the chat message stream (not by GuidedTurn).
//   * `data-testid="interpretation-review-inline-message"` is the discriminator
//     the dispatch-predicate negative-case test (18b §5 test 17) queries to
//     prove this widget did NOT render for non-interpretation inline-blob
//     proposals.
//   * NO focus-on-mount effect.  The guided turn focuses its accept button
//     on mount because mounting the widget IS the AT user's entry to that
//     surface.  In freeform mode the chat input lives BELOW this widget;
//     yanking focus on mount would interrupt a user mid-typing.  Users
//     reach the buttons via Tab from the chat input or by clicking; the
//     buttons themselves are real <button> elements and keyboard-activatable
//     by default.
//
// Accessibility:
//
//   * role="region" with aria-labelledby on the in-bubble heading so AT
//     users navigate to "Interpretation review" by region role + name.
//   * role="status" live-region announces "Your input needs review" on
//     mount; the chat panel's role="log" wraps the messages container, but
//     the status live-region nests safely (per WAI-ARIA, role="status" is
//     a polite live region distinct from the log).
//   * role="alert" error envelope on resolve failures.
//   * Opt-out link is a real <button> (no tabIndex=-1); keyboard-reachable.
// ============================================================================

import { useId, useRef, useEffect } from "react";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import {
  INTERPRETATION_AMENDMENT_MAX_BYTES,
  useInterpretationResolver,
} from "@/hooks/useInterpretationResolver";

export interface InterpretationReviewInlineMessageProps {
  /** The pending interpretation event to review. */
  event: InterpretationEvent;
  /** Owning session id; round-tripped to the store actions. */
  sessionId: string;
  /**
   * Optional callback fired after a successful resolve OR opt-out so the
   * parent can advance its own surface (e.g., scroll the chat to the next
   * exchange).  Errors do NOT fire onResolved — the widget stays mounted
   * with an error banner.
   */
  onResolved?: (newState: CompositionState | null) => void;
}

export function InterpretationReviewInlineMessage({
  event,
  sessionId,
  onResolved,
}: InterpretationReviewInlineMessageProps) {
  const {
    mode,
    amendText,
    setAmendText,
    resolveInFlight,
    showOptOutConfirm,
    displayedError,
    amendByteLength,
    amendIsTooLong,
    submitDisabled,
    primaryButtonsDisabled,
    handleUseMine,
    handleOpenAmend,
    handleCancelAmend,
    handleSubmitAmend,
    handleRequestOptOut,
    handleCancelOptOut,
    handleConfirmOptOut,
  } = useInterpretationResolver({ event, sessionId, onResolved });

  // useId scopes DOM IDs per-instance so multiple inline messages co-existing
  // in the chat stream (two pending events) don't produce colliding header
  // IDs.
  const reactId = useId();
  const headerId = `${reactId}-header`;
  const statusId = `${reactId}-status`;
  const amendInputId = `${reactId}-amend`;
  const errorId = `${reactId}-error`;

  // Focus on view toggle ONLY (not on mount, unlike the guided turn).
  // When the user opens the amend textarea, focus moves there; on Cancel,
  // focus returns to the "Change it" button so the keyboard user is not
  // dumped back at the top of the page.  Mount-focus is deliberately
  // omitted — see the file-level comment for rationale.
  const amendTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const changeItButtonRef = useRef<HTMLButtonElement | null>(null);
  const firstRunRef = useRef(true);
  useEffect(() => {
    if (firstRunRef.current) {
      firstRunRef.current = false;
      return;
    }
    if (mode === "amend") {
      amendTextareaRef.current?.focus();
    } else {
      changeItButtonRef.current?.focus();
    }
  }, [mode]);

  const userTerm = event.user_term ?? "this term";
  const llmDraft = event.llm_draft ?? "";

  return (
    <section
      role="region"
      aria-labelledby={headerId}
      data-testid="interpretation-review-inline-message"
      className="chat-message chat-message--assistant interpretation-review-inline-message"
    >
      {/*
        role="status" live-region.  Implies aria-live="polite"; AT users
        focused in the chat input below hear "Your input needs review" when
        the message appears in the stream.  Visually hidden — the body text
        already explains the affordance for sighted users.
      */}
      <div id={statusId} role="status" className="visually-hidden">
        Your input needs review
      </div>

      <h3
        id={headerId}
        className="interpretation-review-inline-message-heading"
      >
        Interpretation review
      </h3>
      <p className="interpretation-review-inline-message-body">
        When you said{" "}
        <em className="interpretation-review-inline-message-user-term">
          {userTerm}
        </em>
        , I read that as roughly{" "}
        <em className="interpretation-review-inline-message-llm-draft">
          {llmDraft}
        </em>
        . Want to adjust the definition, or use mine?
      </p>

      {displayedError !== null && (
        <div
          id={errorId}
          role="alert"
          className="interpretation-review-inline-message-error"
        >
          <strong className="interpretation-review-inline-message-error-heading">
            {displayedError.heading}
          </strong>
          <span className="interpretation-review-inline-message-error-body">
            {displayedError.body}
          </span>
        </div>
      )}

      {mode === "choose" ? (
        <div className="interpretation-review-inline-message-actions">
          <button
            type="button"
            className="btn btn-primary interpretation-review-inline-message-accept-btn"
            aria-label={`Accept the LLM's interpretation of ${userTerm}`}
            onClick={() => void handleUseMine()}
            disabled={primaryButtonsDisabled}
          >
            {resolveInFlight ? (
              <>
                <span
                  className="interpretation-review-inline-message-spinner"
                  aria-hidden="true"
                />
                Saving…
              </>
            ) : (
              "Use my interpretation"
            )}
          </button>
          <button
            ref={changeItButtonRef}
            type="button"
            className="btn interpretation-review-inline-message-amend-btn"
            aria-label={`Edit the interpretation of ${userTerm}`}
            onClick={handleOpenAmend}
            disabled={primaryButtonsDisabled}
          >
            Change it: I meant…
          </button>
        </div>
      ) : (
        <div className="interpretation-review-inline-message-amend">
          <label
            htmlFor={amendInputId}
            className="interpretation-review-inline-message-amend-label"
          >
            What did you mean by{" "}
            <em>{userTerm}</em>?
          </label>
          <textarea
            ref={amendTextareaRef}
            id={amendInputId}
            className="interpretation-review-inline-message-amend-input"
            value={amendText}
            onChange={(e) => setAmendText(e.target.value)}
            rows={4}
            disabled={resolveInFlight}
          />
          {amendIsTooLong && (
            <p
              className="interpretation-review-inline-message-amend-cap-warning"
              role="status"
            >
              Amendment is {amendByteLength} bytes; the maximum is{" "}
              {INTERPRETATION_AMENDMENT_MAX_BYTES} bytes.
            </p>
          )}
          <div className="interpretation-review-inline-message-amend-actions">
            <button
              type="button"
              className="btn interpretation-review-inline-message-cancel-btn"
              onClick={handleCancelAmend}
              disabled={resolveInFlight}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary interpretation-review-inline-message-submit-btn"
              onClick={() => void handleSubmitAmend()}
              disabled={submitDisabled}
            >
              {resolveInFlight ? (
                <>
                  <span
                    className="interpretation-review-inline-message-spinner"
                    aria-hidden="true"
                  />
                  Saving…
                </>
              ) : (
                "Submit"
              )}
            </button>
          </div>
        </div>
      )}

      {/*
        Session-scope opt-out.  Rendered as a de-emphasised text button —
        same wording and contract as the guided counterpart so the user's
        muscle memory transfers between modes.  Real <button>, no
        tabIndex="-1", reachable by Tab.
      */}
      <div className="interpretation-review-inline-message-opt-out">
        <button
          type="button"
          className="interpretation-review-inline-message-opt-out-link"
          onClick={handleRequestOptOut}
          disabled={primaryButtonsDisabled}
        >
          Stop reviewing interpretations this session
        </button>
      </div>

      {showOptOutConfirm && (
        <ConfirmDialog
          title="Stop reviewing interpretations for this session?"
          message={
            "For the rest of this session, I'll bake interpretations in " +
            "automatically without asking you to review each one.  You can " +
            "audit what was baked from the session's audit-readiness panel."
          }
          confirmLabel="Stop reviewing for this session"
          cancelLabel="Keep reviewing"
          variant="default"
          onConfirm={() => void handleConfirmOptOut()}
          onCancel={handleCancelOptOut}
        />
      )}
    </section>
  );
}

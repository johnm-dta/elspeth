// ============================================================================
// InterpretationReviewTurn.tsx — Phase 5b Task 4 of 18b-phase-5b-frontend.md
//
// Guided-mode widget that surfaces the LLM's interpretation of a user term
// and asks the user to either accept it as drafted ("Use my interpretation"),
// amend it ("Change it: I meant…"), or stop being asked about interpretations
// for the rest of the session ("Stop reviewing interpretations this session").
//
// Pattern follows the convention established by InspectAndConfirmTurn
// (sibling file) and InlineSourceDisambiguationTurn (Phase 5a Task 4):
//
//   - Real <button> elements (no <div onClick>); aria-label on the two
//     primary buttons embeds event.user_term so screen-reader users hear
//     which term they're approving (sighted users see the same label via
//     visible button text).
//   - On mount, focus moves to the "Use my interpretation" button (parity
//     with InlineSourceDisambiguationTurn's F-19 focus-handoff contract).
//     When the user toggles into amend-mode the focus shifts to the
//     textarea (focus-restoration on view toggle — same convention used by
//     InspectAndConfirmTurn).
//   - role="region" with aria-labelledby pointing at the header so AT users
//     navigate to the widget by region role + a stable accessible name
//     ("Interpretation review").
//   - role="status" live-region announces "Your input needs review" on
//     mount; this is independent of (and not nested inside) any parent
//     live-region — the parent ChatPanel wraps GuidedTurn in a role="log"
//     region, but the status announcement here is structural to the
//     widget's own contract and must survive ChatPanel refactors.
//
// Behavioural state machine, error-mapping, and resolve/opt-out wiring are
// shared with the freeform-mode counterpart (InterpretationReviewInlineMessage)
// via the `useInterpretationResolver` hook in hooks/useInterpretationResolver.ts.
// This component owns only its rendering, ARIA wiring, and focus management;
// the wire/store logic and 8 KB byte cap live in the hook.
// ============================================================================

import { useEffect, useId, useRef } from "react";
import type { CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import {
  INTERPRETATION_AMENDMENT_MAX_BYTES,
  useInterpretationResolver,
} from "@/hooks/useInterpretationResolver";

// Re-export the byte-cap constant so existing callers / tests that imported
// it from this module continue to compile after the hook extraction.
export { INTERPRETATION_AMENDMENT_MAX_BYTES };

export interface InterpretationReviewTurnProps {
  /** The pending interpretation event to review. */
  event: InterpretationEvent;
  /** Owning session id; round-tripped to the store actions. */
  sessionId: string;
  /**
   * Optional callback fired after a successful resolve OR opt-out so the
   * parent can advance its own surface (e.g., scroll to the next turn,
   * dismiss the widget).  Errors do NOT fire onResolved — the widget
   * stays mounted with an error banner.
   */
  onResolved?: (newState: CompositionState | null) => void;
}

export function InterpretationReviewTurn({
  event,
  sessionId,
  onResolved,
}: InterpretationReviewTurnProps) {
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

  // useId scopes DOM IDs per-instance so multiple widgets coexisting in
  // GuidedHistory (or two open tabs of the same session in development)
  // don't produce colliding header IDs.
  const reactId = useId();
  const headerId = `${reactId}-header`;
  const statusId = `${reactId}-status`;
  const amendInputId = `${reactId}-amend`;
  const errorId = `${reactId}-error`;

  // Refs for focus management.
  const useMineButtonRef = useRef<HTMLButtonElement | null>(null);
  const amendTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const changeItButtonRef = useRef<HTMLButtonElement | null>(null);

  // Focus on mount: the "Use my interpretation" button.  Mirrors
  // InlineSourceDisambiguationTurn's F-19 focus-handoff so keyboard
  // users don't tab from the top of the chat panel.
  //
  // This is a guided-mode-only contract: mounting the widget IS the AT
  // user's entry to the surface.  The freeform-mode inline-message
  // counterpart intentionally does NOT focus on mount because it lives
  // inside the chat log and would yank focus from a user typing in the
  // chat input below.
  useEffect(() => {
    useMineButtonRef.current?.focus();
    // Empty dep array: only fire on initial mount.  Toggling between
    // choose/amend modes handles its own focus via the mode effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Focus on view toggle.  Skip the first run (handled by the mount effect
  // above); on subsequent toggles, move focus to the new view's primary
  // control.
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
      className="guided-turn guided-interpretation-review-turn"
    >
      {/*
        Live-region announcement on mount.  role="status" implies
        aria-live="polite"; AT users focused elsewhere (e.g. the chat
        input) hear "Your input needs review" when the widget appears.
      */}
      <div id={statusId} role="status" className="visually-hidden">
        Your input needs review
      </div>

      <h3
        id={headerId}
        className="guided-interpretation-review-heading"
      >
        Interpretation review
      </h3>
      <p className="guided-interpretation-review-body">
        Before we finalise: when you said{" "}
        <em className="guided-interpretation-review-user-term">{userTerm}</em>,
        I read that as roughly{" "}
        <em className="guided-interpretation-review-llm-draft">{llmDraft}</em>.
      </p>

      {displayedError !== null && (
        <div
          id={errorId}
          role="alert"
          className="guided-interpretation-review-error"
        >
          <strong className="guided-interpretation-review-error-heading">
            {displayedError.heading}
          </strong>
          <span className="guided-interpretation-review-error-body">
            {displayedError.body}
          </span>
        </div>
      )}

      {mode === "choose" ? (
        <div className="guided-interpretation-review-actions">
          <button
            ref={useMineButtonRef}
            type="button"
            className="btn btn-primary guided-interpretation-review-accept-btn"
            aria-label={`Accept the LLM's interpretation of ${userTerm}`}
            onClick={() => void handleUseMine()}
            disabled={primaryButtonsDisabled}
          >
            {resolveInFlight ? (
              <>
                <span
                  className="guided-interpretation-review-spinner"
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
            className="btn guided-interpretation-review-amend-btn"
            aria-label={`Edit the interpretation of ${userTerm}`}
            onClick={handleOpenAmend}
            disabled={primaryButtonsDisabled}
          >
            Change it: I meant…
          </button>
        </div>
      ) : (
        <div className="guided-interpretation-review-amend">
          <label
            htmlFor={amendInputId}
            className="guided-interpretation-review-amend-label"
          >
            What did you mean by{" "}
            <em>{userTerm}</em>?
          </label>
          <textarea
            ref={amendTextareaRef}
            id={amendInputId}
            className="guided-interpretation-review-amend-input"
            value={amendText}
            onChange={(e) => setAmendText(e.target.value)}
            rows={4}
            disabled={resolveInFlight}
          />
          {amendIsTooLong && (
            <p
              className="guided-interpretation-review-amend-cap-warning"
              role="status"
            >
              Amendment is {amendByteLength} bytes; the maximum is{" "}
              {INTERPRETATION_AMENDMENT_MAX_BYTES} bytes.
            </p>
          )}
          <div className="guided-interpretation-review-amend-actions">
            <button
              type="button"
              className="btn guided-interpretation-review-cancel-btn"
              onClick={handleCancelAmend}
              disabled={resolveInFlight}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn btn-primary guided-interpretation-review-submit-btn"
              onClick={() => void handleSubmitAmend()}
              disabled={submitDisabled}
            >
              {resolveInFlight ? (
                <>
                  <span
                    className="guided-interpretation-review-spinner"
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
        Session-scope opt-out.  Rendered as a de-emphasised but Tab-reachable
        text button so the most common per-term action (accepting the draft)
        remains visually distinct from the session-wide opt-out.  Must NOT
        have tabIndex="-1" — keyboard users reach it by Tab; Enter/Space
        activates it via the native <button> contract.
      */}
      <div className="guided-interpretation-review-opt-out">
        <button
          type="button"
          className="guided-interpretation-review-opt-out-link"
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

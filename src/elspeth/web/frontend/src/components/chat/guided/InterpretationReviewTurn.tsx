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
// Wire contract:
//
//   - "Use my interpretation" → resolveEvent(sessionId, event.id,
//     { choice: 'accepted_as_drafted' }).
//   - "Change it" reveals a textarea pre-filled with event.llm_draft.
//     Submit → resolveEvent(sessionId, event.id,
//     { choice: 'amended', amended_value: <text> }).
//   - "Stop reviewing interpretations this session" opens a ConfirmDialog
//     with explicit session-scope copy; on confirm → optOut(sessionId).
//
// Error handling:
//
//   - The store's resolveEvent / optOut surface the underlying ApiError
//     on failure.  The widget catches and routes:
//       * status === 409  → "This interpretation was already resolved in
//                            another tab — reload to see the latest"
//                            (multi-tab TOCTOU; F-12).
//       * status === 422  → surface the validation error detail.
//       * any other status → generic "Could not resolve interpretation"
//                            with the detail text.
//
// Client-side validation:
//
//   - Empty amendment → Submit disabled (no request issued).
//   - Amendment exceeding INTERPRETATION_AMENDMENT_MAX_BYTES bytes
//     (UTF-8) → client-side validation error before the request is
//     issued.  The 8 KB cap mirrors the backend pydantic validator in
//     contracts/composer_interpretation.py.
// ============================================================================

import { useEffect, useId, useRef, useState } from "react";
import type {
  ApiError,
  CompositionState,
} from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";

/**
 * Maximum amendment length in UTF-8 bytes.  Mirrors the backend's
 * InterpretationResolveRequest.amended_value validator (see
 * contracts/composer_interpretation.py).  Encoding-byte length is the
 * correct unit: JavaScript .length counts UTF-16 code units, which
 * under-reports for non-BMP characters and over-reports relative to the
 * wire-side byte budget for ASCII.  We measure with TextEncoder so the
 * client-side check matches the server's evaluation exactly.
 */
export const INTERPRETATION_AMENDMENT_MAX_BYTES = 8192;

const TEXT_ENCODER = new TextEncoder();

function byteLength(text: string): number {
  return TEXT_ENCODER.encode(text).byteLength;
}

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

/** Local error envelope — narrow enough to render without leaking ApiError shape. */
interface DisplayedError {
  /** Heading text shown above the body. */
  heading: string;
  /** Body text; may contain server-provided detail. */
  body: string;
}

/**
 * Type guard: distinguishes ApiError (plain object thrown by parseResponse)
 * from arbitrary Error instances.  parseResponse throws a raw object literal
 * with a numeric `status` field; that's the signal we use here.  Falling
 * through to the generic branch is correct for AbortError or unexpected
 * runtime errors — we don't want to confuse "the network failed" with
 * "the server returned a structured 4xx".
 */
function isApiError(err: unknown): err is ApiError {
  return (
    typeof err === "object" &&
    err !== null &&
    "status" in err &&
    typeof (err as { status: unknown }).status === "number"
  );
}

function describeError(err: unknown): DisplayedError {
  if (isApiError(err)) {
    if (err.status === 409) {
      // F-12 multi-tab TOCTOU: the event was already resolved on the
      // server (typically by another browser tab on the same session).
      return {
        heading: "Already resolved",
        body:
          "This interpretation was already resolved in another tab — " +
          "reload to see the latest.",
      };
    }
    if (err.status === 422) {
      return {
        heading: "Invalid amendment",
        body: err.detail || "The server rejected the amendment.",
      };
    }
    return {
      heading: "Could not resolve interpretation",
      body: err.detail || `Server returned ${err.status}.`,
    };
  }
  return {
    heading: "Could not resolve interpretation",
    body: "An unexpected error occurred.",
  };
}

export function InterpretationReviewTurn({
  event,
  sessionId,
  onResolved,
}: InterpretationReviewTurnProps) {
  // Store actions.  Selecting individual actions (rather than the whole
  // state) keeps re-renders scoped: this widget never reads pendingBySession
  // / resolvedCountBySession / optedOutBySession directly — the parent's
  // re-render on resolution will unmount it.
  const resolveEvent = useInterpretationEventsStore((s) => s.resolveEvent);
  const optOut = useInterpretationEventsStore((s) => s.optOut);

  // useId scopes DOM IDs per-instance so multiple widgets coexisting in
  // GuidedHistory (or two open tabs of the same session in development)
  // don't produce colliding header IDs.
  const reactId = useId();
  const headerId = `${reactId}-header`;
  const statusId = `${reactId}-status`;
  const amendInputId = `${reactId}-amend`;
  const errorId = `${reactId}-error`;

  // ── Local UI state ───────────────────────────────────────────────────────
  // mode: which sub-view is showing.
  //   "choose"   → two primary buttons (Use mine / Change it).
  //   "amend"    → textarea + Submit / Cancel.
  // Distinct flag from `inFlight` because the textarea must stay editable
  // until the user clicks Submit; we don't conflate "showing the textarea"
  // with "request in flight".
  const [mode, setMode] = useState<"choose" | "amend">("choose");
  // Amendment draft; initialised lazily on first transition to amend-mode
  // so re-renders in choose-mode don't snapshot the LLM draft into local
  // state until the user actually opts to edit.
  const [amendText, setAmendText] = useState<string>("");
  // In-flight guards.  Two separate booleans rather than one shared flag
  // because the opt-out modal can be open while a resolve is pending and
  // we want each surface's spinner / disabled state to track its own
  // request.
  const [resolveInFlight, setResolveInFlight] = useState(false);
  const [optOutInFlight, setOptOutInFlight] = useState(false);
  // Confirm-modal visibility for the session-scope opt-out.
  const [showOptOutConfirm, setShowOptOutConfirm] = useState(false);
  // Last error to display.  Cleared when the user starts another action.
  const [displayedError, setDisplayedError] = useState<DisplayedError | null>(
    null,
  );

  // Refs for focus management.
  const useMineButtonRef = useRef<HTMLButtonElement | null>(null);
  const amendTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const changeItButtonRef = useRef<HTMLButtonElement | null>(null);

  // Focus on mount: the "Use my interpretation" button.  Mirrors
  // InlineSourceDisambiguationTurn's F-19 focus-handoff so keyboard
  // users don't tab from the top of the chat panel.
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

  // ── Derived state ────────────────────────────────────────────────────────
  const userTerm = event.user_term ?? "this term";
  const llmDraft = event.llm_draft ?? "";
  const trimmedAmendText = amendText.trim();
  const amendByteLength = byteLength(amendText);
  const amendIsEmpty = trimmedAmendText.length === 0;
  const amendIsTooLong = amendByteLength > INTERPRETATION_AMENDMENT_MAX_BYTES;
  const submitDisabled = amendIsEmpty || amendIsTooLong || resolveInFlight;
  const primaryButtonsDisabled = resolveInFlight || optOutInFlight;

  // ── Handlers ─────────────────────────────────────────────────────────────

  async function handleUseMine() {
    if (primaryButtonsDisabled) return;
    setDisplayedError(null);
    setResolveInFlight(true);
    try {
      const { new_state } = await resolveEvent(sessionId, event.id, {
        choice: "accepted_as_drafted",
      });
      onResolved?.(new_state);
    } catch (err) {
      setDisplayedError(describeError(err));
    } finally {
      setResolveInFlight(false);
    }
  }

  function handleOpenAmend() {
    if (primaryButtonsDisabled) return;
    setDisplayedError(null);
    setAmendText(llmDraft);
    setMode("amend");
  }

  function handleCancelAmend() {
    if (resolveInFlight) return;
    setDisplayedError(null);
    setMode("choose");
  }

  async function handleSubmitAmend() {
    if (submitDisabled) return;
    // Belt-and-suspenders: even if the Submit button is disabled, an
    // Enter-key press in the textarea (or a test that bypasses pointer
    // events) could route here with empty / too-long text.  Surface the
    // client-side validation error and don't issue the request.
    if (amendIsEmpty) {
      setDisplayedError({
        heading: "Invalid amendment",
        body: "Amendment cannot be empty.",
      });
      return;
    }
    if (amendIsTooLong) {
      setDisplayedError({
        heading: "Amendment too long",
        body:
          `Amendment is ${amendByteLength} bytes; the maximum is ` +
          `${INTERPRETATION_AMENDMENT_MAX_BYTES} bytes.`,
      });
      return;
    }
    setDisplayedError(null);
    setResolveInFlight(true);
    try {
      const { new_state } = await resolveEvent(sessionId, event.id, {
        choice: "amended",
        amended_value: amendText,
      });
      onResolved?.(new_state);
    } catch (err) {
      setDisplayedError(describeError(err));
    } finally {
      setResolveInFlight(false);
    }
  }

  function handleRequestOptOut() {
    if (primaryButtonsDisabled) return;
    setDisplayedError(null);
    setShowOptOutConfirm(true);
  }

  function handleCancelOptOut() {
    setShowOptOutConfirm(false);
  }

  async function handleConfirmOptOut() {
    setShowOptOutConfirm(false);
    setOptOutInFlight(true);
    try {
      await optOut(sessionId);
      onResolved?.(null);
    } catch (err) {
      setDisplayedError(describeError(err));
    } finally {
      setOptOutInFlight(false);
    }
  }

  // ── Render ───────────────────────────────────────────────────────────────

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
            onClick={handleUseMine}
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
              onClick={handleSubmitAmend}
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
          onConfirm={handleConfirmOptOut}
          onCancel={handleCancelOptOut}
        />
      )}
    </section>
  );
}

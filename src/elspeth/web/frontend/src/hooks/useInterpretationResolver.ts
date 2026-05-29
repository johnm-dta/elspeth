// ============================================================================
// useInterpretationResolver.ts — shared state machine for the interpretation
// review widgets (Phase 5b.18b.4 + 5b.18b.5).
//
// Two surfaces consume the interpretation-review affordances:
//
//   1. InterpretationReviewTurn (guided mode) — a turn-card widget rendered
//      inside the GuidedTurn dispatch.
//   2. InterpretationReviewInlineMessage (freeform mode) — a chat-styled
//      inline message rendered inside the ChatPanel message list.
//
// The two widgets differ ONLY in their visual rendering and a handful of
// presentational concerns (mount-focus, region heading, button copy
// placement).  The behaviour — when each handler is fired, how it interacts
// with the store, how API errors are mapped, the 8 KB amendment cap, the
// opt-out confirmation flow — is identical.  Duplicating that behaviour in
// two components would be a maintenance hazard: a fix to the 409 multi-tab
// recovery path on one widget would silently miss the other.
//
// This hook encapsulates the shared logic:
//
//   * The four-state UI state machine (mode, amendText, two in-flight
//     flags, displayed error, opt-out confirm dialog visibility).
//   * The four handlers (handleUseMine, handleSubmitAmend, handleConfirmOptOut,
//     and the open/cancel helpers).
//   * The error-shape mapper (409 → multi-tab, 422 → validation detail,
//     other → generic with detail).
//   * The 8 KB amendment cap measured in UTF-8 bytes (mirrors the backend
//     pydantic validator in contracts/composer_interpretation.py).
//
// What stays in each component:
//
//   * The actual DOM (region wrapper vs. inline-message wrapper).
//   * ARIA wiring (the headerId / statusId / errorId IDs are owned by the
//     component because their placement depends on the surface layout).
//   * Focus management on mount (the guided turn focuses the accept button
//     immediately because mounting the widget IS the AT user's entry to
//     that surface; the inline message lives inside the chat log under
//     role="log" and must NOT yank focus away from a user typing in the
//     chat input).
//
// Tier-discipline note: the hook does no I/O directly — it dispatches to
// the interpretationEventsStore's resolveEvent / optOut actions and maps
// thrown ApiError shapes back into a display-friendly envelope.  The
// store actions handle the wire write and atomicity invariants.
// ============================================================================

import { useState } from "react";
import type { ApiError, CompositionState } from "@/types/index";
import type { InterpretationEvent } from "@/types/interpretation";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";

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

export function byteLength(text: string): number {
  return TEXT_ENCODER.encode(text).byteLength;
}

/** Local error envelope — narrow enough to render without leaking ApiError shape. */
export interface DisplayedError {
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

export function describeError(err: unknown): DisplayedError {
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
      if (err.error_type === "interpretation_placeholder_unavailable") {
        return {
          heading: "Stale review",
          body:
            "This prompt template has already changed. Reload the session " +
            "and review the current prompt template again.",
        };
      }
      if (err.error_type === "interpretation_node_missing") {
        return {
          heading: "Review target removed",
          body:
            "The affected node was removed from the composition. Reload the " +
            "session to review the current draft.",
        };
      }
      if (err.error_type === "interpretation_node_mutated") {
        return {
          heading: "Review target changed",
          body:
            "The affected node is no longer the same LLM transform. Reload " +
            "the session and review the current draft.",
        };
      }
      if (err.error_type === "interpretation_resolution_unsupported") {
        return {
          heading: "Unsupported review",
          body:
            "This interpretation kind cannot be accepted inline in this " +
            "release.",
        };
      }
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

/**
 * Sub-view modes for the widget.
 *
 *   "choose"   → two primary buttons (Use mine / Change it).
 *   "amend"    → textarea + Submit / Cancel.
 */
export type ReviewMode = "choose" | "amend";

export interface UseInterpretationResolverParams {
  event: InterpretationEvent;
  sessionId: string;
  onResolved?: (newState: CompositionState | null) => void;
}

export interface UseInterpretationResolverResult {
  // ── State ────────────────────────────────────────────────────────────────
  /** Current sub-view mode. */
  mode: ReviewMode;
  /** Amendment text in the textarea (only meaningful in amend mode). */
  amendText: string;
  /** Setter passed to the textarea's onChange. */
  setAmendText: (next: string) => void;
  /** A resolve request is in flight. */
  resolveInFlight: boolean;
  /** An opt-out request is in flight. */
  optOutInFlight: boolean;
  /** The opt-out confirm modal is currently visible. */
  showOptOutConfirm: boolean;
  /** Mapped error envelope to render (null when no error). */
  displayedError: DisplayedError | null;

  // ── Derived ──────────────────────────────────────────────────────────────
  /** UTF-8 byte length of the current amendText. */
  amendByteLength: number;
  /** Trimmed amendment text is empty. */
  amendIsEmpty: boolean;
  /** Amendment exceeds INTERPRETATION_AMENDMENT_MAX_BYTES. */
  amendIsTooLong: boolean;
  /** Submit button should be disabled. */
  submitDisabled: boolean;
  /** Both primary buttons should be disabled (resolve OR opt-out in flight). */
  primaryButtonsDisabled: boolean;

  // ── Handlers ─────────────────────────────────────────────────────────────
  handleUseMine: () => Promise<void>;
  handleOpenAmend: () => void;
  handleCancelAmend: () => void;
  handleSubmitAmend: () => Promise<void>;
  handleRequestOptOut: () => void;
  handleCancelOptOut: () => void;
  handleConfirmOptOut: () => Promise<void>;
}

/**
 * Behavioural state machine for the interpretation review widgets.
 *
 * Returns the full reactive state + handler bundle so the calling
 * component focuses purely on rendering.  The hook is identity-stable
 * across renders for primitive flags (boolean / string / number) — but
 * handler references rebuild on every render (no useCallback inside).
 * Rendering components must not memoise on handler identity.
 */
export function useInterpretationResolver({
  event,
  sessionId,
  onResolved,
}: UseInterpretationResolverParams): UseInterpretationResolverResult {
  // Store actions.  Selecting individual actions (rather than the whole
  // state) keeps re-renders scoped: this hook never reads pendingBySession
  // / resolvedCountBySession / optedOutBySession directly — the parent's
  // re-render on resolution will unmount the consumer widget.
  const resolveEvent = useInterpretationEventsStore((s) => s.resolveEvent);
  const optOut = useInterpretationEventsStore((s) => s.optOut);

  // ── Local UI state ───────────────────────────────────────────────────────
  // mode: which sub-view is showing.
  // Distinct flag from `inFlight` because the textarea must stay editable
  // until the user clicks Submit; we don't conflate "showing the textarea"
  // with "request in flight".
  const [mode, setMode] = useState<ReviewMode>("choose");
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

  const llmDraft = event.llm_draft ?? "";
  const trimmedAmendText = amendText.trim();
  const amendByteLength = byteLength(amendText);
  const amendIsEmpty = trimmedAmendText.length === 0;
  const amendIsTooLong = amendByteLength > INTERPRETATION_AMENDMENT_MAX_BYTES;
  const submitDisabled = amendIsEmpty || amendIsTooLong || resolveInFlight;
  const primaryButtonsDisabled = resolveInFlight || optOutInFlight;

  async function handleUseMine(): Promise<void> {
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

  function handleOpenAmend(): void {
    if (primaryButtonsDisabled) return;
    setDisplayedError(null);
    setAmendText(llmDraft);
    setMode("amend");
  }

  function handleCancelAmend(): void {
    if (resolveInFlight) return;
    setDisplayedError(null);
    setMode("choose");
  }

  async function handleSubmitAmend(): Promise<void> {
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

  function handleRequestOptOut(): void {
    if (primaryButtonsDisabled) return;
    setDisplayedError(null);
    setShowOptOutConfirm(true);
  }

  function handleCancelOptOut(): void {
    setShowOptOutConfirm(false);
  }

  async function handleConfirmOptOut(): Promise<void> {
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

  return {
    mode,
    amendText,
    setAmendText,
    resolveInFlight,
    optOutInFlight,
    showOptOutConfirm,
    displayedError,
    amendByteLength,
    amendIsEmpty,
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
  };
}

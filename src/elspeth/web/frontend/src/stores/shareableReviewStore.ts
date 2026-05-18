/**
 * Zustand store for the Phase 6B shareable-reviews dialog state.
 *
 * Holds:
 *
 * * ``dialogOpen`` — whether the SaveForReviewDialog is mounted.
 * * ``latestResponse`` — the most recent MarkReadyForReviewResponse, used
 *   by the dialog to render the share URL + copy-to-clipboard affordance.
 * * ``inFlight`` — true while the POST is being awaited; the
 *   CompletionBar's "Save for review" button disables itself.
 * * ``error`` — the error message from the most recent failed mark
 *   attempt (typed-parse failure, 409, 404). Generic; the dialog
 *   surfaces it as a banner.
 *
 * The store is deliberately session-scoped via the caller's ``sessionId``
 * argument — opening the dialog clears the previous session's response
 * so the dialog never shows alice's token while bob is composing.
 */

import { create } from "zustand";

import { markReadyForReview } from "../api/shareableReviews";
import type { ApiError, MarkReadyForReviewResponse } from "../types/api";

export interface ShareableReviewState {
  /** Whether the SaveForReviewDialog is mounted. */
  dialogOpen: boolean;
  /** Most recent successful response from POST /mark-ready-for-review. */
  latestResponse: MarkReadyForReviewResponse | null;
  /** True while the POST is in-flight. The bar button reads this to
   *  disable itself; the dialog renders a spinner while true. */
  inFlight: boolean;
  /** Error message from the most recent attempt, or null on success. */
  error: string | null;
  /** Session_id the latestResponse belongs to, or null. Used to clear
   *  stale responses when the user switches sessions. */
  sessionIdForResponse: string | null;

  /** Open the dialog and POST mark-ready-for-review.  Resolves the
   *  in-flight promise on success/failure; state reflects the outcome
   *  via ``latestResponse`` or ``error``. */
  openAndMark: (sessionId: string) => Promise<void>;
  /** Close the dialog. Does NOT clear ``latestResponse`` — the user can
   *  reopen the dialog to see the same response. ``reset()`` is the
   *  explicit clear path. */
  close: () => void;
  /** Reset the store entirely (clears response + error + dialog state). */
  reset: () => void;
}

const _initial: Omit<ShareableReviewState, "openAndMark" | "close" | "reset"> = {
  dialogOpen: false,
  latestResponse: null,
  inFlight: false,
  error: null,
  sessionIdForResponse: null,
};

function _isApiError(value: unknown): value is ApiError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { status?: unknown }).status === "number"
  );
}

export const useShareableReviewStore = create<ShareableReviewState>((set, get) => ({
  ..._initial,

  async openAndMark(sessionId: string) {
    // Clear any stale response from a previous session before opening.
    // Without this the dialog could flash the previous session's URL.
    const current = get();
    if (current.sessionIdForResponse !== null && current.sessionIdForResponse !== sessionId) {
      set({ latestResponse: null, sessionIdForResponse: null, error: null });
    }
    set({ dialogOpen: true, inFlight: true, error: null });
    try {
      const response = await markReadyForReview(sessionId);
      set({
        latestResponse: response,
        sessionIdForResponse: sessionId,
        inFlight: false,
        error: null,
      });
    } catch (exc) {
      let message = "Could not mint a shareable link";
      if (_isApiError(exc)) {
        if (exc.status === 409) {
          message =
            exc.detail ??
            "The composition is not ready for sharing — fix the surfaced errors and try again.";
        } else if (exc.status === 404) {
          message = "Session not found.";
        } else if (exc.status === 401) {
          message = "Authentication required.";
        } else if (typeof exc.detail === "string") {
          message = exc.detail;
        }
      } else if (exc instanceof Error) {
        message = exc.message;
      }
      set({ inFlight: false, error: message, latestResponse: null });
    }
  },

  close() {
    set({ dialogOpen: false });
  },

  reset() {
    set(_initial);
  },
}));

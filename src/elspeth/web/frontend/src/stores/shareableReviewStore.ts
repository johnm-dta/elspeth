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
   *  via ``latestResponse`` or ``error``.
   *
   *  Re-entrancy: if a POST is already in flight, the call is a no-op
   *  (returns immediately without minting a second audit row). The
   *  CompletionBar button's ``inFlight`` disable handles UI; this guard
   *  closes the programmatic-re-entry hole (Enter spam, double-click
   *  before re-render).
   *
   *  Session safety: the sessionId is captured at call entry. If the
   *  store's tracking sessionId changes before the POST resolves (the
   *  user switched sessions, or another openAndMark for a different
   *  session completed first), the late response is dropped on the
   *  floor — the audit row is still minted server-side, but we never
   *  surface session A's token while session B is composing. */
  openAndMark: (sessionId: string) => Promise<void>;
  /** Close the dialog. Does NOT clear ``latestResponse`` — the user can
   *  reopen the dialog to see the same response. ``reset()`` is the
   *  explicit clear path. */
  close: () => void;
  /** Clear all dialog/response state if ``sessionId`` matches the
   *  session that owns the current response. No-op if the id doesn't
   *  match (we don't own that session's state). Use from
   *  session-switch handlers — the documented "switching sessions
   *  clears the store" contract. */
  clearForSession: (sessionId: string) => void;
  /** Reset the store entirely (clears response + error + dialog state). */
  reset: () => void;
}

const _initial: Omit<
  ShareableReviewState,
  "openAndMark" | "close" | "clearForSession" | "reset"
> = {
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

/**
 * Module-scoped state for ``openAndMark`` race control.
 *
 * - ``_openAndMarkEpoch`` is a monotonic counter; each invocation captures
 *   its epoch at start, then re-checks after the network await. Mismatch
 *   means a later call superseded us and our response must be dropped
 *   (gap 1 — stale-response race).
 * - ``_inFlightSessionId`` records which session currently has a POST in
 *   flight, used by the double-click guard (gap 2). Tracking this in
 *   module scope rather than store state keeps it out of the public
 *   ``ShareableReviewState`` type (consumers shouldn't care) and avoids
 *   needing a separate field next to the user-visible ``inFlight`` /
 *   ``sessionIdForResponse`` pair.
 */
let _openAndMarkEpoch = 0;
let _inFlightSessionId: string | null = null;

export const useShareableReviewStore = create<ShareableReviewState>((set, get) => ({
  ..._initial,

  async openAndMark(sessionId: string) {
    // DC-7 gap 2: double-click race. If a POST is already in-flight for
    // the SAME session, drop the re-entrant call — minting a second token
    // would create a duplicate audit row (append-only, can't undo). A
    // different session is allowed to proceed; the in-flight call's
    // post-await check will discard its stale response (gap 1).
    //
    // The button's UI inFlight-disable handles single-session double-click
    // at the view layer; this guard closes the programmatic-re-entry hole
    // (rapid Enter, keyboard repeat, click-before-render).
    if (_inFlightSessionId === sessionId) {
      return;
    }
    // DC-7 gap 1 prep: capture this call's epoch. After the await we
    // re-check against the live counter; mismatch means a later call
    // superseded us and our response must be dropped.
    _openAndMarkEpoch += 1;
    const myEpoch = _openAndMarkEpoch;
    _inFlightSessionId = sessionId;

    // Clear any stale response from a previous session before opening,
    // so the dialog never flashes the previous session's URL while the
    // new POST is in flight.
    const current = get();
    const staleSessionResponse =
      current.sessionIdForResponse !== null && current.sessionIdForResponse !== sessionId;
    set({
      dialogOpen: true,
      inFlight: true,
      error: null,
      latestResponse: staleSessionResponse ? null : current.latestResponse,
      sessionIdForResponse: staleSessionResponse ? sessionId : current.sessionIdForResponse,
    });

    try {
      const response = await markReadyForReview(sessionId);
      // DC-7 gap 1: post-await captured-epoch check. If another
      // openAndMark started after us (different session, or our own
      // entry was overtaken), our epoch is stale — drop the response.
      // Audit row was minted server-side regardless; we just don't
      // surface session A's token while session B is composing.
      // Note: do NOT touch _inFlightSessionId here — the superseding
      // call already owns it.
      if (_openAndMarkEpoch !== myEpoch) {
        return;
      }
      _inFlightSessionId = null;
      set({
        latestResponse: response,
        sessionIdForResponse: sessionId,
        inFlight: false,
        error: null,
      });
    } catch (exc) {
      // Same epoch check on the error path: a stale failure must not
      // overwrite the current session's successful response with an
      // error banner from a prior session.
      if (_openAndMarkEpoch !== myEpoch) {
        return;
      }
      _inFlightSessionId = null;
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

  clearForSession(sessionId: string) {
    // No-op unless the current response belongs to this session. We
    // don't blow away state that came from a different session — the
    // caller has only authority to clear what they own.
    const current = get();
    if (current.sessionIdForResponse !== sessionId) {
      return;
    }
    // Bump the epoch so any pending openAndMark for this session sees a
    // stale epoch on resume and drops its response. Also clear the
    // in-flight tracker so a subsequent openAndMark for the same id
    // isn't accidentally suppressed by the double-click guard.
    _openAndMarkEpoch += 1;
    if (_inFlightSessionId === sessionId) {
      _inFlightSessionId = null;
    }
    set(_initial);
  },

  reset() {
    // Bump epoch + clear in-flight tracker for the same reason as
    // clearForSession: any pending await that resumes after reset
    // must see a stale epoch and drop its response.
    _openAndMarkEpoch += 1;
    _inFlightSessionId = null;
    set(_initial);
  },
}));

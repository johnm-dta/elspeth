/**
 * Zustand store for the shareable-reviews dialog state.
 *
 * Canonical internal model: a discriminated union over the request
 * lifecycle. The legacy flat fields (`inFlight`, `error`, `latestResponse`,
 * `sessionIdForResponse`) are derived from the union on every transition
 * so existing consumers (CompletionBar, SaveForReviewDialog, integration
 * tests) keep working unchanged.
 *
 *   request.kind === "idle"      ↔ no request has run since the last reset
 *   request.kind === "pending"   ↔ a POST is in-flight; carries its session
 *                                  id + monotonic epoch (used to drop stale
 *                                  late responses)
 *   request.kind === "resolved"  ↔ the most recent attempt produced a
 *                                  token; carries the response + session
 *   request.kind === "error"     ↔ the most recent attempt failed; carries
 *                                  the user-facing message + session
 *
 * The epoch lives inside the pending state — when it would have been a
 * module-scope scalar in the old design, it now cannot exist outside the
 * phase that actually has a request in flight. Stale-response detection is
 * `request.kind === "pending" && request.epoch === myEpoch` at await
 * resume; if either side mismatches, the late response is dropped.
 *
 * ``dialogOpen`` is orthogonal to the request phase: closing the dialog
 * preserves the previous response so reopening shows the same URL. The
 * union models the *request lifecycle*, not the UI dialog state.
 */

import { create } from "zustand";

import { markReadyForReview } from "../api/shareableReviews";
import type { ApiError, MarkReadyForReviewResponse } from "../types/api";

/**
 * Request-lifecycle discriminated union.
 *
 * Exported for tests that want to assert on the canonical state shape;
 * normal consumers read the derived flat fields below.
 */
export type RequestPhase =
  | { kind: "idle" }
  | { kind: "pending"; sessionId: string; epoch: number }
  | { kind: "resolved"; sessionId: string; response: MarkReadyForReviewResponse }
  | { kind: "error"; sessionId: string; message: string };

export interface ShareableReviewState {
  /** Whether the SaveForReviewDialog is mounted (orthogonal to request phase). */
  dialogOpen: boolean;
  /** Canonical request-lifecycle state (discriminated union). */
  request: RequestPhase;

  // ── Derived (kept in lockstep with `request` on every set()) ──
  /** True while a POST is being awaited. Derived from request.kind === "pending". */
  inFlight: boolean;
  /** Latest successful response, preserved across close()/error so reopening
   *  the dialog can show the same share URL. Derived: present whenever the
   *  most recent terminal phase was "resolved" for the active session. */
  latestResponse: MarkReadyForReviewResponse | null;
  /** Session id of the latestResponse. Tracks the session that minted it. */
  sessionIdForResponse: string | null;
  /** Error message from the most recent attempt. Derived from request.kind === "error". */
  error: string | null;

  // ── Mutators ──
  openAndMark: (sessionId: string) => Promise<void>;
  close: () => void;
  clearForSession: (sessionId: string) => void;
  reset: () => void;
}

interface InternalState {
  dialogOpen: boolean;
  request: RequestPhase;
  /** Cached last-successful response so closing the dialog (or hitting an
   *  error) does not blank the URL the user already saw. Promoted from
   *  `request.kind === "resolved"` when that phase exits. */
  lastSuccess: { sessionId: string; response: MarkReadyForReviewResponse } | null;
}

const _internalInitial: InternalState = {
  dialogOpen: false,
  request: { kind: "idle" },
  lastSuccess: null,
};

function _isApiError(value: unknown): value is ApiError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { status?: unknown }).status === "number"
  );
}

/** Project canonical InternalState onto the public flat-field surface. */
function _project(internal: InternalState): Omit<
  ShareableReviewState,
  "openAndMark" | "close" | "clearForSession" | "reset"
> {
  const inFlight = internal.request.kind === "pending";
  const error = internal.request.kind === "error" ? internal.request.message : null;
  // latestResponse precedence: the *cached* lastSuccess wins for view-state
  // continuity (so a transient error or a close()/reopen does not blank
  // the URL). When the current request resolved successfully, lastSuccess
  // is already updated to the same payload, so the two agree.
  const latestResponse = internal.lastSuccess?.response ?? null;
  const sessionIdForResponse = internal.lastSuccess?.sessionId ?? null;
  return {
    dialogOpen: internal.dialogOpen,
    request: internal.request,
    inFlight,
    latestResponse,
    sessionIdForResponse,
    error,
  };
}

/** Stash for the internal source-of-truth state. Zustand stores the
 *  projected (flat) surface so consumers can subscribe to `inFlight` etc.
 *  directly; the internal record is updated alongside on each transition. */
let _internal: InternalState = _internalInitial;

export const useShareableReviewStore = create<ShareableReviewState>((set, _get) => {
  function _commit(next: InternalState): void {
    _internal = next;
    set(_project(next));
  }

  return {
    ..._project(_internal),

    async openAndMark(sessionId: string) {
      // Double-click guard: if a POST is already in-flight for the same
      // session, drop the re-entrant call. Audit rows are append-only —
      // minting a second token would create a duplicate audit row.
      if (_internal.request.kind === "pending" && _internal.request.sessionId === sessionId) {
        return;
      }

      // Capture this call's epoch BEFORE awaiting. The next openAndMark
      // (same or different session) will bump the epoch when it sets its
      // own pending phase; on resume we re-check our captured epoch against
      // the live one. Mismatch ⇒ we were superseded; drop our late response.
      const myEpoch =
        _internal.request.kind === "pending" ? _internal.request.epoch + 1 : 1;

      // Switching sessions while a previous response is still cached:
      // wipe the cached lastSuccess so the dialog does not flash session
      // A's URL while session B's POST is in flight. Same-session reentry
      // keeps the cache.
      const sessionSwitch =
        _internal.lastSuccess !== null && _internal.lastSuccess.sessionId !== sessionId;

      _commit({
        ..._internal,
        dialogOpen: true,
        request: { kind: "pending", sessionId, epoch: myEpoch },
        lastSuccess: sessionSwitch ? null : _internal.lastSuccess,
      });

      try {
        const response = await markReadyForReview(sessionId);
        // Stale-epoch check: a later openAndMark superseded us while we
        // were awaiting. Our response must NOT clobber the new pending /
        // resolved state. The server-side audit row was minted regardless;
        // we just suppress the late client-side surfacing.
        if (
          _internal.request.kind !== "pending" ||
          _internal.request.epoch !== myEpoch ||
          _internal.request.sessionId !== sessionId
        ) {
          return;
        }
        _commit({
          ..._internal,
          request: { kind: "resolved", sessionId, response },
          lastSuccess: { sessionId, response },
        });
      } catch (exc) {
        if (
          _internal.request.kind !== "pending" ||
          _internal.request.epoch !== myEpoch ||
          _internal.request.sessionId !== sessionId
        ) {
          return;
        }
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
        } else {
          // Fallback: exc is neither an ApiError nor an Error instance.
          // The generic banner remains, but record the actual exception
          // for debugging — a silent fallback would hide misbehaviour at
          // a lower layer (e.g. a `throw "string"` from a fetch wrapper).
          // eslint-disable-next-line no-console
          console.warn(
            "shareableReviewStore: openAndMark failed with unexpected exception shape:",
            exc,
          );
        }
        _commit({
          ..._internal,
          // Cached lastSuccess is wiped on error to match the previous
          // store's behaviour (the original code set
          // `latestResponse: null` on the error path).
          request: { kind: "error", sessionId, message },
          lastSuccess: null,
        });
      }
    },

    close() {
      _commit({ ..._internal, dialogOpen: false });
    },

    clearForSession(sessionId: string) {
      // No-op unless the cached response belongs to this session. We
      // deliberately do NOT consult `request` here: a pending POST for
      // sessionId is allowed to continue; the stale-epoch check on its
      // resume will see the reset and drop its result.
      if (_internal.lastSuccess?.sessionId !== sessionId) {
        return;
      }
      // Mint a "fresh" epoch by way of resetting the request — any
      // pending POST for this session will re-check and find phase=idle
      // (epoch mismatch by construction), so its response drops.
      _commit({ ..._internalInitial });
    },

    reset() {
      _commit({ ..._internalInitial });
    },
  };
});

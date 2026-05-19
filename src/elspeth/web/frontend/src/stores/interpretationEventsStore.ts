// ============================================================================
// interpretationEventsStore — per-session view of interpretation review events.
//
// Phase 5b Task 3 of 18b-phase-5b-frontend.md.  Holds the local projection
// of interpretation-event state for the LLM-interpretation review surface.
// Wraps the API client (api/client.ts Phase 5b.18b.2 methods) and updates
// store state atomically on success.  Errors from the API surface as
// rejected promises and DO NOT mutate the store (the wire write didn't
// happen, so the local projection must not pretend it did).
//
// Three slices of state per session:
//
//  1. pendingBySession[sid][eventId] — InterpretationEvent
//     Active "review me" affordances.  Cleared on resolve/opt-out.
//
//  2. resolvedCountBySession[sid] — { accepted_as_drafted, amended, opted_out }
//     Audit-readiness-panel counters.  We don't keep the full resolved
//     event list in memory for the readiness panel (it only needs counts);
//     the wire route is the source of truth for individual resolved rows.
//
//  2a. resolvedBySession[sid] — InterpretationEvent[]
//     Resolved events kept in memory for the Phase 6B NarrativeResults
//     overlay (19b-phase-6b-frontend.md Task 6, load-bearing run-window
//     filter). The overlay needs to filter resolved events to the active
//     run's wall-clock window so it does not surface resolutions from
//     prior runs in the same session. The audit-readiness panel
//     deliberately reads `resolvedCountBySession`, not this slice — the
//     two surfaces have different invariants and the count-only view
//     remains canonical for the readiness gauge.
//
//  3. optedOutBySession[sid] — boolean
//     Mirror of sessions.interpretation_review_disabled.  When true,
//     pending events for that session are cleared (the UX is
//     "interpretations silenced from now on for this session"); the
//     backend may still hold pending DB rows but they're no longer
//     surfaced.
//
// Tier-discipline note: InterpretationEvent arriving from the API client
// has already passed pydantic strict-mode validation server-side and
// typed parseResponse<T>() client-side.  Inside this store it's Tier-2
// data — direct field access is correct.  No defensive `.get()` /
// `?.optional()` patterns; if a wire field is missing the upstream
// invariant has already broken and a crash is the right answer.
//
// Telemetry: NONE.  The backend Landscape is the canonical record for
// every interpretation-event mutation; emitting client-side telemetry
// would be redundant duplication of audit truth.
// ============================================================================

import { create } from "zustand";
import * as api from "@/api/client";
import type { CompositionState } from "@/types/api";
import type {
  InterpretationEvent,
  InterpretationResolveRequest,
} from "@/types/interpretation";

// ── Types ────────────────────────────────────────────────────────────────────

/**
 * Per-session counters for resolved interpretation events.  Mirrors the
 * three terminal-from-the-store-perspective `InterpretationChoice` values
 * the audit-readiness panel cares about.  `pending` is not a count here
 * (it's derivable from `pendingBySession[sid]` size); `abandoned` is a
 * Phase-11 orphan-cleanup outcome the readiness panel surfaces separately
 * if needed and is not yet wired into this store.
 */
export interface ResolvedCounts {
  accepted_as_drafted: number;
  amended: number;
  opted_out: number;
}

const EMPTY_COUNTS: ResolvedCounts = {
  accepted_as_drafted: 0,
  amended: 0,
  opted_out: 0,
};

interface InterpretationEventsState {
  // ── Primary projections ──────────────────────────────────────────────────
  pendingBySession: Record<string, Record<string, InterpretationEvent>>;
  resolvedCountBySession: Record<string, ResolvedCounts>;
  /**
   * Resolved interpretation events keyed by session_id, populated by
   * `refreshAll` (and incrementally by `resolveEvent`). Used by Phase 6B
   * NarrativeResults to filter events by run wall-clock window for
   * overlay rendering — see 19b-phase-6b-frontend.md Task 6.
   *
   * Events arrive in the API's order (chronological by `created_at` /
   * `resolved_at`). The component performs the wall-clock filter; the
   * store does not — keeping ordering responsibilities at the read site
   * means a future schema change (e.g. adding `run_id` to events) only
   * touches the consumer.
   */
  resolvedBySession: Record<string, InterpretationEvent[]>;
  optedOutBySession: Record<string, boolean>;

  // ── Actions ──────────────────────────────────────────────────────────────

  /**
   * Refresh the pending-event projection for a session.
   *
   * Calls GET /interpretations?status=pending and overwrites
   * pendingBySession[sessionId] with the result.  Does NOT touch the
   * resolved counts or opt-out flag — use `refreshAll` if those need
   * rehydrating too.
   */
  refreshPending: (sessionId: string) => Promise<void>;

  /**
   * Refresh pending events, resolved counts, and opt-out state for a session.
   *
   * Calls GET /interpretations?status=all and partitions the result:
   *   - pending rows → pendingBySession[sessionId]
   *   - non-pending rows → counted into resolvedCountBySession[sessionId]
   *   - auto_interpreted_opt_out row → optedOutBySession[sessionId] = true
   *
   * Used on session load to bootstrap the per-session view in one round-trip.
   */
  refreshAll: (sessionId: string) => Promise<void>;

  /**
   * Resolve an interpretation event.
   *
   * On success: removes the event from pendingBySession[sessionId],
   * increments the matching counter in resolvedCountBySession[sessionId],
   * returns the new composition state so the caller can update its
   * own composition-state view atomically.
   *
   * On error: throws (rejection propagated); store state is untouched
   * (atomicity — if the wire write didn't happen, the projection must
   * not pretend it did).
   */
  resolveEvent: (
    sessionId: string,
    eventId: string,
    body: InterpretationResolveRequest,
  ) => Promise<{ new_state: CompositionState }>;

  /**
   * Record the per-session opt-out decision.
   *
   * On success: flips optedOutBySession[sessionId] to true, clears
   * pendingBySession[sessionId], increments the opted_out counter by 1
   * (representing the opt-out event itself; subsequent auto-baked
   * opt-out rows are NOT pre-counted here — the audit-readiness panel
   * fetches /opt_out_summary on demand).
   *
   * On error: throws; store state untouched.
   */
  optOut: (sessionId: string) => Promise<void>;

  /**
   * Add a single pending event to the store.
   *
   * Used when a fresh interpretation event arrives via the compose-loop
   * response (the backend creates it inline and we'd rather not round-trip
   * GET /interpretations to surface it).  Idempotent — re-adding the same
   * event ID overwrites the previous entry.
   */
  addPendingEvent: (sessionId: string, event: InterpretationEvent) => void;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Increment the per-session resolved counter for the given choice.
 * Returns a new record (immutable update).  Unknown choice values throw
 * (offensive: the wire-side InterpretationChoice union is closed; an
 * unknown value here is a bug in the API client or the backend, not
 * a user-data issue).
 */
function incrementResolvedCount(
  prev: Record<string, ResolvedCounts>,
  sessionId: string,
  choice: InterpretationEvent["choice"],
): Record<string, ResolvedCounts> {
  const current = prev[sessionId] ?? EMPTY_COUNTS;
  let next: ResolvedCounts;
  switch (choice) {
    case "accepted_as_drafted":
      next = { ...current, accepted_as_drafted: current.accepted_as_drafted + 1 };
      break;
    case "amended":
      next = { ...current, amended: current.amended + 1 };
      break;
    case "opted_out":
      next = { ...current, opted_out: current.opted_out + 1 };
      break;
    case "pending":
      // Pending is not a resolved state; the caller should not have
      // dispatched here.  Throw offensively (CLAUDE.md decision test:
      // bug in our code → let it crash with a meaningful message).
      throw new Error(
        "incrementResolvedCount: 'pending' is not a resolved state",
      );
    case "abandoned":
      // Abandoned is a Phase-11 orphan-cleanup outcome written by the
      // session-end job, not by user resolve actions.  It does not flow
      // through resolveEvent / optOut; if we ever see it here, the
      // call site is wrong.
      throw new Error(
        "incrementResolvedCount: 'abandoned' is not produced by user action",
      );
    default: {
      // Exhaustiveness check: a future widening of InterpretationChoice
      // turns this into a compile error.
      const _exhaustive: never = choice;
      throw new Error(
        `incrementResolvedCount: unhandled choice ${String(_exhaustive)}`,
      );
    }
  }
  return { ...prev, [sessionId]: next };
}

// ── Store ────────────────────────────────────────────────────────────────────

export const useInterpretationEventsStore = create<InterpretationEventsState>(
  (set) => ({
    pendingBySession: {},
    resolvedCountBySession: {},
    resolvedBySession: {},
    optedOutBySession: {},

    async refreshPending(sessionId: string) {
      // The API client returns InterpretationEvent[] (envelope already
      // unwrapped).  Project into the {eventId: event} map.
      const events = await api.listInterpretationEvents(sessionId, "pending");
      const map: Record<string, InterpretationEvent> = {};
      for (const event of events) {
        map[event.id] = event;
      }
      set((state) => ({
        pendingBySession: { ...state.pendingBySession, [sessionId]: map },
      }));
    },

    async refreshAll(sessionId: string) {
      const events = await api.listInterpretationEvents(sessionId, "all");
      const pendingMap: Record<string, InterpretationEvent> = {};
      const counts: ResolvedCounts = { ...EMPTY_COUNTS };
      // Resolved-event list for the Phase 6B NarrativeResults overlay.
      // Preserves API order (chronological); the consumer (NarrativeResults)
      // owns the wall-clock filter. We include every non-pending choice
      // that the counts surface tracks; 'abandoned' rows are skipped here
      // for the same reason they're skipped from counts.
      const resolvedList: InterpretationEvent[] = [];
      let optedOutFromHistory = false;
      for (const event of events) {
        if (event.interpretation_source === "auto_interpreted_opt_out") {
          optedOutFromHistory = true;
        }
        if (event.choice === "pending") {
          pendingMap[event.id] = event;
        } else if (
          event.choice === "accepted_as_drafted" ||
          event.choice === "amended" ||
          event.choice === "opted_out"
        ) {
          // Direct field bump rather than calling incrementResolvedCount
          // — we're building a fresh ResolvedCounts in one pass, not
          // accumulating onto a prior store state.
          counts[event.choice] += 1;
          resolvedList.push(event);
        }
        // 'abandoned' rows are not counted in this store; the audit-readiness
        // panel surfaces them via a separate code path if needed.
      }
      set((state) => {
        const optedOut =
          optedOutFromHistory || state.optedOutBySession[sessionId] === true;
        return {
          pendingBySession: {
            ...state.pendingBySession,
            [sessionId]: optedOut ? {} : pendingMap,
          },
          resolvedCountBySession: {
            ...state.resolvedCountBySession,
            [sessionId]: counts,
          },
          resolvedBySession: {
            ...state.resolvedBySession,
            [sessionId]: resolvedList,
          },
          optedOutBySession: optedOut
            ? { ...state.optedOutBySession, [sessionId]: true }
            : state.optedOutBySession,
        };
      });
    },

    async resolveEvent(
      sessionId: string,
      eventId: string,
      body: InterpretationResolveRequest,
    ) {
      // Call the API first; on error the throw propagates and the store
      // is untouched (atomicity invariant).
      const response = await api.resolveInterpretation(sessionId, eventId, body);
      const resolvedChoice = response.event.choice;
      const resolvedEvent = response.event;

      set((state) => {
        // Remove the event from pending.  Use a fresh inner map rather
        // than mutating the existing one (Zustand selectors compare by
        // reference identity).
        const sessionPending = state.pendingBySession[sessionId];
        const nextSessionPending: Record<string, InterpretationEvent> = {};
        if (sessionPending) {
          for (const [id, evt] of Object.entries(sessionPending)) {
            if (id !== eventId) {
              nextSessionPending[id] = evt;
            }
          }
        }

        // Append the newly resolved event into the resolved-list slice
        // so NarrativeResults (Phase 6B Task 6) can render it without
        // forcing a refreshAll round-trip after every resolve.
        const priorResolved = state.resolvedBySession[sessionId] ?? [];
        const nextResolved = [...priorResolved, resolvedEvent];

        return {
          pendingBySession: {
            ...state.pendingBySession,
            [sessionId]: nextSessionPending,
          },
          resolvedCountBySession: incrementResolvedCount(
            state.resolvedCountBySession,
            sessionId,
            resolvedChoice,
          ),
          resolvedBySession: {
            ...state.resolvedBySession,
            [sessionId]: nextResolved,
          },
        };
      });

      return { new_state: response.new_state };
    },

    async optOut(sessionId: string) {
      // API first; on error, throw and leave store untouched.
      await api.optOutOfInterpretations(sessionId);

      set((state) => ({
        optedOutBySession: { ...state.optedOutBySession, [sessionId]: true },
        // Clear pending events for the session — the UX is "interpretations
        // silenced from now on".  The backend may still hold pending rows
        // in the DB; we don't fetch /opt_out_summary here because the panel
        // surface owns that fetch (lazy, on demand).
        pendingBySession: { ...state.pendingBySession, [sessionId]: {} },
        // Bump the opted_out counter to represent the opt-out event itself.
        resolvedCountBySession: incrementResolvedCount(
          state.resolvedCountBySession,
          sessionId,
          "opted_out",
        ),
      }));
    },

    addPendingEvent(sessionId: string, event: InterpretationEvent) {
      // Defensive note: this entry point exists for inline-add from the
      // compose-loop response path.  The caller is responsible for only
      // dispatching pending events here; counts/opt-out are not touched.
      set((state) => {
        const sessionPending = state.pendingBySession[sessionId] ?? {};
        return {
          pendingBySession: {
            ...state.pendingBySession,
            [sessionId]: { ...sessionPending, [event.id]: event },
          },
        };
      });
    },
  }),
);

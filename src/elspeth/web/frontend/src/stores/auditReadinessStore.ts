/**
 * Zustand store for the audit-readiness panel (Phase 2B).
 *
 * Caches snapshots and explain narratives by sessionId, keyed by the
 * composition_version each carries. `loadSnapshot` is a no-op when the
 * cached snapshot's version matches the requested version — this is what
 * makes auto-validate-on-composition-change cheap.
 *
 * The store never coerces server payloads — Phase 2A's `_StrictResponse`
 * is the contract. If a literal value doesn't match the TypeScript union,
 * that's a backend/frontend version skew and the panel renderer will
 * surface it via the exhaustive `never` arm in AuditReadinessPanel.tsx.
 */
import { create } from "zustand";

import {
  fetchAuditReadiness,
  fetchAuditReadinessExplain,
} from "../api/auditReadiness";
import type {
  AuditReadinessSnapshot,
  AuditReadinessExplain,
  ApiError,
} from "../types/api";

export interface LoadSnapshotOptions {
  force: boolean;
}

export interface AuditReadinessState {
  snapshotsBySession: Record<string, AuditReadinessSnapshot>;
  explainsBySession: Record<string, AuditReadinessExplain>;
  /** In-flight AbortController keyed by sessionId for snapshot fetches.
   *  loadSnapshot aborts the previous controller before starting a new fetch,
   *  preventing stale responses from overwriting a more recent result.
   *  clearSession also aborts any in-flight request for the cleared session. */
  abortControllers: Record<string, AbortController>;
  /** In-flight AbortController keyed by sessionId for explain fetches.
   *  Parallel to abortControllers — kept separate so aborting an explain fetch
   *  does not cancel a concurrent snapshot fetch for the same session.
   *  clearSession aborts both controllers. */
  explainAbortControllers: Record<string, AbortController>;
  /** Per-session loading and error state. Global scalars would contaminate
   *  cross-session UI — switching from a failing session A to a healthy
   *  session B would render A's error banner on B. Keyed-by-sessionId mirrors
   *  the data fields (snapshotsBySession etc.) and makes the store's contract
   *  internally consistent. */
  isLoadingBySession: Record<string, boolean>;
  isLoadingExplainBySession: Record<string, boolean>;
  errorBySession: Record<string, string | null>;
  explainErrorBySession: Record<string, string | null>;
  /** Per-session user-expand preference. Kept in the store so the preference
   *  survives right-rail remounts. Keyed-by-sessionId mirrors the other six
   *  per-session maps; clearSession drops the entry alongside them. */
  userExpandedBySession: Record<string, boolean>;

  loadSnapshot: (
    sessionId: string,
    compositionVersion: number,
    options?: LoadSnapshotOptions,
  ) => Promise<void>;
  loadExplain: (sessionId: string, compositionVersion: number) => Promise<void>;
  clearSession: (sessionId: string) => void;
  reset: () => void;
  setUserExpanded: (sessionId: string, value: boolean) => void;
}

export const getInitialState = (): Omit<AuditReadinessState, "loadSnapshot" | "loadExplain" | "clearSession" | "reset" | "setUserExpanded"> => ({
  snapshotsBySession: {},
  explainsBySession: {},
  abortControllers: {},
  explainAbortControllers: {},
  isLoadingBySession: {},
  isLoadingExplainBySession: {},
  errorBySession: {},
  explainErrorBySession: {},
  userExpandedBySession: {},
});

export const useAuditReadinessStore = create<AuditReadinessState>((set, get) => ({
  ...getInitialState(),

  async loadSnapshot(
    sessionId: string,
    compositionVersion: number,
    options: LoadSnapshotOptions = { force: false },
  ) {
    const cached = get().snapshotsBySession[sessionId];
    if (!options.force && cached && cached.composition_version === compositionVersion) {
      return;
    }

    // Abort any in-flight request for this session before starting a new one.
    const prev = get().abortControllers[sessionId];
    if (prev) prev.abort();
    const controller = new AbortController();
    set((state) => ({
      abortControllers: { ...state.abortControllers, [sessionId]: controller },
      isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: true },
      errorBySession: { ...state.errorBySession, [sessionId]: null },
    }));

    try {
      const snapshot = await fetchAuditReadiness(sessionId, controller.signal);
      // Monotonic write guard: discard the response if a newer version has
      // already been stored while this fetch was in flight.
      set((state) => {
        const current = state.snapshotsBySession[sessionId];
        if (current && current.composition_version > snapshot.composition_version) {
          // Stale response arrived after a newer one was already cached —
          // discard. Also drop our resolved controller from abortControllers
          // (the invariant is "abortControllers holds only in-flight
          // controllers"). Clear our session's loading flag so the UI does
          // not hang.
          const { [sessionId]: _staleCtrl, ...restStaleCtrl } = state.abortControllers;
          return {
            abortControllers: restStaleCtrl,
            isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          };
        }
        const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
        return {
          snapshotsBySession: { ...state.snapshotsBySession, [sessionId]: snapshot },
          abortControllers: restCtrl,
          isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          // Clear any prior error for this session — a successful fetch
          // supersedes the previous failure.
          errorBySession: { ...state.errorBySession, [sessionId]: null },
        };
      });
    } catch (err) {
      // AbortError is the expected path for an aborted-by-newer-fetch scenario;
      // the loading indicator must clear because no successor fetch may arrive
      // (e.g., session navigation away). Per-session: only this session's
      // flag clears.
      //
      // Controller-identity guard: if our controller is no longer the tracked
      // one for this session, clearSession or a newer fetch has taken over —
      // do not resurrect per-session state we no longer own.
      if ((err as { name?: string }).name === "AbortError") {
        set((state) => {
          if (state.abortControllers[sessionId] !== controller) {
            return state;
          }
          const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
          return {
            abortControllers: restCtrl,
            isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          };
        });
        return;
      }
      const apiErr = err as ApiError;
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
        return {
          abortControllers: restCtrl,
          isLoadingBySession: { ...state.isLoadingBySession, [sessionId]: false },
          errorBySession: {
            ...state.errorBySession,
            [sessionId]: apiErr.detail ?? "Failed to load audit readiness.",
          },
        };
      });
    }
  },

  async loadExplain(sessionId: string, compositionVersion: number) {
    const cached = get().explainsBySession[sessionId];
    if (cached && cached.composition_version === compositionVersion) {
      return;
    }

    // Abort any in-flight explain fetch for this session before starting a new
    // one. Uses a parallel controller dict (explainAbortControllers) so that
    // aborting an explain fetch does not cancel a concurrent snapshot fetch.
    const prevExplain = get().explainAbortControllers[sessionId];
    if (prevExplain) prevExplain.abort();
    const explainController = new AbortController();
    set((state) => ({
      explainAbortControllers: { ...state.explainAbortControllers, [sessionId]: explainController },
      isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: true },
      explainErrorBySession: { ...state.explainErrorBySession, [sessionId]: null },
    }));

    try {
      const explain = await fetchAuditReadinessExplain(sessionId, explainController.signal);
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.explainAbortControllers;
        return {
          explainsBySession: {
            ...state.explainsBySession,
            [sessionId]: explain,
          },
          explainAbortControllers: restCtrl,
          isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          // Clear any prior explain error for this session — a successful
          // fetch supersedes the previous failure.
          explainErrorBySession: { ...state.explainErrorBySession, [sessionId]: null },
        };
      });
    } catch (err) {
      // Mirror the snapshot AbortError pattern: clear isLoadingExplain so the
      // UI does not hang on "Loading explain…" after navigation away. Per-
      // session: only this session's flag clears.
      //
      // Controller-identity guard: if our explainController is no longer the
      // tracked one for this session, clearSession or a newer fetch has taken
      // over — do not resurrect per-session state we no longer own.
      if ((err as { name?: string }).name === "AbortError") {
        set((state) => {
          if (state.explainAbortControllers[sessionId] !== explainController) {
            return state;
          }
          const { [sessionId]: _ctrl, ...restCtrl } = state.explainAbortControllers;
          return {
            explainAbortControllers: restCtrl,
            isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          };
        });
        return;
      }
      const apiErr = err as ApiError;
      set((state) => {
        const { [sessionId]: _ctrl, ...restCtrl } = state.explainAbortControllers;
        return {
          explainAbortControllers: restCtrl,
          isLoadingExplainBySession: { ...state.isLoadingExplainBySession, [sessionId]: false },
          explainErrorBySession: {
            ...state.explainErrorBySession,
            [sessionId]: apiErr.detail ?? "Failed to load the explain narrative.",
          },
        };
      });
    }
  },

  clearSession(sessionId: string) {
    // Abort any in-flight snapshot and explain fetches for this session.
    const ctrl = get().abortControllers[sessionId];
    if (ctrl) ctrl.abort();
    const explainCtrl = get().explainAbortControllers[sessionId];
    if (explainCtrl) explainCtrl.abort();
    set((state) => {
      const { [sessionId]: _snap, ...restSnap } = state.snapshotsBySession;
      const { [sessionId]: _expl, ...restExpl } = state.explainsBySession;
      const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
      const { [sessionId]: _eCtrl, ...restECtrl } = state.explainAbortControllers;
      const { [sessionId]: _il, ...restIL } = state.isLoadingBySession;
      const { [sessionId]: _ilx, ...restILX } = state.isLoadingExplainBySession;
      const { [sessionId]: _err, ...restErr } = state.errorBySession;
      const { [sessionId]: _errx, ...restErrX } = state.explainErrorBySession;
      const { [sessionId]: _ue, ...restUE } = state.userExpandedBySession;
      return {
        snapshotsBySession: restSnap,
        explainsBySession: restExpl,
        abortControllers: restCtrl,
        explainAbortControllers: restECtrl,
        isLoadingBySession: restIL,
        isLoadingExplainBySession: restILX,
        errorBySession: restErr,
        explainErrorBySession: restErrX,
        userExpandedBySession: restUE,
      };
    });
  },

  reset() {
    set(getInitialState());
  },

  setUserExpanded(sessionId: string, value: boolean) {
    set((s) => ({
      userExpandedBySession: { ...s.userExpandedBySession, [sessionId]: value },
    }));
  },
}));

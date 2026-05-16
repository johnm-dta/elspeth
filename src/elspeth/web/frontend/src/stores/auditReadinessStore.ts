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
  isLoading: boolean;
  isLoadingExplain: boolean;
  error: string | null;
  explainError: string | null;

  loadSnapshot: (sessionId: string, compositionVersion: number) => Promise<void>;
  loadExplain: (sessionId: string, compositionVersion: number) => Promise<void>;
  clearSession: (sessionId: string) => void;
  reset: () => void;
}

export const getInitialState = (): Omit<AuditReadinessState, "loadSnapshot" | "loadExplain" | "clearSession" | "reset"> => ({
  snapshotsBySession: {},
  explainsBySession: {},
  abortControllers: {},
  explainAbortControllers: {},
  isLoading: false,
  isLoadingExplain: false,
  error: null,
  explainError: null,
});

export const useAuditReadinessStore = create<AuditReadinessState>((set, get) => ({
  ...getInitialState(),

  async loadSnapshot(sessionId: string, compositionVersion: number) {
    const cached = get().snapshotsBySession[sessionId];
    if (cached && cached.composition_version === compositionVersion) {
      return;
    }

    // Abort any in-flight request for this session before starting a new one.
    const prev = get().abortControllers[sessionId];
    if (prev) prev.abort();
    const controller = new AbortController();
    set((state) => ({
      abortControllers: { ...state.abortControllers, [sessionId]: controller },
      isLoading: true,
      error: null,
    }));

    try {
      const snapshot = await fetchAuditReadiness(sessionId, controller.signal);
      // Monotonic write guard: discard the response if a newer version has
      // already been stored while this fetch was in flight.
      set((state) => {
        const current = state.snapshotsBySession[sessionId];
        if (current && current.composition_version > snapshot.composition_version) {
          return { isLoading: false };
        }
        const { [sessionId]: _ctrl, ...restCtrl } = state.abortControllers;
        return {
          snapshotsBySession: { ...state.snapshotsBySession, [sessionId]: snapshot },
          abortControllers: restCtrl,
          isLoading: false,
        };
      });
    } catch (err) {
      // AbortError is the expected path for an aborted-by-newer-fetch scenario;
      // the loading indicator must clear because no successor fetch may arrive
      // (e.g., session navigation away). isLoading is global, not per-session.
      if ((err as { name?: string }).name === "AbortError") {
        set({ isLoading: false });
        return;
      }
      const apiErr = err as ApiError;
      set({
        isLoading: false,
        error: apiErr.detail ?? "Failed to load audit readiness.",
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
      isLoadingExplain: true,
      explainError: null,
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
          isLoadingExplain: false,
        };
      });
    } catch (err) {
      // Mirror the snapshot AbortError pattern: clear isLoadingExplain so the
      // UI does not hang on "Loading explain…" after navigation away.
      if ((err as { name?: string }).name === "AbortError") {
        set({ isLoadingExplain: false });
        return;
      }
      const apiErr = err as ApiError;
      set({
        isLoadingExplain: false,
        explainError: apiErr.detail ?? "Failed to load the explain narrative.",
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
      return {
        snapshotsBySession: restSnap,
        explainsBySession: restExpl,
        abortControllers: restCtrl,
        explainAbortControllers: restECtrl,
        isLoading: false,
        isLoadingExplain: false,
      };
    });
  },

  reset() {
    set(getInitialState());
  },
}));

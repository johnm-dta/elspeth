// ============================================================================
// inlineSourceStore — projected per-session view of the inline-blob source
// attached to the current composition state (Phase 5a Task 2).
//
// This store is a caching/projection layer, NOT a source of truth. The
// derivation that populates it lives in the ChatPanel wiring landed in
// Task 3 (computed from compositionState.source + blob metadata). Three
// downstream consumers read from here:
//   - InlineSourceCreatedTurn (Task 3) — confirmation widget after creation.
//   - InlineSourceDisambiguationTurn (Task 4) — ambiguous-input picker.
//   - Audit-readiness panel surface (Task 7) — readiness row for inline source.
//
// The store has THREE responsibilities, intentionally co-located in one
// container per the Phase 5a plan:
//   1. Per-session inline-source summary projection (summariesBySession).
//   2. Disambiguation-related message-ID sets (F-11 re-fire guard for
//      "treat as single row"; F-10 escape for "this isn't source data").
//   3. Per-session dismissal timestamp for the fallback prompt (F-20),
//      so a dismissed prompt does not re-fire within the same session
//      regardless of predicate re-evaluation.
// ============================================================================

import { create } from "zustand";
import type { InlineSourceSummary } from "@/types/api";

interface InlineSourceState {
  // --- Primary projection: per-session inline-source summary ---
  summariesBySession: Record<string, InlineSourceSummary>;
  setSummary: (sessionId: string, summary: InlineSourceSummary) => void;
  clearSummary: (sessionId: string) => void;
  getSummary: (sessionId: string) => InlineSourceSummary | null;

  // --- Disambiguation re-fire guard (F-11) ---
  // Message IDs for which the user explicitly chose "treat as 1 row".
  // The disambiguation predicate in ChatPanel skips these message IDs.
  userRequestedSingleRowForMessageIds: Set<string>;
  addUserRequestedSingleRow: (messageId: string) => void;

  // --- "Not source data" escape (F-10) ---
  // Message IDs for which the user explicitly chose "this isn't source data".
  // The disambiguation predicate and fallback-prompt predicate skip these.
  nonSourceMessageIds: Set<string>;
  addNonSourceMessage: (messageId: string) => void;

  // --- Fallback-prompt dismiss persistence (F-20) ---
  // Keyed by sessionId. A dismissed fallback prompt must not re-fire
  // within the same session regardless of predicate re-evaluation.
  dismissedAt: Map<string, number>;
  markDismissed: (sessionId: string) => void;
  isDismissed: (sessionId: string) => boolean;
}

export const useInlineSourceStore = create<InlineSourceState>((set, get) => ({
  summariesBySession: {},
  setSummary: (sessionId, summary) =>
    set((s) => ({
      summariesBySession: { ...s.summariesBySession, [sessionId]: summary },
    })),
  clearSummary: (sessionId) =>
    set((s) => {
      const next = { ...s.summariesBySession };
      delete next[sessionId];
      return { summariesBySession: next };
    }),
  getSummary: (sessionId) => get().summariesBySession[sessionId] ?? null,

  userRequestedSingleRowForMessageIds: new Set(),
  addUserRequestedSingleRow: (messageId) =>
    set((s) => ({
      userRequestedSingleRowForMessageIds: new Set([
        ...s.userRequestedSingleRowForMessageIds,
        messageId,
      ]),
    })),

  nonSourceMessageIds: new Set(),
  addNonSourceMessage: (messageId) =>
    set((s) => ({
      nonSourceMessageIds: new Set([...s.nonSourceMessageIds, messageId]),
    })),

  dismissedAt: new Map(),
  markDismissed: (sessionId) =>
    set((s) => {
      const next = new Map(s.dismissedAt);
      next.set(sessionId, Date.now());
      return { dismissedAt: next };
    }),
  isDismissed: (sessionId) => get().dismissedAt.has(sessionId),
}));

// ============================================================================
// preferencesStore — account-level composer preferences (Phase 1B Task 2).
//
// One Zustand store per user-account row at /api/composer-preferences.
// Reads via bootstrap() on auth-success (App.tsx) and via resolveDefaultMode()
// at session-create time (sessionStore.createSession). Writes are optimistic
// with revert-on-error, gated by a single `writing` flag that serialises
// concurrent setDefaultMode / dismissDefaultChangedBanner calls (both go
// through the same PATCH; an unguarded race would let the second call's
// optimistic set + revert overwrite the first call's pending result).
//
// defaultMode is null before bootstrap completes. Components MUST gate on
// `loaded === true` before assuming a non-null mode; the typed getter
// resolveDefaultMode() does this for non-component callers.
//
// Cross-tab coordination (Phase 1B Panel: banner cluster):
//   When one tab calls dismissDefaultChangedBanner(), we write the resolved
//   timestamp to localStorage under BANNER_DISMISSED_STORAGE_KEY. Other
//   tabs subscribe (via initCrossTabSync) and update their local
//   bannerDismissedAt without making a second PATCH. The storage event
//   does not fire in the originating tab, so there is no echo loop.
//   Mirrors src/hooks/useTheme.test.tsx's cross-tab pattern.
//
// Banner timing watermark (Phase 1B Panel: "first session after opt-out"):
//   When setDefaultMode("freeform") is called and a session is currently
//   active, we capture the activeSessionId into optedOutAtSessionId. The
//   DefaultModeChangedBanner suppresses itself while the user is still in
//   that session — they see the banner only after navigating to a new
//   session or reloading. The watermark is in-memory only; a reload
//   surfaces the banner (matches "I opted out, refreshed, see
//   confirmation" intuition).
// ============================================================================

import { create } from "zustand";
import {
  fetchUserComposerPreferences,
  updateUserComposerPreferences,
} from "@/api/client";
import type { ComposerMode } from "@/types/api";

// localStorage key for cross-tab banner-dismiss broadcasts. Versioned
// (`v1`) so a future schema change can add a `v2` key without colliding
// with stale tabs that pre-date the upgrade.
const BANNER_DISMISSED_STORAGE_KEY = "elspeth_prefs_banner_dismissed_v1";

interface PreferencesState {
  defaultMode: ComposerMode | null;
  bannerDismissedAt: string | null;
  tutorialCompletedAt: string | null;
  tutorialCompleted: boolean;
  loaded: boolean;
  writing: boolean;
  // Most-recent error from a setDefaultMode or dismiss call. Components
  // render this as an accessible role="alert" region (Panel a11y F2).
  // Cleared on the next successful write or by explicit clearError().
  writeError: string | null;
  // Session in which the user opted out of guided. Banner suppresses while
  // activeSessionId matches this watermark. See module comment.
  optedOutAtSessionId: string | null;

  bootstrap: () => Promise<void>;
  resolveDefaultMode: () => Promise<ComposerMode>;
  setDefaultMode: (mode: ComposerMode, activeSessionId?: string | null) => Promise<void>;
  markTutorialCompleted: (mode: ComposerMode) => Promise<void>;
  resetTutorial: () => Promise<void>;
  dismissDefaultChangedBanner: () => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

function tutorialCompletedFrom(value: string | null): boolean {
  return value !== null;
}

const INITIAL_STATE = {
  defaultMode: null as ComposerMode | null,
  bannerDismissedAt: null as string | null,
  tutorialCompletedAt: null as string | null,
  tutorialCompleted: false,
  loaded: false,
  writing: false,
  writeError: null as string | null,
  optedOutAtSessionId: null as string | null,
};

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  ...INITIAL_STATE,

  bootstrap: async () => {
    const payload = await fetchUserComposerPreferences();
    set({
      defaultMode: payload.default_mode,
      bannerDismissedAt: payload.banner_dismissed_at,
      tutorialCompletedAt: payload.tutorial_completed_at,
      tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
      loaded: true,
    });
  },

  resolveDefaultMode: async () => {
    const current = get();
    if (current.loaded && current.defaultMode !== null) {
      return current.defaultMode;
    }
    await get().bootstrap();
    const after = get();
    if (after.defaultMode === null) {
      // bootstrap resolved without populating defaultMode — backend contract
      // violation (Phase 1A's GET always returns a row, defaulting to guided).
      throw new Error(
        "preferencesStore: bootstrap completed but defaultMode is null",
      );
    }
    return after.defaultMode;
  },

  setDefaultMode: async (mode, activeSessionId = null) => {
    if (get().writing) return;
    const previous = get().defaultMode;
    const wasOptOut = mode === "freeform" && previous !== "freeform";
    // Capture the timing watermark BEFORE the async write so a reload
    // mid-write still suppresses the banner for the current session if
    // the write succeeds. Reset to null on opt-IN (mode === "guided") so
    // an opt-out → opt-in → opt-out cycle gets a fresh watermark.
    set({
      defaultMode: mode,
      writing: true,
      writeError: null,
      optedOutAtSessionId: wasOptOut
        ? activeSessionId
        : mode === "guided"
          ? null
          : get().optedOutAtSessionId,
    });
    try {
      const payload = await updateUserComposerPreferences({
        default_mode: mode,
      });
      set({
        defaultMode: payload.default_mode,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        writing: false,
      });
    } catch (err) {
      set({
        defaultMode: previous,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't save your preference: ${err.message}`
            : "Couldn't save your preference.",
        // Revert the watermark on failure so the banner doesn't suppress
        // for a write that didn't land.
        optedOutAtSessionId: wasOptOut ? null : get().optedOutAtSessionId,
      });
      throw err;
    }
  },

  markTutorialCompleted: async (mode) => {
    if (get().writing) return;
    const stamp = new Date().toISOString();
    const previous = {
      defaultMode: get().defaultMode,
      tutorialCompletedAt: get().tutorialCompletedAt,
      tutorialCompleted: get().tutorialCompleted,
      optedOutAtSessionId: get().optedOutAtSessionId,
    };
    set({
      defaultMode: mode,
      writing: true,
      writeError: null,
      optedOutAtSessionId: mode === "guided" ? null : get().optedOutAtSessionId,
    });
    try {
      const payload = await updateUserComposerPreferences({
        default_mode: mode,
        tutorial_completed_at: stamp,
      });
      set({
        defaultMode: payload.default_mode,
        bannerDismissedAt: payload.banner_dismissed_at,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        writing: false,
      });
    } catch (err) {
      set({
        defaultMode: previous.defaultMode,
        tutorialCompletedAt: previous.tutorialCompletedAt,
        tutorialCompleted: previous.tutorialCompleted,
        optedOutAtSessionId: previous.optedOutAtSessionId,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't save tutorial completion: ${err.message}`
            : "Couldn't save tutorial completion.",
      });
      throw err;
    }
  },

  resetTutorial: async () => {
    if (get().writing) return;
    const previous = {
      tutorialCompletedAt: get().tutorialCompletedAt,
      tutorialCompleted: get().tutorialCompleted,
    };
    set({
      writing: true,
      writeError: null,
    });
    try {
      const payload = await updateUserComposerPreferences({
        tutorial_completed_at: null,
      });
      set({
        defaultMode: payload.default_mode,
        bannerDismissedAt: payload.banner_dismissed_at,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        writing: false,
      });
    } catch (err) {
      set({
        tutorialCompletedAt: previous.tutorialCompletedAt,
        tutorialCompleted: previous.tutorialCompleted,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't reset the tutorial: ${err.message}`
            : "Couldn't reset the tutorial.",
      });
      throw err;
    }
  },

  dismissDefaultChangedBanner: async () => {
    if (get().writing) return;
    const stamp = new Date().toISOString();
    const previous = get().bannerDismissedAt;
    set({ bannerDismissedAt: stamp, writing: true, writeError: null });
    try {
      const payload = await updateUserComposerPreferences({
        banner_dismissed_at: stamp,
      });
      const resolved = payload.banner_dismissed_at;
      set({
        bannerDismissedAt: resolved,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        writing: false,
      });
      // Cross-tab broadcast (Panel banner cluster): write the resolved
      // value to localStorage so peer tabs update their local state
      // without making a second PATCH. The storage event does not fire
      // in this tab.
      if (typeof window !== "undefined" && resolved !== null) {
        try {
          window.localStorage.setItem(
            BANNER_DISMISSED_STORAGE_KEY,
            resolved,
          );
        } catch {
          // localStorage may be disabled (private browsing strict mode);
          // in-tab state is correct, peer tabs will catch up on next
          // bootstrap. No user-visible failure.
        }
      }
    } catch (err) {
      set({
        bannerDismissedAt: previous,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't dismiss the banner: ${err.message}`
            : "Couldn't dismiss the banner.",
      });
      throw err;
    }
  },

  clearError: () => set({ writeError: null }),

  reset: () => set(INITIAL_STATE),
}));

// ── Cross-tab sync wiring ────────────────────────────────────────────────
// Idempotent at-most-once subscription, attached at module load. Mirrors
// the useTheme storage-event pattern but for the banner-dismiss key.
// Guarded against re-execution (e.g. HMR) and SSR (no window).
let crossTabSyncInitialised = false;

export function initCrossTabSync(): void {
  if (crossTabSyncInitialised) return;
  if (typeof window === "undefined") return;
  crossTabSyncInitialised = true;

  window.addEventListener("storage", (event: StorageEvent) => {
    if (event.key !== BANNER_DISMISSED_STORAGE_KEY) return;
    if (event.newValue === null) return;
    // Update local state to match the broadcast value WITHOUT making
    // another PATCH (the originating tab already wrote it). If the
    // current store already has a non-null dismissed_at, prefer the
    // peer's value (idempotent — the peer's value won the race).
    usePreferencesStore.setState({ bannerDismissedAt: event.newValue });
  });
}

// Auto-initialise on module load in browser environments. SSR/test
// environments without window are gracefully skipped.
initCrossTabSync();

export function selectTutorialCompleted(state: PreferencesState): boolean {
  return state.tutorialCompleted;
}

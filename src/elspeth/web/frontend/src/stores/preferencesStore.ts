// ============================================================================
// preferencesStore — account-level composer preferences (Phase 1B Task 2).
//
// One Zustand store per user-account row at /api/composer-preferences.
// Reads via bootstrap() on auth-success (App.tsx) and via resolveDefaultMode()
// at session-create time (sessionStore.createSession). Writes are optimistic
// with revert-on-error, gated by a single `writing` flag that serialises
// concurrent setDefaultMode / saveTutorialMode /
// markTutorialGraduated / dismissDefaultChangedBanner calls (all go
// through PATCH; an unguarded race would let the second call's
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
import type {
  ApiError,
  ComposerMode,
  PersistedTutorialStage,
} from "@/types/api";

// localStorage key for cross-tab banner-dismiss broadcasts. Versioned
// (`v1`) so a future schema change can add a `v2` key without colliding
// with stale tabs that pre-date the upgrade.
const BANNER_DISMISSED_STORAGE_KEY = "elspeth_prefs_banner_dismissed_v1";
const FREEFORM_INTRO_DISMISSED_STORAGE_KEY =
  "elspeth_prefs_freeform_intro_dismissed_v1";

/**
 * In-progress tutorial resume state (elspeth-918f4434b3), server-persisted
 * on every stage transition so a reload resumes at the persisted stage with
 * the SAME session instead of restarting at Welcome (and silently
 * re-spending LLM budget). All four are null when no tutorial is in
 * progress; runId/sourceDataHash appear once the tutorial run completes.
 */
export interface TutorialProgress {
  stage: PersistedTutorialStage | null;
  sessionId: string | null;
  runId: string | null;
  sourceDataHash: string | null;
}

interface PreferencesState {
  defaultMode: ComposerMode | null;
  bannerDismissedAt: string | null;
  freeformIntroDismissedAt: string | null;
  tutorialCompletedAt: string | null;
  tutorialCompleted: boolean;
  tutorialStage: PersistedTutorialStage | null;
  tutorialSessionId: string | null;
  tutorialRunId: string | null;
  tutorialSourceDataHash: string | null;
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
  saveTutorialProgress: (progress: TutorialProgress) => Promise<void>;
  resolveDefaultMode: () => Promise<ComposerMode>;
  setDefaultMode: (mode: ComposerMode, activeSessionId?: string | null) => Promise<void>;
  saveTutorialMode: (mode: ComposerMode) => Promise<void>;
  markTutorialGraduated: (options?: {
    publishLocally?: boolean;
    // Telemetry discriminator: an explicit in-tutorial exit (the wire-stage
    // exit terminal or the shell-chrome "Exit tutorial" control) rides the
    // PATCH as tutorial_completed_via so the backend does not bucket it as
    // "skip" (elspeth-61591e64bb).
    via?: "exit";
  }) => Promise<string | null>;
  publishTutorialGraduation: (completedAt: string | null) => void;
  resetTutorial: () => Promise<void>;
  dismissDefaultChangedBanner: () => Promise<void>;
  dismissFreeformIntro: () => Promise<void>;
  clearError: () => void;
  reset: () => void;
}

function tutorialCompletedFrom(value: string | null): boolean {
  return value !== null;
}

const INITIAL_STATE = {
  defaultMode: null as ComposerMode | null,
  bannerDismissedAt: null as string | null,
  freeformIntroDismissedAt: null as string | null,
  tutorialCompletedAt: null as string | null,
  tutorialCompleted: false,
  tutorialStage: null as PersistedTutorialStage | null,
  tutorialSessionId: null as string | null,
  tutorialRunId: null as string | null,
  tutorialSourceDataHash: null as string | null,
  loaded: false,
  writing: false,
  writeError: null as string | null,
  optedOutAtSessionId: null as string | null,
};

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  ...INITIAL_STATE,

  bootstrap: async () => {
    // I5 — silent-failure-hunter remediation. bootstrap() is contracted
    // to NEVER reject: callers (App.tsx, resolveDefaultMode) treat
    // bootstrap as "load OR record why we couldn't". On failure we
    // degrade to the guided default and surface the failure via
    // writeError so the role="alert" region (Phase 1B-round-2) shows
    // the user something is wrong. Loaded is set true on the failure
    // branch as well so the UI doesn't block on a condition that will
    // never become true (a corrupt-row user would otherwise be unable
    // to even create a session).
    //
    // The error_type discriminator distinguishes a CorruptPreferencesError
    // (the row is structurally invalid — needs operator action) from a
    // transient network failure (will probably recover). The frontend
    // does not show bad_value (the backend handler strips it), only the
    // user-actionable framing.
    try {
      const payload = await fetchUserComposerPreferences();
      set({
        defaultMode: payload.default_mode,
        bannerDismissedAt: payload.banner_dismissed_at,
        freeformIntroDismissedAt: payload.freeform_intro_dismissed_at,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        tutorialStage: payload.tutorial_stage,
        tutorialSessionId: payload.tutorial_session_id,
        tutorialRunId: payload.tutorial_run_id,
        tutorialSourceDataHash: payload.tutorial_source_data_hash,
        loaded: true,
      });
    } catch (err) {
      // No-fabrication shape (CLAUDE.md "fabrication test"). Leave
      // defaultMode and tutorialCompletedAt at null — we genuinely
      // don't know what they were. Setting defaultMode="guided" here
      // would attribute a preference choice to the user that they
      // never made; the audit trail would later carry a confident
      // answer to a question the system never resolved.
      //
      // Set loaded:true so the UI unblocks (gating the whole UI on
      // loaded would leave a corrupt-row user unable to create a
      // session at all — strictly worse than presenting them with an
      // accurate "we couldn't load your preferences" banner).
      //
      // resolveDefaultMode() continues to throw when defaultMode is
      // null after a bootstrap pass; sessionStore.createSession()
      // catches that and presents the "couldn't apply your default
      // mode, you're in freeform" message — honest about not knowing.
      const apiError = err as Partial<ApiError>;
      const isCorrupt = apiError?.error_type === "corrupt_preferences";
      const message = isCorrupt
        ? "Your saved preferences are corrupted. Contact your administrator to restore them."
        : err instanceof Error
          ? `Couldn't load your preferences (${err.message}).`
          : "Couldn't load your preferences.";
      set({
        loaded: true,
        writeError: message,
      });
    }
  },

  saveTutorialProgress: async (progress) => {
    // Deliberately NOT gated on the `writing` serialisation flag: stage
    // transitions must never be silently dropped because an unrelated
    // preferences write is in flight, and this PATCH touches only the
    // disjoint tutorial_* resume fields (the backend upsert writes only
    // the supplied fields, so there is no clobber hazard with a
    // concurrent default_mode/banner write). No optimistic set/revert
    // either — the local tutorial UI state lives in the tutorial
    // machine; this store only mirrors the server's persisted copy, so
    // it updates from the response.
    const payload = await updateUserComposerPreferences({
      tutorial_stage: progress.stage,
      tutorial_session_id: progress.sessionId,
      tutorial_run_id: progress.runId,
      tutorial_source_data_hash: progress.sourceDataHash,
    });
    set({
      tutorialStage: payload.tutorial_stage,
      tutorialSessionId: payload.tutorial_session_id,
      tutorialRunId: payload.tutorial_run_id,
      tutorialSourceDataHash: payload.tutorial_source_data_hash,
    });
  },

  resolveDefaultMode: async () => {
    const current = get();
    if (current.loaded) {
      // bootstrap has already run. If defaultMode is null at this point,
      // bootstrap failed (writeError is set) and a second bootstrap pass
      // would just re-fail against the same broken backend. Throw
      // immediately so sessionStore.createSession surfaces the honest
      // secondary-failure attribution to the user without an extra
      // round-trip.
      if (current.defaultMode === null) {
        throw new Error(
          "preferencesStore: loaded=true but defaultMode is null — bootstrap failed or backend returned a null default_mode (contract violation)",
        );
      }
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

  saveTutorialMode: async (mode) => {
    if (get().writing) return;
    const previous = {
      defaultMode: get().defaultMode,
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
        optedOutAtSessionId: previous.optedOutAtSessionId,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't save your preference: ${err.message}`
            : "Couldn't save your preference.",
      });
      throw err;
    }
  },

  markTutorialGraduated: async (options = {}) => {
    // A graduation opt-out (skip or exit) must never be silently dropped
    // because an unrelated preferences write is in flight — the old
    // `if (writing) return` no-op did exactly that, stranding exit clicks
    // (elspeth-61591e64bb). Wait for the in-flight write to settle before
    // sending; bounded so a hung PATCH cannot wedge the exit (a rare
    // interleaved write beats a dropped opt-out).
    for (let waitedMs = 0; get().writing && waitedMs < 5000; waitedMs += 50) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    // Re-read AFTER the wait: the write just waited out may itself have been
    // this same graduation (a double-clicked Exit, or the chrome exit racing
    // the wizard's onExited hand-off). Proceeding unconditionally would stamp
    // a second completion PATCH and double-count completion_path telemetry —
    // the backend counts every via=exit write, not null→set transitions
    // (preferences/service.py). publishLocally=false writes (skip) never set
    // tutorialCompleted, so the graduation card's deliberate re-persist
    // still goes through.
    const settled = get();
    if (settled.tutorialCompleted && settled.tutorialCompletedAt !== null) {
      return settled.tutorialCompletedAt;
    }
    const publishLocally = options.publishLocally ?? true;
    const stamp = new Date().toISOString();
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
        tutorial_completed_at: stamp,
        ...(options.via !== undefined
          ? { tutorial_completed_via: options.via }
          : {}),
      });
      set({
        ...(publishLocally
          ? {
              tutorialCompletedAt: payload.tutorial_completed_at,
              tutorialCompleted: tutorialCompletedFrom(
                payload.tutorial_completed_at,
              ),
            }
          : {}),
        // Completion-clears-progress (backend rule): the server just
        // terminated any in-progress resume state; mirror it. Safe to
        // publish even when publishLocally=false — the resume fields do
        // not drive the tutorial-vs-composer gate.
        tutorialStage: payload.tutorial_stage,
        tutorialSessionId: payload.tutorial_session_id,
        tutorialRunId: payload.tutorial_run_id,
        tutorialSourceDataHash: payload.tutorial_source_data_hash,
        writing: false,
      });
      return payload.tutorial_completed_at;
    } catch (err) {
      set({
        tutorialCompletedAt: previous.tutorialCompletedAt,
        tutorialCompleted: previous.tutorialCompleted,
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't save tutorial completion: ${err.message}`
            : "Couldn't save tutorial completion.",
      });
      throw err;
    }
  },

  publishTutorialGraduation: (completedAt) => {
    set({
      tutorialCompletedAt: completedAt,
      tutorialCompleted: tutorialCompletedFrom(completedAt),
    });
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
        // Explicitly clear the resume fields too. The completion-clears-
        // progress rule covers the graduated path, but Reset is now also
        // offered MID-tutorial (the wedged-resume escape hatch) — there the
        // stale stage/session must not survive the reset, or the next load
        // resumes straight back into the state being escaped.
        tutorial_stage: null,
        tutorial_session_id: null,
        tutorial_run_id: null,
        tutorial_source_data_hash: null,
      });
      set({
        defaultMode: payload.default_mode,
        bannerDismissedAt: payload.banner_dismissed_at,
        tutorialCompletedAt: payload.tutorial_completed_at,
        tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
        // A retake restarts cleanly at Welcome: the reset PATCH also
        // cleared any lingering resume state server-side
        // (completion-clears-progress rule); mirror it locally.
        tutorialStage: payload.tutorial_stage,
        tutorialSessionId: payload.tutorial_session_id,
        tutorialRunId: payload.tutorial_run_id,
        tutorialSourceDataHash: payload.tutorial_source_data_hash,
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
    if (get().writing) {
      // Offensive guard (CLAUDE.md): the dismiss button is disabled
      // while `writing` is true, so reaching this branch means a UI
      // guard was bypassed (programmatic call, keyboard race, or
      // future caller that doesn't read the writing flag). Throw a
      // named error so the regression surfaces loudly instead of
      // silently failing to dismiss.
      throw new Error(
        "preferencesStore: dismissDefaultChangedBanner called while a write was in flight — UI must disable the trigger before invoking this action",
      );
    }
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

  dismissFreeformIntro: async () => {
    if (get().writing) {
      throw new Error(
        "preferencesStore: dismissFreeformIntro called while a write was in flight",
      );
    }
    const stamp = new Date().toISOString();
    set({ writing: true, writeError: null });
    try {
      const payload = await updateUserComposerPreferences({
        freeform_intro_dismissed_at: stamp,
      });
      const resolved = payload.freeform_intro_dismissed_at;
      set({
        freeformIntroDismissedAt: resolved,
        writing: false,
      });
      if (typeof window !== "undefined" && resolved !== null) {
        try {
          window.localStorage.setItem(
            FREEFORM_INTRO_DISMISSED_STORAGE_KEY,
            resolved,
          );
        } catch {
          // The account-level server value remains authoritative; peer tabs
          // catch up on their next preference bootstrap.
        }
      }
    } catch (err) {
      set({
        writing: false,
        writeError:
          err instanceof Error
            ? `Couldn't hide the freeform introduction: ${err.message}`
            : "Couldn't hide the freeform introduction.",
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
    if (event.newValue === null) return;
    if (event.key === FREEFORM_INTRO_DISMISSED_STORAGE_KEY) {
      usePreferencesStore.setState({
        freeformIntroDismissedAt: event.newValue,
      });
      return;
    }
    if (event.key !== BANNER_DISMISSED_STORAGE_KEY) return;
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

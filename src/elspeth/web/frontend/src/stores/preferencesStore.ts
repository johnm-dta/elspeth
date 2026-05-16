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
// ============================================================================

import { create } from "zustand";
import {
  fetchUserComposerPreferences,
  updateUserComposerPreferences,
} from "@/api/client";
import type { ComposerMode } from "@/types/api";

interface PreferencesState {
  defaultMode: ComposerMode | null;
  bannerDismissedAt: string | null;
  loaded: boolean;
  writing: boolean;

  bootstrap: () => Promise<void>;
  resolveDefaultMode: () => Promise<ComposerMode>;
  setDefaultMode: (mode: ComposerMode) => Promise<void>;
  dismissDefaultChangedBanner: () => Promise<void>;
  reset: () => void;
}

const INITIAL_STATE = {
  defaultMode: null as ComposerMode | null,
  bannerDismissedAt: null as string | null,
  loaded: false,
  writing: false,
};

export const usePreferencesStore = create<PreferencesState>((set, get) => ({
  ...INITIAL_STATE,

  bootstrap: async () => {
    const payload = await fetchUserComposerPreferences();
    set({
      defaultMode: payload.default_mode,
      bannerDismissedAt: payload.banner_dismissed_at,
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

  setDefaultMode: async (mode) => {
    if (get().writing) return;
    const previous = get().defaultMode;
    set({ defaultMode: mode, writing: true });
    try {
      const payload = await updateUserComposerPreferences({
        default_mode: mode,
      });
      set({ defaultMode: payload.default_mode, writing: false });
    } catch (err) {
      set({ defaultMode: previous, writing: false });
      throw err;
    }
  },

  dismissDefaultChangedBanner: async () => {
    if (get().writing) return;
    const stamp = new Date().toISOString();
    const previous = get().bannerDismissedAt;
    set({ bannerDismissedAt: stamp, writing: true });
    try {
      const payload = await updateUserComposerPreferences({
        banner_dismissed_at: stamp,
      });
      set({
        bannerDismissedAt: payload.banner_dismissed_at,
        writing: false,
      });
    } catch (err) {
      set({ bannerDismissedAt: previous, writing: false });
      throw err;
    }
  },

  reset: () => set(INITIAL_STATE),
}));

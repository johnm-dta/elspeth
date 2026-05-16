/**
 * Tests for src/stores/preferencesStore.ts (Phase 1B Task 2).
 *
 * Mocking convention: vi.mock("@/api/client", () => ({...})) at module load,
 * vi.mocked(fn).mockResolvedValueOnce(...) per test. Mirrors
 * sessionStore.test.ts.
 *
 * Store-isolation convention: resetStore(usePreferencesStore) in beforeEach,
 * because Zustand stores are module-level singletons and any earlier
 * test in the same vitest worker leaks state otherwise (the Phase 1A
 * finding-7 pattern).
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import { usePreferencesStore } from "./preferencesStore";
import { resetStore } from "@/test/store-helpers";
import {
  fetchUserComposerPreferences,
  updateUserComposerPreferences,
} from "@/api/client";

vi.mock("@/api/client", () => ({
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

const mockFetch = vi.mocked(fetchUserComposerPreferences);
const mockUpdate = vi.mocked(updateUserComposerPreferences);

describe("preferencesStore", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.clearAllMocks();
  });

  it("loads preferences from API on bootstrap", async () => {
    mockFetch.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().bootstrap();

    const state = usePreferencesStore.getState();
    expect(state.defaultMode).toBe("freeform");
    expect(state.bannerDismissedAt).toBeNull();
    expect(state.loaded).toBe(true);
  });

  it("setDefaultMode updates state optimistically and persists", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().setDefaultMode("freeform");

    expect(usePreferencesStore.getState().defaultMode).toBe("freeform");
    expect(mockUpdate).toHaveBeenCalledWith({ default_mode: "freeform" });
  });

  it("setDefaultMode reverts on error", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockRejectedValueOnce(new Error("network failure"));

    await expect(
      usePreferencesStore.getState().setDefaultMode("freeform"),
    ).rejects.toThrow("network failure");

    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
    expect(usePreferencesStore.getState().writing).toBe(false);
  });

  it("setDefaultMode ignores concurrent calls while writing", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      writing: true,
    });

    await usePreferencesStore.getState().setDefaultMode("freeform");

    expect(mockUpdate).not.toHaveBeenCalled();
    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
  });

  it("dismissDefaultChangedBanner persists timestamp", async () => {
    const stamp = "2026-05-15T12:00:00Z";
    usePreferencesStore.setState({ loaded: true, defaultMode: "freeform" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: stamp,
      updated_at: stamp,
    });

    await usePreferencesStore.getState().dismissDefaultChangedBanner();

    expect(usePreferencesStore.getState().bannerDismissedAt).toBe(stamp);
    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ banner_dismissed_at: expect.any(String) }),
    );
  });

  it("dismissDefaultChangedBanner is no-op while another write is in flight", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
      writing: true,
    });

    await usePreferencesStore.getState().dismissDefaultChangedBanner();

    expect(mockUpdate).not.toHaveBeenCalled();
    expect(usePreferencesStore.getState().bannerDismissedAt).toBeNull();
  });

  it("banner reappears if backend dismiss fails (revert-on-error)", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
    });
    mockUpdate.mockRejectedValueOnce(new Error("server error"));

    await expect(
      usePreferencesStore.getState().dismissDefaultChangedBanner(),
    ).rejects.toThrow("server error");

    expect(usePreferencesStore.getState().bannerDismissedAt).toBeNull();
    expect(usePreferencesStore.getState().writing).toBe(false);
  });

  it("bootstrap is re-entrant safe (always fetches; two calls = two API calls)", async () => {
    mockFetch.mockResolvedValue({
      default_mode: "guided",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().bootstrap();
    await usePreferencesStore.getState().bootstrap();

    expect(usePreferencesStore.getState().loaded).toBe(true);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("resolveDefaultMode returns cached value if already loaded", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });

    const mode = await usePreferencesStore.getState().resolveDefaultMode();

    expect(mode).toBe("guided");
    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("resolveDefaultMode awaits bootstrap when not yet loaded", async () => {
    mockFetch.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    const mode = await usePreferencesStore.getState().resolveDefaultMode();

    expect(mode).toBe("freeform");
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(usePreferencesStore.getState().loaded).toBe(true);
  });
});

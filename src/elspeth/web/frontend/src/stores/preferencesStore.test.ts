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
  // Phase 1B Task 4.5 — the integration test below drives the real
  // sessionStore.createSession, which calls api.createSession; mock here so
  // the module-load wiring for that test resolves cleanly.
  createSession: vi.fn(),
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

// ── Phase 1B Task 4.5: preferences → session integration ───────────────────
// Both real stores. Only the API layer is mocked (and enterGuided is stubbed
// to avoid pulling in the GET /guided machinery). Proves the inter-store
// contract — sessionStore.createSession actually consults the live
// preferencesStore via resolveDefaultMode() rather than reading a stale
// closure or its own state.
describe("preferences → session integration (real stores, API mocked)", () => {
  beforeEach(async () => {
    resetStore(usePreferencesStore);
    const { useSessionStore } = await import("@/stores/sessionStore");
    resetStore(useSessionStore);
    vi.clearAllMocks();
  });

  it("createSession enters guided when the live preference is guided (smoke)", async () => {
    const { useSessionStore } = await import("@/stores/sessionStore");
    const api = await import("@/api/client");
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      bannerDismissedAt: null,
      writing: false,
    });
    (api.createSession as ReturnType<typeof vi.fn>).mockResolvedValueOnce({
      id: "sess-int",
      title: "untitled",
      created_at: "2026-05-15T00:00:00Z",
      updated_at: "2026-05-15T00:00:00Z",
    });
    const enterGuided = vi
      .spyOn(useSessionStore.getState(), "enterGuided")
      .mockResolvedValue();

    await useSessionStore.getState().createSession();

    expect(enterGuided).toHaveBeenCalledTimes(1);
  });
});

// ── Phase 1B panel banner-cluster + writeError additions ─────────────────
describe("preferencesStore — banner cluster + error surface (Phase 1B Panel)", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.clearAllMocks();
    // Clean localStorage so the cross-tab broadcast tests don't see
    // residue from prior tests in the same worker.
    try {
      window.localStorage.clear();
    } catch {
      // localStorage may be unavailable in some test envs; fine.
    }
  });

  it("setDefaultMode('freeform', activeSessionId) captures optedOutAtSessionId watermark", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-16T00:00:00Z",
    });

    await usePreferencesStore.getState().setDefaultMode("freeform", "sess-a");

    expect(usePreferencesStore.getState().optedOutAtSessionId).toBe("sess-a");
  });

  it("setDefaultMode('guided', ...) clears optedOutAtSessionId (re-opt-in resets)", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      optedOutAtSessionId: "sess-old",
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      updated_at: "2026-05-16T00:00:00Z",
    });

    await usePreferencesStore.getState().setDefaultMode("guided", "sess-b");

    expect(usePreferencesStore.getState().optedOutAtSessionId).toBeNull();
  });

  it("setDefaultMode failure clears the watermark on revert (banner doesn't suppress for a write that didn't land)", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockRejectedValueOnce(new Error("503"));

    await expect(
      usePreferencesStore.getState().setDefaultMode("freeform", "sess-c"),
    ).rejects.toThrow("503");

    expect(usePreferencesStore.getState().optedOutAtSessionId).toBeNull();
    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
  });

  it("setDefaultMode populates writeError on failure (role=alert surface for AT users)", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockRejectedValueOnce(new Error("network down"));

    await expect(
      usePreferencesStore.getState().setDefaultMode("freeform", null),
    ).rejects.toThrow("network down");

    expect(usePreferencesStore.getState().writeError).toMatch(/network down/);
  });

  it("setDefaultMode clears writeError on next success (recovery semantics)", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      writeError: "prior failure",
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      updated_at: "2026-05-16T00:00:00Z",
    });

    await usePreferencesStore.getState().setDefaultMode("freeform", null);

    expect(usePreferencesStore.getState().writeError).toBeNull();
  });

  it("dismissDefaultChangedBanner writes resolved value to localStorage (cross-tab broadcast)", async () => {
    const stamp = "2026-05-16T01:00:00Z";
    usePreferencesStore.setState({ loaded: true, defaultMode: "freeform" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: stamp,
      updated_at: stamp,
    });

    await usePreferencesStore.getState().dismissDefaultChangedBanner();

    expect(window.localStorage.getItem("elspeth_prefs_banner_dismissed_v1")).toBe(stamp);
  });

  it("storage event from a peer tab updates bannerDismissedAt without making a PATCH", () => {
    // Simulate a peer tab having just dismissed: the storage event fires
    // in THIS tab, and our store listener updates local state.
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
    });
    const peerStamp = "2026-05-16T02:00:00Z";

    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "elspeth_prefs_banner_dismissed_v1",
        newValue: peerStamp,
      }),
    );

    expect(usePreferencesStore.getState().bannerDismissedAt).toBe(peerStamp);
    // No PATCH issued — the peer tab already wrote the value.
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("storage event for unrelated keys is ignored", () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
    });

    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "some-other-key",
        newValue: "value",
      }),
    );

    expect(usePreferencesStore.getState().bannerDismissedAt).toBeNull();
  });

  it("dismissDefaultChangedBanner populates writeError on failure", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
    });
    mockUpdate.mockRejectedValueOnce(new Error("503 Service Unavailable"));

    await expect(
      usePreferencesStore.getState().dismissDefaultChangedBanner(),
    ).rejects.toThrow("503");

    expect(usePreferencesStore.getState().writeError).toMatch(/503/);
  });

  it("clearError() returns writeError to null", () => {
    usePreferencesStore.setState({ writeError: "some error" });
    usePreferencesStore.getState().clearError();
    expect(usePreferencesStore.getState().writeError).toBeNull();
  });
});

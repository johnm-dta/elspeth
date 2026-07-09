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
import { selectTutorialCompleted, usePreferencesStore } from "./preferencesStore";
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    await usePreferencesStore.getState().bootstrap();

    const state = usePreferencesStore.getState();
    expect(state.defaultMode).toBe("freeform");
    expect(state.bannerDismissedAt).toBeNull();
    expect(state.tutorialCompletedAt).toBeNull();
    expect(selectTutorialCompleted(state)).toBe(false);
    expect(state.loaded).toBe(true);
  });

  it("selectTutorialCompleted derives true when tutorialCompletedAt is set", () => {
    usePreferencesStore.setState({
      tutorialCompletedAt: "2026-05-19T12:00:00Z",
      tutorialCompleted: true,
    });

    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it("setDefaultMode updates state optimistically and persists", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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

  it("saveTutorialMode PATCHes only the default mode", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-19T12:30:00Z",
    });

    await usePreferencesStore.getState().saveTutorialMode("freeform");

    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(mockUpdate.mock.calls[0][0]).toEqual({ default_mode: "freeform" });
    expect(usePreferencesStore.getState().defaultMode).toBe("freeform");
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull();
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(false);
  });

  it("markTutorialGraduated PATCHes only the completion timestamp", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-05-19T12:30:00Z",
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-19T12:30:00Z",
    });

    await usePreferencesStore.getState().markTutorialGraduated();

    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(mockUpdate.mock.calls[0][0]).toEqual({
      tutorial_completed_at: expect.any(String),
    });
    expect(usePreferencesStore.getState().defaultMode).toBe("freeform");
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBe(
      "2026-05-19T12:30:00Z",
    );
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it("markTutorialGraduated can defer the local completion flip until the caller publishes it", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      tutorialCompletedAt: null,
      tutorialCompleted: false,
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-05-19T12:30:00Z",
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-19T12:30:00Z",
    });

    const completedAt = await usePreferencesStore
      .getState()
      .markTutorialGraduated({ publishLocally: false });

    expect(completedAt).toBe("2026-05-19T12:30:00Z");
    expect(mockUpdate.mock.calls[0][0]).toEqual({
      tutorial_completed_at: expect.any(String),
    });
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull();
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(false);

    usePreferencesStore.getState().publishTutorialGraduation(completedAt);

    expect(usePreferencesStore.getState().tutorialCompletedAt).toBe(
      "2026-05-19T12:30:00Z",
    );
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it("saveTutorialMode respects the writing guard", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      writing: true,
    });

    await usePreferencesStore.getState().saveTutorialMode("freeform");

    expect(mockUpdate).not.toHaveBeenCalled();
    expect(usePreferencesStore.getState().defaultMode).toBe("guided");
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(false);
  });

  it("markTutorialGraduated waits out an in-flight write instead of dropping the opt-out", async () => {
    // The old `if (writing) return` no-op silently dropped the graduation
    // PATCH — an exit/skip click landing while another preferences write was
    // in flight simply never persisted (elspeth-61591e64bb). The store now
    // waits for the in-flight write to settle, then sends the PATCH.
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      writing: true,
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-07-09T00:00:00Z",
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-09T00:00:00Z",
    });
    // Simulate the in-flight write settling shortly after the click.
    setTimeout(() => {
      usePreferencesStore.setState({ writing: false });
    }, 120);

    await usePreferencesStore.getState().markTutorialGraduated();

    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ tutorial_completed_at: expect.any(String) }),
    );
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it("markTutorialGraduated does not re-PATCH a completion that landed while it waited", async () => {
    // Double-clicked Exit (or the chrome exit racing the wizard's onExited
    // hand-off): the write this call waits out IS the same graduation. The
    // backend counts every via=exit PATCH (no prior-state check), so an
    // unconditional second PATCH double-counts completion_path telemetry.
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      writing: true,
    });
    // Simulate the first exit click's PATCH landing while this one waits.
    setTimeout(() => {
      usePreferencesStore.setState({
        writing: false,
        tutorialCompletedAt: "2026-07-09T00:00:00Z",
        tutorialCompleted: true,
      });
    }, 120);

    const completedAt = await usePreferencesStore
      .getState()
      .markTutorialGraduated({ via: "exit" });

    expect(completedAt).toBe("2026-07-09T00:00:00Z");
    expect(mockUpdate).not.toHaveBeenCalled();
  });

  it("two rapid exit clicks send exactly one completion PATCH", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    let resolveFirst!: (payload: {
      default_mode: "guided";
      banner_dismissed_at: null;
      tutorial_completed_at: string;
      tutorial_stage: null;
      tutorial_session_id: null;
      tutorial_run_id: null;
      tutorial_source_data_hash: null;
      updated_at: string;
    }) => void;
    mockUpdate.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveFirst = resolve;
        }),
    );

    const first = usePreferencesStore
      .getState()
      .markTutorialGraduated({ via: "exit" });
    const second = usePreferencesStore
      .getState()
      .markTutorialGraduated({ via: "exit" });
    resolveFirst({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-07-09T00:00:00Z",
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-09T00:00:00Z",
    });

    const [a, b] = await Promise.all([first, second]);

    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(a).toBe("2026-07-09T00:00:00Z");
    expect(b).toBe("2026-07-09T00:00:00Z");
  });

  it("markTutorialGraduated sends the exit discriminator when asked", async () => {
    // The backend infers first_time/skip from payload shape; an explicit
    // exit must carry tutorial_completed_via so it is not bucketed as
    // "skip" (elspeth-61591e64bb telemetry correction).
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-07-09T00:00:00Z",
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-09T00:00:00Z",
    });

    await usePreferencesStore.getState().markTutorialGraduated({ via: "exit" });

    expect(mockUpdate).toHaveBeenCalledWith({
      tutorial_completed_at: expect.any(String),
      tutorial_completed_via: "exit",
    });
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(true);
  });

  it("resetTutorial clears tutorial_completed_at through the PATCH contract", async () => {
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "guided",
      tutorialCompletedAt: "2026-05-19T12:00:00Z",
      tutorialCompleted: true,
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-19T12:30:00Z",
    });

    await usePreferencesStore.getState().resetTutorial();

    // Completion AND the resume fields clear in one PATCH: Reset is also
    // offered mid-tutorial (the wedged-resume escape hatch), where a stale
    // stage/session surviving the reset would resume straight back into the
    // state being escaped.
    expect(mockUpdate).toHaveBeenCalledWith({
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
    });
    expect(usePreferencesStore.getState().tutorialCompletedAt).toBeNull();
    expect(selectTutorialCompleted(usePreferencesStore.getState())).toBe(false);
  });

  it("dismissDefaultChangedBanner persists timestamp", async () => {
    const stamp = "2026-05-15T12:00:00Z";
    usePreferencesStore.setState({ loaded: true, defaultMode: "freeform" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "freeform",
      banner_dismissed_at: stamp,
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: stamp,
    });

    await usePreferencesStore.getState().dismissDefaultChangedBanner();

    expect(usePreferencesStore.getState().bannerDismissedAt).toBe(stamp);
    expect(mockUpdate).toHaveBeenCalledWith(
      expect.objectContaining({ banner_dismissed_at: expect.any(String) }),
    );
  });

  it("dismissDefaultChangedBanner throws if invoked while another write is in flight (P0.8 offensive guard)", async () => {
    // P0.8: the prior silent `if (writing) return;` short-circuit
    // hid a UI-guard bypass behind a no-op. Now the store throws a
    // named error so any caller that fails to disable its trigger
    // (the DefaultModeChangedBanner dismiss button) gets a loud
    // failure surface. In normal flow the disabled button prevents
    // this path entirely; reaching the throw means a regression.
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: null,
      writing: true,
    });

    await expect(
      usePreferencesStore.getState().dismissDefaultChangedBanner(),
    ).rejects.toThrow(
      /called while a write was in flight — UI must disable the trigger/,
    );
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-05-15T00:00:00Z",
    });

    const mode = await usePreferencesStore.getState().resolveDefaultMode();

    expect(mode).toBe("freeform");
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(usePreferencesStore.getState().loaded).toBe(true);
  });

  // I5 — silent-failure-hunter remediation. Before this fix, bootstrap()
  // would reject and the App.tsx caller's `.catch(console.error)` would
  // swallow the failure. `loaded` stayed false, gating the UI on a
  // condition that would never become true — a CorruptPreferencesError
  // (named for incident response) was completely invisible to the user.
  //
  // The new contract: bootstrap() NEVER rejects. On failure it sets
  // loaded:true + writeError but LEAVES defaultMode null (the
  // no-fabrication shape per CLAUDE.md). Setting defaultMode="guided"
  // on failure would attribute a preference choice to the user that
  // they never made. resolveDefaultMode() continues to throw on the
  // null branch; sessionStore.createSession catches that and tells the
  // user "you're in freeform; we couldn't apply your default mode" —
  // an honest secondary-failure attribution.

  it("bootstrap sets loaded+writeError on generic API failure without fabricating a defaultMode", async () => {
    mockFetch.mockRejectedValueOnce(new Error("network failure"));

    await expect(
      usePreferencesStore.getState().bootstrap(),
    ).resolves.toBeUndefined();

    const state = usePreferencesStore.getState();
    expect(state.loaded).toBe(true);
    // Honest: we don't know what mode the user had set.
    expect(state.defaultMode).toBeNull();
    expect(state.writeError).not.toBeNull();
    expect(state.writeError).toMatch(/network failure/);
  });

  it("bootstrap surfaces a corrupt-preferences message when the backend signals error_type=corrupt_preferences", async () => {
    // ApiError shape produced by parseResponse() in @/api/client when the
    // backend returns a structured 5xx with error_type. The store branches
    // on error_type to distinguish a corrupt-row failure (needs operator
    // action) from a transient unavailability.
    const apiError = {
      status: 500,
      detail: "Saved preferences are corrupt; the composer is using defaults.",
      error_type: "corrupt_preferences",
    };
    mockFetch.mockRejectedValueOnce(apiError);

    await expect(
      usePreferencesStore.getState().bootstrap(),
    ).resolves.toBeUndefined();

    const state = usePreferencesStore.getState();
    expect(state.loaded).toBe(true);
    expect(state.defaultMode).toBeNull();
    expect(state.writeError).not.toBeNull();
    expect(state.writeError).toMatch(/corrupt/i);
    expect(state.writeError).toMatch(/administrator|operator|contact/i);
  });

  it("resolveDefaultMode throws immediately without re-bootstrapping when loaded=true and defaultMode=null", async () => {
    // P0.9: the prior implementation guarded as
    // `if (current.loaded && current.defaultMode !== null) return …`,
    // then fell through to a second `bootstrap()` call when that guard
    // failed because of `defaultMode === null`. A bootstrap that already
    // produced `loaded:true, defaultMode:null` is in a known-broken
    // state (writeError is set); re-running it just re-fails. Throw
    // immediately so sessionStore.createSession surfaces the honest
    // secondary-failure message to the user without an extra round-trip.
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: null,
      writeError: "Saved preferences are corrupt; using defaults.",
    });

    await expect(
      usePreferencesStore.getState().resolveDefaultMode(),
    ).rejects.toThrow(/loaded=true but defaultMode is null/);

    expect(mockFetch).not.toHaveBeenCalled();
  });

  it("resolveDefaultMode still throws after a failed bootstrap (preserves sessionStore secondary-failure attribution)", async () => {
    // mockRejected is consumed twice because resolveDefaultMode awaits
    // bootstrap() once and that bootstrap consumes one mockRejectedValueOnce.
    // Use mockRejectedValue (not Once) so the implementation can retry
    // without exhausting the queue.
    mockFetch.mockRejectedValue(new Error("server unreachable"));

    await expect(
      usePreferencesStore.getState().resolveDefaultMode(),
    ).rejects.toThrow(/bootstrap completed but defaultMode is null/);

    // The writeError is set on the failure path even though the throw
    // bubbles out of resolveDefaultMode — that's the channel
    // sessionStore.createSession surfaces to the user.
    expect(usePreferencesStore.getState().writeError).not.toBeNull();
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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
      tutorial_completed_at: null,
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
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

describe("preferencesStore — tutorial resume state (elspeth-918f4434b3)", () => {
  beforeEach(() => {
    resetStore(usePreferencesStore);
    vi.clearAllMocks();
  });

  it("bootstrap loads the persisted tutorial progress fields", async () => {
    mockFetch.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: "run",
      tutorial_session_id: "sess-1",
      tutorial_run_id: "run-1",
      tutorial_source_data_hash: "hash-1",
      updated_at: "2026-07-02T00:00:00Z",
    });

    await usePreferencesStore.getState().bootstrap();

    const state = usePreferencesStore.getState();
    expect(state.tutorialStage).toBe("run");
    expect(state.tutorialSessionId).toBe("sess-1");
    expect(state.tutorialRunId).toBe("run-1");
    expect(state.tutorialSourceDataHash).toBe("hash-1");
  });

  it("saveTutorialProgress PATCHes the four resume fields and mirrors the response", async () => {
    usePreferencesStore.setState({ loaded: true, defaultMode: "guided" });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: "guided",
      tutorial_session_id: "sess-2",
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-02T00:00:00Z",
    });

    await usePreferencesStore.getState().saveTutorialProgress({
      stage: "guided",
      sessionId: "sess-2",
      runId: null,
      sourceDataHash: null,
    });

    expect(mockUpdate).toHaveBeenCalledWith({
      tutorial_stage: "guided",
      tutorial_session_id: "sess-2",
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
    });
    const state = usePreferencesStore.getState();
    expect(state.tutorialStage).toBe("guided");
    expect(state.tutorialSessionId).toBe("sess-2");
  });

  it("saveTutorialProgress is not blocked by the writing serialisation flag", async () => {
    // A stage transition must never be silently dropped because an
    // unrelated preferences write is in flight — the PATCH touches only
    // the disjoint tutorial_* fields.
    usePreferencesStore.setState({ loaded: true, writing: true });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: null,
      tutorial_stage: "run",
      tutorial_session_id: "sess-3",
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-02T00:00:00Z",
    });

    await usePreferencesStore.getState().saveTutorialProgress({
      stage: "run",
      sessionId: "sess-3",
      runId: null,
      sourceDataHash: null,
    });

    expect(mockUpdate).toHaveBeenCalledTimes(1);
    expect(usePreferencesStore.getState().tutorialStage).toBe("run");
  });

  it("markTutorialGraduated mirrors the server clearing the resume fields", async () => {
    usePreferencesStore.setState({
      loaded: true,
      tutorialStage: "graduation",
      tutorialSessionId: "sess-4",
      tutorialRunId: "run-4",
      tutorialSourceDataHash: "hash-4",
    });
    mockUpdate.mockResolvedValueOnce({
      default_mode: "guided",
      banner_dismissed_at: null,
      tutorial_completed_at: "2026-07-02T00:00:00Z",
      // Completion-clears-progress: the backend terminated the resume state.
      tutorial_stage: null,
      tutorial_session_id: null,
      tutorial_run_id: null,
      tutorial_source_data_hash: null,
      updated_at: "2026-07-02T00:00:00Z",
    });

    await usePreferencesStore.getState().markTutorialGraduated();

    const state = usePreferencesStore.getState();
    expect(state.tutorialCompleted).toBe(true);
    expect(state.tutorialStage).toBeNull();
    expect(state.tutorialSessionId).toBeNull();
    expect(state.tutorialRunId).toBeNull();
    expect(state.tutorialSourceDataHash).toBeNull();
  });
});

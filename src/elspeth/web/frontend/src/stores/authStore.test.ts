import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "./authStore";
import { usePreferencesStore } from "./preferencesStore";
import { resetStore } from "@/test/store-helpers";

vi.mock("@/api/client", () => ({
  fetchCurrentUser: vi.fn(),
  login: vi.fn(),
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
}));

describe("authStore account-scoped store reset", () => {
  beforeEach(() => {
    resetStore(useAuthStore);
    resetStore(usePreferencesStore);
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("logout clears cached composer preferences before another account can reuse them", async () => {
    useAuthStore.setState({
      token: "token-for-alice",
      user: {
        user_id: "alice",
        username: "alice",
        display_name: "Alice",
        email: null,
        groups: [],
      },
      isLoading: false,
    });
    usePreferencesStore.setState({
      loaded: true,
      defaultMode: "freeform",
      bannerDismissedAt: "2026-05-16T00:00:00Z",
      tutorialCompletedAt: "2026-05-19T00:00:00Z",
      tutorialCompleted: true,
      optedOutAtSessionId: "alice-session",
    });

    await useAuthStore.getState().logout();

    expect(usePreferencesStore.getState()).toMatchObject({
      loaded: false,
      defaultMode: null,
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      optedOutAtSessionId: null,
      writeError: null,
    });
  });
});

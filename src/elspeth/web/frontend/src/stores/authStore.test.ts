import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "./authStore";
import { useBlobStore } from "./blobStore";
import { usePreferencesStore } from "./preferencesStore";
import { useSecretsStore } from "./secretsStore";
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
    resetStore(useBlobStore);
    resetStore(usePreferencesStore);
    resetStore(useSecretsStore);
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("logout clears account-scoped cached stores before another account can reuse them", async () => {
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
    useBlobStore.setState({
      blobs: [
        {
          id: "alice-blob",
          session_id: "alice-session",
          filename: "alice.csv",
          mime_type: "text/csv",
          size_bytes: 128,
          content_hash: "abc123",
          created_at: "2026-05-16T00:00:00Z",
          created_by: "user",
          source_description: null,
          status: "ready",
          creation_modality: "verbatim",
          created_from_message_id: null,
          creating_model_identifier: null,
          creating_model_version: null,
          creating_provider: null,
          creating_composer_skill_hash: null,
          creating_arguments_hash: null,
        },
      ],
      error: "old blob error",
    });
    useSecretsStore.setState({
      secrets: [
        {
          name: "ALICE_API_KEY",
          scope: "user",
          available: true,
          source_kind: "database",
          reason: null,
        },
      ],
      error: "old secret error",
    });

    await useAuthStore.getState().logout();

    expect(useBlobStore.getState()).toMatchObject({
      blobs: [],
      isLoading: false,
      error: null,
    });
    expect(usePreferencesStore.getState()).toMatchObject({
      loaded: false,
      defaultMode: null,
      bannerDismissedAt: null,
      tutorialCompletedAt: null,
      tutorialCompleted: false,
      optedOutAtSessionId: null,
      writeError: null,
    });
    expect(useSecretsStore.getState()).toMatchObject({
      secrets: [],
      isLoading: false,
      error: null,
    });
  });
});

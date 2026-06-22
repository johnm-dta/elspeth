import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "./authStore";
import { useBlobStore } from "./blobStore";
import { usePreferencesStore } from "./preferencesStore";
import { useSecretsStore } from "./secretsStore";
import { useShareableReviewStore } from "./shareableReviewStore";
import * as shareableReviewsApi from "../api/shareableReviews";
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
    useShareableReviewStore.getState().reset();
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
    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: false,
      latestResponse: null,
      sessionIdForResponse: null,
      error: null,
      inFlight: false,
    });
  });

  it("logout clears a minted share-review token before another account can see it", async () => {
    vi.spyOn(shareableReviewsApi, "markReadyForReview").mockResolvedValueOnce({
      token: "alice-share-token",
      share_url: "/#/shared/alice-share-token",
      expires_at: "2026-06-19T00:00:00+00:00",
      payload_digest: "sha256:" + "ab".repeat(32),
    });
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
    await useShareableReviewStore.getState().openAndMark("alice-session");
    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: true,
      latestResponse: expect.objectContaining({
        token: "alice-share-token",
        share_url: "/#/shared/alice-share-token",
      }),
      sessionIdForResponse: "alice-session",
    });

    await useAuthStore.getState().logout();

    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: false,
      latestResponse: null,
      sessionIdForResponse: null,
      error: null,
      inFlight: false,
    });
  });

  it("logout prevents an in-flight share-review response from repopulating the store", async () => {
    let resolveMint: (
      response: Awaited<ReturnType<typeof shareableReviewsApi.markReadyForReview>>,
    ) => void = () => {};
    const pendingMint = new Promise<
      Awaited<ReturnType<typeof shareableReviewsApi.markReadyForReview>>
    >((resolve) => {
      resolveMint = resolve;
    });
    vi.spyOn(shareableReviewsApi, "markReadyForReview").mockReturnValueOnce(
      pendingMint,
    );
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

    const mintPromise =
      useShareableReviewStore.getState().openAndMark("alice-session");
    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: true,
      inFlight: true,
      sessionIdForResponse: "alice-session",
    });

    await useAuthStore.getState().logout();
    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: false,
      latestResponse: null,
      sessionIdForResponse: null,
      error: null,
      inFlight: false,
    });

    resolveMint({
      token: "late-alice-share-token",
      share_url: "/#/shared/late-alice-share-token",
      expires_at: "2026-06-19T00:00:00+00:00",
      payload_digest: "sha256:" + "cd".repeat(32),
    });
    await mintPromise;

    expect(useShareableReviewStore.getState()).toMatchObject({
      dialogOpen: false,
      latestResponse: null,
      sessionIdForResponse: null,
      error: null,
      inFlight: false,
    });
  });
});

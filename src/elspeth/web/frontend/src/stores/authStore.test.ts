import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "./authStore";
import { useBlobStore } from "./blobStore";
import { usePreferencesStore } from "./preferencesStore";
import { useSecretsStore } from "./secretsStore";
import { useShareableReviewStore } from "./shareableReviewStore";
import { usePluginCatalogStore } from "./pluginCatalogStore";
import * as shareableReviewsApi from "../api/shareableReviews";
import * as apiClient from "@/api/client";
import { resetStore } from "@/test/store-helpers";
import { GUIDED_RETRY_STORAGE_KEY } from "./guidedOperationRetry";

vi.mock("@/api/client", () => ({
  fetchCurrentUser: vi.fn(),
  login: vi.fn(),
  fetchUserComposerPreferences: vi.fn(),
  updateUserComposerPreferences: vi.fn(),
  fetchPluginPolicy: vi.fn(),
  listSources: vi.fn(),
  listTransforms: vi.fn(),
  listSinks: vi.fn(),
  getPluginSchema: vi.fn(),
}));

describe("authStore interactive login", () => {
  beforeEach(() => {
    resetStore(useAuthStore);
    // Simulate a completed boot: loadFromStorage has already resolved.
    useAuthStore.setState({ isLoading: false });
    localStorage.clear();
    sessionStorage.clear();
    vi.clearAllMocks();
  });

  it("failed login sets the generic error, returns false, and never flips isLoading", async () => {
    // Regression for elspeth-d49f8ad511: login() used to set isLoading=true,
    // which drove AuthGuard's "Checking authentication" spinner and
    // UNMOUNTED the LoginPage mid-attempt — a failed login then remounted
    // a blank form, wiping the username (WCAG 3.3.7 Redundant Entry).
    vi.mocked(apiClient.login).mockRejectedValue({
      status: 401,
      detail: "Invalid credentials",
    });
    const isLoadingSamples: boolean[] = [];
    const unsubscribe = useAuthStore.subscribe((state) => {
      isLoadingSamples.push(state.isLoading);
    });

    const succeeded = await useAuthStore.getState().login("alice", "wrong");
    unsubscribe();

    expect(succeeded).toBe(false);
    expect(isLoadingSamples).not.toContain(true);
    expect(useAuthStore.getState()).toMatchObject({
      token: null,
      user: null,
      loginError: "Invalid username or password.",
      isLoading: false,
    });
    expect(localStorage.getItem("auth_token")).toBeNull();
  });

  it("successful login returns true, stores the token, and loads the user", async () => {
    vi.mocked(apiClient.login).mockResolvedValue({ access_token: "tok-1" });
    vi.mocked(apiClient.fetchCurrentUser).mockResolvedValue({
      user_id: "u-1",
      username: "alice",
      display_name: "Alice",
      email: null,
      groups: [],
    });

    const succeeded = await useAuthStore.getState().login("alice", "pw");

    expect(succeeded).toBe(true);
    expect(useAuthStore.getState().token).toBe("tok-1");
    expect(useAuthStore.getState().user).toMatchObject({ username: "alice" });
    expect(useAuthStore.getState().loginError).toBeNull();
    expect(localStorage.getItem("auth_token")).toBe("tok-1");
  });
});

describe("authStore account-scoped store reset", () => {
  beforeEach(() => {
    resetStore(useAuthStore);
    resetStore(useBlobStore);
    resetStore(usePreferencesStore);
    resetStore(useSecretsStore);
    usePluginCatalogStore.getState().clear();
    useShareableReviewStore.getState().reset();
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("logout clears account-scoped cached stores before another account can reuse them", async () => {
    sessionStorage.setItem(GUIDED_RETRY_STORAGE_KEY, "stale-retry-custody");
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
    usePluginCatalogStore.setState({
      key: "local:alice:alice-snapshot",
      principal: "local:alice",
      fingerprint: "alice-snapshot",
      sources: [],
      transforms: [],
      sinks: [],
    });

    await useAuthStore.getState().logout();

    expect(sessionStorage.getItem(GUIDED_RETRY_STORAGE_KEY)).toBeNull();

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
    expect(usePluginCatalogStore.getState()).toMatchObject({
      key: null,
      principal: null,
      fingerprint: null,
      sources: null,
      transforms: null,
      sinks: null,
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

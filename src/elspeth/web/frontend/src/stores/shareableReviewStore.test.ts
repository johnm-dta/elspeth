import { describe, it, expect, vi, beforeEach } from "vitest";
import { useShareableReviewStore } from "./shareableReviewStore";
import * as api from "../api/shareableReviews";

const SESSION_A = "00000000-0000-0000-0000-00000000000a";
const SESSION_B = "00000000-0000-0000-0000-00000000000b";

const validResponseFor = (suffix: string) => ({
  token: `tk-${suffix}`,
  share_url: `/#/shared/tk-${suffix}`,
  expires_at: "2026-06-19T00:00:00+00:00",
  payload_digest: "sha256:" + "ab".repeat(32),
});

describe("shareableReviewStore", () => {
  beforeEach(() => {
    useShareableReviewStore.getState().reset();
    vi.restoreAllMocks();
  });

  it("openAndMark sets latestResponse + dialogOpen on success", async () => {
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(validResponseFor("1"));
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    const state = useShareableReviewStore.getState();
    expect(state.dialogOpen).toBe(true);
    expect(state.inFlight).toBe(false);
    expect(state.error).toBeNull();
    expect(state.latestResponse?.token).toBe("tk-1");
    expect(state.sessionIdForResponse).toBe(SESSION_A);
  });

  it("openAndMark marks inFlight=true during the POST", async () => {
    let resolveFn: (v: ReturnType<typeof validResponseFor>) => void = () => {};
    const pending = new Promise<ReturnType<typeof validResponseFor>>((resolve) => {
      resolveFn = resolve;
    });
    vi.spyOn(api, "markReadyForReview").mockReturnValueOnce(pending);
    const callPromise = useShareableReviewStore.getState().openAndMark(SESSION_A);
    expect(useShareableReviewStore.getState().inFlight).toBe(true);
    expect(useShareableReviewStore.getState().dialogOpen).toBe(true);
    resolveFn(validResponseFor("1"));
    await callPromise;
    expect(useShareableReviewStore.getState().inFlight).toBe(false);
  });

  it("openAndMark sets error and clears latestResponse on 409", async () => {
    vi.spyOn(api, "markReadyForReview").mockRejectedValueOnce({
      status: 409,
      detail: "composition validation failed; fix errors before sharing",
    });
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    const state = useShareableReviewStore.getState();
    expect(state.dialogOpen).toBe(true); // dialog stays open to show the error
    expect(state.inFlight).toBe(false);
    expect(state.error).toContain("composition validation failed");
    expect(state.latestResponse).toBeNull();
  });

  it("openAndMark falls back to a generic 401 message when detail absent", async () => {
    vi.spyOn(api, "markReadyForReview").mockRejectedValueOnce({ status: 401 });
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    const state = useShareableReviewStore.getState();
    expect(state.error).toBe("Authentication required.");
  });

  it("openAndMark clears stale response when switching sessions", async () => {
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(validResponseFor("A"));
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    expect(useShareableReviewStore.getState().sessionIdForResponse).toBe(SESSION_A);
    // Now switch to a different session. Latest response from A must NOT leak.
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(validResponseFor("B"));
    await useShareableReviewStore.getState().openAndMark(SESSION_B);
    const state = useShareableReviewStore.getState();
    expect(state.sessionIdForResponse).toBe(SESSION_B);
    expect(state.latestResponse?.token).toBe("tk-B");
  });

  it("close() keeps latestResponse so reopening shows the same URL", async () => {
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(validResponseFor("1"));
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    useShareableReviewStore.getState().close();
    const state = useShareableReviewStore.getState();
    expect(state.dialogOpen).toBe(false);
    expect(state.latestResponse?.token).toBe("tk-1"); // preserved
  });

  it("reset() clears everything", async () => {
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(validResponseFor("1"));
    await useShareableReviewStore.getState().openAndMark(SESSION_A);
    useShareableReviewStore.getState().reset();
    const state = useShareableReviewStore.getState();
    expect(state.dialogOpen).toBe(false);
    expect(state.latestResponse).toBeNull();
    expect(state.sessionIdForResponse).toBeNull();
    expect(state.error).toBeNull();
  });
});

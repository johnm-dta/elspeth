/**
 * Cross-cutting integration test for the Phase 6B completion-gesture flow.
 *
 * Phase 6B Task 11. The unit tests for CompletionBar / SaveForReviewDialog /
 * shareableReviewStore cover each piece in isolation. This test exercises
 * the full chain: click Save for review in the bar → store dispatches
 * markReadyForReview → dialog opens, shows the spinner, then the success
 * panel with the share URL → user copies the URL → dialog closes.
 *
 * Mocked surfaces:
 *
 * * `api/shareableReviews.markReadyForReview` — returns a fixed response.
 * * `navigator.clipboard.writeText` — succeeds, captures the argument.
 *
 * NOT mocked:
 *
 * * `useShareableReviewStore` — the real store; the test exercises its
 *   dispatch + state-transition logic end-to-end.
 * * `CompletionBar`, `SaveForReviewDialog` — the real components.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { CompletionBar } from "./CompletionBar";
import { SaveForReviewDialog } from "./SaveForReviewDialog";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";
import { useSessionStore } from "@/stores/sessionStore";
import { useExecutionStore } from "@/stores/executionStore";
import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
import { resetStore } from "@/test/store-helpers";
import * as api from "@/api/shareableReviews";

function _validValidation() {
  return {
    is_valid: true,
    checks: [],
    errors: [],
  } as never;
}

function _withOrigin(origin: string, fn: () => void) {
  const original = window.location;
  Object.defineProperty(window, "location", {
    writable: true,
    value: { ...original, origin } as Location,
  });
  try {
    fn();
  } finally {
    Object.defineProperty(window, "location", { writable: true, value: original });
  }
}

describe("Phase 6B completion-flow (CompletionBar + Dialog + store)", () => {
  beforeEach(() => {
    useSessionStore.setState({ activeSessionId: "sess-integration" } as never);
    useExecutionStore.setState({
      validationResult: _validValidation(),
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    useShareableReviewStore.getState().reset();
    resetStore(useInterpretationEventsStore);
    vi.restoreAllMocks();
  });

  it("end-to-end: click Save for review → dialog opens → success URL shown → copy works → close preserves response", async () => {
    const writeText = vi.fn().mockResolvedValueOnce(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    const apiResponse = {
      token: "tk-int",
      share_url: "/#/shared/tk-int",
      expires_at: "2026-06-19T00:00:00+00:00",
      payload_digest: "sha256:" + "11".repeat(32),
    };
    const apiSpy = vi
      .spyOn(api, "markReadyForReview")
      .mockResolvedValueOnce(apiResponse);

    _withOrigin("https://elspeth.example", () => {
      render(
        <>
          <CompletionBar />
          <SaveForReviewDialog />
        </>,
      );
    });

    // Before click: bar visible, dialog absent.
    expect(screen.getByTestId("completion-bar")).toBeInTheDocument();
    expect(screen.queryByTestId("save-for-review-dialog")).toBeNull();

    // Click Save for review.
    fireEvent.click(screen.getByTestId("completion-bar-save-for-review"));

    // Immediately: dialog mounted, spinner visible.
    expect(screen.getByTestId("save-for-review-dialog")).toBeInTheDocument();
    expect(screen.getByTestId("save-for-review-spinner")).toBeInTheDocument();

    // API was called with the active session id.
    expect(apiSpy).toHaveBeenCalledWith("sess-integration");

    // After the API resolves: success panel visible.
    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-success")).toBeInTheDocument(),
    );
    const urlInput = screen.getByTestId("save-for-review-url-input") as HTMLInputElement;
    expect(urlInput.value).toContain("/#/shared/tk-int");

    // Copy works (the urlInput's absolute URL is whatever the dialog
    // resolved at render time; assert clipboard received that value).
    fireEvent.click(screen.getByTestId("save-for-review-copy"));
    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toBe(urlInput.value);

    // Close button: dialog unmounts but store retains the response.
    fireEvent.click(screen.getByTestId("save-for-review-close"));
    await waitFor(() =>
      expect(screen.queryByTestId("save-for-review-dialog")).toBeNull(),
    );
    expect(useShareableReviewStore.getState().latestResponse?.token).toBe("tk-int");
  });

  it("end-to-end: 409 from backend renders the error panel with retry", async () => {
    vi.spyOn(api, "markReadyForReview").mockRejectedValueOnce({
      status: 409,
      detail: "composition validation failed",
    });

    render(
      <>
        <CompletionBar />
        <SaveForReviewDialog />
      </>,
    );

    fireEvent.click(screen.getByTestId("completion-bar-save-for-review"));

    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-error")).toBeInTheDocument(),
    );
    expect(screen.getByTestId("save-for-review-error")).toHaveTextContent(
      "composition validation failed",
    );
    expect(screen.getByTestId("save-for-review-retry")).toBeInTheDocument();
  });

  it("clicking Save for review is a no-op when validation is invalid", () => {
    useExecutionStore.setState({
      validationResult: { is_valid: false, checks: [], errors: [{ component_id: "n", component_type: "transform", message: "x", suggestion: null }] } as never,
      isExecuting: false,
      progress: null,
      execute: vi.fn(),
    } as never);
    const apiSpy = vi.spyOn(api, "markReadyForReview");

    render(
      <>
        <CompletionBar />
        <SaveForReviewDialog />
      </>,
    );

    // Button is disabled — fireEvent.click on a disabled button is a no-op.
    const btn = screen.getByTestId("completion-bar-save-for-review") as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    fireEvent.click(btn);
    expect(apiSpy).not.toHaveBeenCalled();
    expect(screen.queryByTestId("save-for-review-dialog")).toBeNull();
  });

  it("opening the dialog after a previous response from a different session clears the stale URL", async () => {
    // First, open dialog for session A with a successful response.
    const respA = {
      token: "tk-A",
      share_url: "/#/shared/tk-A",
      expires_at: "2026-06-19T00:00:00+00:00",
      payload_digest: "sha256:" + "aa".repeat(32),
    };
    vi.spyOn(api, "markReadyForReview").mockResolvedValueOnce(respA);

    render(
      <>
        <CompletionBar />
        <SaveForReviewDialog />
      </>,
    );
    fireEvent.click(screen.getByTestId("completion-bar-save-for-review"));
    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-success")).toBeInTheDocument(),
    );
    expect(useShareableReviewStore.getState().sessionIdForResponse).toBe("sess-integration");

    // Switch to session B and trigger again. The store should clear the stale
    // response for A so the user never sees A's URL in B's context.
    act(() => {
      useSessionStore.setState({ activeSessionId: "sess-B" } as never);
    });
    fireEvent.click(screen.getByTestId("completion-bar-save-for-review"));
    // While inFlight: success panel from session A should NOT be visible.
    expect(screen.queryByText("/#/shared/tk-A")).toBeNull();
  });
});

import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { SaveForReviewDialog } from "./SaveForReviewDialog";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";
import * as api from "@/api/shareableReviews";

expect.extend(toHaveNoViolations);

const _validResponse = {
  token: "abc",
  share_url: "/#/shared/abc",
  expires_at: "2026-06-19T00:00:00+00:00",
  payload_digest: "sha256:" + "ab".repeat(32),
};

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

describe("SaveForReviewDialog", () => {
  beforeEach(() => {
    useShareableReviewStore.getState().reset();
  });

  it("renders nothing when dialogOpen is false", () => {
    const { container } = render(<SaveForReviewDialog />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the spinner when inFlight is true", () => {
    useShareableReviewStore.setState({ dialogOpen: true, inFlight: true } as never);
    render(<SaveForReviewDialog />);
    expect(screen.getByTestId("save-for-review-spinner")).toBeInTheDocument();
  });

  it("renders as a modal dialog with an accessible name", () => {
    useShareableReviewStore.setState({ dialogOpen: true, inFlight: true } as never);
    render(<SaveForReviewDialog />);

    const dialog = screen.getByRole("dialog", { name: "Share for review" });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby", "save-for-review-dialog-title");
  });

  it("shows the error banner and retry button when error is set", () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      error: "composition validation failed",
    } as never);
    render(<SaveForReviewDialog />);
    expect(screen.getByTestId("save-for-review-error")).toHaveTextContent(
      "composition validation failed",
    );
    expect(screen.getByTestId("save-for-review-retry")).toBeInTheDocument();
  });

  it("retries a failed first mint attempt with the failed session id", async () => {
    const apiSpy = vi.spyOn(api, "markReadyForReview");
    apiSpy
      .mockRejectedValueOnce({
        status: 409,
        detail: "composition validation failed",
      })
      .mockResolvedValueOnce(_validResponse);

    await useShareableReviewStore.getState().openAndMark("sess-retry");
    render(<SaveForReviewDialog />);

    fireEvent.click(screen.getByTestId("save-for-review-retry"));

    await waitFor(() => expect(apiSpy).toHaveBeenCalledTimes(2));
    expect(apiSpy).toHaveBeenLastCalledWith("sess-retry");
    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-success")).toBeInTheDocument(),
    );
  });

  it("shows the share URL and prepends location.origin on success", () => {
    _withOrigin("https://elspeth.example", () => {
      useShareableReviewStore.setState({
        dialogOpen: true,
        latestResponse: _validResponse,
        sessionIdForResponse: "sess-1",
      } as never);
      render(<SaveForReviewDialog />);
      const input = screen.getByTestId("save-for-review-url-input") as HTMLInputElement;
      expect(input.value).toBe("https://elspeth.example/#/shared/abc");
    });
  });

  it("copy button writes the absolute URL to the clipboard and shows feedback", async () => {
    _withOrigin("https://elspeth.example", () => {
      useShareableReviewStore.setState({
        dialogOpen: true,
        latestResponse: _validResponse,
        sessionIdForResponse: "sess-1",
      } as never);
    });
    const writeText = vi.fn().mockResolvedValueOnce(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(<SaveForReviewDialog />);
    fireEvent.click(screen.getByTestId("save-for-review-copy"));
    await waitFor(() => expect(writeText).toHaveBeenCalled());
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("/#/shared/abc"),
    );
    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-copy")).toHaveTextContent(/copied/i),
    );
  });

  it("copy button surfaces failure when clipboard API is unavailable", async () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      latestResponse: _validResponse,
      sessionIdForResponse: "sess-1",
    } as never);
    const writeText = vi.fn().mockRejectedValueOnce(new Error("not allowed"));
    Object.assign(navigator, { clipboard: { writeText } });

    render(<SaveForReviewDialog />);
    fireEvent.click(screen.getByTestId("save-for-review-copy"));
    await waitFor(() =>
      expect(screen.getByTestId("save-for-review-copy")).toHaveTextContent(/copy failed/i),
    );
  });

  it("Close button dispatches close()", () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      latestResponse: _validResponse,
      sessionIdForResponse: "sess-1",
    } as never);
    render(<SaveForReviewDialog />);
    fireEvent.click(screen.getByTestId("save-for-review-close"));
    expect(useShareableReviewStore.getState().dialogOpen).toBe(false);
  });

  it("URL input is readonly and selects on focus", () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      latestResponse: _validResponse,
      sessionIdForResponse: "sess-1",
    } as never);
    render(<SaveForReviewDialog />);
    const input = screen.getByTestId("save-for-review-url-input") as HTMLInputElement;
    expect(input.readOnly).toBe(true);
  });

  // Plan line 276 (`docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`)
  // names jest-axe accessibility-clean assertion on SaveForReviewDialog as a
  // load-bearing GO condition for Phase 6B (a11y was a multi-reviewer GO
  // gate). The success state is the user-visible terminal state of the
  // happy path — the share URL display, the Copy button, and the Close
  // button must be accessible. axe-core inspects roles, labels, focus
  // management, and contrast-of-context against the rendered DOM.
  it("has no accessibility violations in the success state", async () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      latestResponse: _validResponse,
      sessionIdForResponse: "sess-1",
    } as never);
    const { container } = render(<SaveForReviewDialog />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  // Plan line 278 (`docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`)
  // requires Esc-to-close focus management on the modal. WAI-ARIA Authoring
  // Practices: any element with `role="dialog"` + `aria-modal="true"` must
  // close on Escape. The listener is attached to `document` rather than the
  // dialog element itself because the dialog does not auto-focus on open in
  // jsdom (and would not reliably receive keydown otherwise); document-level
  // keydown is the canonical pattern for modal Esc handling in React.
  it("closes the dialog when Escape is pressed", () => {
    useShareableReviewStore.setState({
      dialogOpen: true,
      latestResponse: _validResponse,
      sessionIdForResponse: "sess-1",
    } as never);
    render(<SaveForReviewDialog />);
    expect(screen.getByTestId("save-for-review-dialog")).toBeInTheDocument();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(useShareableReviewStore.getState().dialogOpen).toBe(false);
    expect(screen.queryByTestId("save-for-review-dialog")).not.toBeInTheDocument();
  });

  // Plan line 272 AC 6 (`docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`):
  // "Copied!" feedback after clicking copy auto-clears after 2000ms. The
  // timer is named COPY_FEEDBACK_TIMEOUT_MS in the component; vi fake
  // timers exercise the setTimeout branch end-to-end.
  it("auto-clears the 'Copied!' confirmation after 2000ms", async () => {
    // Fake setTimeout/clearTimeout only — leave the microtask queue
    // (Promise / queueMicrotask) on real timers so the awaited
    // navigator.clipboard.writeText promise can resolve. Then flush
    // microtasks via a real-timer `setImmediate`-style yield, assert
    // the "Copied!" state, advance the fake setTimeout by 2000ms inside
    // `act(...)`, and assert the auto-clear has fired.
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      useShareableReviewStore.setState({
        dialogOpen: true,
        latestResponse: _validResponse,
        sessionIdForResponse: "sess-1",
      } as never);
      const writeText = vi.fn().mockResolvedValueOnce(undefined);
      Object.assign(navigator, { clipboard: { writeText } });

      render(<SaveForReviewDialog />);
      fireEvent.click(screen.getByTestId("save-for-review-copy"));
      // Flush the awaited writeText promise + React state update.
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      expect(screen.getByTestId("save-for-review-copy")).toHaveTextContent(
        /copied/i,
      );
      act(() => {
        vi.advanceTimersByTime(2000);
      });
      expect(screen.getByTestId("save-for-review-copy")).toHaveTextContent(
        /^copy$/i,
      );
    } finally {
      vi.useRealTimers();
    }
  });

  // Plan line 273 AC 7 (`docs/composer/ux-redesign-2026-05/19b-phase-6b-frontend.md`):
  // Open-link affordance opens in a new tab with rel="noopener noreferrer"
  // — security hardening for the user-shared URL.
  it("renders an open-in-new-tab anchor with noopener/noreferrer and correct href", () => {
    _withOrigin("https://elspeth.example", () => {
      useShareableReviewStore.setState({
        dialogOpen: true,
        latestResponse: _validResponse,
        sessionIdForResponse: "sess-1",
      } as never);
      render(<SaveForReviewDialog />);
      const link = screen.getByTestId("save-for-review-open-link");
      expect(link.tagName).toBe("A");
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
      expect(link).toHaveAttribute(
        "href",
        "https://elspeth.example/#/shared/abc",
      );
    });
  });
});

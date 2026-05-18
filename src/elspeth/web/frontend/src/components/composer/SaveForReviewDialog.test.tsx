import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";
import { SaveForReviewDialog } from "./SaveForReviewDialog";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";

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
});

/**
 * SaveForReviewDialog — confirmation + share-URL display (Phase 6B Task 4).
 *
 * Mounted by App.tsx (top-level) and reads `dialogOpen`, `latestResponse`,
 * `inFlight`, and `error` from `useShareableReviewStore`. The dialog has
 * three observable states:
 *
 *   * inFlight=true  → "Minting share link..." spinner.
 *   * error set      → error banner with a "Try again" affordance.
 *   * latestResponse → share URL + expiry + copy-to-clipboard button.
 *
 * The share_url returned from the backend is path-only (`/#/shared/{token}`);
 * the dialog prepends `location.origin` for the copy-to-clipboard
 * affordance and the "Open in new tab" link so the user always gets an
 * absolute URL.
 *
 * Copy-to-clipboard uses `navigator.clipboard.writeText`. The plan calls
 * out that environments without Clipboard API support fall back to
 * selectable text — the input element used to display the URL IS
 * selectable, so users on locked-down browsers can still copy manually.
 */

import { useEffect, useRef, useState } from "react";

import { useFocusTrap } from "@/hooks/useFocusTrap";
import { useShareableReviewStore } from "@/stores/shareableReviewStore";

const COPY_FEEDBACK_TIMEOUT_MS = 2000;

/** Convert a backend path-only share_url to an absolute URL on the current
 *  origin. The frontend handles this rather than the backend so the same
 *  build can be deployed under any origin. */
function _toAbsoluteShareUrl(pathOnly: string): string {
  if (pathOnly.startsWith("http://") || pathOnly.startsWith("https://")) {
    return pathOnly;
  }
  return `${window.location.origin}${pathOnly}`;
}

export function SaveForReviewDialog(): JSX.Element | null {
  const dialogOpen = useShareableReviewStore((s) => s.dialogOpen);
  const latestResponse = useShareableReviewStore((s) => s.latestResponse);
  const inFlight = useShareableReviewStore((s) => s.inFlight);
  const error = useShareableReviewStore((s) => s.error);
  const close = useShareableReviewStore((s) => s.close);
  const openAndMark = useShareableReviewStore((s) => s.openAndMark);

  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const copyTimeoutRef = useRef<number | null>(null);
  const dialogRef = useRef<HTMLElement>(null);

  // Real modal behaviour consistent with GraphModal / ConfirmDialog: trap Tab
  // focus inside the dialog, move initial focus to the always-present Close
  // button, and restore focus to the opener on unmount. `dialogOpen` gates the
  // trap; the component unmounts (returns null) when it closes, so the trap's
  // restore-on-cleanup runs.
  useFocusTrap(dialogRef, dialogOpen, ".save-for-review-close");

  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current !== null) {
        window.clearTimeout(copyTimeoutRef.current);
      }
    };
  }, []);

  // WAI-ARIA Authoring Practices require role="dialog" + aria-modal="true"
  // modals to close on Escape. The listener is attached to `document` rather
  // than the dialog element — the canonical React modal pattern, matching
  // GraphModal / ConfirmDialog — so Escape closes regardless of which trapped
  // control currently holds focus. Effect re-runs when `dialogOpen` or `close`
  // change so the handler only listens while the dialog is open.
  useEffect(() => {
    if (!dialogOpen) return undefined;
    function _onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        close();
      }
    }
    document.addEventListener("keydown", _onKeyDown);
    return () => document.removeEventListener("keydown", _onKeyDown);
  }, [dialogOpen, close]);

  if (!dialogOpen) return null;

  const absoluteShareUrl = latestResponse !== null
    ? _toAbsoluteShareUrl(latestResponse.share_url)
    : null;

  async function _onCopy() {
    if (absoluteShareUrl === null) return;
    try {
      await navigator.clipboard.writeText(absoluteShareUrl);
      setCopyState("copied");
    } catch {
      // Clipboard API may be unavailable (HTTP / restrictive permissions).
      // The URL input is still selectable; user can copy manually.
      setCopyState("failed");
    }
    if (copyTimeoutRef.current !== null) {
      window.clearTimeout(copyTimeoutRef.current);
    }
    copyTimeoutRef.current = window.setTimeout(() => {
      setCopyState("idle");
    }, COPY_FEEDBACK_TIMEOUT_MS);
  }

  function _onRetry() {
    // The store has the session id captured from the original openAndMark
    // call. If we lost it (resetSession between attempts), the retry path
    // would no-op; the user closes and re-opens via the CompletionBar.
    const sessionId = useShareableReviewStore.getState().sessionIdForResponse;
    if (sessionId !== null) {
      void openAndMark(sessionId);
    }
  }

  return (
    <>
      <div
        className="save-for-review-dialog-backdrop"
        data-testid="save-for-review-dialog-backdrop"
        onClick={close}
        aria-hidden="true"
      />
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        className="save-for-review-dialog"
        aria-labelledby="save-for-review-dialog-title"
        data-testid="save-for-review-dialog"
      >
        <div className="save-for-review-dialog-content">
          <header>
            <h2 id="save-for-review-dialog-title">Share for review</h2>
          </header>

          {inFlight && (
            <div role="status" data-testid="save-for-review-spinner">
              Minting share link...
            </div>
          )}

          {!inFlight && error !== null && (
            <div role="alert" className="save-for-review-error" data-testid="save-for-review-error">
              <p>{error}</p>
              <button
                type="button"
                className="btn btn-compact"
                onClick={_onRetry}
                data-testid="save-for-review-retry"
              >
                Try again
              </button>
            </div>
          )}

          {!inFlight && error === null && latestResponse !== null && absoluteShareUrl !== null && (
            <div className="save-for-review-success" data-testid="save-for-review-success">
              <p>
                Share this link with your reviewer. It expires on{" "}
                <time dateTime={latestResponse.expires_at}>
                  {new Date(latestResponse.expires_at).toLocaleString()}
                </time>
                .
              </p>
              <div className="save-for-review-url-row">
                <label htmlFor="save-for-review-url">Share URL</label>
                <input
                  id="save-for-review-url"
                  type="text"
                  readOnly
                  value={absoluteShareUrl}
                  onFocus={(e) => e.currentTarget.select()}
                  data-testid="save-for-review-url-input"
                />
                <button
                  type="button"
                  className="btn btn-compact"
                  onClick={() => void _onCopy()}
                  data-testid="save-for-review-copy"
                  aria-label="Copy share URL to clipboard"
                >
                  {copyState === "idle" && "Copy"}
                  {copyState === "copied" && "Copied!"}
                  {copyState === "failed" && "Copy failed — select & copy manually"}
                </button>
                <a
                  href={absoluteShareUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  data-testid="save-for-review-open-link"
                >
                  Open in new tab
                </a>
              </div>
              <p className="save-for-review-tip">
                The link grants a different authenticated user read-only access
                to a frozen snapshot of this pipeline. Recipients must have an
                account on this deployment.
              </p>
            </div>
          )}

          <footer>
            <button
              type="button"
              className="btn save-for-review-close"
              onClick={close}
              data-testid="save-for-review-close"
            >
              Close
            </button>
          </footer>
        </div>
      </section>
    </>
  );
}

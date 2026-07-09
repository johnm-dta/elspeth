// src/components/chat/guided/GuidedPendingStrip.tsx

import { useEffect, useState, type RefObject } from "react";

import type { ComposerProgressSnapshot } from "@/types/api";
import { ElapsedReadout } from "../ComposingIndicator";

/**
 * Pointer-arming delay for the Stop button. A mouse double-click on Send is
 * common muscle memory; the strip mounts in the space Send just vacated, so
 * without this the second click of a double-click aborts the request the
 * user just started. CSS `pointer-events: none` for this window — keyboard
 * access is deliberately unaffected (focus never lands on Stop; see the
 * focus contract in ChatPanel).
 */
export const STOP_ARMING_DELAY_MS = 300;

interface GuidedPendingStripProps {
  /** Live compose progress; drives the adaptive headline when backend-sourced. */
  composerProgress: ComposerProgressSnapshot | null;
  /**
   * Abort the in-flight chat. Absent → no Stop rendered (a pending state
   * with nothing abortable must not offer a dead interrupt —
   * elspeth-fb4464cdf0).
   */
  onStop?: () => void;
  /**
   * Focus target for ChatPanel's pending-swap focus contract: the wrapper is
   * tabIndex=-1 (programmatic-only) and deliberately NOT the Stop button —
   * a habitual double-Enter after Send must land on a non-activatable node.
   */
  stripRef?: RefObject<HTMLDivElement>;
  substeps?: readonly string[];
  activeSubstepIndex?: number;
}

/**
 * The backend headline is only trustworthy while a compose is actually in
 * flight; an idle or terminal snapshot is stale carry-over from a previous
 * turn (the progress poller lags the pending flags). Everything else — the
 * keyword-guessed "(estimated)" evidence the full ComposingIndicator shows —
 * is deliberately cut here: low-confidence filler doesn't earn the composer
 * slot (UX-critic spec 2026-07-03). The rich card remains in freeform.
 */
function pendingHeadline(progress: ComposerProgressSnapshot | null): string {
  if (
    progress !== null &&
    progress.phase !== "idle" &&
    progress.phase !== "complete" &&
    progress.phase !== "failed" &&
    progress.phase !== "cancelled"
  ) {
    return progress.headline;
  }
  return "Working on it...";
}

/**
 * Lean single-row working strip that REPLACES the guided ChatInput while a
 * /guided/chat request is in flight (elspeth-6a9673ecd3). The swap — rather
 * than a disabled input — is the honest state: the old arrangement kept a
 * fully-alive-looking textarea with a quietly dead Send and grew the dock
 * with a detached status card below (clipped at wide viewports).
 *
 * Structure contract:
 * - role="status" wraps ONLY the announcement content (dots + headline +
 *   elapsed), never the Stop button — a control inside a live region gets
 *   re-announced on unrelated content mutations.
 * - The elapsed readout stays aria-hidden inside the status region (its
 *   once-per-second tick would spam AT; the headline is the AT signal).
 * - Height ≈ the idle input strip so the conversation column above barely
 *   reflows on the swap.
 */
export function GuidedPendingStrip({
  composerProgress,
  onStop,
  stripRef,
  substeps,
  activeSubstepIndex = 0,
}: GuidedPendingStripProps): JSX.Element {
  const [stopArmed, setStopArmed] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setStopArmed(true), STOP_ARMING_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div ref={stripRef} tabIndex={-1} className="guided-pending-strip">
      <div role="status" className="guided-pending-strip-status">
        <span className="composing-pulse" aria-hidden="true">
          <span className="composing-dot" />
          <span className="composing-dot" />
          <span className="composing-dot" />
        </span>
        <span className="guided-pending-strip-headline">
          {pendingHeadline(composerProgress)}
        </span>
        {/* Mount lifecycle doubles as the reset: the strip only exists while
            pending, so each compose counts from 00:00. */}
        <ElapsedReadout />
      </div>
      {substeps !== undefined && substeps.length > 0 && (
        <ol className="guided-pending-substeps" aria-label="Tutorial step progress">
          {substeps.map((substep, index) => (
            <li
              key={substep}
              className={
                index === activeSubstepIndex
                  ? "guided-pending-substep guided-pending-substep--current"
                  : "guided-pending-substep"
              }
              aria-current={index === activeSubstepIndex ? "step" : undefined}
            >
              {substep}
            </li>
          ))}
        </ol>
      )}
      {onStop !== undefined && (
        <button
          type="button"
          onClick={onStop}
          aria-label="Stop composing"
          className={`chat-input-cancel-btn guided-pending-strip-stop${
            stopArmed ? "" : " guided-pending-strip-stop--arming"
          }`}
        >
          Stop
        </button>
      )}
    </div>
  );
}

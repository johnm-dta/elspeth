// src/components/chat/guided/GuidedPendingStrip.tsx

import { useEffect, useState } from "react";

import type { ComposerProgressSnapshot } from "@/types/api";
import { ElapsedReadout } from "../ComposingIndicator";

/**
 * Pointer-arming delay for the Stop button. A mouse double-click on Send is
 * common muscle memory; guarding the first ~300ms after mount means a stray
 * second click can never abort the request the user just started. With the
 * strip riding in the conversation flow (not mounting where Send sat, as the
 * retired composer swap did) the window is belt-and-braces, but it stays:
 * the cost is nil and layout shifts can still move Stop under a pointer.
 * CSS `pointer-events: none` for this window — keyboard access is
 * deliberately unaffected (focus never lands on Stop; see the focus
 * contract in ChatPanel).
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
 * Lean single-row working strip shown while a /guided/chat request is in
 * flight. Placement (operator pass 2026-07-23): it rides IN the conversation
 * flow — the bottom of the guided scroll column, the provisional reply slot,
 * matching the decision-submit indicator — and never replaces or overlays
 * the message input (the retired elspeth-6a9673ecd3 swap read as a panel
 * occluding the typing area; the input now keeps its place with its
 * ordinary disabled state).
 *
 * Structure contract:
 * - role="status" wraps ONLY the announcement content (dots + headline +
 *   elapsed), never the Stop button — a control inside a live region gets
 *   re-announced on unrelated content mutations. The mount must stay
 *   OUTSIDE every role="log" container (elspeth-76a0cc485e).
 * - The elapsed readout stays aria-hidden inside the status region (its
 *   once-per-second tick would spam AT; the headline is the AT signal).
 */
export function GuidedPendingStrip({
  composerProgress,
  onStop,
  substeps,
  activeSubstepIndex = 0,
}: GuidedPendingStripProps): JSX.Element {
  const [stopArmed, setStopArmed] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => setStopArmed(true), STOP_ARMING_DELAY_MS);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="guided-pending-strip">
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

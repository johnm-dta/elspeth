// src/components/chat/guided/GuidedDecisionPendingIndicator.tsx

import type { ComposerProgressSnapshot } from "@/types/api";
import { ElapsedReadout } from "../ComposingIndicator";

/**
 * The live headline is only trustworthy while a compose is actually in
 * flight; an idle or terminal snapshot is stale carry-over from a previous
 * turn. Mirrors GuidedPendingStrip's pendingHeadline guard.
 *
 * NOTE: today the guided /respond route does not publish composer-progress
 * snapshots (no progress sink is wired into its planner calls), so a decision
 * submit normally resolves to the fallback; the guided /chat route does
 * publish. Rendering through this guard means the phase text lights up
 * automatically wherever the backend publishes, without any client change.
 */
export function decisionPendingHeadline(
  progress: ComposerProgressSnapshot | null | undefined,
  fallback: string,
): string {
  if (
    progress != null &&
    progress.phase !== "idle" &&
    progress.phase !== "complete" &&
    progress.phase !== "failed" &&
    progress.phase !== "cancelled"
  ) {
    return progress.headline;
  }
  return fallback;
}

interface GuidedDecisionPendingIndicatorProps {
  /** Static copy shown when no live non-terminal progress snapshot exists. */
  fallback: string;
  composerProgress?: ComposerProgressSnapshot | null;
}

/**
 * Shared content for a pending guided decision submit (wire confirm /
 * correction, proposal decision): pulse dots + adaptive headline + elapsed
 * readout — the chat Send pending idiom (GuidedPendingStrip) reused inside
 * the caller's own role="status" element. Rendered as a fragment so callers
 * keep their existing live-region wrapper (role, ref, focus semantics).
 *
 * The elapsed readout stays aria-hidden inside the caller's status region
 * (its once-per-second tick would spam AT; the headline is the AT signal) —
 * same contract as GuidedPendingStrip / ComposingIndicator. Mount lifecycle
 * doubles as the reset: the indicator exists only while the submit is
 * pending, so each decision counts from 00:00.
 */
export function GuidedDecisionPendingIndicator({
  fallback,
  composerProgress = null,
}: GuidedDecisionPendingIndicatorProps): JSX.Element {
  return (
    <>
      <span className="composing-pulse" aria-hidden="true">
        <span className="composing-dot" />
        <span className="composing-dot" />
        <span className="composing-dot" />
      </span>
      <span className="guided-decision-pending-headline">
        {decisionPendingHeadline(composerProgress, fallback)}
      </span>
      <ElapsedReadout />
    </>
  );
}

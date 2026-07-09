// src/components/chat/guided/guidedBuildActive.ts
//
// The ONE definition of "a guided build is on screen". Two components must
// agree on it exactly:
//
//   - ChatPanel's guided-active branch renders the two-column workspace
//     (conversation column + pipeline rail) when this holds;
//   - App suppresses the freeform SideRail while it holds — the workspace
//     rail replaces it, and rendering both puts two rails side by side.
//
// If the two ever computed this independently they would drift, and a drift
// shows as either a double rail or a freeform surface with no rail at all.
// ChatPanel re-states the null/terminal checks inline purely for TypeScript
// narrowing; the truth lives here.

import type { GuidedSession, TurnPayload } from "@/types/guided";

/**
 * True while an ACTIVE guided build should occupy the panel.
 *
 * Mirrors ChatPanel's branch precedence:
 *   - a terminal session (completed / exited_to_freeform) is NOT active —
 *     those render the CompletionSummary surface or fall through to
 *     freeform, and the SideRail must return for both (Run / Export live
 *     there post-completion);
 *   - a non-terminal session is active when the server has posed a turn, OR
 *     at step_3_transforms with no turn yet (the per-stage transforms prompt
 *     drives a cold start through /guided/chat — the composer must render so
 *     the operator can describe the transforms);
 *   - any other turn-less non-terminal state is a transient (e.g. the gap
 *     during startGuided) and falls through to the freeform surface, rail
 *     included.
 */
export function isGuidedBuildActive(
  guidedSession: GuidedSession | null,
  guidedNextTurn: TurnPayload | null,
): boolean {
  return (
    guidedSession != null &&
    guidedSession.terminal == null &&
    (guidedNextTurn != null || guidedSession.step === "step_3_transforms")
  );
}

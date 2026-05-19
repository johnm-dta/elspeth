import { useState } from "react";

import type { CompositionProposal } from "@/types/api";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";

interface PendingProposalsBannerProps {
  proposals: CompositionProposal[];
  staleProposalIds: string[];
  proposalActionPendingIds: string[];
  onAccept: (proposalId: string) => void;
  onReject: (proposalId: string) => void;
}

/**
 * Persistent banner above the chat input that surfaces pending composer
 * proposals with inline Accept/Reject controls.
 *
 * Pending proposals are also rendered inside ToolCallCard on the originating
 * assistant message, but that surface is buried inside the message's
 * collapsible "Tool calls (N)" panel and the user must scroll up from the
 * agent's most recent prose to reach it. The banner keeps the action visible
 * regardless of scroll position, which matches the user's mental model: when
 * the agent says "this change needs approval", the button to approve it is
 * right there.
 *
 * Stale proposals (base_state_id no longer matches the current committed
 * state) are excluded — the user cannot accept them, and the ToolCallCard
 * surface already shows the rebase prompt for them.
 */
export function PendingProposalsBanner({
  proposals,
  staleProposalIds,
  proposalActionPendingIds,
  onAccept,
  onReject,
}: PendingProposalsBannerProps) {
  const [rejectConfirmId, setRejectConfirmId] = useState<string | null>(null);
  const actionable = proposals.filter(
    (p) => p.status === "pending" && !staleProposalIds.includes(p.id),
  );
  if (actionable.length === 0) {
    return null;
  }
  const rejectTarget =
    rejectConfirmId !== null
      ? actionable.find((p) => p.id === rejectConfirmId)
      : undefined;

  return (
    <section
      className="pending-proposals-banner"
      aria-label={`Pending changes (${actionable.length})`}
    >
      <header className="pending-proposals-banner-header">
        <strong>
          Pending change{actionable.length === 1 ? "" : "s"} (
          {actionable.length})
        </strong>
        <span className="pending-proposals-banner-help">
          The composer prepared {actionable.length === 1 ? "a change" : "changes"}{" "}
          but needs your approval before applying.
        </span>
      </header>
      <ul className="pending-proposals-banner-list">
        {actionable.map((proposal) => {
          const isBusy = proposalActionPendingIds.includes(proposal.id);
          return (
            <li
              key={proposal.id}
              className="pending-proposals-banner-item"
            >
              <div className="pending-proposals-banner-item-body">
                <p className="pending-proposals-banner-summary">
                  {proposal.summary}
                </p>
                {proposal.affects.length > 0 && (
                  <p className="pending-proposals-banner-affects">
                    Affects: {proposal.affects.join(", ")}
                  </p>
                )}
              </div>
              <div className="pending-proposals-banner-actions">
                <button
                  type="button"
                  onClick={() => onAccept(proposal.id)}
                  aria-label={`Accept proposal: ${proposal.summary}`}
                  disabled={isBusy}
                  className="btn btn-primary"
                >
                  Accept
                </button>
                <button
                  type="button"
                  onClick={() => setRejectConfirmId(proposal.id)}
                  aria-label={`Reject proposal: ${proposal.summary}`}
                  disabled={isBusy}
                  className="btn btn-danger"
                >
                  Reject
                </button>
              </div>
            </li>
          );
        })}
      </ul>
      {rejectTarget && (
        <ConfirmDialog
          title="Reject this proposal?"
          message="The composer's proposed change will be discarded. You can ask the composer to revise the proposal afterwards."
          confirmLabel="Reject proposal"
          cancelLabel="Keep open"
          variant="danger"
          onConfirm={() => {
            onReject(rejectTarget.id);
            setRejectConfirmId(null);
          }}
          onCancel={() => setRejectConfirmId(null)}
        />
      )}
    </section>
  );
}

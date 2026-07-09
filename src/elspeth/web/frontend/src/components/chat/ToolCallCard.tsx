import { useMemo, useState } from "react";

import type { CompositionProposal, CompositionState, ToolCall } from "@/types/api";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { ArgumentFields, buildProposalDiff, ProposalChanges } from "./ProposalDiff";
import { describeToolCall } from "./toolCallDescriptions";

/**
 * A button-triggered tooltip explaining what a composer tool call does in
 * general — not what it did to the current pipeline. Hover or keyboard-focus
 * the "i" button to reveal. Uses CSS-only show/hide; no JS state.
 */
function ToolCallInfo({
  toolName,
  describedById,
}: {
  toolName: string;
  describedById: string;
}) {
  return (
    <span className="tool-call-info">
      <button
        type="button"
        className="tool-call-info-trigger"
        aria-label={`What does ${toolName} do?`}
        aria-describedby={describedById}
      >
        i
      </button>
      <span
        id={describedById}
        role="tooltip"
        className="tool-call-info-bubble"
      >
        <strong className="tool-call-info-bubble-name">{toolName}</strong>
        <span className="tool-call-info-bubble-body">
          {describeToolCall(toolName)}
        </span>
      </span>
    </span>
  );
}

interface ToolCallCardProps {
  toolCall: ToolCall;
  proposal: CompositionProposal | null;
  /**
   * Current composition state — the "before" side of the proposal diff
   * (elspeth-10f76f9250). Only consulted for pending, non-stale proposals;
   * for anything else the current state is no longer the state the proposal
   * targeted and the card falls back to structured argument fields.
   */
  currentState?: CompositionState | null;
  isStale?: boolean;
  isBusy?: boolean;
  onAccept: (proposalId: string) => void;
  onReject: (proposalId: string) => void;
}

export function ToolCallCard({
  toolCall,
  proposal,
  currentState = null,
  isStale = false,
  isBusy = false,
  onAccept,
  onReject,
}: ToolCallCardProps) {
  const [rejectConfirmOpen, setRejectConfirmOpen] = useState(false);
  // Fragment-level before/after projection of the proposal against the
  // current pipeline. null = not derivable (unknown tool, malformed args, no
  // state) → structured argument fields render instead. Computed before the
  // no-proposal early return to keep hooks unconditional.
  const diffEntries = useMemo(() => {
    if (proposal === null || proposal.status !== "pending" || isStale) {
      return null;
    }
    return buildProposalDiff(
      proposal.tool_name,
      proposal.arguments_redacted_json,
      currentState,
    );
  }, [proposal, isStale, currentState]);
  if (!proposal) {
    return (
      <div className="tool-call-ribbon">
        <ToolCallInfo
          toolName={toolCall.function.name}
          describedById={`tool-call-info-${toolCall.id}`}
        />
        <span>Looked up: {toolCall.function.name}</span>
      </div>
    );
  }

  const isPending = proposal.status === "pending";
  const heading =
    proposal.status === "pending"
      ? `Proposed: ${proposal.tool_name}`
      : proposal.status === "committed"
        ? `Applied: ${proposal.tool_name}`
        : `Rejected: ${proposal.tool_name}`;

  return (
    <article className={`tool-call-card tool-call-card--${proposal.status}`}>
      <header className="tool-call-card-header">
        <strong>{heading}</strong>
        <ToolCallInfo
          toolName={proposal.tool_name}
          describedById={`tool-call-info-${proposal.id}`}
        />
        {proposal.audit_event_id && (
          <code className="tool-call-audit-id">
            audit {proposal.audit_event_id.slice(0, 8)}
          </code>
        )}
      </header>
      <p className="tool-call-summary">{proposal.summary}</p>
      <p className="tool-call-rationale">
        <strong>Why:</strong> {proposal.rationale}
      </p>
      {proposal.affects.length > 0 && (
        <p className="tool-call-affects">
          <strong>Affects:</strong> {proposal.affects.join(", ")}
        </p>
      )}
      {/* Primary change surface (elspeth-10f76f9250): a before/after diff of
          the affected pipeline fragment when derivable, otherwise structured
          per-argument fields. The raw JSON stays available behind the
          details expander below in both cases. */}
      {diffEntries !== null ? (
        <ProposalChanges entries={diffEntries} />
      ) : (
        <ArgumentFields args={proposal.arguments_redacted_json} />
      )}
      <details className="tool-call-details">
        <summary>View arguments (JSON)</summary>
        <pre
          tabIndex={0}
          aria-label="Tool call arguments (scrollable)"
        >{JSON.stringify(proposal.arguments_redacted_json, null, 2)}</pre>
      </details>
      {isStale && (
        <p className="tool-call-stale">
          Stale proposal. Ask the composer to rebase or revise this proposal.
        </p>
      )}
      {isPending && !isStale && (
        <div className="tool-call-actions">
          <button
            type="button"
            onClick={() => onAccept(proposal.id)}
            aria-label={`Accept proposal: ${proposal.summary}`}
            disabled={isBusy}
            className="btn btn-primary btn-small"
          >
            Accept
          </button>
          <button
            type="button"
            onClick={() => setRejectConfirmOpen(true)}
            aria-label={`Reject proposal: ${proposal.summary}`}
            disabled={isBusy}
            className="btn btn-danger btn-small"
          >
            Reject
          </button>
        </div>
      )}
      {rejectConfirmOpen && proposal && (
        <ConfirmDialog
          title="Reject this proposal?"
          message="The composer's proposed change will be discarded. You can ask the composer to revise the proposal afterwards."
          confirmLabel="Reject proposal"
          cancelLabel="Keep open"
          variant="danger"
          onConfirm={() => {
            onReject(proposal.id);
            setRejectConfirmOpen(false);
          }}
          onCancel={() => setRejectConfirmOpen(false)}
        />
      )}
    </article>
  );
}

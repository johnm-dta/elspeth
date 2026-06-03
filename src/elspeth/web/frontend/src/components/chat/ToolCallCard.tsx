import { useState } from "react";

import type { CompositionProposal, ToolCall } from "@/types/api";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
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
  isStale?: boolean;
  isBusy?: boolean;
  onAccept: (proposalId: string) => void;
  onReject: (proposalId: string) => void;
}

export function ToolCallCard({
  toolCall,
  proposal,
  isStale = false,
  isBusy = false,
  onAccept,
  onReject,
}: ToolCallCardProps) {
  const [rejectConfirmOpen, setRejectConfirmOpen] = useState(false);
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

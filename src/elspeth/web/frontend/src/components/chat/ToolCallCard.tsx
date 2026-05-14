import type { CompositionProposal, ToolCall } from "@/types/api";

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
  if (!proposal) {
    return (
      <div className="tool-call-ribbon">
        <span aria-hidden="true" className="tool-call-ribbon-icon">
          i
        </span>
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
        <pre>{JSON.stringify(proposal.arguments_redacted_json, null, 2)}</pre>
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
            onClick={() => onReject(proposal.id)}
            aria-label={`Reject proposal: ${proposal.summary}`}
            disabled={isBusy}
            className="btn btn-danger btn-small"
          >
            Reject
          </button>
        </div>
      )}
    </article>
  );
}

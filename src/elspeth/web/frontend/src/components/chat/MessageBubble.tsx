// src/components/chat/MessageBubble.tsx
import { useState, useCallback, useRef, useEffect } from "react";
import type { ChatMessage, CompositionProposal, InlineSourceSummary } from "@/types/api";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { ToolCallCard } from "./ToolCallCard";
import { InlineSourceCreatedTurn } from "./InlineSourceCreatedTurn";

interface MessageBubbleProps {
  message: ChatMessage;
  isComposing?: boolean;
  onRetry?: (messageId: string) => void;
  onFork?: (messageId: string, newContent: string) => void;
  proposalsByToolCallId?: Map<string, CompositionProposal>;
  staleProposalIds?: string[];
  proposalActionPendingIds?: string[];
  onAcceptProposal?: (proposalId: string) => void;
  onRejectProposal?: (proposalId: string) => void;
  /**
   * Inline source summaries attached to this turn — rendered as a second
   * collapsible group below the tool-calls group, separated by a horizontal
   * ruler. The bubble is the natural home for these because they represent
   * something the agent did *as part of this turn* (created dynamic sources
   * from the user's message). The store currently holds at most one summary
   * per session, but the prop is a list so multiple-source turns work without
   * a future refactor here.
   */
  sourcesCreated?: ReadonlyArray<InlineSourceSummary>;
  onEditInlineSource?: (summary: InlineSourceSummary) => void;
}

export function MessageBubble({
  message,
  isComposing,
  onRetry,
  onFork,
  proposalsByToolCallId,
  staleProposalIds = [],
  proposalActionPendingIds = [],
  onAcceptProposal = () => undefined,
  onRejectProposal = () => undefined,
  sourcesCreated,
  onEditInlineSource,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const [toolsExpanded, setToolsExpanded] = useState(false);
  const hasToolCalls = !!(message.tool_calls && message.tool_calls.length > 0);
  const hasSourcesCreated = !!(sourcesCreated && sourcesCreated.length > 0);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const editRef = useRef<HTMLTextAreaElement>(null);
  const hasProposalToolCall =
    message.tool_calls?.some(
      (tc) => tc.id && proposalsByToolCallId?.has(tc.id),
    ) ?? false;
  const showToolCalls = toolsExpanded || hasProposalToolCall;

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API may fail in insecure contexts
    }
  }, [message.content]);

  const handleEditStart = useCallback(() => {
    setEditContent(message.content);
    setIsEditing(true);
  }, [message.content]);

  const handleEditCancel = useCallback(() => {
    setIsEditing(false);
    setEditContent(message.content);
  }, [message.content]);

  const handleForkSubmit = useCallback(() => {
    if (onFork && editContent.trim()) {
      onFork(message.id, editContent.trim());
      setIsEditing(false);
    }
  }, [onFork, message.id, editContent]);

  useEffect(() => {
    if (isEditing && editRef.current) {
      editRef.current.focus();
      editRef.current.setSelectionRange(
        editRef.current.value.length,
        editRef.current.value.length,
      );
    }
  }, [isEditing]);

  // System messages: centre-aligned full-width banner, muted colour,
  // italic text, no sender label. Used for audit markers like
  // "Pipeline reverted to version N."
  if (isSystem) {
    return (
      <div
        className="message-bubble message-bubble--system message-row message-row--system"
      >
        <div
          className="bubble bubble-system"
          role="status"
        >
          <MarkdownRenderer content={message.content} />
        </div>
      </div>
    );
  }

  return (
    <div
      className={`message-bubble message-bubble--${message.role} message-row ${isUser ? "message-row--user" : "message-row--assistant"}`}
    >
      <div
        className={`bubble ${isUser ? "bubble-user" : "bubble-assistant"} message-bubble-content${isUser ? " message-bubble-content--user" : ""}`}
      >
        {/* Copy button — visible on hover via CSS, always accessible on touch */}
        {!isSystem && (
          <button
            onClick={handleCopy}
            aria-label={copied ? "Copied to clipboard" : "Copy message"}
            className="bubble-copy-btn bubble-action-overlay bubble-action-overlay--copy"
            style={{
              opacity: copied ? 1 : undefined,
            }}
          >
            {copied ? "Copied!" : "\u2398"}
          </button>
        )}

        {isUser && isEditing ? (
          <div className="message-edit-form">
            <textarea
              ref={editRef}
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                  handleForkSubmit();
                } else if (e.key === "Escape") {
                  handleEditCancel();
                }
              }}
              aria-label="Edit message"
              className="message-edit-textarea"
            />
            <div className="message-edit-actions">
              <button
                onClick={handleEditCancel}
                className="message-edit-cancel"
              >
                Cancel
              </button>
              <button
                onClick={handleForkSubmit}
                disabled={!editContent.trim()}
                className="message-edit-fork"
              >
                Fork
              </button>
            </div>
          </div>
        ) : isUser ? (
          message.content
        ) : (
          <MarkdownRenderer content={message.content} />
        )}

        {/* Edit/fork button — user messages only, not pending/failed */}
        {isUser && !isEditing && !message.local_status && onFork && (
          <button
            onClick={handleEditStart}
            aria-label="Edit and fork from this message"
            className="bubble-edit-btn bubble-action-overlay bubble-action-overlay--edit"
          >
            &#9998;
          </button>
        )}

        {isUser && message.local_status === "failed" && onRetry && (
          <div className="message-failed-row">
            <span className="message-failed-text">
              {message.local_error ?? "Failed to send message. Please try again."}
            </span>
            <button
              onClick={() => onRetry(message.id)}
              className="message-retry-btn"
            >
              Retry
            </button>
          </div>
        )}

        {isUser && message.local_status === "pending" && !isComposing && (
          <div className="message-pending">
            Sending...
          </div>
        )}

        {/* Tool calls section (assistant messages only) */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="message-tools">
            <button
              onClick={() => setToolsExpanded(!toolsExpanded)}
              aria-expanded={showToolCalls}
              aria-label={`Tool calls (${message.tool_calls.length})`}
              className="message-tools-toggle"
            >
              {showToolCalls ? "\u25BC" : "\u25B6"} Tool calls (
              {message.tool_calls.length})
            </button>
            {showToolCalls && (
              <div className="message-tools-list">
                {message.tool_calls.map((tc, i) => (
                  <ToolCallCard
                    key={tc.id ?? i}
                    toolCall={tc}
                    proposal={
                      tc.id
                        ? proposalsByToolCallId?.get(tc.id) ?? null
                        : null
                    }
                    isStale={
                      tc.id
                        ? staleProposalIds.includes(
                            proposalsByToolCallId?.get(tc.id)?.id ?? "",
                          )
                        : false
                    }
                    isBusy={
                      tc.id
                        ? proposalActionPendingIds.includes(
                            proposalsByToolCallId?.get(tc.id)?.id ?? "",
                          )
                        : false
                    }
                    onAccept={onAcceptProposal}
                    onReject={onRejectProposal}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Horizontal ruler between Tool calls and Sources created — only
            rendered when both groups exist. Without this guard the bubble
            would carry a stray separator when there are zero tool calls but
            one source-created event (e.g. a hello-world first message that
            creates a dynamic source without invoking any tools). */}
        {hasToolCalls && hasSourcesCreated && (
          <hr className="message-group-separator" aria-hidden="true" />
        )}

        {/* Sources created section (assistant messages only).
            Deliberately NOT a collapsible disclosure (unlike Tool calls
            above). Source creation is a notification of an action that
            just got attached to the composition — the user needs to see
            it to decide whether to amend or proceed. Burying it behind
            a twisty would defer an actionable moment behind a click,
            which is the opposite of "hey, this happened, you need to
            know". The visual heading uses the same styling as the tool-
            calls toggle (.message-tools-toggle) so the two groups still
            read as siblings in the bubble, but the heading is a static
            <div> rather than a button — no aria-expanded, nothing to
            toggle. The inner InlineSourceCreatedTurn widget still has
            its own audit-info <details> disclosure for the SHA-256 hash;
            that nested twisty shows the cryptographic detail on demand
            without hiding the notification itself. */}
        {hasSourcesCreated && (
          <div className="message-sources-created">
            <div className="message-tools-toggle message-sources-created-heading">
              Sources ({sourcesCreated!.length})
            </div>
            <div className="message-sources-created-list">
              {sourcesCreated!.map((summary) => (
                <InlineSourceCreatedTurn
                  key={summary.blobId}
                  summary={summary}
                  onEdit={onEditInlineSource ?? (() => undefined)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

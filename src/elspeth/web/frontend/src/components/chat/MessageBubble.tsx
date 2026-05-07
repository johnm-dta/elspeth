// src/components/chat/MessageBubble.tsx
import { useState, useCallback, useRef, useEffect } from "react";
import type { ChatMessage } from "@/types/api";
import { MarkdownRenderer } from "./MarkdownRenderer";

interface MessageBubbleProps {
  message: ChatMessage;
  isComposing?: boolean;
  onRetry?: (messageId: string) => void;
  onFork?: (messageId: string, newContent: string) => void;
}

export function MessageBubble({ message, isComposing, onRetry, onFork }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";
  const [toolsExpanded, setToolsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const editRef = useRef<HTMLTextAreaElement>(null);

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
              aria-expanded={toolsExpanded}
              aria-label={`Tool calls (${message.tool_calls.length})`}
              className="message-tools-toggle"
            >
              {toolsExpanded ? "\u25BC" : "\u25B6"} Tool calls (
              {message.tool_calls.length})
            </button>
            {toolsExpanded && (
              <ul className="message-tools-list">
                {message.tool_calls.map((tc, i) => (
                  <li
                    key={tc.id ?? i}
                    className="message-tools-item"
                  >
                    <strong>{tc.function.name}</strong>
                    {tc.function.arguments && (
                      <details className="message-tools-details">
                        <summary className="message-tools-summary">
                          Arguments
                        </summary>
                        <pre className="message-tools-pre">
                          {tc.function.arguments}
                        </pre>
                      </details>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

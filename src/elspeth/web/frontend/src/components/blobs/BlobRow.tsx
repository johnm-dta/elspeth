// src/components/blobs/BlobRow.tsx
import { useState } from "react";
import { previewBlobContent } from "@/api/client";
import type { BlobMetadata } from "@/types/api";

const PREVIEWABLE_MIME_TYPES = new Set([
  "text/plain",
  "text/csv",
  "application/json",
  "application/x-jsonlines",
]);

const MAX_PREVIEW_CHARS = 5000;

interface BlobRowProps {
  blob: BlobMetadata;
  sessionId: string;
  onDownload: (blobId: string) => void;
  onDelete: (blobId: string) => void;
  onUseAsInput: (blob: BlobMetadata) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function creatorBadge(createdBy: string): string {
  switch (createdBy) {
    case "user":
      return "\u{1F4E4}";
    case "assistant":
      return "\u{1F916}";
    case "pipeline":
      return "\u{2699}\u{FE0F}";
    default:
      return "";
  }
}

function statusIndicator(status: string): { color: string; label: string } {
  switch (status) {
    case "ready":
      return { color: "var(--color-success)", label: "Ready" };
    case "pending":
      return { color: "var(--color-warning)", label: "Pending" };
    case "error":
      return { color: "var(--color-error)", label: "Error" };
    default:
      return { color: "var(--color-text-muted)", label: status };
  }
}

export function BlobRow({ blob, sessionId, onDownload, onDelete, onUseAsInput }: BlobRowProps) {
  const status = statusIndicator(blob.status);
  const canPreview = PREVIEWABLE_MIME_TYPES.has(blob.mime_type);

  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  const handleTogglePreview = async () => {
    if (previewOpen) {
      setPreviewOpen(false);
      return;
    }

    setPreviewOpen(true);

    // Only fetch if we haven't cached the content yet
    if (previewContent !== null) return;

    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const text = await previewBlobContent(sessionId, blob.id);
      setPreviewContent(text);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load preview";
      setPreviewError(message);
    } finally {
      setPreviewLoading(false);
    }
  };

  const truncated = previewContent !== null && previewContent.length > MAX_PREVIEW_CHARS;
  const displayContent = truncated
    ? previewContent.slice(0, MAX_PREVIEW_CHARS)
    : previewContent;

  return (
    <div>
      <div
        className="blob-row blob-row-container"
        style={{
          borderBottom: previewOpen ? "none" : "1px solid var(--color-border)",
        }}
      >
        {/* Status dot */}
        <span
          className="blob-row-status-dot"
          title={status.label}
          style={{
            backgroundColor: status.color,
          }}
        />

        {/* Creator badge */}
        <span className="blob-row-creator" title={`Created by ${blob.created_by}`}>
          {creatorBadge(blob.created_by)}
        </span>

        {/* Filename */}
        <span
          className="blob-row-filename"
          title={blob.filename}
        >
          {blob.filename}
        </span>

        {/* Size */}
        <span className="blob-row-size">
          {formatBytes(blob.size_bytes)}
        </span>

        {/* Actions */}
        <div className="blob-row-actions">
          {canPreview && blob.status === "ready" && (
            <button
              onClick={handleTogglePreview}
              title={previewOpen ? "Hide preview" : "Preview content"}
              aria-label={`${previewOpen ? "Hide" : "Preview"} ${blob.filename}`}
              aria-expanded={previewOpen}
              className="blob-action-btn"
            >
              {"\uD83D\uDC41"}
            </button>
          )}
          {blob.status === "ready" && (
            <>
              <button
                onClick={() => onUseAsInput(blob)}
                title="Use as pipeline input"
                aria-label={`Use ${blob.filename} as input`}
                className="blob-action-btn"
              >
                {"\u25B6"}
              </button>
              <button
                onClick={() => onDownload(blob.id)}
                title="Download"
                aria-label={`Download ${blob.filename}`}
                className="blob-action-btn"
              >
                {"\u2B07"}
              </button>
            </>
          )}
          <button
            onClick={() => onDelete(blob.id)}
            title="Delete"
            aria-label={`Delete ${blob.filename}`}
            className="blob-action-btn"
          >
            {"\u2715"}
          </button>
        </div>
      </div>

      {/* Preview panel */}
      {previewOpen && (
        <div className="blob-row-preview">
          {previewLoading && (
            <div className="blob-row-preview-loading">
              Loading preview...
            </div>
          )}
          {previewError && (
            <div className="blob-row-preview-error">
              {previewError}
            </div>
          )}
          {displayContent !== null && !previewLoading && (
            <pre className="blob-row-preview-pre">
              {displayContent}
              {truncated && (
                <span className="blob-row-preview-truncated">
                  {"\n... (truncated)"}
                </span>
              )}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// src/components/blobs/BlobRow.tsx
import { useMemo, useState } from "react";
import { previewBlobContentSnippet } from "@/api/client";
import { Icon, StructuredJsonPreview, type IconName } from "@/components/ui";
import type { BlobMetadata } from "@/types/api";
import { describeStructuralSummary, summarizeContentStructure } from "@/utils/contentStructure";

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

function creatorBadge(createdBy: BlobMetadata["created_by"]): IconName {
  switch (createdBy) {
    case "user":
      return "user";
    case "assistant":
      return "assistant";
    case "pipeline":
      return "pipeline";
  }
}

function creatorBadgeLabel(createdBy: BlobMetadata["created_by"]): string {
  switch (createdBy) {
    case "user":
      return "Created by user";
    case "assistant":
      return "Created by assistant";
    case "pipeline":
      return "Created by pipeline";
  }
}

function statusIndicator(status: string): {
  color: string;
  label: string;
  icon: IconName;
} {
  switch (status) {
    case "ready":
      return { color: "var(--color-success)", label: "Ready", icon: "status-ready" };
    case "pending":
      return { color: "var(--color-warning)", label: "Pending", icon: "status-pending" };
    case "error":
      return { color: "var(--color-error)", label: "Error", icon: "status-error" };
    default:
      return {
        color: "var(--color-text-muted)",
        label: status,
        icon: "status-unknown",
      };
  }
}

export function BlobRow({ blob, sessionId, onDownload, onDelete, onUseAsInput }: BlobRowProps) {
  const status = statusIndicator(blob.status);
  const creatorLabel = creatorBadgeLabel(blob.created_by);
  const normalizedMimeType = blob.mime_type.split(";")[0].trim().toLowerCase();
  const canPreview = PREVIEWABLE_MIME_TYPES.has(normalizedMimeType);

  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [previewTruncated, setPreviewTruncated] = useState(false);
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
      const preview = await previewBlobContentSnippet(
        sessionId,
        blob.id,
        MAX_PREVIEW_CHARS,
      );
      setPreviewContent(preview.text);
      setPreviewTruncated(preview.truncated);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load preview";
      setPreviewError(message);
    } finally {
      setPreviewLoading(false);
    }
  };

  const truncated =
    previewContent !== null &&
    (previewTruncated || previewContent.length > MAX_PREVIEW_CHARS);
  const displayContent =
    previewContent !== null && truncated
      ? previewContent.slice(0, MAX_PREVIEW_CHARS)
      : previewContent;

  // Structural self-disclosure (T-3): static introspection of the preview
  // content already fetched above — no additional network call. Honest by
  // design: a truncated/ragged/oversized/unparseable body surfaces a plain
  // caveat instead of a guessed row count (see contentStructure.ts).
  const structuralSummary = useMemo(() => {
    if (previewContent === null) return null;
    return summarizeContentStructure(normalizedMimeType, previewContent, { truncated });
  }, [previewContent, truncated, normalizedMimeType]);
  const structuralSummaryLine = structuralSummary
    ? describeStructuralSummary(structuralSummary)
    : null;

  return (
    <div>
      <div
        className="blob-row blob-row-container"
        style={{
          borderBottom: previewOpen ? "none" : "1px solid var(--color-border)",
        }}
      >
        {/* Status indicator: shape icon (non-colour cue) + accessible name. */}
        <span
          className="blob-row-status-dot"
          role="img"
          aria-label={status.label}
          title={status.label}
          style={{
            color: status.color,
          }}
        >
          <Icon name={status.icon} />
        </span>

        {/* Creator badge */}
        <span
          className="blob-row-creator"
          role="img"
          aria-label={creatorLabel}
          title={creatorLabel}
        >
          <Icon name={creatorBadge(blob.created_by)} />
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
              <Icon name="eye" />
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
                <Icon name="play" />
              </button>
              <button
                onClick={() => onDownload(blob.id)}
                title="Download"
                aria-label={`Download ${blob.filename}`}
                className="blob-action-btn"
              >
                <Icon name="download" />
              </button>
            </>
          )}
          <button
            onClick={() => onDelete(blob.id)}
            title="Delete"
            aria-label={`Delete ${blob.filename}`}
            className="blob-action-btn"
          >
            <Icon name="trash" />
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
          {structuralSummary &&
            structuralSummary.format !== "unsupported" &&
            !previewLoading &&
            !previewError && (
              <div className="blob-row-structure" data-testid="blob-row-structure">
                {structuralSummary.caveat && (
                  <p className="blob-row-structure-caveat">{structuralSummary.caveat}</p>
                )}
                {structuralSummaryLine && (
                  <p className="blob-row-structure-summary">{structuralSummaryLine}</p>
                )}
              </div>
            )}
          {displayContent !== null &&
            !previewLoading &&
            normalizedMimeType === "application/json" && (
              <StructuredJsonPreview
                text={displayContent}
                truncated={truncated}
              />
            )}
          {displayContent !== null &&
            !previewLoading &&
            normalizedMimeType !== "application/json" && (
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

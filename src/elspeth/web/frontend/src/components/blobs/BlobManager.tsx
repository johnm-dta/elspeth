// src/components/blobs/BlobManager.tsx
import { useEffect, useRef, useCallback, useMemo } from "react";
import { useBlobStore } from "@/stores/blobStore";
import { useSessionStore } from "@/stores/sessionStore";
import { BlobRow } from "./BlobRow";
import type { BlobMetadata, BlobCategory } from "@/types/api";

interface BlobManagerProps {
  onUseAsInput: (blob: BlobMetadata) => void;
}

/**
 * Categorize a blob into source/sink/other based on who created it.
 * - User uploads → source files (pipeline inputs)
 * - Pipeline outputs → sink files (results)
 * - Assistant-created → other (prompts, templates, config)
 */
function categorizeBlob(blob: BlobMetadata): BlobCategory {
  if (blob.created_by === "user") return "source";
  if (blob.created_by === "pipeline") return "sink";
  return "other";
}

const CATEGORY_LABELS: Record<BlobCategory, string> = {
  source: "Source files",
  sink: "Output files",
  other: "Other files",
};

const CATEGORY_ORDER: BlobCategory[] = ["source", "sink", "other"];

/**
 * Collapsible blob manager panel with categorized folders.
 * Shows session-scoped files grouped by source/output/other
 * with upload, download, delete, and "use as input" actions.
 */
export function BlobManager({ onUseAsInput }: BlobManagerProps) {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const { blobs, isLoading, error, loadBlobs, uploadBlob, deleteBlob, downloadBlob } =
    useBlobStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (activeSessionId) {
      loadBlobs(activeSessionId);
    }
  }, [activeSessionId, loadBlobs]);

  const grouped = useMemo(() => {
    const groups: Record<BlobCategory, BlobMetadata[]> = {
      source: [],
      sink: [],
      other: [],
    };
    for (const blob of blobs) {
      groups[categorizeBlob(blob)].push(blob);
    }
    return groups;
  }, [blobs]);

  const handleUpload = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !activeSessionId) return;
      try {
        await uploadBlob(activeSessionId, file);
      } catch {
        // Error is already in the store
      } finally {
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [activeSessionId, uploadBlob],
  );

  const handleDelete = useCallback(
    (blobId: string) => {
      if (!activeSessionId) return;
      deleteBlob(activeSessionId, blobId);
    },
    [activeSessionId, deleteBlob],
  );

  const handleDownload = useCallback(
    (blobId: string) => {
      if (!activeSessionId) return;
      downloadBlob(activeSessionId, blobId);
    },
    [activeSessionId, downloadBlob],
  );

  if (!activeSessionId) return null;

  return (
    <div
      className="blob-manager blob-manager-container"
    >
      {/* Header */}
      <div className="blob-manager-header">
        <span className="blob-manager-title">
          Files ({blobs.length})
        </span>
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn blob-manager-upload-btn"
          aria-label="Upload file"
        >
          + Upload
        </button>
        <input
          ref={fileInputRef}
          type="file"
          onChange={handleUpload}
          style={{ display: "none" }}
          aria-hidden="true"
          tabIndex={-1}
        />
      </div>

      {/* Error */}
      {error && (
        <div
          role="alert"
          className="blob-manager-error"
        >
          {error}
        </div>
      )}

      {/* Categorized file list */}
      <div className="blob-manager-list">
        {isLoading ? (
          <div className="blob-manager-loading">
            Loading...
          </div>
        ) : blobs.length === 0 ? (
          <div className="blob-manager-empty">
            No files yet. Upload a file to get started.
          </div>
        ) : (
          CATEGORY_ORDER.map((category) => {
            const categoryBlobs = grouped[category];
            if (categoryBlobs.length === 0) return null;
            return (
              <div key={category}>
                <div className="blob-manager-category-header">
                  {CATEGORY_LABELS[category]}
                </div>
                {categoryBlobs.map((blob) => (
                  <BlobRow
                    key={blob.id}
                    blob={blob}
                    sessionId={activeSessionId}
                    onDownload={handleDownload}
                    onDelete={handleDelete}
                    onUseAsInput={onUseAsInput}
                  />
                ))}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

// ============================================================================
// RunOutputsPanel
//
// Per-run audit-evidence inventory of every sink-write artefact, surfaced
// through the GET /api/runs/{rid}/outputs manifest endpoint. Each row
// renders the artefact's type, basename/URI, size, and short content
// hash, plus (for downloadable file artefacts):
//
//   * a Download anchor that hits /content
//   * an Expand-preview toggle that lazy-fetches /preview
//
// Distinct from the diagnostics-panel artifact list (capped at 20 for
// operator-UI pacing): this panel is the unbounded canonical view.
//
// Non-file artefacts (DatabaseSink, Dataverse webhook, Azure blob
// without filesystem mirror) are listed as metadata-only — honest
// evidence the run produced them, no fake action buttons.
// ============================================================================

import { useEffect, useRef, useState } from "react";
import {
  downloadRunOutputContent,
  fetchRunOutputPreview,
  fetchRunOutputs,
} from "@/api/client";
import {
  PreviewTable,
  StructuredJsonPreview,
  type PreviewTableModel,
} from "@/components/ui";
import { absoluteTime } from "@/utils/time";
import type {
  ApiError,
  RunOutputArtifact,
  RunOutputArtifactPreview,
  RunOutputsResponse,
} from "@/types/index";

function isApiError(value: unknown): value is ApiError {
  return (
    typeof value === "object" &&
    value !== null &&
    "status" in value &&
    "detail" in value
  );
}

function formatError(value: unknown, fallback: string): string {
  if (isApiError(value)) {
    return value.detail || fallback;
  }
  if (value instanceof Error) {
    return value.message;
  }
  return fallback;
}

/**
 * Trigger a browser download from an in-memory Blob. Uses a
 * synthetic anchor + object URL because the `/content` endpoint
 * requires Authorization headers — a plain `<a href>` would 401 on
 * top-level navigation. Mirrors `blobStore.downloadBlob`.
 */
function triggerBrowserDownload(data: Blob, filename: string): void {
  const url = URL.createObjectURL(data);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

interface RunOutputsPanelProps {
  runId: string;
}

interface PreviewState {
  status: "loading" | "loaded" | "error" | "purged";
  preview?: RunOutputArtifactPreview;
  error?: string;
}

const HASH_DISPLAY_LENGTH = 12;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MiB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GiB`;
}

function basenameOf(pathOrUri: string): string {
  // Strip file:// prefix, then take the last segment.
  const stripped = pathOrUri.startsWith("file://") ? pathOrUri.slice(7) : pathOrUri;
  const idx = stripped.lastIndexOf("/");
  return idx === -1 ? stripped : stripped.slice(idx + 1);
}

/**
 * "blob" and "payload" are elspeth's own opaque internal storage — a
 * content hash or blob id as a filename, meaningless to an operator.
 * Server-classified (see storage_kind on the wire type) against the
 * REAL storage layouts, not guessed from path shape: replaces a former
 * frontend regex heuristic that matched a layout no repo code actually
 * produced (elspeth-52af16f9ae).
 */
function isInternalStoragePath(artifact: RunOutputArtifact): boolean {
  return (
    artifact.storage_kind === "blob" || artifact.storage_kind === "payload"
  );
}

function artifactDisplayName(artifact: RunOutputArtifact): string {
  if (!isFileArtifact(artifact)) {
    return artifact.path_or_uri;
  }
  if (isInternalStoragePath(artifact)) {
    return artifact.sink_node_id || "artifact";
  }
  return basenameOf(artifact.path_or_uri);
}

function artifactDisplayTitle(artifact: RunOutputArtifact): string {
  if (isFileArtifact(artifact) && isInternalStoragePath(artifact)) {
    return `Recorded artifact for ${artifact.sink_node_id || "artifact"}`;
  }
  return artifact.path_or_uri;
}

function isFileArtifact(artifact: RunOutputArtifact): boolean {
  // The backend reports artifact_type="file" for filesystem outputs;
  // anything else (database, webhook) is non-file evidence.
  return artifact.artifact_type === "file";
}

export function RunOutputsPanel({ runId }: RunOutputsPanelProps) {
  const [manifest, setManifest] = useState<RunOutputsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [previewByArtifactId, setPreviewByArtifactId] = useState<
    Record<string, PreviewState>
  >({});
  // Tracks which artifact rows the operator has expanded for preview.
  const [expandedArtifactIds, setExpandedArtifactIds] = useState<Set<string>>(
    new Set(),
  );
  const activeRunIdRef = useRef(runId);
  const manifestRequestSeqRef = useRef(0);
  const previewRunGenerationRef = useRef(0);

  const loadManifest = async (
    targetRunId: string,
    options: { clearRunScopedState?: boolean } = {},
  ) => {
    const requestSeq = ++manifestRequestSeqRef.current;
    activeRunIdRef.current = targetRunId;
    if (options.clearRunScopedState) {
      previewRunGenerationRef.current += 1;
      setManifest(null);
      setPreviewByArtifactId({});
      setExpandedArtifactIds(new Set());
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetchRunOutputs(targetRunId);
      if (
        requestSeq !== manifestRequestSeqRef.current ||
        targetRunId !== activeRunIdRef.current
      ) {
        return;
      }
      setManifest(response);
    } catch (err) {
      if (
        requestSeq !== manifestRequestSeqRef.current ||
        targetRunId !== activeRunIdRef.current
      ) {
        return;
      }
      setError(formatError(err, "Failed to load outputs"));
    } finally {
      if (
        requestSeq === manifestRequestSeqRef.current &&
        targetRunId === activeRunIdRef.current
      ) {
        setIsLoading(false);
      }
    }
  };

  const handleDownload = async (artifact: RunOutputArtifact) => {
    try {
      const { data, filename } = await downloadRunOutputContent(
        runId,
        artifact.artifact_id,
      );
      triggerBrowserDownload(data, filename);
    } catch (err) {
      // Surface the failure inline against the manifest banner; less
      // intrusive than a toast, and the operator can click Refresh.
      setError(formatError(err, "Download failed"));
    }
  };

  useEffect(() => {
    void loadManifest(runId, { clearRunScopedState: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  const togglePreview = async (artifact: RunOutputArtifact) => {
    const next = new Set(expandedArtifactIds);
    if (next.has(artifact.artifact_id)) {
      next.delete(artifact.artifact_id);
      setExpandedArtifactIds(next);
      return;
    }
    next.add(artifact.artifact_id);
    setExpandedArtifactIds(next);

    // Skip refetch if already loaded.
    if (previewByArtifactId[artifact.artifact_id]?.status === "loaded") {
      return;
    }
    setPreviewByArtifactId((prev) => ({
      ...prev,
      [artifact.artifact_id]: { status: "loading" },
    }));
    const targetRunId = runId;
    const previewRunGeneration = previewRunGenerationRef.current;
    try {
      const preview = await fetchRunOutputPreview(targetRunId, artifact.artifact_id);
      if (
        previewRunGeneration !== previewRunGenerationRef.current ||
        targetRunId !== activeRunIdRef.current
      ) {
        return;
      }
      setPreviewByArtifactId((prev) => ({
        ...prev,
        [artifact.artifact_id]: { status: "loaded", preview },
      }));
    } catch (err) {
      if (
        previewRunGeneration !== previewRunGenerationRef.current ||
        targetRunId !== activeRunIdRef.current
      ) {
        return;
      }
      // The preview endpoint returns 410 + error_type=artifact_purged_or_moved
      // when the file existed at manifest time but is gone now (purge race).
      // We surface this as a per-row "no longer available on disk" state
      // rather than a generic toast. Match on the structured error_type
      // field rather than string-matching the human-readable detail —
      // detail wording can change without breaking the contract.
      const isPurgedRace =
        isApiError(err) && err.error_type === "artifact_purged_or_moved";
      setPreviewByArtifactId((prev) => ({
        ...prev,
        [artifact.artifact_id]: isPurgedRace
          ? { status: "purged" }
          : {
              status: "error",
              error: formatError(err, "Preview failed"),
            },
      }));
    }
  };

  return (
    <section
      aria-label="Run outputs"
      className="run-outputs-panel"
    >
      <div className="run-outputs-panel-header">
        <span className="run-outputs-panel-title">Outputs</span>
        <button
          type="button"
          className="btn-compact"
          onClick={() => void loadManifest(runId)}
          disabled={isLoading}
        >
          {isLoading ? "Loading…" : "Refresh"}
        </button>
      </div>

      {error && (
        <div role="alert" className="run-outputs-panel-error">
          {error}
        </div>
      )}

      {!manifest && !error && isLoading && (
        <div className="run-outputs-panel-muted">Loading outputs…</div>
      )}

      {manifest && manifest.artifacts.length === 0 && !isLoading && (
        <div className="run-outputs-panel-muted">
          This run produced no outputs.
        </div>
      )}

      {manifest && manifest.artifacts.length > 0 && (
        <ul className="run-output-artifact-list">
          {manifest.artifacts.map((artifact) => {
            const expanded = expandedArtifactIds.has(artifact.artifact_id);
            const previewState = previewByArtifactId[artifact.artifact_id];
            const isFile = isFileArtifact(artifact);
            return (
              <li
                key={artifact.artifact_id}
                className="run-output-artifact-item"
              >
                <ArtifactRow
                  artifact={artifact}
                  expanded={expanded}
                  onTogglePreview={() => void togglePreview(artifact)}
                  onDownload={() => void handleDownload(artifact)}
                />
                {isFile && expanded && (
                  <ArtifactPreviewView
                    previewState={previewState}
                    onDownload={() => void handleDownload(artifact)}
                  />
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

// ── ArtifactRow ─────────────────────────────────────────────────────────────

interface ArtifactRowProps {
  artifact: RunOutputArtifact;
  expanded: boolean;
  onTogglePreview: () => void;
  onDownload: () => void;
}

function ArtifactRow({ artifact, expanded, onTogglePreview, onDownload }: ArtifactRowProps) {
  const isFile = isFileArtifact(artifact);
  const displayName = artifactDisplayName(artifact);
  const shortHash = artifact.content_hash.slice(0, HASH_DISPLAY_LENGTH);
  const displayTitle = artifactDisplayTitle(artifact);

  return (
    <div className="run-output-artifact-row">
      <span className="run-output-artifact-kind">
        {artifact.artifact_type}
      </span>
      <span
        className="run-output-artifact-name"
        title={displayTitle}
      >
        {displayName}
      </span>
      <span className="run-output-artifact-meta">
        {formatBytes(artifact.size_bytes)}
      </span>
      {/* Run timestamp — same class family as file size so the two pieces
          of inline per-artifact metadata read as siblings. Tooltip carries the
          unmodified wire-format timestamp (with timezone marker) for
          anyone diffing against the audit DB directly. */}
      <span
        className="run-output-artifact-meta run-output-artifact-time"
        title={artifact.created_at}
      >
        {absoluteTime(artifact.created_at)}
      </span>
      <span
        className="run-output-artifact-hash"
        title={`SHA-256 ${artifact.content_hash}`}
      >
        {shortHash}…
      </span>
      {isFile && (
        <ArtifactActions
          artifact={artifact}
          expanded={expanded}
          onTogglePreview={onTogglePreview}
          onDownload={onDownload}
        />
      )}
    </div>
  );
}

interface ArtifactActionsProps {
  artifact: RunOutputArtifact;
  expanded: boolean;
  onTogglePreview: () => void;
  onDownload: () => void;
}

function ArtifactActions({ artifact, expanded, onTogglePreview, onDownload }: ArtifactActionsProps) {
  if (!artifact.exists_now) {
    return (
      <span
        className="run-output-artifact-unavailable"
        title={artifactDisplayTitle(artifact)}
      >
        no longer available on disk
      </span>
    );
  }
  if (!artifact.downloadable) {
    // File exists on disk but is outside the sink-allowlist that the
    // /content endpoint enforces. The audit row is honest evidence;
    // the download is refused for defence-in-depth.
    return (
      <span
        className="run-output-artifact-unavailable"
        title="The recorded path is outside the sink output allowlist; the server refuses to serve its bytes."
      >
        outside allowed sink directories
      </span>
    );
  }
  return (
    <span className="run-output-artifact-actions">
      <button
        type="button"
        className="btn-compact"
        onClick={onTogglePreview}
        aria-expanded={expanded}
      >
        {expanded ? "Hide preview" : "Preview"}
      </button>
      <button
        type="button"
        className="btn-compact"
        onClick={onDownload}
      >
        Download
      </button>
    </span>
  );
}

// ── ArtifactPreviewView ────────────────────────────────────────────────────

interface ArtifactPreviewViewProps {
  previewState?: PreviewState;
  onDownload: () => void;
}

function ArtifactPreviewView({
  previewState,
  onDownload,
}: ArtifactPreviewViewProps) {
  if (!previewState || previewState.status === "loading") {
    return (
      <div style={{ marginTop: 6, color: "var(--color-text-muted)" }}>Loading preview…</div>
    );
  }
  if (previewState.status === "purged") {
    return (
      <div style={{ marginTop: 6, color: "var(--color-text-muted)", fontStyle: "italic" }}>
        File is no longer available on disk (purged or moved between manifest fetch and preview).
      </div>
    );
  }
  if (previewState.status === "error") {
    return (
      <div role="alert" style={{ marginTop: 6, color: "var(--color-error)" }}>
        {previewState.error ?? "Preview failed"}
      </div>
    );
  }
  const preview = previewState.preview;
  if (!preview) return null;
  if (preview.content_type === "binary") {
    return (
      <div style={{ marginTop: 6, color: "var(--color-text-muted)", fontStyle: "italic" }}>
        Binary file — no inline preview available. Use the Download button to inspect.
      </div>
    );
  }
  return (
    <div style={{ marginTop: 6 }}>
      {preview.content_type === "json" ? (
        <StructuredJsonPreview
          text={preview.preview_text}
          truncated={preview.truncated}
        />
      ) : preview.content_type === "csv" || preview.content_type === "jsonl" ? (
        <TabularPreview text={preview.preview_text} contentType={preview.content_type} />
      ) : (
        <pre
          style={{
            maxHeight: 300,
            overflow: "auto",
            backgroundColor: "var(--color-surface-hover)",
            padding: 8,
            borderRadius: "var(--radius-sm)",
            margin: 0,
            fontSize: 11,
            whiteSpace: "pre-wrap",
            overflowWrap: "anywhere",
          }}
        >
          {preview.preview_text}
        </pre>
      )}
      {preview.truncated && (
        <div
          style={{ marginTop: 4, color: "var(--color-text-muted)", fontSize: 11 }}
        >
          Preview truncated
          {preview.row_count_preview != null && ` to ${preview.row_count_preview} rows`}
          {" — "}
          <button
            type="button"
            onClick={onDownload}
            style={{
              background: "none",
              border: "none",
              padding: 0,
              color: "var(--color-link)",
              textDecoration: "underline",
              cursor: "pointer",
              font: "inherit",
            }}
          >
            download for full file
          </button>
          {" "}({formatBytes(preview.total_size_bytes)} total).
        </div>
      )}
    </div>
  );
}

// ── TabularPreview ─────────────────────────────────────────────────────────

interface TabularPreviewProps {
  text: string;
  contentType: "csv" | "jsonl";
}

/**
 * Builds a headers+rows model for the shared PreviewTable out of raw
 * csv/jsonl preview text. Tolerant of malformed rows — this is
 * deliberately not a full CSV parser (no quoted-comma handling); preview
 * is best-effort, not a data-loading path.
 *
 * Two content types feed this:
 *   * csv  — backend tags both `.csv` and `.tsv` files as content_type
 *            "csv" (see web/execution/preview._CSV_EXTENSIONS), so we
 *            sniff the first line for tab vs comma rather than
 *            hardcoding `,`. Without this, TSV rows collapse into a
 *            single column. The first line is the real header row.
 *   * jsonl — each line is a JSON object that must NOT be split on
 *             commas (that fragments the JSON across cells). Each line
 *             is rendered as a single-column row under one synthetic
 *             "value" header — jsonl has no column structure of its own,
 *             but every PreviewTable still gets a real th scope="col"
 *             header cell rather than the old bold-td fake header
 *             (elspeth-611a05668e).
 */
function buildTabularPreviewModel(
  text: string,
  contentType: "csv" | "jsonl",
): PreviewTableModel | null {
  const lines = text.split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) {
    return null;
  }
  if (contentType === "jsonl") {
    return {
      headers: ["value"],
      rows: lines.map((line) => [line]),
    };
  }
  const firstLine = lines[0];
  const tabCount = (firstLine.match(/\t/g) ?? []).length;
  const commaCount = (firstLine.match(/,/g) ?? []).length;
  const delimiter = tabCount > commaCount ? "\t" : ",";
  const [headerRow, ...bodyRows] = lines.map((line) => line.split(delimiter));
  const columnCount =
    bodyRows.length === 0
      ? headerRow.length
      : Math.max(headerRow.length, ...bodyRows.map((row) => row.length));
  const headers = Array.from({ length: columnCount }, (_, i) => headerRow[i] ?? "");
  const rows = bodyRows.map((row) =>
    Array.from({ length: columnCount }, (_, i) => row[i] ?? ""),
  );
  return { headers, rows };
}

function TabularPreview({ text, contentType }: TabularPreviewProps) {
  const table = buildTabularPreviewModel(text, contentType);
  if (!table) {
    return null;
  }
  return <PreviewTable table={table} />;
}

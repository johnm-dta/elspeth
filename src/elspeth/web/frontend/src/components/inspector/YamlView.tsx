// ============================================================================
// YamlView
//
// Read-only syntax-highlighted YAML display using prism-react-renderer.
// The YAML is fetched from GET /api/sessions/{id}/state/yaml on version change.
//
// Features:
// - Syntax highlighting with line numbers
// - Copy-to-clipboard button
// - Download button for YAML export
// - Theme-aware (light/dark)
// - Explicit blocked/error state when YAML export fails
//
// Empty state when no composition state exists.
// ============================================================================

import { useState, useEffect, useCallback, useMemo } from "react";
import { Highlight, themes } from "prism-react-renderer";
import { useSessionStore } from "@/stores/sessionStore";
import { useTheme } from "@/hooks/useTheme";
import * as api from "@/api/client";
import type { ApiError } from "@/types/index";

interface YamlFetchError {
  title: string;
  detail: string;
}

function describeYamlFetchError(error: unknown): YamlFetchError {
  const apiError = error as Partial<ApiError>;
  const detail =
    typeof apiError.detail === "string" && apiError.detail.trim().length > 0
      ? apiError.detail
      : "Please try again.";

  if (apiError.status === 409) {
    return {
      title: "YAML export is blocked by validation errors.",
      detail,
    };
  }

  return {
    title: "Failed to load YAML.",
    detail,
  };
}

export function YamlView() {
  const compositionState = useSessionStore((s) => s.compositionState);
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const compositionProposals = useSessionStore((s) => s.compositionProposals);
  const proposalActionPendingIds = useSessionStore(
    (s) => s.proposalActionPendingIds,
  );
  const staleProposalIds = useSessionStore((s) => s.staleProposalIds);
  const acceptProposal = useSessionStore((s) => s.acceptProposal);
  const rejectProposal = useSessionStore((s) => s.rejectProposal);
  const pendingYamlProposals = useMemo(
    () =>
      compositionProposals.filter(
        (proposal) =>
          proposal.status === "pending" && proposal.affects.includes("yaml"),
      ),
    [compositionProposals],
  );
  const [yaml, setYaml] = useState<string | null>(null);
  const [yamlError, setYamlError] = useState<YamlFetchError | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const { resolvedTheme } = useTheme();

  // Fetch YAML from the backend whenever composition state version changes
  const version = compositionState?.version ?? null;

  useEffect(() => {
    if (!activeSessionId || version === null) {
      setYaml(null);
      setYamlError(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setYamlError(null);

    api
      .fetchYaml(activeSessionId)
      .then(({ yaml: text }) => {
        if (!cancelled) {
          setYaml(text);
          setYamlError(null);
          setIsLoading(false);
        }
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setYaml(null);
          setYamlError(describeYamlFetchError(error));
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId, version]);

  const handleCopy = useCallback(async () => {
    if (!yaml) return;
    try {
      await navigator.clipboard.writeText(yaml);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API may fail in some contexts (e.g. insecure origin)
    }
  }, [yaml]);

  const handleDownload = useCallback(() => {
    if (!yaml) return;
    const blob = new Blob([yaml], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeline-v${compositionState?.version ?? 1}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [yaml, compositionState?.version]);

  const pendingYamlProposal = pendingYamlProposals[0] ?? null;
  const pendingYamlProposalIsBusy =
    pendingYamlProposal !== null &&
    proposalActionPendingIds.includes(pendingYamlProposal.id);
  const pendingYamlProposalIsStale =
    pendingYamlProposal !== null && staleProposalIds.includes(pendingYamlProposal.id);
  const pendingYamlProposalPanel =
    pendingYamlProposal === null ? null : (
      <div className="yaml-pending-summary" role="note">
        <span>Pending YAML change: {pendingYamlProposal.summary}</span>
        {pendingYamlProposalIsStale ? (
          <span className="tool-call-stale">
            Stale proposal. Ask the composer to rebase or revise this proposal.
          </span>
        ) : (
          <span className="tool-call-actions">
            <button
              type="button"
              className="btn btn-primary btn-small"
              disabled={pendingYamlProposalIsBusy}
              onClick={() => void acceptProposal(pendingYamlProposal.id)}
              aria-label={`Accept YAML proposal: ${pendingYamlProposal.summary}`}
            >
              Accept
            </button>
            <button
              type="button"
              className="btn btn-danger btn-small"
              disabled={pendingYamlProposalIsBusy}
              onClick={() => void rejectProposal(pendingYamlProposal.id)}
              aria-label={`Reject YAML proposal: ${pendingYamlProposal.summary}`}
            >
              Reject
            </button>
          </span>
        )}
      </div>
    );

  // Empty state
  if (!compositionState || version === null) {
    return (
      <div className="empty-state">
        YAML will appear here once your pipeline has components.
      </div>
    );
  }

  // Loading state
  if (isLoading && !yaml) {
    return (
      <div
        role="status"
        aria-live="polite"
        className="yaml-loading"
      >
        Loading YAML...
      </div>
    );
  }

  if (yamlError && !yaml) {
    return (
      <div className="yaml-view">
        {pendingYamlProposalPanel}
        <div
          role="alert"
          className="validation-banner validation-banner-fail"
        >
          <div className="validation-banner-content">
            <div className="validation-banner-summary">{yamlError.title}</div>
            <div>{yamlError.detail}</div>
          </div>
        </div>
      </div>
    );
  }

  // No YAML returned (unexpected empty response)
  if (!yaml) {
    return (
      <div className="empty-state">
        YAML will appear here once your pipeline has components.
      </div>
    );
  }

  const highlightTheme = resolvedTheme === "dark" ? themes.vsDark : themes.vsLight;

  return (
    <div className="yaml-view">
      {pendingYamlProposalPanel}

      {/* Toolbar: Copy + Download buttons */}
      <div className="yaml-view-toolbar">
        <button
          onClick={handleCopy}
          aria-label={copied ? "Copied to clipboard" : "Copy YAML to clipboard"}
          className="btn yaml-toolbar-btn"
          data-copied={copied ? "true" : "false"}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
        <button
          onClick={handleDownload}
          aria-label="Download YAML file"
          className="btn yaml-toolbar-btn"
        >
          Download
        </button>
      </div>

      {/* Syntax-highlighted YAML */}
      <div className="yaml-view-content">
        <Highlight theme={highlightTheme} code={yaml} language="yaml">
          {({ tokens, getLineProps, getTokenProps }) => (
            <pre className="yaml-view-pre">
              {tokens.map((line, i) => (
                <div key={i} {...getLineProps({ line })} className="yaml-view-line">
                  <span className="yaml-view-line-number" aria-hidden="true">{i + 1}</span>
                  <span className="yaml-view-line-content">
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token })} />
                    ))}
                  </span>
                </div>
              ))}
            </pre>
          )}
        </Highlight>
      </div>
    </div>
  );
}

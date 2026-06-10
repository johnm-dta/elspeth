// ============================================================================
// YamlView
//
// Session-store-coupled YAML view for the composer surface.
//
// The YAML is fetched from GET /api/sessions/{id}/state/yaml on version
// change. Once loaded, display is delegated to `YamlDisplay`, which
// owns the syntax-highlight + Copy + Download chrome (Phase 6B FIX-C
// extracted that pure primitive so the SharedInspectView can render a
// frozen YAML blob without the fetch/proposal machinery here).
//
// YamlView retains:
// - the fetch effect (on composition-version change)
// - the empty/loading/error states
// - the pending-YAML-proposal panel (with Accept/Reject buttons)
// - the 409 validation-blocked alert
//
// SharedInspectView mounts `<YamlDisplay yaml={...} />` directly with
// the wire YAML, bypassing all of the above.
// ============================================================================

import { useState, useEffect, useMemo } from "react";
import { useSessionStore } from "@/stores/sessionStore";
import * as api from "@/api/client";
import type { ApiError } from "@/types/index";
import { YamlDisplay } from "./YamlDisplay";
import { ConfirmDialog } from "@/components/common/ConfirmDialog";
import { hasCompositionContent } from "@/utils/compositionState";

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
  const [rejectConfirmId, setRejectConfirmId] = useState<string | null>(null);

  // Fetch YAML from the backend whenever composition state version changes
  const version = compositionState?.version ?? null;
  const hasPipelineContent = hasCompositionContent(compositionState);

  useEffect(() => {
    if (!activeSessionId || version === null || !hasPipelineContent) {
      setYaml(null);
      setYamlError(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;
    setIsLoading(true);
    setYaml(null);
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
  }, [activeSessionId, version, hasPipelineContent]);

  const pendingYamlProposal = pendingYamlProposals[0] ?? null;
  const pendingYamlProposalIsBusy =
    pendingYamlProposal !== null &&
    proposalActionPendingIds.includes(pendingYamlProposal.id);
  const pendingYamlProposalIsStale =
    pendingYamlProposal !== null && staleProposalIds.includes(pendingYamlProposal.id);
  const pendingYamlProposalPanel =
    pendingYamlProposal === null ? null : (
      <>
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
                onClick={() => setRejectConfirmId(pendingYamlProposal.id)}
                aria-label={`Reject YAML proposal: ${pendingYamlProposal.summary}`}
              >
                Reject
              </button>
            </span>
          )}
        </div>
        {rejectConfirmId !== null && (
          <ConfirmDialog
            title="Reject this YAML proposal?"
            message="The composer's proposed change will be discarded. You can ask the composer to revise the proposal afterwards."
            confirmLabel="Reject proposal"
            cancelLabel="Keep open"
            variant="danger"
            onConfirm={() => {
              void rejectProposal(rejectConfirmId);
              setRejectConfirmId(null);
            }}
            onCancel={() => setRejectConfirmId(null)}
          />
        )}
      </>
    );

  // Empty state
  if (!compositionState || version === null || !hasPipelineContent) {
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

  return (
    <div className="yaml-view">
      {pendingYamlProposalPanel}
      <YamlDisplay
        yaml={yaml}
        filename={`pipeline-v${compositionState?.version ?? 1}.yaml`}
      />
    </div>
  );
}

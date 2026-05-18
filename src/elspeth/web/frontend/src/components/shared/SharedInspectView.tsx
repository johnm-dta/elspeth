/**
 * SharedInspectView — Phase 6B Task 8.
 *
 * Read-only inspect view rendered when the URL hash is in the form
 * `#/shared/{token}`. Mounted by App.tsx as a top-level branch that
 * short-circuits the regular composer UI.
 *
 * The token is a CAPABILITY, not an authenticator — the reviewer must
 * still be logged in. AuthGuard wraps this view so unauthenticated
 * users hit the login flow; on successful auth they land back here
 * via the persisted hash (the AuthGuard preserves the hash through
 * the login redirect).
 *
 * The view renders:
 *
 * * Pipeline metadata (name, description).
 * * The Phase 2 / Phase 18 six-row Audit Readiness panel, read-only,
 *   served verbatim from the frozen blob (no live re-fetch).
 * * The rendered YAML (read-only, copy-only).
 * * The composition's nodes/edges/outputs as a structural summary.
 *
 * The composer chat panel, run controls, and edit affordances are
 * deliberately ABSENT — this is an inspect-only view, not a fork
 * surface (multi-user collaborative editing is out of scope per
 * plan 19a §"Scope boundaries").
 *
 * Error handling:
 *
 * * 401 (tampered/expired token) → "This share link is no longer valid"
 *   with a "Return to my workspace" CTA.
 * * 404 (blob reaped) → "This shared snapshot is no longer available"
 *   with a "Ask the sender for a fresh link" message.
 * * Other / network → "Couldn't load the shared pipeline" with retry.
 */

import { useEffect, useState } from "react";

import { fetchSharedInspect } from "@/api/shareableReviews";
import type { ApiError, SharedInspectResponse } from "@/types/api";

interface SharedInspectViewProps {
  token: string;
}

type LoadState =
  | { kind: "loading" }
  | { kind: "loaded"; response: SharedInspectResponse }
  | { kind: "error"; status: number | null; message: string };

function _isApiError(value: unknown): value is ApiError {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { status?: unknown }).status === "number"
  );
}

function _classifyError(exc: unknown): { status: number | null; message: string } {
  if (_isApiError(exc)) {
    if (exc.status === 401) {
      return {
        status: 401,
        message:
          "This share link is no longer valid. The link may have expired or " +
          "been revoked. Ask the sender for a fresh link.",
      };
    }
    if (exc.status === 404) {
      return {
        status: 404,
        message:
          "This shared snapshot is no longer available. The original " +
          "operator may need to re-share — ask the sender for a fresh link.",
      };
    }
    return {
      status: exc.status,
      message: exc.detail ?? "Couldn't load the shared pipeline.",
    };
  }
  if (exc instanceof Error) {
    return { status: null, message: exc.message };
  }
  return { status: null, message: "Couldn't load the shared pipeline." };
}

function _returnToWorkspaceUrl(): string {
  return `${window.location.origin}${window.location.pathname}`;
}

export function SharedInspectView({ token }: SharedInspectViewProps): JSX.Element {
  const [state, setState] = useState<LoadState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setState({ kind: "loading" });
    fetchSharedInspect(token, controller.signal)
      .then((response) => {
        if (cancelled) return;
        setState({ kind: "loaded", response });
      })
      .catch((exc: unknown) => {
        if (cancelled) return;
        const { status, message } = _classifyError(exc);
        setState({ kind: "error", status, message });
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [token]);

  if (state.kind === "loading") {
    return (
      <main
        role="main"
        className="shared-inspect-view"
        data-testid="shared-inspect-loading"
        aria-busy="true"
      >
        <p>Loading shared pipeline…</p>
      </main>
    );
  }

  if (state.kind === "error") {
    return (
      <main
        role="main"
        className="shared-inspect-view shared-inspect-view--error"
        data-testid="shared-inspect-error"
      >
        <h1>Shared link unavailable</h1>
        <p role="alert">{state.message}</p>
        <p>
          <a href={_returnToWorkspaceUrl()} data-testid="shared-inspect-return-link">
            Return to my workspace
          </a>
        </p>
      </main>
    );
  }

  const { response } = state;
  const pipelineName =
    typeof response.pipeline_metadata.name === "string"
      ? response.pipeline_metadata.name
      : "Untitled pipeline";
  const pipelineDescription =
    typeof response.pipeline_metadata.description === "string"
      ? response.pipeline_metadata.description
      : "";

  return (
    <main
      role="main"
      className="shared-inspect-view shared-inspect-view--loaded"
      data-testid="shared-inspect-loaded"
    >
      <header>
        <p className="shared-inspect-banner" role="status">
          Read-only shared view. Shared by{" "}
          <strong>{response.created_by_user_id}</strong> on{" "}
          <time dateTime={response.created_at}>
            {new Date(response.created_at).toLocaleString()}
          </time>
          ; expires on{" "}
          <time dateTime={response.expires_at}>
            {new Date(response.expires_at).toLocaleString()}
          </time>
          .
        </p>
        <h1 data-testid="shared-inspect-pipeline-name">{pipelineName}</h1>
        {pipelineDescription !== "" && (
          <p data-testid="shared-inspect-pipeline-description">{pipelineDescription}</p>
        )}
      </header>

      <section aria-label="Audit readiness panel (read-only)" data-testid="shared-inspect-audit-readiness">
        <h2>Audit readiness</h2>
        <p>
          The owner reviewed this readiness panel at the moment of marking
          the pipeline ready for review. The values below are frozen — they
          reflect the owner's mark-time view, not the live state.
        </p>
        <table className="shared-inspect-readiness-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Status</th>
              <th>Summary</th>
            </tr>
          </thead>
          <tbody>
            {response.audit_readiness.rows.map((row) => (
              <tr key={row.id} data-testid={`shared-inspect-readiness-row-${row.id}`}>
                <td>{row.label}</td>
                <td>{row.status}</td>
                <td>{row.summary}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section aria-label="Pipeline YAML (read-only)" data-testid="shared-inspect-yaml">
        <h2>Pipeline YAML</h2>
        <pre className="shared-inspect-yaml-source">
          <code>{response.yaml}</code>
        </pre>
      </section>

      <footer>
        <p>
          <a href={_returnToWorkspaceUrl()} data-testid="shared-inspect-return-link">
            Return to my workspace
          </a>
        </p>
      </footer>
    </main>
  );
}
